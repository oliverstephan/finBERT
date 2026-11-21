"""Offline tests for the optional training module, using a tiny randomly initialised BERT."""

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import torch  # noqa: E402
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast  # noqa: E402

from finbert.predict import FinBertPredictor  # noqa: E402
from finbert.training import Config, FinBert  # noqa: E402

LABELS = ["positive", "negative", "neutral"]
WORDS = "profit rose fell loss growth decline revenue the company reported results shares".split()
SENTENCES = [
    ("profit rose", "positive"),
    ("revenue growth", "positive"),
    ("shares fell", "negative"),
    ("loss decline", "negative"),
    ("the company reported results", "neutral"),
    ("the results", "neutral"),
] * 2


@pytest.fixture
def tiny_setup(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    tok_dir = tmp_path / "tokenizer"
    BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True).save_pretrained(tok_dir)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for phase in ("train", "validation", "test"):
        rows = ["\ttext\tlabel"] + [f"{i}\t{t}\t{label}" for i, (t, label) in enumerate(SENTENCES)]
        (data_dir / f"{phase}.csv").write_text("\n".join(rows) + "\n")

    torch.manual_seed(0)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=5 + len(WORDS),
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            num_labels=3,
            id2label=dict(enumerate(LABELS)),
            label2id={label: i for i, label in enumerate(LABELS)},
        )
    )
    return tmp_path, tok_dir, data_dir, model


def make_finbert(tmp_path, tok_dir, data_dir, model, **overrides):
    config = Config(
        data_dir=str(data_dir),
        bert_model=model,
        model_dir=tmp_path / "model_out",
        max_seq_length=8,
        train_batch_size=4,
        num_train_epochs=2,
        learning_rate=1e-3,
        no_cuda=True,
        encoder_no=2,
        base_model=str(tok_dir),
        **overrides,
    )
    fb = FinBert(config)
    fb.prepare_model(label_list=LABELS)
    return fb


def test_gradient_accumulation_matches_full_batch(tiny_setup):
    """Regression test: with accumulation the original code never called backward()."""
    tmp_path, tok_dir, data_dir, model = tiny_setup
    fb = make_finbert(tmp_path, tok_dir, data_dir, model, gradient_accumulation_steps=2)
    model.eval()  # no dropout, so both passes are deterministic

    # The loader yields micro-batches of 2 (effective batch 4 / 2 accumulation steps); glue two together.
    micro_batches = list(fb.get_loader(fb.get_data("train")[:4], "eval"))
    assert [len(b[0]) for b in micro_batches] == [2, 2]
    input_ids, attention_mask, token_type_ids, label_ids, _ = (torch.cat(t) for t in zip(*micro_batches, strict=True))

    def loss_on(sl):
        logits = model(input_ids[sl], attention_mask=attention_mask[sl], token_type_ids=token_type_ids[sl]).logits
        return fb._loss(logits, label_ids[sl])

    model.zero_grad()
    loss_on(slice(0, 4)).backward()
    full = model.classifier.weight.grad.clone()

    model.zero_grad()
    fb._backward(loss_on(slice(0, 2)))
    fb._backward(loss_on(slice(2, 4)))
    accumulated = model.classifier.weight.grad

    assert accumulated.abs().sum() > 0
    torch.testing.assert_close(accumulated, full)


@pytest.mark.parametrize("accumulation", [1, 2])
def test_train_end_to_end_and_reload_with_pipeline(tiny_setup, accumulation):
    tmp_path, tok_dir, data_dir, model = tiny_setup
    fb = make_finbert(tmp_path, tok_dir, data_dir, model, gradient_accumulation_steps=accumulation)
    before = model.classifier.weight.detach().clone()

    train_examples = fb.get_data("train")
    trained = fb.train(train_examples, fb.create_the_model())

    assert not torch.equal(before, trained.classifier.weight.detach())
    assert len(fb.validation_losses) == 2

    df = fb.evaluate(trained, fb.get_data("test"))
    assert len(df) == len(SENTENCES)

    # The saved directory is a regular Hugging Face model and works with the prediction pipeline.
    predictor = FinBertPredictor(str(tmp_path / "model_out"), device="cpu")
    results = predictor.predict("Profit rose. Shares fell.")
    assert len(results) == 2
    assert all(r.prediction in LABELS for r in results)


def test_rejects_non_empty_output_dir(tiny_setup):
    tmp_path, tok_dir, data_dir, model = tiny_setup
    (tmp_path / "model_out").mkdir()
    (tmp_path / "model_out" / "x").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        make_finbert(tmp_path, tok_dir, data_dir, model)
