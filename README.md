# FinBERT: Financial Sentiment Analysis with BERT

FinBERT is a BERT model further trained on financial text and fine-tuned for sentiment classification
(positive / negative / neutral). See [FinBERT: Financial Sentiment Analysis with Pre-trained Language
Models](https://arxiv.org/pdf/1908.10063.pdf) for details. The fine-tuned model is published on the Hugging Face
Hub as [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert).

This repository wraps that model in a small, modern Python package:

- **Prediction** via the `transformers` text-classification pipeline, sentence by sentence
- **REST API** with FastAPI (OpenAPI docs at `/docs`)
- **CLI** for scoring text files into CSV
- **Optional training module** with the original discriminative fine-tuning and gradual unfreezing

Each sentence gets the three class probabilities, the predicted label and a
**sentiment score = P(positive) − P(negative)** in [-1, 1].

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # prediction, API and CLI
uv sync --extra train   # additionally pandas, scikit-learn, tqdm for training
```

The model (~440 MB) is downloaded from the Hugging Face Hub on first use and cached under `~/.cache/huggingface`.

## Usage

### Python

```python
from finbert import FinBertPredictor

predictor = FinBertPredictor()  # ProsusAI/finbert; pass a local directory to use your own model
for r in predictor.predict("Operating profit rose to EUR 13.1 mn. The net loss widened sharply."):
    print(r.prediction, round(r.sentiment_score, 3), r.sentence)
```

### Command line

```bash
uv run finbert predict --text-file examples/sample.txt                 # print to stdout
uv run finbert predict --text-file examples/sample.txt --out output/predictions.csv
```

### REST API

```bash
uv run finbert serve --port 8000        # or: uv run uvicorn finbert.api:create_app --factory
curl -s localhost:8000/predict -H 'content-type: application/json' \
     -d '{"text": "Revenue rose 12% year on year. Margins came under pressure."}'
```

| Endpoint        | Description                                                        |
|-----------------|--------------------------------------------------------------------|
| `POST /predict` | `{"text": "...", "split_sentences": true}` → per-sentence scores and average |
| `GET /health`   | Status, model id and package version                               |
| `GET /docs`     | Interactive OpenAPI documentation                                  |

Environment variables: `FINBERT_MODEL` (model id or path, default `ProsusAI/finbert`), `FINBERT_DEVICE`
(`cpu`, `cuda`, `mps`), `FINBERT_CORS_ORIGINS` (comma separated, default `*`).

### Docker

```bash
docker build -t finbert .
docker run -p 8000:8000 finbert
```

The image uses CPU-only PyTorch and bakes the model in at build time, so the container needs no network access.

## Datasets
There are two datasets used for FinBERT. The language model further training is done on a subset of Reuters TRC2 
dataset. This dataset is not public, but researchers can apply for access 
[here](https://trec.nist.gov/data/reuters/reuters.html).

For the sentiment analysis, we used Financial PhraseBank from [Malo et al. (2014)](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts).
 The dataset can be downloaded from this [link](https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip?origin=publication_list).
 If you want to train the model on the same dataset, after downloading it, you should create three files under the 
 `data/sentiment_data` folder as `train.csv`, `validation.csv`, `test.csv`. 
To create these files, do the following steps:
- Download the Financial PhraseBank from the above link.
- Get the path of `Sentences_50Agree.txt` file in the `FinancialPhraseBank-v1.0` zip.
- Run the datasets script:
```bash
uv run --extra train python -m finbert.training.datasets --data-path <path to Sentences_50Agree.txt>
```

## Training the model

Training lives in the optional `finbert.training` module (install with `uv sync --extra train`):

```python
from transformers import AutoModelForSequenceClassification
from finbert.training import Config, FinBert

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
config = Config(
    data_dir="data/sentiment_data",
    bert_model=model,
    model_dir="models/classifier_model/finbert-sentiment",
    num_train_epochs=4,
    max_seq_length=64,
    train_batch_size=32,
    learning_rate=2e-5,
    warm_up_proportion=0.2,
    discriminate=True,       # discriminative fine-tuning
    gradual_unfreeze=True,   # gradual unfreezing
)
finbert = FinBert(config)
finbert.prepare_model(label_list=["positive", "negative", "neutral"])
trained = finbert.train(finbert.get_data("train"), finbert.create_the_model())
```

The output directory is a standard Hugging Face model directory and can be used directly with
`FinBertPredictor("models/classifier_model/finbert-sentiment")`.

Changes compared with the original training code:

- `gradient_accumulation_steps > 1` now works. Previously `loss.backward()` was skipped whenever accumulation was
  enabled, so no gradients were computed and the model did not learn.
- `torch.optim.AdamW` replaces the removed `transformers.AdamW`, the best epoch is kept in memory and the model is
  saved with `save_pretrained` (including the tokenizer).
- fp16 and distributed training are not supported.

`notebooks/finbert_training.ipynb` is kept as a reference for the original experiments and is not maintained.

## Development

```bash
uv sync --extra train
uv run pytest              # fast, offline unit tests (fake classifier and a tiny BERT)
uv run pytest -m slow      # smoke test against the real ProsusAI/finbert model (needs network)
uv run ruff check . && uv run ruff format --check .
```

## Disclaimer
This is not an official Prosus product. It is the outcome of an intern research project in Prosus AI team.
### About Prosus 
Prosus is a global consumer internet group and one of the largest technology investors in the world. Operating and
 investing globally in markets with long-term growth potential, Prosus builds leading consumer internet companies that empower people and enrich communities.
For more information, please visit [www.prosus.com](www.prosus.com).

## Contact information
Please contact Dogu Araci `dogu.araci[at]naspers[dot]com` and Zulkuf Genc `zulkuf.genc[at]naspers[dot]com` about
 any FinBERT related issues and questions.
