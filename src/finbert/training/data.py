# The classes used for data processing and convert_examples_to_features are very similar versions of the ones \
# found in Hugging Face's scripts in the transformers library. For more BERT or similar language model implementation \
# examples, we would highly recommend checking that library as well.


"""Data handling for FinBERT training (ported from the original ``finbert/utils.py``)."""

import csv
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


# Classes regarding input and data handling


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, agree=None):
        """
        Constructs an InputExample
        Parameters
        ----------
        guid: str
            Unique id for the examples
        text: str
            Text for the first sequence.
        label: str, optional
            Label for the example.
        agree: str, optional
            For FinBERT , inter-annotator agreement level.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.agree = agree


class InputFeatures:
    """
    A single set of features for the data.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, agree=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.agree = agree


class DataProcessor:
    """Base class to read data files."""

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
        return lines


class FinSentProcessor(DataProcessor):
    """
    Data processor for FinBERT.
    """

    def get_examples(self, data_dir, phase):
        """
        Get examples from the data directory.

        Parameters
        ----------
        data_dir: str
            Path for the data directory.
        phase: str
            Name of the .csv file to be loaded.
        """
        return self._create_examples(self._read_tsv(os.path.join(data_dir, (phase + ".csv"))), phase)

    def get_labels(self):
        return ["positive", "negative", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text = line[1]
            label = line[2]
            agree = line[3] if len(line) > 3 else None
            examples.append(InputExample(guid=guid, text=text, label=label, agree=agree))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode="classification"):
    """
    Loads a data file into a list of InputBatch's. With this function, the InputExample's are converted to features
    that can be used for the model. Text is tokenized, converted to ids and zero-padded. Labels are mapped to integers.

    Parameters
    ----------
    examples: list
        A list of InputExample's.
    label_list: list
        The list of labels.
    max_seq_length: int
        The maximum sequence length.
    tokenizer: BertTokenizer
        The tokenizer to be used.
    mode: str, optional
        The task type: 'classification' or 'regression'. Default is 'classification'

    Returns
    -------
    features: list
        A list of InputFeature's, which is an InputBatch.
    """

    if mode == "classification":
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map[None] = 9090

    features = []
    for ex_index, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: (max_seq_length // 4) - 1] + tokens[len(tokens) - (3 * max_seq_length // 4) + 1 :]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding

        token_type_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        if mode == "classification":
            label_id = label_map[example.label]
        elif mode == "regression":
            label_id = float(example.label)
        else:
            raise ValueError("The mode should either be classification or regression. You entered: " + mode)

        agree = example.agree
        mapagree = {"0.5": 1, "0.66": 2, "0.75": 3, "1.0": 4}
        agree = mapagree.get(agree, 0)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join(tokens))
            logger.info("input_ids: %s", input_ids)
            logger.info("label: %s (id = %s)", example.label, label_id)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id,
                agree=agree,
            )
        )
    return features


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]


def get_metrics(df):
    "Computes accuracy and precision-recall for different sentiments."

    df.loc[:, "guess"] = df.predictions.apply(np.argmax)
    df.loc[:, "accurate"] = df.apply(lambda x: x["guess"] == x["labels"], axis=1)
    accuracy = df.accurate.sum() / df.shape[0]

    pos_recall = df[df["labels"] == 0].accurate.sum() / df[df["labels"] == 0].shape[0]
    neg_recall = df[df["labels"] == 1].accurate.sum() / df[df["labels"] == 1].shape[0]
    net_recall = df[df["labels"] == 2].accurate.sum() / df[df["labels"] == 2].shape[0]

    pos_precision = df[df["guess"] == 0].accurate.sum() / df[df["guess"] == 0].shape[0]
    neg_precision = df[df["guess"] == 1].accurate.sum() / df[df["guess"] == 1].shape[0]
    net_precision = df[df["guess"] == 2].accurate.sum() / df[df["guess"] == 2].shape[0]

    pos_f1score = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
    neg_f1score = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
    net_f1score = 2 * (net_precision * net_recall) / (net_precision + net_recall)

    return {
        "Accuracy": accuracy,
        "Positive": {"precision": pos_precision, "recall": pos_recall, "f1-score": pos_f1score},
        "Negative": {"precision": neg_precision, "recall": neg_recall, "f1-score": neg_f1score},
        "Neutral": {"precision": net_precision, "recall": net_recall, "f1-score": net_f1score},
    }


def get_prediction(text, model, tokenizer):
    """
    Get one prediction.

    Parameters
    ----------
    text: str
        The text to be analyzed.
    model: BertModel
        The model to be used.
    tokenizer: BertTokenizer
        The tokenizer to be used.

    Returns
    -------
    predition: np.array
        An array that includes probabilities for each class.
    """

    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_type_ids = [0] * len(tokens)
    attention_mask = [1] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding = [0] * (64 - len(input_ids))
    input_ids += padding
    attention_mask += padding
    token_type_ids += padding

    features = []
    features.append(
        InputFeatures(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, label_id=None)
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logits = model(all_input_ids, attention_mask=all_attention_mask, token_type_ids=all_token_type_ids).logits
    prediction = softmax(logits.cpu().numpy())
    return prediction


def chunks(items, n):
    """Split a list into consecutive chunks of length ``n``."""
    for i in range(0, len(items), n):
        yield items[i : i + n]
