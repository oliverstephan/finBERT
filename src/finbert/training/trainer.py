"""FinBERT fine-tuning (ported from the original ``finbert/finbert.py``).

Optional module: install with ``uv sync --extra train``. Implements discriminative fine-tuning and
gradual unfreezing as described in the FinBERT paper (Araci, 2019).
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from finbert.training.data import FinSentProcessor, convert_examples_to_features

logger = logging.getLogger(__name__)


class Config:
    """The configuration class for training."""

    def __init__(
        self,
        data_dir,
        bert_model,
        model_dir,
        max_seq_length=64,
        train_batch_size=32,
        eval_batch_size=32,
        learning_rate=5e-5,
        num_train_epochs=10.0,
        warm_up_proportion=0.1,
        no_cuda=False,
        do_lower_case=True,
        seed=42,
        local_rank=-1,
        gradient_accumulation_steps=1,
        fp16=False,
        output_mode="classification",
        discriminate=True,
        gradual_unfreeze=True,
        encoder_no=12,
        base_model="bert-base-uncased",
    ):
        """
        Parameters
        ----------
        data_dir: str
            Path for the training and evaluation datasets.
        bert_model: BertModel
            The BERT model to be used. For example: BertForSequenceClassification.from_pretrained(...)
        model_dir: str
            The path where the resulting model will be saved.
        max_seq_length: int
            The maximum length of the sequence to be used. Default value is 64.
        train_batch_size: int
            The batch size for the training. Default value is 32.
        eval_batch_size: int
            The batch size for the evaluation. Default value is 32.
        learning_rate: float
            The learning rate. Default value is 5e5.
        num_train_epochs: int
            Number of epochs to train. Default value is 4.
        warm_up_proportion: float
            During the training, the learning rate is linearly increased. This value determines when the learning rate
            reaches the intended learning rate. Default value is 0.1.
        no_cuda: bool
            Determines whether to use gpu. Default is False.
        do_lower_case: bool
            Determines whether to make all training and evaluation examples lower case. Default is True.
        seed: int
            Random seed. Defaults to 42.
        local_rank: int
            Kept for backwards compatibility; distributed training is not supported by this port.
        gradient_accumulation_steps: int
            Number of gradient accumulations steps. Defaults to 1.
        fp16: bool
            Not supported by this port; must be False.
        output_mode: 'classification' or 'regression'
            Determines whether the task is classification or regression.
        discriminate: bool
            Determines whether to apply discriminative fine-tuning.
        gradual_unfreeze: bool
            Determines whether to gradually unfreeze lower and lower layers as the training goes on.
        encoder_no: int
            Starting from which layer the model is going to be finetuned. If set 12, whole model is going to be
            fine-tuned. If set, for example, 6, only the last 6 layers will be fine-tuned.
        base_model: str
            Hugging Face id or path of the tokenizer to use. Default is 'bert-base-uncased'.
        """
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.model_dir = model_dir
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.local_rank = local_rank
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warm_up_proportion = warm_up_proportion
        self.no_cuda = no_cuda
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_mode = output_mode
        self.fp16 = fp16
        self.discriminate = discriminate
        self.gradual_unfreeze = gradual_unfreeze
        self.encoder_no = encoder_no
        self.base_model = base_model


class FinBert:
    """
    The main class for FinBERT training and evaluation.
    """

    def __init__(self, config):
        self.config = config

    def prepare_model(self, label_list):
        """
        Sets some of the components of the model: Dataset processor, number of labels, device, gradient
        accumulation steps and tokenizer.

        Parameters
        ----------
        label_list: list
            The list of labels values in the dataset. For example: ['positive','negative','neutral']
        """
        if self.config.fp16:
            raise NotImplementedError("fp16 training is not supported by this port; use fp16=False")

        self.processor = FinSentProcessor()

        if self.config.no_cuda or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count() if self.device.type == "cuda" else 0
        logger.info("device: %s n_gpu: %d", self.device, self.n_gpu)

        if self.config.gradient_accumulation_steps < 1:
            raise ValueError(
                f"Invalid gradient_accumulation_steps parameter: {self.config.gradient_accumulation_steps}, "
                "should be >= 1"
            )

        # The configured batch size is the effective batch size per optimizer step; each forward pass
        # sees train_batch_size / gradient_accumulation_steps examples.
        self.config.train_batch_size = self.config.train_batch_size // self.config.gradient_accumulation_steps

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.config.seed)

        self.config.model_dir = Path(self.config.model_dir)
        if self.config.model_dir.exists() and any(self.config.model_dir.iterdir()):
            raise ValueError(f"Output directory ({self.config.model_dir}) already exists and is not empty.")
        self.config.model_dir.mkdir(parents=True, exist_ok=True)

        self.num_labels = len(label_list)
        self.label_list = label_list

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, do_lower_case=self.config.do_lower_case)

    def get_data(self, phase):
        """
        Gets the data for training or evaluation. In the data directory, there should be a tab separated file
        named <phase>.csv.

        Parameters
        ----------
        phase: str
            Name of the dataset that will be used in that phase, e.g. 'train', 'validation' or 'test'.

        Returns
        -------
        examples: list
            A list of InputExample's.
        """
        examples = self.processor.get_examples(self.config.data_dir, phase)

        if phase == "train":
            self.num_train_optimization_steps = int(
                len(examples) / self.config.train_batch_size / self.config.gradient_accumulation_steps
            ) * int(self.config.num_train_epochs)

            train = pd.read_csv(os.path.join(self.config.data_dir, "train.csv"), sep="\t", index_col=False)
            class_weights = [train.shape[0] / train[train.label == label].shape[0] for label in self.label_list]
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

        return examples

    def create_the_model(self):
        """
        Creates the model. Sets the model to be trained and the optimizer.
        """
        model = self.config.bert_model
        model.to(self.device)

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        lr = self.config.learning_rate
        dft_rate = 1.2

        def groups(module, layer_lr):
            params = list(module.named_parameters())
            return [
                {
                    "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                    "lr": layer_lr,
                },
                {
                    "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": layer_lr,
                },
            ]

        if self.config.discriminate:
            # Discriminative fine-tuning: lower layers get geometrically smaller learning rates.
            layers = model.bert.encoder.layer
            n_layers = len(layers)
            optimizer_grouped_parameters = [
                *groups(model.bert.embeddings, lr / (dft_rate ** (n_layers + 1))),
                *groups(model.bert.pooler, lr),
                *groups(model.classifier, lr),
            ]
            for i in range(n_layers):
                optimizer_grouped_parameters.extend(groups(layers[i], lr / (dft_rate ** (n_layers - i))))
        else:
            optimizer_grouped_parameters = groups(model, lr)

        # Drop empty groups (e.g. a module without bias parameters).
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]

        self.num_warmup_steps = int(float(self.num_train_optimization_steps) * self.config.warm_up_proportion)

        # transformers.AdamW was removed; torch.optim.AdamW always applies bias correction, whereas the
        # original code used correct_bias=False. The difference only matters in the first few steps.
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_train_optimization_steps,
        )
        return model

    def get_loader(self, examples, phase):
        """
        Creates a data loader object for a dataset.

        Parameters
        ----------
        examples: list
            The list of InputExample's.
        phase: 'train' or 'eval'
            Determines whether to use random sampling or sequential sampling depending on the phase.
        """
        features = convert_examples_to_features(
            examples, self.label_list, self.config.max_seq_length, self.tokenizer, self.config.output_mode
        )

        logger.info("***** Loading data *****")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", self.config.train_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        if self.config.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        else:
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        all_agree_ids = torch.tensor([f.agree or 0 for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, all_agree_ids)
        sampler = RandomSampler(data) if phase == "train" else SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.config.train_batch_size)

    def _loss(self, logits, label_ids, weights=None):
        if self.config.output_mode == "classification":
            return CrossEntropyLoss(weight=weights)(logits.view(-1, self.num_labels), label_ids.view(-1))
        return MSELoss()(logits.view(-1), label_ids.view(-1))

    def _backward(self, loss):
        """Scale the loss for gradient accumulation and back-propagate.

        The original implementation only called ``backward()`` when ``gradient_accumulation_steps == 1``;
        with accumulation enabled no gradients were ever computed and the model did not learn.
        """
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        return loss

    def _optimizer_step(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def train(self, train_examples, model):
        """
        Trains the model, keeping the weights of the epoch with the lowest validation loss.

        Parameters
        ----------
        train_examples: list
            Contains the data as a list of InputExample's
        model: BertForSequenceClassification
            The model to be trained (as returned by ``create_the_model``).

        Returns
        -------
        model: BertForSequenceClassification
            The trained model.
        """
        validation_examples = self.get_data("validation")
        self.validation_losses = []

        train_dataloader = self.get_loader(train_examples, "train")
        weights = self.class_weights.to(self.device)
        step_number = len(train_dataloader)
        accumulation = self.config.gradient_accumulation_steps

        best_state = None
        best_epoch = None
        i = 0
        for epoch in trange(int(self.config.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0.0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if self.config.gradual_unfreeze and i == 0:
                    for param in model.bert.parameters():
                        param.requires_grad = False

                # Gradual unfreezing: one more encoder layer is unfrozen every third of an epoch.
                if step % max(1, step_number // 3) == 0:
                    i += 1

                if self.config.gradual_unfreeze and 1 < i < self.config.encoder_no:
                    for k in range(i - 1):
                        layer_idx = self.config.encoder_no - 1 - k
                        if 0 <= layer_idx < len(model.bert.encoder.layer):
                            for param in model.bert.encoder.layer[layer_idx].parameters():
                                param.requires_grad = True

                if self.config.gradual_unfreeze and i > self.config.encoder_no + 1:
                    for param in model.bert.embeddings.parameters():
                        param.requires_grad = True

                input_ids, attention_mask, token_type_ids, label_ids, _ = (t.to(self.device) for t in batch)
                logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
                loss = self._backward(self._loss(logits, label_ids, weights))
                tr_loss += loss.item()

                if (step + 1) % accumulation == 0 or (step + 1) == step_number:
                    self._optimizer_step(model)

            valid_loss = self._validation_loss(model, validation_examples, weights)
            self.validation_losses.append(valid_loss)
            logger.info("Epoch %d: train loss %.4f, validation losses %s", epoch, tr_loss, self.validation_losses)

            if valid_loss == min(self.validation_losses):
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

        if best_state is not None:
            logger.info("Restoring weights from epoch %d", best_epoch)
            model.load_state_dict(best_state)

        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(self.config.model_dir)
        self.tokenizer.save_pretrained(self.config.model_dir)
        return model

    def _validation_loss(self, model, examples, weights):
        loader = self.get_loader(examples, phase="eval")
        model.eval()
        total, steps = 0.0, 0
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, label_ids, _ in tqdm(loader, desc="Validating"):
                input_ids, attention_mask, token_type_ids, label_ids = (
                    t.to(self.device) for t in (input_ids, attention_mask, token_type_ids, label_ids)
                )
                logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
                total += self._loss(logits, label_ids, weights).item()
                steps += 1
        return total / max(steps, 1)

    def evaluate(self, model, examples):
        """
        Evaluate the model.

        Returns
        -------
        evaluation_df: pd.DataFrame
            A dataframe that includes for each example the logits, the label and the agreement level.
        """
        eval_loader = self.get_loader(examples, phase="eval")
        logger.info("***** Running evaluation ***** ")
        logger.info("  Num examples = %d", len(examples))

        model.eval()
        predictions, labels, agree_levels = [], [], []
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, label_ids, agree_ids in tqdm(eval_loader, desc="Testing"):
                logits = model(
                    input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    token_type_ids=token_type_ids.to(self.device),
                ).logits
                predictions.extend(np.array(logits.cpu().numpy()))
                labels.extend(label_ids.tolist())
                agree_levels.extend(agree_ids.tolist())

        return pd.DataFrame({"predictions": predictions, "labels": labels, "agree_levels": agree_levels})
