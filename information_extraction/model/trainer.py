from preprocessing import get_base_config
import numpy as np
import pandas as pd
from paddlenlp.trainer import Trainer
from paddlenlp.transformers.model_utils import PretrainedModel
from paddle.io import Dataset, DataLoader
from paddlenlp.utils.log import logger
from paddlenlp.trainer.utils.helper import nested_concat, nested_numpify, nested_truncate
from paddlenlp.trainer.training_args import TrainingArguments
from paddlenlp.data import DataCollator
from paddlenlp.utils.batch_sampler import DistributedBatchSampler as NlpDistributedBatchSampler
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.trainer.trainer_utils import (
    EvalPrediction,
    EvalLoopOutput,
    has_length,
    find_batch_size,
    IterableDatasetShard,
)
from paddlenlp.trainer.trainer_callback import TrainerCallback
from paddle import nn

import paddle
from typing import Union, Optional, Callable, Dict, Tuple, List, NamedTuple

base_config = get_base_config()
# TODO if loss太大，print出文本


class IEEvalPrediction(NamedTuple):

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]
    eval_group: Union[np.ndarray, Tuple[np.ndarray]]
    dataloader: Union[np.ndarray, Tuple[np.ndarray]]


class IETrainer(Trainer):
    """改的地方標記為: Modification

    Args:
        Trainer (_type_): _description_
        do_experiment: 實驗階段會override父類別的方法，以監控更多指標，因此當do_experiment=True時，會使用override方法，
        當do_experiment=False時，會直接套用父類別的方法。
    """

    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[IEEvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = ...,
        preprocess_logits_for_metrics: Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor] = None,
        do_experiment: bool = True,
        mlflow_training_step: int = 0,
        mlflow_eval_step: int = 0,
    ):
        self.do_experiment = do_experiment
        self.mlflow_training_step = mlflow_training_step
        self.mlflow_eval_step = mlflow_eval_step

        super().__init__(
            model,
            criterion,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        # Modification
        # 這裡接到的inputs是split後的input，label是整個文本的label，兩者長度不同
        # ex. inputs['input_ids'] = 7 by 2048, label = 1 by 14700

        if self.criterion is not None:
            if "labels" in inputs:
                labels = inputs.pop("labels")
            elif "start_positions" in inputs and "end_positions" in inputs:
                labels = (inputs.pop("start_positions"), inputs.pop("end_positions"))
            elif self.args.label_names is not None:
                labels = []
                for label in self.label_names:
                    labels.append(inputs.pop(label))
                labels = tuple(labels)
            elif "generator_labels" in inputs:
                labels = inputs["generator_labels"]
        else:
            labels = None

        # Modification
        is_test_full_content = True
        if is_test_full_content:
            paddle.set_device(self.args.device)
            # model.config.hidden_size
            model_max_len = 256
            model_input = {}
            last_sequence_output = []

            # model loop
            while inputs["input_ids"].shape[1] > 0:
                for key in inputs:
                    model_input[key] = inputs[key][:, :model_max_len]
                    inputs[key] = inputs[key][:, model_max_len:]
                sequence_output, _ = model(**model_input)
                # output[0] = 1 * max_len * hidden_size, outputs[1] = 1 * hidden_size
                if len(last_sequence_output) > 0:
                    last_sequence_output = paddle.concat([last_sequence_output, sequence_output], axis=1)
                else:
                    last_sequence_output = sequence_output

            # breakpoint()
            start_logits = model.linear_start(last_sequence_output)
            start_logits = paddle.squeeze(start_logits, -1)
            start_prob = model.sigmoid(start_logits)
            end_logits = model.linear_end(last_sequence_output)
            end_logits = paddle.squeeze(end_logits, -1)
            end_prob = model.sigmoid(end_logits)
            outputs = (start_prob, end_prob)

            # breakpoint()
        else:
            outputs = model(**inputs)

        # for group evaluation
        min_word = self.__get_min_word_in_ner_type()
        group = self.__get_eval_group(min_word, inputs)

        if self.criterion is not None:
            # Modification
            if model.training:
                loss = self.criterion(
                    outputs, labels, group, mlflow_key="Training loss", mlflow_step=self.mlflow_training_step
                )
                self.mlflow_training_step += 1

            else:
                loss = self.criterion(
                    outputs, labels, group, mlflow_key="Evaluation loss", mlflow_step=self.mlflow_eval_step
                )
                self.mlflow_eval_step += 1

            outputs = (loss, outputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        """
        Modification: 為了能看不同標籤類別的指標，這邊刻意override evaluation_loop，缺點是後續paddle更新時沒辦法同步，
        因此若不需要實驗時，可以直接使用父類別的evaluation_loop。
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Modification
        if not self.do_experiment:
            return super().evaluation_loop(
                dataloader=dataloader,
                description=description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                max_eval_iters=max_eval_iters,
            )
        # min_word 為用來判斷group的最小字元，例如min_word=4，代表用每個ner_type的前四個字來區隔其他的type
        min_word = self.__get_min_word_in_ner_type()

        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self.model

        if isinstance(dataloader, paddle.io.DataLoader):
            batch_size = dataloader.batch_sampler.batch_size
        elif isinstance(dataloader, paddle.fluid.dataloader.dataloader_iter._DataLoaderIterBase):
            # support for inner dataloader
            batch_size = dataloader._batch_sampler.batch_size
            # alias for inner dataloader
            dataloader.dataset = dataloader._dataset
        else:
            raise ValueError("Only support for paddle.io.DataLoader")

        num_samples = None
        if max_eval_iters > 0:
            # on eval limit steps
            num_samples = batch_size * self.args.world_size * max_eval_iters
            if isinstance(dataloader, paddle.fluid.dataloader.dataloader_iter._DataLoaderIterBase) and isinstance(
                dataloader._batch_sampler, NlpDistributedBatchSampler
            ):
                consumed_samples = (
                    ((self.state.global_step) // args.eval_steps)
                    * max_eval_iters
                    * args.per_device_eval_batch_size
                    * args.world_size
                )
                dataloader._batch_sampler.set_epoch(consumed_samples=consumed_samples)

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")
            else:
                logger.info(f"  Total prediction steps = {len(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")

        logger.info(f"  Pre device batch size = {batch_size}")
        logger.info(f"  Total Batch size = {batch_size * self.args.world_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        losses = []
        # Modification
        eval_group = []
        for step, inputs in enumerate(dataloader):
            # Modification
            group = self.__get_eval_group(min_word, inputs)

            # breakpoint()
            eval_group.extend(group)

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step, output = (batch_size, max_len)
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                # losses = self._nested_gather(loss.repeat(batch_size))
                losses = self._nested_gather(paddle.tile(loss, repeat_times=[batch_size, 1]))
                losses_host = losses if losses_host is None else paddle.concat((losses_host, losses), axis=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            if max_eval_iters > 0 and step >= max_eval_iters - 1:
                break

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if num_samples is not None:
            pass
        elif has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        model.train()

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            # Modification
            # test metrics
            metrics = self.compute_metrics(
                IEEvalPrediction(
                    predictions=all_preds, label_ids=all_labels, eval_group=np.array(eval_group), dataloader=dataloader
                )
            )
        else:
            metrics = {}

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def __get_min_word_in_ner_type(self):
        min_word = np.min([len(i) for i in base_config.ner_type])
        ner_type = [i[:min_word] for i in base_config.ner_type]
        if len(pd.unique(ner_type)) != len(base_config.ner_type):
            raise ValueError(f"The first word in ner_type is repeat. Please adjust it.")
        return min_word

    def __get_eval_group(self, min_word, inputs):
        group = self.tokenizer.convert_ids_to_tokens(np.array(inputs["input_ids"][:, 1 : (min_word + 1)]).flatten())
        group = [
            "".join(group)[start:end]
            for start, end in zip(
                range(0, len(group), min_word),
                range(min_word, len(group) + min_word, min_word),
            )
        ]
        return group
