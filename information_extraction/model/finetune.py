from preprocessing import *
from paddlenlp.trainer import (
    PdArgumentParser,
    get_last_checkpoint,
    Trainer,
    TrainingArguments,
)

from paddlenlp.transformers import export_model
from paddle import set_device, optimizer
from paddle.static import InputSpec
from paddlenlp.utils.log import logger
from config import BaseConfig
from typing import Optional, List, Any, Callable, Dict, Union, Tuple
from callbacks import *
from functools import partial
from paddlenlp.datasets import load_dataset
import os


# Add MLflow for experiment # TODO change mlflow to False
MLFLOW = True
if MLFLOW:
    from setup_mlflow import ML_Flow_Handler

    mlflow_handler = ML_Flow_Handler()
    uie_loss_func = mlflow_handler.loss_func
    logger.debug("Success to set up mlflow.")


# main function
def finetune(
    train_path: str,
    dev_path: str,
    max_seq_len: int = 512,
    model_name_or_path: str = "uie-base",
    export_model_dir: Optional[str] = None,
    multilingual: Optional[bool] = False,
    convert_and_tokenize_function: Optional[
        Callable[[Dict[str, str], Any, int], Dict[str, Union[str, float]]]
    ] = convert_to_uie_format,
    criterion=uie_loss_func,
    compute_metrics=SpanEvaluator_metrics,
    optimizers: Optional[Tuple[optimizer.Optimizer, optimizer.lr.LRScheduler]] = (None, None),
    training_args: Optional[TrainingArguments] = None,
) -> None:

    # Check arguments Legal or not
    if not os.path.exists(train_path):
        raise ValueError(f"Training data not found in {train_path}. Please input the correct path of training data.")

    if not os.path.exists(dev_path):
        if training_args.do_eval == True:
            logger.warning(
                f"Evaluation data not found in {dev_path}. \
                Please input the correct path of evaluation data.\
                    Auto-training without evaluation data..."
            )
        training_args.do_eval = False

    if model_name_or_path in ["uie-m-base", "uie-m-large"]:
        multilingual = True

    set_device(training_args.device)

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    train_dataset, dev_dataset = (
        load_dataset(read_finetune_data, data_path=data, max_seq_len=512, lazy=False) for data in (train_path, dev_path)
    )

    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    # TODO implement soft prompt
    """
        if train_for_soft_prompt:
            model = only_open_wordembeddings_layers_for_soft_prompt_train(model)
    """

    # Tokenization and Convert the data into a dataset that aligns with the format of prompt learning input..
    convert_function = partial(
        convert_and_tokenize_function,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        multilingual=multilingual,
    )

    # TODO solve none dev_dataset
    train_dataset, dev_dataset = (data.map(convert_function) for data in (train_dataset, dev_dataset))

    trainer = Trainer(
        model=model,
        criterion=criterion,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=dev_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
    )
    trainer.optimizers = (
        optimizer.AdamW(learning_rate=training_args.learning_rate, parameters=model.parameters())
        if optimizers[0] is None
        else optimizers[0]
    )

    # Detecting last checkpoint.
    checkpoint, last_checkpoint = None, None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    logger.debug(f"chechpoint: {checkpoint}")
    logger.debug(f"last_checkpoint: {last_checkpoint}")

    # Training
    if training_args.do_train:
        if MLFLOW:

            train_result = mlflow_handler.mlflow_train(
                trainer,
                checkpoint,
                log_parms_dict=dict(
                    model_name_or_path=model_name_or_path,
                    batch_size_train=training_args.per_device_train_batch_size,
                    learning_rate=training_args.learning_rate,
                    n_epoch=training_args.num_train_epochs,
                    optimizer=trainer.optimizers,
                    train_data_len=len(train_dataset),
                ),
            )

        else:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate and tests model
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)

    # export inference model
    if training_args.do_export:
        # You can also load from certain checkpoint
        # trainer.load_state_dict_from_checkpoint("/path/to/checkpoint/")
        if multilingual:
            input_spec = [
                InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
                InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            ]
        else:
            input_spec = [
                InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
                InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
                InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
                InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
            ]
        if export_model_dir is None:
            logger.warning(f"Missing export_model_dir path. Using {training_args.output_dir}export as default.")
            export_model_dir = os.path.join(training_args.output_dir)
        try:
            export_model(model=trainer.model, input_spec=input_spec, path=export_model_dir)
        except Exception as e:
            logger.error(f"Fail to export model. Error in export_model: {e.__class__.__name__}: {e}.")


if __name__ == "__main__":
    base_config = BaseConfig()
    parser = PdArgumentParser((ModelArguments, DataArguments, IETrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    if base_config.root_dir:
        if data_args.train_path is None and training_args.do_train:
            data_args.train_path = base_config.root_dir + base_config.experiment_data_path + "training_data.txt"
            logger.warning(
                f"Missing 'train_path' argument. " + "Automatically use {data_args.train_path} as training data."
            )
        if data_args.dev_path is None and training_args.do_eval:
            data_args.dev_path = base_config.root_dir + base_config.experiment_data_path + "eval_data.txt"
            logger.warning(
                f"Missing 'dev_path' argument. " + "Automatically use {data_args.dev_path} as evaluation data."
            )

    finetune(
        train_path=data_args.train_path,
        dev_path=data_args.dev_path,
        max_seq_len=data_args.max_seq_len,
        model_name_or_path=model_args.model_name_or_path,
        export_model_dir=model_args.export_model_dir,
        multilingual=model_args.multilingual,
        training_args=training_args,
    )
