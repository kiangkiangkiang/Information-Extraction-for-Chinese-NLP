from preprocessing import (
    DataArguments,
    ModelArguments,
    IETrainingArguments,
    read_finetune_data,
    convert_to_uie_format,
)
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint
from paddlenlp.transformers import AutoModel, AutoTokenizer, export_model

from paddle import set_device
from config import generate_logger, BaseConfig
from typing import Optional, List, Any, Callable, Dict, Union
from functools import partial
from paddlenlp.datasets import load_dataset
import os


logger = generate_logger(name=__name__)
ML_FLOW = True  # Add MLflow for experiment # TODO change mlflow to False

# main function
def finetune(
    train_path: str,
    dev_path: str,
    max_seq_length: int = 512,
    model_name_or_path: str = "uie-base",
    dynamic_max_length: Optional[List[int]] = None,
    export_model_dir: Optional[str] = None,
    convert_and_tokenize_function: Optional[
        Callable[[Dict[str, str], Any, int], Dict[str, Union[str, float]]]
    ] = convert_to_uie_format,
    multilingual: Optional[bool] = False,
    **kwargs: Any,
) -> None:

    # Check arguments Legal or not
    if not os.path.exists(train_path):
        raise ValueError(f"Training data not found in {train_path}. Please input the correct path of training data.")

    if not os.path.exists(dev_path):
        if kwargs.do_eval == True:
            logger.warning(
                f"Evaluation data not found in {dev_path}. \
                Please input the correct path of evaluation data.\
                    Auto-training without evaluation data..."
            )
        kwargs.do_eval = False

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

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(kwargs.output_dir) and kwargs.do_train and not kwargs.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and kwargs.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # from pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # from pretrained model
    # TODO testing other model
    model = AutoModel.from_pretrained(model_name_or_path)

    # TODO implement soft prompt
    """
        if train_for_soft_prompt:
            model = only_open_wordembeddings_layers_for_soft_prompt_train(model)
    """

    # TODO Tokenization and Convert the data into a dataset that aligns with the format of prompt learning input..
    convert_function = partial(
        convert_and_tokenize_function,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        dynamic_max_length=dynamic_max_length,
        multilingual=multilingual,
    )

    train_dataset, dev_dataset = (data.map(convert_function) for data in (train_dataset, dev_dataset))

    # TODO loss function

    # TODO metrices

    # TODO add log and mlflow in metrics

    # TODO optimizer optional (do not fix)


if __name__ == "__main__":
    parser = PdArgumentParser((ModelArguments, DataArguments, IETrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    base_config = BaseConfig()

    training_args.print_config(model_args, "Model")
    training_args.print_config(model_args, "Data")

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
        max_seq_length=data_args.max_seq_length,
        dynamic_max_length=data_args.dynamic_max_length,
        model_name_or_path=model_args.model_name_or_path,
        export_model_dir=model_args.export_model_dir,
        multilingual=model_args.multilingual,
        kwargs=training_args,
    )
