from utils.data_utils import read_data_by_chunk, convert_to_uie_format
from utils.model_utils import uie_loss_func, compute_metrics
from config.base_config import UIE_input_spec
from paddlenlp.transformers import UIE, AutoTokenizer
from paddlenlp.trainer import Trainer, get_last_checkpoint, TrainingArguments, PdArgumentParser
from paddlenlp.trainer.trainer_callback import DefaultFlowCallback, EarlyStoppingCallback
from paddlenlp.transformers import export_model
from paddle import set_device, optimizer
from paddlenlp.utils.log import logger
from typing import Optional, List, Any, Callable, Dict, Union, Tuple, Literal
from functools import partial
from paddlenlp.datasets import load_dataset
import os
from dataclasses import dataclass, field, asdict


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    dataset_path: str = field(
        default="./data/model_input_data/",
        metadata={"help": "Local dataset directory including train.txt, dev.txt and test.txt (optional)."},
    )

    train_file: str = field(
        default="train.txt",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    dev_file: str = field(
        default="dev.txt",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    test_file: str = field(
        default="test.txt",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    max_seq_len: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default="uie-base",
        metadata={
            "help": "Path to pretrained model, such as 'uie-base', 'uie-tiny', "
            "'uie-medium', 'uie-mini', 'uie-micro', 'uie-nano', 'uie-base-en', "
            "'uie-m-base', 'uie-m-large', or finetuned model path."
        },
    )
    export_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the exported inference model."},
    )


# main function
def finetune(
    dataset_path: str,
    train_file: str,
    dev_file: str = None,
    test_file: str = None,
    max_seq_len: int = 512,
    model_name_or_path: str = "uie-base",
    export_model_dir: Optional[str] = None,
    convert_and_tokenize_function: Optional[
        Callable[[Dict[str, str], Any, int], Dict[str, Union[str, float]]]
    ] = convert_to_uie_format,
    criterion=uie_loss_func,
    compute_metrics=compute_metrics,
    optimizers: Optional[Tuple[optimizer.Optimizer, optimizer.lr.LRScheduler]] = (None, None),
    training_args: Optional[TrainingArguments] = None,
    trainer_callbacks=[DefaultFlowCallback],
) -> None:

    train_path, dev_path, test_path = (os.path.join(dataset_path, file) for file in (train_file, dev_file, test_file))
    # Path Checking
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
    if not os.path.exists(test_path):
        if training_args.do_predict == True:
            logger.warning(
                f"Testing data not found in {test_path}. \
                Please input the correct path of testing data.\
                    Auto-training without testing data..."
            )
        training_args.do_predict = False
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Model & Data Setup
    set_device(training_args.device)
    train_dataset, dev_dataset, test_dataset = (
        load_dataset(
            read_data_by_chunk,
            data_path=data,
            max_seq_len=max_seq_len,
            lazy=False,
        )
        for data in (train_path, dev_path, test_path)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = UIE.from_pretrained(model_name_or_path)
    convert_function = partial(
        convert_and_tokenize_function,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    # TODO solve none dev_dataset
    train_dataset, dev_dataset, test_dataset = (
        data.map(convert_function) for data in (train_dataset, dev_dataset, test_dataset)
    )

    # Trainer Setup
    trainer = Trainer(
        model=model,
        criterion=criterion,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=dev_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        callbacks=trainer_callbacks,
    )
    trainer.optimizers = (
        optimizer.AdamW(learning_rate=training_args.learning_rate, parameters=model.parameters())
        if optimizers[0] is None
        else optimizers[0]
    )

    # Checkpoint Setup
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

    # Start Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Start Evaluate and tests model
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)

    # Start Testing
    if training_args.do_predict:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("test", test_metrics)

    # export inference model
    if training_args.do_export:
        if export_model_dir is None:
            export_model_dir = os.path.join(training_args.output_dir, "export")
        export_model(model=trainer.model, input_spec=UIE_input_spec, path=export_model_dir)
        trainer.tokenizer.save_pretrained(export_model_dir)


if __name__ == "__main__":
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    finetune(
        dataset_path=data_args.dataset_path,
        train_file=data_args.train_file,
        dev_file=data_args.dev_file,
        test_file=data_args.test_file,
        max_seq_len=data_args.max_seq_len,
        model_name_or_path=model_args.model_name_or_path,
        export_model_dir=model_args.export_model_dir,
        training_args=training_args,
    )
