from preprocessing import (
    DataArguments,
    ModelArguments,
    IETrainingArguments,
    read_finetune_data,
    convert_to_uie_input,
)
from paddlenlp.trainer import PdArgumentParser
from config import generate_logger, BaseConfig
from typing import Optional, List, Any
from functools import partial
from paddlenlp.datasets import load_dataset
import os

logger = generate_logger(name=__name__)
ML_FLOW = True  # Add MLflow for experiment # TODO change mlflow to False

# main function
def finetune(
    train_path: str = None,
    dev_path: str = None,
    max_seq_length: Optional[int] = 512,
    dynamic_max_length: Optional[List[int]] = None,
    model_name_or_path: Optional[str] = "uie-base",
    export_model_dir: Optional[str] = None,
    multilingual: bool = False,
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

    # TODO read data from train/dev/testing,
    train_dataset, dev_dataset = (
        load_dataset(read_finetune_data, data_path=data, max_seq_len=512, lazy=False) for data in (train_path, dev_path)
    )

    # TODO from pretrained tokenizer

    # TODO from pretrained model

    # TODO from pretrained other model

    # TODO implement soft prompt

    # TODO Convert the data into a dataset that aligns with the format of the model input..
    convert_function = partial(
        convert_to_uie_input,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        multilingual=multilingual,
        dynamic_max_length=dynamic_max_length,
    )

    train_dataset, dev_dataset = (data.map(convert_function) for data in (train_dataset, dev_dataset))

    # TODO loss function

    # TODO metrices

    # TODO add log and mlflow in metrics

    # TODO optimizer optional (do not fix)

    print(kwargs)


if __name__ == "__main__":
    parser = PdArgumentParser((ModelArguments, DataArguments, IETrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    base_config = BaseConfig()

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
