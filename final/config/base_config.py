from paddle.static import InputSpec
from dataclasses import dataclass, field
from typing import Optional, List
from paddlenlp.utils.log import logger

entity_type = ["精神慰撫金額", "醫療費用", "薪資收入"]

UIE_input_spec = [
    InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
    InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
    InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
    InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
]

logger.set_level("DEBUG")

regularized_token = ["\n", " ", "\u3000"]


@dataclass
class ConvertArguments:
    labelstudio_file: str = field(
        default="./data/label_studio_data/label_studio_output.json",
        metadata={"help": "The export file from label studio. Only support the JSON format."},
    )

    save_dir: str = field(
        default="./data/model_input_data/",
        metadata={"help": "The path of converted data (train/dev/test) that you want to save."},
    )

    seed: int = field(
        default=1000,
        metadata={"help": "Random seed for shuffle."},
    )

    split_ratio: float = field(
        default_factory=lambda: [0.8, 0.1, 0.1],
        metadata={
            "help": "The ratio of samples in datasets. [0.7, 0.2, 0.1] means 70% samples used for training, 20% for evaluation and 10% for testing."
        },
    )

    is_shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the labeled dataset. Defaults to True."},
    )

    is_regularize_data: bool = field(
        default=True,
        metadata={"help": "Whether to regularize data (remove special tokens likes \\n). Defaults to True"},
    )


@dataclass
class TrainModelArguments:
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

    max_seq_len: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class TrainDataArguments:
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

    export_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the exported inference model."},
    )


@dataclass
class EvaluationArguments(TrainModelArguments):

    dev_file: str = field(
        default="./data/model_input_data/test.txt",
        metadata={"help": "The path of file that you want to evaluate."},
    )

    batch_size: int = field(
        default=16,
        metadata={"help": "Evaluation batch size."},
    )

    device: str = field(
        default="cpu",
        metadata={"help": "The device for evaluate."},
    )

    is_eval_by_class: bool = field(
        default=False,
        metadata={
            "help": "Precision, recall and F1 score are calculated for each class separately if this option is enabled."
        },
    )


@dataclass
class InferenceDataArguments:
    data_file: str = field(
        default="./data/model_infer_data/example.txt",
        metadata={"help": "The path of data that you wanna inference."},
    )

    text_list: List[str] = field(
        default=None,
        metadata={"help": "The path of data that you wanna inference."},
    )

    save_dir: str = field(
        default=None,
        metadata={"help": "The path where you wanna to save results of inference. If None, model won't write data."},
    )

    is_regularize_data: bool = field(
        default=False,
        metadata={"help": "Whether to regularize data (remove special tokens likes \\n). Defaults to False"},
    )


@dataclass
class InferenceTaskflowArguments:

    device_id: int = field(
        default=0,
        metadata={
            "help": "The device id for the gpu. The defalut value is 0. If you want to use CPU, set the device_id to -1."
        },
    )

    precision: str = field(
        default="fp32",
        metadata={
            "help": "fp16 (Only GPU) or fp32. Default 'fp32', which is slower than 'fp16'. If 'fp16' is applied, make sure your CUDA>=11.2 and cuDNN>=8.1.1."
            "If there is warning when using fp16, pip install onnxruntime-gpu onnx onnxconverter-common."
        },
    )

    batch_size: int = field(
        default=1,
        metadata={"help": "Inference batch size."},
    )

    model: str = field(
        default="uie-base",
        metadata={
            "help": "Pretrained model, such as 'uie-base', 'uie-tiny', "
            "'uie-medium', 'uie-mini', 'uie-micro', 'uie-nano'. Only applied when task_path is None."
        },
    )

    task_path: str = field(
        default=None,
        metadata={"help": "The checkpoint you want to use on inference."},
    )


@dataclass
class InferenceStrategyArguments:

    select_strategy: str = field(
        default="all",
        metadata={
            "help": "'all' or 'max' or 'threshold'. Strategy of getting results. max: Only get the max prob. result. all: Get all results. "
            "threshold: Ｇet the results whose prob. is larger than threshold. "
        },
    )

    select_strategy_threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold for probability. Only applied when select_strategy='threshold'. "},
    )

    select_key: List[str] = field(
        default_factory=lambda: ["text", "start", "end", "probability"],
        metadata={
            "help": "UIE will output ['text', 'start', 'end', 'probability']. --select_key is to select which key in the list you want to return."
        },
    )
