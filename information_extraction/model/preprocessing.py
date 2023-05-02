import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Union
from paddlenlp.trainer import TrainingArguments
from paddlenlp.transformers import AutoTokenizer

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from config import generate_logger, BaseConfig
from utils.exceptions import DataError

logger = generate_logger(name=__name__)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    train_path: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    dev_path: str = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    dynamic_max_length: Optional[List[int]] = field(
        default=None,
        metadata={"help": "dynamic max length from batch, it can be array of length, eg: 16 32 64 128"},
    )

    def __iter__(self):
        return iter((self.train_path, self.dev_path, self.max_seq_length, self.dynamic_max_length))


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
    multilingual: bool = field(default=False, metadata={"help": "Whether the model is a multilingual model."})


# information extraction training arguments
@dataclass
class IETrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="./",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


def read_finetune_data(data_path: str, max_seq_len: int = 512) -> Dict[str, str]:
    """讀「透過 utils/split_labelstudio.py 分割的 .txt檔」，此 txt 檔格式和 UIE官方提供的doccano.py轉換後的格式一樣。

    Args:
        data_path (str): 資料路徑（轉換後的training/eval/testing資料）。
        max_seq_len (int, optional): 模型input最大長度. Defaults to 512.

    Raises:
        ValueError: max_seq_len太小或prompt太長。
        DataError: 原始資料有問題（output of label studio），可能是entity太長或end的位置 < start的位置。

    Yields:
        Iterator[Dict[str, str]]: 每個batch所吃的原始文本（Before tokenization）。
    """

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            json_line = json.loads(line)
            content = json_line["content"].strip()
            prompt = json_line["prompt"]
            # Model Input is aslike: [CLS] Prompt [SEP] Content [SEP]
            # It include three summary tokens.
            if max_seq_len <= len(prompt) + 3:
                raise ValueError("The value of max_seq_len is too small. Please set a larger value.")
            result_list = json_line["result_list"]
            accumulate_token = 0

            # start pop all content
            while len(content) > 0:
                max_content_len = max_seq_len - len(prompt) - 3
                current_content_result = []
                current_content = []

                # pop result in current content
                while len(result_list) > 0:
                    print("result[start]: ", result_list[0]["start"])
                    if (
                        result_list[0]["start"] > result_list[0]["end"]
                        or result_list[0]["end"] - result_list[0]["start"] > max_content_len
                    ):
                        raise DataError(
                            f"Error in result list. Invalid start or end location\
                                (start: {result_list[0]['start']}, end: {result_list[0]['end']}).\
                                    Please check the data in line {i} of {data_path}."
                        )
                    if result_list[0]["start"] < max_content_len:
                        if result_list[0]["end"] > max_content_len:
                            # Result-Cross case: result cross interval of content
                            # dynamic adjust max_content_len to escape Result-Cross case
                            logger.debug(
                                f"Result-Cross: content: {json_line['content']}, prompt: {prompt}, result_list: {json_line['result_list']}.\n\
                                Result-Cross in {result_list[0]}."
                            )
                            max_content_len = result_list[0]["start"]
                            result_list[0]["start"] -= max_content_len
                            result_list[0]["end"] -= max_content_len
                            break
                        else:
                            current_content_result.append(result_list.pop(0))
                    else:
                        result_list[0]["start"] -= max_content_len
                        result_list[0]["end"] -= max_content_len
                        break  # result list is sorted by start

                current_content = content[:max_content_len]
                content = content[max_content_len:]
                accumulate_token += max_content_len

                yield {
                    "content": current_content,
                    "result_list": current_content_result,
                    "prompt": prompt,
                }


def get_dynamic_max_length(examples, default_max_length: int, dynamic_max_length: List[int]) -> int:
    """get max_length by examples which you can change it by examples in batch"""
    cur_length = len(examples[0]["input_ids"])
    max_length = default_max_length
    for max_length_option in sorted(dynamic_max_length):
        if cur_length <= max_length_option:
            max_length = max_length_option
            break
    return max_length


def convert_to_uie_format(
    data: Dict[str, str] = None,
    tokenizer: Any = None,
    max_seq_length: int = 512,
    dynamic_max_length: Optional[List[int]] = None,
    multilingual: Optional[bool] = False,
) -> Dict[str, Union[str, float]]:

    pass
