import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List, Any
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


def read_finetune_data(data_path, max_seq_len=512):
    """
    read json
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
            # max_content_len = max_seq_len - len(prompt) - 3
            # if len(content) <= max_content_len:
            #    yield json_line
            # else:
            result_list = json_line["result_list"]
            accumulate_token = 0

            # start pop all content
            # TODO test len(content) <= max_content_len:
            while len(content) > 0:
                max_content_len = max_seq_len - len(prompt) - 3
                current_content_result = []
                current_content = []

                # pop result in current content
                while len(result_list) > 0:
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
                            max_content_len = result_list[0]["start"]
                            break
                        else:
                            current_content_result.append(result_list.pop(0))
                    else:
                        break  # result list is sorted by start

                current_content = content[:max_content_len]
                content = content[max_content_len:]

                # update result index of start/end
                for result in current_content_result:
                    result["start"] -= accumulate_token
                    result["end"] -= accumulate_token

                accumulate_token += max_content_len

                yield {
                    "content": current_content,
                    "result_list": current_content_result,
                    "prompt": prompt,
                }


# en read_finetune_data
9902 / 512
9728
512 * 19
16 // 17
19 % 17
[(i, interval_start * 512) for i, interval_start in enumerate(range(9802 // 512))]


def reader(data_path, max_seq_len=512):
    """
    read json
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
            max_content_len = max_seq_len - len(prompt) - 3
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line["result_list"]
                json_lines = []
                accumulate = 0
                while True:
                    # content 會一直被砍, result_list也會一直被砍
                    # 反正這邊所有 while true 的邏輯應該都是慢慢砍掉的概念

                    cur_result_list = []
                    # 前處理result (跑遍所有result): 1. test不合理的result, 2. 測試有沒有
                    for result in result_list:
                        if result["end"] - result["start"] > max_content_len:
                            raise DataError(
                                f"result['end'] - result['start'] exceeds max_content_len,\
                                     which will result an invalid instance being returned.\
                                        Please check the data in line {i} of {data_path}."
                            )
                        # 這段要看完才懂，因為會一直砍，所以要看到後面他到底改了啥
                        if (
                            result["start"] + 1 <= max_content_len < result["end"]
                            and result["end"] - result["start"] <= max_content_len
                        ):
                            max_content_len = result["start"]
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]

                    while True:
                        # 砍完了 -> break
                        if len(result_list) == 0:
                            break

                        # 如果還沒砍完
                        elif result_list[0]["end"] <= max_content_len:
                            if result_list[0]["end"] > 0:
                                cur_result = result_list.pop(0)
                                cur_result_list.append(cur_result)
                            else:
                                cur_result_list = [result for result in result_list]
                                break
                        else:
                            break

                    json_line = {"content": cur_content, "result_list": cur_result_list, "prompt": prompt}
                    json_lines.append(json_line)

                    for result in result_list:
                        if result["end"] <= 0:
                            break
                        result["start"] -= max_content_len
                        result["end"] -= max_content_len
                    accumulate += max_content_len
                    max_content_len = max_seq_len - len(prompt) - 3
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        json_line = {"content": res_content, "result_list": result_list, "prompt": prompt}
                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content

                for json_line in json_lines:
                    yield json_line


def convert_to_uie_input(
    data: str = None,
    tokenizer: Any = None,
    multilingual: bool = False,
    max_seq_length: Optional[int] = 512,
    dynamic_max_length: Optional[List[int]] = None,
):

    pass
