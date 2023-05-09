import os
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any, Dict, Union, Tuple
from paddlenlp.trainer import TrainingArguments

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from paddlenlp.utils.log import logger
from utils.exceptions import DataError, PreprocessingError
from config import BaseConfig

base_config = BaseConfig()


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
    multilingual: bool = field(default=False, metadata={"help": "Whether the model is a multilingual model."})


# information extraction (IE) training arguments
@dataclass
class IETrainingArguments(TrainingArguments):
    output_dir: str = field(default=None, metadata={"help": "The path where the checkpoint of the model is saved."})

    def __post_init__(self):
        if base_config.root_dir and self.output_dir is None:
            self.output_dir = base_config.root_dir + base_config.train_result_path
        return super().__post_init__()


def read_finetune_data(data_path: str, max_seq_len: int = 512) -> Dict[str, str]:
    """
    Summary: 讀「透過 utils/split_labelstudio.py 分割的 .txt檔」，此 txt 檔格式和 UIE官方提供的doccano.py轉換後的格式一樣。
    Model Input Format: [CLS] Prompt [SEP] Content [SEP].
    Result-Cross case: result cross interval of each subcontent.

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

            # 3 means '[CLS] [SEP] [SEP] in [CLS] Prompt [SEP] Content [SEP]
            if max_seq_len <= len(prompt) + 3:
                raise ValueError("The value of max_seq_len is too small. Please set a larger value.")
            result_list = json_line["result_list"]
            accumulate_token = 0

            # start pop all subcontent (segment by max_seq_len)
            while len(content) > 0:
                max_content_len = max_seq_len - len(prompt) - 3
                current_content_result = []

                # pop result in subcontent
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
                            # Result-Cross case: using dynamic adjust max_content_len to escape the problem.
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

                yield {
                    "content": content[:max_content_len],
                    "result_list": current_content_result,
                    "prompt": prompt,
                }

                content = content[max_content_len:]
                accumulate_token += max_content_len


def drift_offsets_mapping(offset_mapping: Tuple[Tuple[int, int]]) -> Tuple[List[List[int]], int]:
    """Scale the offset_mapping in tokenization output to align with the prompt learning format.

    Note: 因為 tokenize 後有些字會被 tokenize 在一起，所以 index 會和原本的有所差異，因此需做調整，將 tokenize 前後的 index 對齊。

    Args:
        offset_mapping (Tuple[Tuple[int, int]]): Tokenization outpu. Use argument 'return_offsets_mapping=True'.

    Returns:
        1. List[List[int, int]]: Scaled format, which is to adjust index after adding '[CLS] prompt [SEP]'.
        2. int: Drift term, which defines the scaling of drift after adjustment.
    """

    offset_mapping = [list(x) for x in offset_mapping]
    drift = 0
    for index in range(1, len(offset_mapping)):
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and drift == 0:
            drift = offset_mapping[index - 1][1] + 1  # [SEP] token
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += drift
        offset_mapping[index][1] += drift
    return offset_mapping, drift


def align_to_offset_mapping(origin_index: int, offset_mapping: List[List[int]]) -> int:
    """Align the original index (start/end index in result_list) to tokenized (offset_mapping) index.

    Args:
        origin_index (int): start/end index in result_list.
        offset_mapping (List[List[int, int]]): offset_mapping index after tokenization.

    Raises:
        PreprocessingError: Cannot find original index in offset_mapping.

    Returns:
        int: Aligned index.
    """

    for index, span in enumerate(offset_mapping):
        if span[0] <= origin_index < span[1]:
            return index

    raise PreprocessingError(f"Not found origin_index: {origin_index} in offset_mapping")


def convert_to_uie_format(
    data: Dict[str, str],
    tokenizer: Any,
    max_seq_len: int = 512,
    multilingual: Optional[bool] = False,
) -> Dict[str, Union[str, float]]:
    """此方法主要做兩件事情：
    1. Tokenization.
    2. 將 result_list 的 start/end index 對齊 tokenization 後的位置。

    Note: 在 finetune.py 中，設定此方法為預設 Callback Function，可根據任務或模型換成自定義方法。

    Args:
        data (Dict[str, str], optional): 切片後的文本，通常來自於 read_finetune_data() 的結果
            格式為 {"content": subcontent, "result_list": result_list_in_subcontent, "prompt": prompt}.
        tokenizer (Any, optional): paddlenlp.transformers.AutoTokenizer
        max_seq_len (int, optional): 切片文本的最大長度，通常與 read_finetune_data() 一致，truncation 預設為 True. Defaults to 512.
        multilingual (Optional[bool], optional): Whether the model is a multilingual model. Defaults to False.

    Returns:
        Dict[str, Union[str, float]]: 模型真正的 input 格式。
    """

    # Tokenization and Concate to the following format: [CLS] prompt [SEP] content [SEP]
    encoded_inputs = tokenizer(
        text=[data["prompt"]],
        text_pair=[data["content"]],
        truncation=True,
        max_seq_len=max_seq_len,
        pad_to_max_seq_len=True,
        return_attention_mask=True,
        return_position_ids=True,
        return_dict=False,
        return_offsets_mapping=True,
    )[0]

    # initialize start_ids, end_ids as 0.0
    start_ids, end_ids = map(lambda x: x * max_seq_len, ([0.0], [0.0]))

    # adjust offset_mapping
    adjusted_offset_mapping, drift = drift_offsets_mapping(offset_mapping=encoded_inputs["offset_mapping"])

    # align original index to tokenized (offset_mapping) index
    for item in data["result_list"]:
        aligned_start_index = align_to_offset_mapping(item["start"] + drift, adjusted_offset_mapping)
        aligned_end_index = align_to_offset_mapping(item["end"] - 1 + drift, adjusted_offset_mapping)
        start_ids[aligned_start_index] = 1.0
        end_ids[aligned_end_index] = 1.0

    return (
        {
            "input_ids": encoded_inputs["input_ids"],
            "position_ids": encoded_inputs["position_ids"],
            "start_positions": start_ids,
            "end_positions": end_ids,
        }
        if multilingual
        else {
            "input_ids": encoded_inputs["input_ids"],
            "token_type_ids": encoded_inputs["token_type_ids"],
            "position_ids": encoded_inputs["position_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
            "start_positions": start_ids,
            "end_positions": end_ids,
        }
    )
