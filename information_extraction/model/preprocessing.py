import os
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any, Dict, Union, Tuple
from paddlenlp.trainer import TrainingArguments
import numpy as np
from collections import defaultdict
import pandas as pd

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from paddlenlp.utils.log import logger
from paddle.io import DistributedBatchSampler, BatchSampler, DataLoader
from utils.exceptions import DataError, PreprocessingError
from config import BaseConfig

base_config = BaseConfig()
SOLATIUM_WORD = ["精神", "慰撫", "撫慰", "非財產"]
# INCOME_WORD = ['薪', '收入', '所得', '俸']
# MEDICAL_WORD = []
# THER_WORD = []
MUST_SAMPLE_LIST = ["元"]  # 有提到錢的樣本都一定要sample進來


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

    down_sampling_ratio: Optional[float] = field(
        default=0.3,
        metadata={"help": "The drop out ratio of negative samples."},
    )
    read_data_method: Optional[str] = field(
        default="chunk",
        metadata={
            "help": "The argument determines how to deal with the input content. If full, "
            "it will read the full content and send it as input to the model. If chunk, it will"
            "cut the content into several chunks and send them as an individual observation to the model."
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


def random_choose(prob: float = 0.5) -> bool:
    criterion = np.random.uniform(0, 1, 1)[0]
    return True if criterion < prob else False


# 目前只有實作精神慰撫金
def decide_if_do_augmentation(
    content,
    result_list,
    aug_data_type="negative",
    aug_criterion=SOLATIUM_WORD,
) -> bool:

    if aug_data_type not in ["negative", "positive", "both"]:
        logger.warning(
            f"Data Augmentation args aug_data_type cannot be {aug_data_type}. \
            Must be one of the following args: ['negative', 'positive', 'both']. \
                Automatically change aug_data_type to 'negative'."
        )
        aug_data_type = "negative"

    if aug_data_type == "negative" or "both":
        if len(result_list) == 0:
            criterion = pd.unique([content.find(special_word) for special_word in aug_criterion])
            if len(criterion) > 1 and content.find("元") != -1:  # 有出現
                logger.debug("in more sample")
                return True

    if aug_data_type == "positive" or "both":
        if len(result_list) > 0:
            return True
    return False


def read_data_by_chunk(data_path: str, max_seq_len: int = 512, data_type="train") -> Dict[str, str]:
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

    debug_for_sample_ratio = {i: 0 for i in base_config.ner_type}
    total_for_sample_ratio = {i: 0 for i in base_config.ner_type}
    total_num = 0
    debug_for_sample_ratio["Total Ratio"] = 0
    AUGMENTATION_LIMIT = 3  # 2: 2倍
    augmentation_counter = AUGMENTATION_LIMIT - 1

    if data_type not in ["train", "evaluation", "test"]:
        raise ValueError(f"data_type must be in ['train', 'evaluation', 'test'].")

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line["content"].strip()
            prompt = json_line["prompt"]

            # 3 means '[CLS] [SEP] [SEP] in [CLS] Prompt [SEP] Content [SEP]
            if max_seq_len <= len(prompt) + 3:
                raise ValueError("The value of max_seq_len is too small. Please set a larger value.")
            result_list = json_line["result_list"]
            accumulate_token = 0

            # start pop all subcontent (segment by max_seq_len)
            # logger.debug(f"max_seq_len = {max_seq_len}")
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
                            logger.debug(f"Result-Cross. result: {result_list[0]}, content:{content}")
                            max_content_len = result_list[0]["start"]
                            result_list[0]["start"] -= max_content_len
                            result_list[0]["end"] -= max_content_len
                            break
                        else:
                            current_content_result.append(result_list.pop(0))
                            if result_list:
                                result_list[0]["start"] -= accumulate_token
                                result_list[0]["end"] -= accumulate_token
                    else:
                        result_list[0]["start"] -= max_content_len
                        result_list[0]["end"] -= max_content_len
                        break  # result list is sorted by start

                # TODO normalize content[:max_content_len]
                # from paddlenlp.transformers.tokenizer_utils import normalize_chars
                # normalize_chars(content[:max_content_len])

                # data augmentation (暫時只實作慰撫金)
                if data_type == "train" and prompt == "精神慰撫金額":
                    if decide_if_do_augmentation(content=content[:max_content_len], result_list=current_content_result):
                        augmentation_counter = 1

                while augmentation_counter < AUGMENTATION_LIMIT:
                    truncate_result = {
                        "content": content[:max_content_len],
                        "result_list": current_content_result,
                        "prompt": prompt,
                    }

                    total_num += 1
                    total_for_sample_ratio[prompt] += 1
                    if len(current_content_result) > 0:
                        debug_for_sample_ratio[prompt] += 1
                        debug_for_sample_ratio["Total Ratio"] += 1

                    for each_result in truncate_result["result_list"]:
                        adjust_data = truncate_result["content"][each_result["start"] : each_result["end"]]
                        true_data = each_result["text"]
                        if adjust_data != true_data:
                            raise PreprocessingError(
                                f"adjust error. adjust_data: {adjust_data}, true_data: {true_data}."
                            )
                    # logger.debug(f"debug for preprocessing, truncate_result={truncate_result}")
                    yield truncate_result
                    augmentation_counter += 1
                # content = content[max_content_len:]
                # accumulate_token += max_content_len

                content = content[max_content_len:]
                accumulate_token += max_content_len
                augmentation_counter = AUGMENTATION_LIMIT - 1

        logger.debug(f"Total number of content (after chunk) of {data_path} is {total_num}. ")
        try:
            for i in range(len(total_for_sample_ratio)):
                k = list(debug_for_sample_ratio.keys())[i]
                v1 = list(debug_for_sample_ratio.values())[i]
                v2 = list(total_for_sample_ratio.values())[i]
                logger.debug(f"Ratio of {k}: {v1}/{v2} = {v1/v2}. ")
            logger.debug(
                f"Total Ratio: {debug_for_sample_ratio['Total Ratio']}/{total_num} = {debug_for_sample_ratio['Total Ratio']/total_num}"
            )
        except Exception as e:
            # Common reason: Some ner_type is not occur in data. (Missing some ner_type)
            logger.error(e)


def read_full_data(data_path: str) -> Tuple[Dict[str, str], int]:
    logger.info(f"Read Full Data method is selected. The max_seq_len argument will be ignore...")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            yield {
                "content": json_line["content"].strip(),
                "result_list": json_line["result_list"],
                "prompt": json_line["prompt"],
            }


def get_max_content_len(data_path_list: List[str]) -> int:
    max_content_len = -1
    for each_data_path in data_path_list:
        with open(each_data_path, "r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line)
                if len(json_line["content"].strip()) > max_content_len:
                    max_content_len = len(json_line["content"].strip())
    return max_content_len


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
    final_index = 0
    for index, span in enumerate(offset_mapping):
        if span[0] <= origin_index < span[1]:
            return index
        if span[0] != 0 and span[1] != 0:
            final_index = index

    # raise PreprocessingError(f"Not found origin_index: {origin_index} in offset_mapping")
    return final_index + 1


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
        data (Dict[str, str], optional): 切片後的文本，通常來自於 () 的結果
            格式為 {"content": subcontent, "result_list": result_list_in_subcontent, "prompt": prompt}.
        tokenizer (Any, optional): paddlenlp.transformers.AutoTokenizer
        max_seq_len (int, optional): 切片文本的最大長度，通常與 () 一致，truncation 預設為 True. Defaults to 512.
        multilingual (Optional[bool], optional): Whether the model is a multilingual model. Defaults to False.

    Returns:
        Dict[str, Union[str, float]]: 模型真正的 input 格式。
    """

    # Tokenization and Concate to the following format: [CLS] prompt [SEP] content [SEP]
    try:
        encoded_inputs = tokenizer(
            text=[data["prompt"]],
            text_pair=[data["content"]],
            truncation=True,
            max_seq_len=max_seq_len,
            pad_to_max_seq_len=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_position_ids=True,
            return_dict=False,
            return_offsets_mapping=True,
        )[0]
    except Exception as e:
        logger.error(f"Tokenizer Error: {e}")
        encoded_inputs = tokenizer(
            text=[data["prompt"]],
            text_pair=["無文本"],
            truncation=True,
            max_seq_len=max_seq_len,
            pad_to_max_seq_len=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_position_ids=True,
            return_dict=False,
            return_offsets_mapping=True,
        )[0]
        data["result_list"] = []

    #  len(tokenizer(text=[data["prompt"]], text_pair=[data["content"]])['input_ids'][0])

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


def convert_to_full_data_format(
    data: Dict[str, str],
    tokenizer: Any,
    max_seq_len: int = 512,
    multilingual: Optional[bool] = False,
) -> Dict[str, Union[str, float]]:

    """TODO
    1. [DONE] split chunk by max_seq_len
    2. [DONE] tokenize each chunk
    3. [DONE] concate all chunk
    4. [DONE] adjust index by chunk (only for loss, model will not input)
    5. [DONE] make label from adjust_index mapping
    """

    max_content_len = (
        max_seq_len - len(data["prompt"]) - 3
    )  # 3 means [CLS] [SEP] [SEP] in [CLS] prompt [SEP] content [SEP]
    encoded_inputs = defaultdict(list)
    result_list = data["result_list"]
    accumulate_token = 0
    start_positions = []
    end_positions = []
    data_copy = data.copy()  # must to do, or data will be remove
    while data_copy["content"]:
        # 1. split chunk by max_content_len
        current_content_result = []
        # pop result in subcontent
        while len(result_list) > 0:
            if (
                result_list[0]["start"] > result_list[0]["end"]
                or result_list[0]["end"] - result_list[0]["start"] > max_content_len
            ):
                raise DataError(
                    f"Error in result list. Invalid start or end location\
                        (start: {result_list[0]['start']}, end: {result_list[0]['end']})."
                )
            if result_list[0]["start"] < max_content_len:
                if result_list[0]["end"] > max_content_len:
                    # Result-Cross case: using dynamic adjust max_content_len to escape the problem.
                    max_content_len = result_list[0]["start"]
                    result_list[0]["start"] -= max_content_len
                    result_list[0]["end"] -= max_content_len
                    break
                else:
                    current_content_result.append(result_list.pop(0))
                    if result_list:
                        result_list[0]["start"] -= accumulate_token
                        result_list[0]["end"] -= accumulate_token
            else:
                result_list[0]["start"] -= max_content_len
                result_list[0]["end"] -= max_content_len
                break  # result list is sorted by start

        # 2. tokenize each chunk
        tmp_inputs = tokenizer(
            text=[data_copy["prompt"]],
            text_pair=[data_copy["content"][:max_content_len]],
            max_seq_len=max_seq_len,
            pad_to_max_seq_len=True,
            return_attention_mask=True,
            return_position_ids=True,
            return_dict=False,
            return_offsets_mapping=True,
        )[0]

        # tokenizer([data_copy["prompt"]], [data_copy["content"][:max_content_len]], return_attention_mask=True, pad_to_max_seq_len=True, max_seq_len=max_seq_len, return_offsets_mapping=True, return_dict=False, return_position_ids=True)
        # 206, 224
        # tokenizer(data_copy["content"][225])
        # tokenizer.convert_ids_to_tokens(tokenizer(data_copy["content"][205])['input_ids'])
        # tokenizer(data_copy["content"][:205], return_offsets_mapping=True, max_seq_len=512)
        # tokenizer(data_copy["content"][206:224], return_offsets_mapping=True, max_seq_len=512)
        # xlnet的vocab沒有‘㈠’這個字，所以用xlnet會有bug

        # 3. concate all chunk
        for key in tmp_inputs:
            encoded_inputs[key].extend(tmp_inputs[key])

        accumulate_token += max_content_len
        data_copy["content"] = data_copy["content"][max_content_len:]

        start_ids, end_ids = map(lambda x: x * max_seq_len, ([0.0], [0.0]))

        if current_content_result:
            # adjust offset_mapping
            adjusted_offset_mapping, drift = drift_offsets_mapping(offset_mapping=tmp_inputs["offset_mapping"])

            # align original index to tokenized (offset_mapping) index
            for item in current_content_result:

                aligned_start_index = align_to_offset_mapping(item["start"] + drift, adjusted_offset_mapping)
                aligned_end_index = align_to_offset_mapping(item["end"] + drift, adjusted_offset_mapping)
                if aligned_start_index == -1 or aligned_end_index == -1:
                    logger.error("-1 is happened")
                adjust_ans = "".join(
                    tokenizer.convert_ids_to_tokens(tmp_inputs["input_ids"][aligned_start_index:aligned_end_index])
                )
                # "".join(tokenizer.convert_ids_to_tokens(encoded_inputs["input_ids"][:]))
                if adjust_ans != item["text"]:
                    logger.error(f"After adjust answer: {adjust_ans}, True answer: {item['text']}")
                start_ids[aligned_start_index] = 1.0
                end_ids[aligned_end_index] = 1.0
        start_positions.extend(start_ids)
        end_positions.extend(end_ids)

    return {
        "input_ids": encoded_inputs["input_ids"],
        "token_type_ids": encoded_inputs["token_type_ids"],
        "position_ids": encoded_inputs["position_ids"],
        "attention_mask": encoded_inputs["attention_mask"],
        "start_positions": start_positions,
        "end_positions": end_positions,
    }


def get_base_config():
    return base_config


def create_data_loader(dataset, batch_size=16, trans_fn=None, shuffle=False):
    """
    Create dataloader.
    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)
    sampler = BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = DataLoader(dataset, batch_sampler=sampler, return_list=True)
    return dataloader


"""
test = ("./Chinese-Verdict-NLP/information_extraction/data/eval_data.txt")
s = next(test);s
s
s["content"][431:437]

len(s["content"])
s["content"][431:437]
"""


def convert_example(
    data: Dict[str, str],
    tokenizer: Any,
    max_seq_len: int = 512,
    multilingual: Optional[bool] = False,
):
    """
    data: {
        title
        prompt
        content
        result_list
    }
    """
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
    )
    start_ids = [0.0 for x in range(max_seq_len)]
    end_ids = [0.0 for x in range(max_seq_len)]

    encoded_inputs = encoded_inputs[0]
    offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
    bias = 0
    for index in range(1, len(offset_mapping)):
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
            bias = offset_mapping[index - 1][1] + 1  # Includes [SEP] token
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += bias
        offset_mapping[index][1] += bias

    def map_offset(ori_offset, offset_mapping):
        """
        map ori offset to token offset
        """
        for index, span in enumerate(offset_mapping):
            if span[0] <= ori_offset < span[1]:
                return index
        return -1

    for item in data["result_list"]:
        start = map_offset(item["start"] + bias, offset_mapping)
        end = map_offset(item["end"] - 1 + bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0
    if multilingual:
        tokenized_output = {
            "input_ids": encoded_inputs["input_ids"],
            "position_ids": encoded_inputs["position_ids"],
            "start_positions": start_ids,
            "end_positions": end_ids,
        }
    else:
        tokenized_output = {
            "input_ids": encoded_inputs["input_ids"],
            "token_type_ids": encoded_inputs["token_type_ids"],
            "position_ids": encoded_inputs["position_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
            "start_positions": start_ids,
            "end_positions": end_ids,
        }
    return tokenized_output


""" Tokenizer list out of range bug
a = '當時根據病例記錄應仍臥床可坐輪椅，但仍有傷口需要照護。臥床病人因呼吸道清除功能失效及排尿能力不完整時，就容易出現感染問題。根據2014年CriticalCareNurseVol34,No.6（附件2）文獻建議，出院後壓瘡照護需提供病人足夠熱量、蛋白質和平衡電解質攝取量，使病人身體皮膚組織增生和增強防禦能力，同時藉由皮膚滋潤與減壓措施才能預防壓瘡加劇。病人出院後於門診追蹤及105年9月19日至急診更換鼻胃管時均可坐輪椅，但因病人分別於105年7月22日及105年12月31日至106年1月14日因尿道感染、肺炎、褥瘡感染及惡病質等原因於中國醫藥大學附設醫院住院，出院後完全臥床，所以病人更容易出現呼吸道清除功能失效及排尿能力不完整，需藉由照護提供營養，翻身與痰液清除及預防脫水等才能避免病情再發生及惡化，治療出院後12天於106年1月26日因敗血症休克呼吸哀竭再次住院，除肺炎外，褥瘡感染及惡病質情形均相當嚴重，雖經院方管灌營養、傷口照護、抗生素使用與呼吸器支持，病人仍因多重器官衰竭心跳停止於2月23日上午8：51分宣布死亡。根據2009年外傷（Injury）雜誌（附件3）整合性文章認為多發傷患者的死'  
len(a)
from paddlenlp.transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("roformer-chinese-base")
t('無文本', return_offsets_mapping=True)


a = '當時根據病例記錄應仍臥床可坐輪椅，但仍有傷口需要照護。臥床病人因呼吸道清除功能失效及排尿能力不完整時，就容易出現感染問題。根據2014年CriticalCareNurseVol34,No.6（附件2）文獻建議，出院後壓瘡照護需提供病人足夠熱量、蛋白質和平衡電解質攝取量，使病人身體皮膚組織增生和增強防禦能力，同時藉由皮膚滋潤與減壓措施才能預防壓瘡加劇。病人出院後於門診追蹤及105年9月19日至急診更換鼻胃管時均可坐輪椅，但因病人分別於105年7月22日及105年12月31日至106年1月14日因尿道感染、肺炎、褥瘡感染及惡病質等原因於中國醫藥大學附設醫院住院，出院後完全臥床，所以病人更容易出現呼吸道清除功能失效及排尿能力不完整，需藉由照護提供營養，翻身與痰液清除及預防脫水等才能避免病情再發生及惡化，治療出院後12天於106年1月26日因敗血症休克呼吸哀竭再次住院，除肺炎外，褥瘡感染及惡病質情形均相當嚴重，雖經院方管灌營養、傷口照護、抗生素使用與呼吸器支持，病人仍因多重器官衰竭心跳停止於2月23日上午8：51分宣布死亡。根據2009年外傷（Injury）雜誌（附件3）整合性文章認為多發傷患者的死'  
from paddlenlp.transformers import AutoTokenizer
t_pd = AutoTokenizer.from_pretrained("roformer-chinese-base")
t_pd(a)['input_ids']
from paddlenlp.transformers.tokenizer_utils import normalize_chars
a2 = normalize_chars(a)
a
a2
t_pd(a2, return_offsets_mapping=True)
t_pd.convert_ids_to_tokens(t_pd(a)['input_ids'])
"""
