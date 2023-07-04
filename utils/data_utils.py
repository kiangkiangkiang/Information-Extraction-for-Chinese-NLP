import json
import os
from typing import Optional, List, Any, Dict, Union, Tuple, Iterator
from paddlenlp.utils.log import logger
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from .exceptions import DataError, PreprocessingError


def read_data_by_chunk(data_path: str, max_seq_len: int = 512) -> Iterator[Dict[str, str]]:
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

    if not os.path.exists(data_path):
        raise ValueError(f"Path not found {data_path}.")

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line["content"].strip()
            prompt = json_line["prompt"]

            # 3 means '[CLS] [SEP] [SEP]' in [CLS] Prompt [SEP] Content [SEP]
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
                            f"Error in result list. Invalid start or end location (start: {result_list[0]['start']}, end: {result_list[0]['end']}). Please check the data in {data_path}."
                        )
                    if result_list[0]["start"] < max_content_len:
                        if result_list[0]["end"] > max_content_len:
                            # Result-Cross case: using dynamic adjust max_content_len to escape the problem.
                            logger.debug(f"Result-Cross. result: {result_list[0]}.")
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
                        break

                for each_result in current_content_result:
                    adjust_data = content[:max_content_len][each_result["start"] : each_result["end"]]
                    true_data = each_result["text"]
                    if adjust_data != true_data:
                        raise PreprocessingError(f"adjust error. adjust_data: {adjust_data}, true_data: {true_data}.")

                yield {
                    "content": content[:max_content_len],
                    "result_list": current_content_result,
                    "prompt": prompt,
                }

                content = content[max_content_len:]
                accumulate_token += max_content_len


def drift_offsets_mapping(offset_mapping: Tuple[Tuple[int, int]]) -> Tuple[List[List[int]], int]:
    """Scale the offset_mapping in tokenization output to align with the prompt learning format.

    Note: 因為 tokenization 後有些字會被 tokenize 在一起，所以 index 會和原本的有所差異，因此需做調整，將 tokenize 前後的 index 對齊。

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
    return final_index + 1


def convert_to_uie_format(
    data: Dict[str, str],
    tokenizer: Any,
    max_seq_len: int = 512,
    multilingual: Optional[bool] = False,
) -> Dict[str, Union[str, float]]:
    """此方法功能如下：
        1. Tokenization.
        2. 將 result_list 的 start/end index 對齊 tokenization 後的位置。

    Note:
        在 finetune.py 中，設定此方法為預設 Callback Function，可根據任務或模型換成自定義方法。

    ** Tokenize Bug **
        - Tokenizer 可能會因為中文的一些字 Unknown 導致 Bug (UIE Pretrained Model 並無此問題)。
        - 可參考：https://github.com/PaddlePaddle/PaddleNLP/issues?q=is%3Aissue+is%3Aopen+out+of+range+
        - 實測後可能有 Bug 的模型包含 xlnet, roformer.
        - 實測後正常的模型包含 uie, bert.

    Args:
        data (Dict[str, str], optional): 切片後的文本，通常來自於 () 的結果
            格式為 {"content": subcontent, "result_list": result_list_in_subcontent, "prompt": prompt}.
        tokenizer (Any, optional): paddlenlp.transformers.AutoTokenizer
        max_seq_len (int, optional): 切片文本的最大長度，通常與 () 一致，truncation 預設為 True. Defaults to 512.
        multilingual (Optional[bool], optional): Whether the model is a multilingual model. Defaults to False.

    Returns:
        Dict[str, Union[str, float]]: 模型真正的 input 格式。
    """
    if not data:
        return None

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
        logger.debug(f"Tokenizer Bug, content: {data['prompt']}")
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

    start_ids, end_ids = map(lambda x: x * max_seq_len, ([0.0], [0.0]))

    # adjust offset_mapping
    adjusted_offset_mapping, drift = drift_offsets_mapping(offset_mapping=encoded_inputs["offset_mapping"])

    # align original index to tokenized (offset_mapping) index
    for item in data["result_list"]:
        aligned_start_index = align_to_offset_mapping(item["start"] + drift, adjusted_offset_mapping)
        aligned_end_index = align_to_offset_mapping(item["end"] - 1 + drift, adjusted_offset_mapping)
        start_ids[aligned_start_index] = 1.0
        end_ids[aligned_end_index] = 1.0

    return {
        "input_ids": encoded_inputs["input_ids"],
        "token_type_ids": encoded_inputs["token_type_ids"],
        "position_ids": encoded_inputs["position_ids"],
        "attention_mask": encoded_inputs["attention_mask"],
        "start_positions": start_ids,
        "end_positions": end_ids,
    }


def create_data_loader(dataset, mode="train", batch_size=16, trans_fn=None, shuffle=False):
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

    shuffle = True if mode == "train" else False
    if mode == "train":
        sampler = DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = DataLoader(dataset, batch_sampler=sampler, return_list=True)
    return dataloader
