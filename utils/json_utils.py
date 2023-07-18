import os
import json
import numpy as np
import paddle
import random
from typing import List, Optional
from paddlenlp.utils.log import logger
from .exceptions import ConvertingError
import re


def set_seed(seed: int) -> None:
    """設定種子

    Args:
        seed (int): 固定種子
    """

    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def shuffle_data(data: list) -> list:
    """shuffle data"""

    indexes = np.random.permutation(len(data))
    return [data[i] for i in indexes]


def convert_format(dataset: List[dict], entity_type: List[str], is_shuffle: bool = True) -> List[dict]:
    """轉換格式邏輯程式，將 label studio output 轉換成 UIE 模型所吃的格式。

    Args:
        dataset (List[dict]): label studio output的json檔。
        entity_type (List[str]): label studio 標注時所定義的所有 entity，例如「薪資收入」。
        is_shuffle (bool, optional): 是否隨機打亂資料. Defaults to True.

    Raises:
        ValueError: 任務格式錯誤（此邏輯程式只處理 NER 任務，有 Relation 標籤無法處理）。

    Returns:
        List[dict]: 模型所吃的訓練格式。
    """

    logger.info(f"Length of converting data: {len(dataset)}...")
    results = []
    for data in dataset:
        uie_format = {
            output_type: {"content": data["data"]["text"], "result_list": [], "prompt": output_type}
            for output_type in entity_type
        }
        for label_result in data["annotations"][0]["result"]:
            if label_result["type"] != "labels":
                raise ValueError(
                    "Now we only deal with NER tasks, "
                    "which means the type of label studio result is 'labels'. Please fix the input data type."
                )

            uie_format[label_result["value"]["labels"][0]]["result_list"].append(
                {
                    "text": label_result["value"]["text"],
                    "start": label_result["value"]["start"],
                    "end": label_result["value"]["end"],
                }
            )
        results.extend(uie_format.values())
    return shuffle_data(results) if is_shuffle else results


def read_json(json_file: str) -> None:
    """Read JSON files.

    Args:
        json_file (str): JSON files path and name.

    Raises:
        ValueError: File not found.

    """

    if os.path.exists(json_file):
        result = []
        with open(json_file, "r", encoding="utf-8") as infile:
            for f in infile:
                all_content = json.loads(f)
                for content in all_content:
                    result.append(content)
        return result
    else:
        raise ValueError(f"Cannot found the path {json_file}")


def regularize_content(
    single_json: str,
    regularize_text: Optional[List[str]] = ["\n", " ", "\u3000"],
    special_result_case: Optional[List[str]] = [r"\\n"],
) -> dict:
    """Regularize the keys of 'content' in the JSON file.
        The content may have several special tokens such as '\n', which does not want to be exist in the content.
        Therefore, this function will remove the special tokens and adjust the relatively index in the JSON file.

    Args:
        single_json (str): A JSON file which wants to be regularized. This file must be the same format with label studio output, and have the labels of NER tasks such as 'start' and 'end' index.
        regularize_text ((Optional[List[str]], optional): List of the special tokens. Each string in the list must be only one character (len(TOKEN) == 1).  Defaults to ["\n", " ", "\u3000"].
        special_result_case (Optional[List[str]], optional): Other special case which cannot be regularize in regularize_text. Defaults to [r"\n"].

    Returns:
        dict: Regularized file.
    """

    tmp = ""
    for i in regularize_text:
        tmp = tmp + i + "|"
    pattern = re.compile(tmp[:-1])

    if len(single_json["annotations"][0]["result"]) > 0:
        result_index = []
        # sorted result
        single_json["annotations"][0]["result"] = sorted(
            single_json["annotations"][0]["result"], key=lambda item: item["value"]["start"]
        )
        for i in single_json["annotations"][0]["result"]:
            result_index.append(i["value"]["start"])
            result_index.append(i["value"]["end"])

        logger.debug(f"result_index = {result_index}")

        # count the scale for result index
        special_token_counter = 0
        result_index_pointer = 0
        result_index_len = len(result_index)
        for i, char in enumerate(single_json["data"]["text"]):

            if i == result_index[result_index_pointer]:
                result_index[result_index_pointer] -= special_token_counter
                result_index_pointer += 1
                if result_index_pointer == result_index_len:
                    break
            if char in regularize_text:
                special_token_counter += 1

        # adjust result index
        result_index_pointer = 0
        for i in range(len(single_json["annotations"][0]["result"])):
            single_json["annotations"][0]["result"][i]["value"]["start"] = result_index[result_index_pointer]
            single_json["annotations"][0]["result"][i]["value"]["end"] = result_index[result_index_pointer + 1]
            single_json["annotations"][0]["result"][i]["value"]["text"] = re.sub(
                pattern, "", single_json["annotations"][0]["result"][i]["value"]["text"]
            )
            for u in special_result_case:
                single_json["annotations"][0]["result"][i]["value"]["text"] = re.sub(
                    u, "", single_json["annotations"][0]["result"][i]["value"]["text"]
                )

            result_index_pointer += 2

    single_json["data"]["text"] = re.sub(pattern, "", single_json["data"]["text"])
    return single_json


def regularize_json_file(
    json_file: str,
    out_variable: bool = False,
    output_path: str = "./",
    regularize_text: Optional[List[str]] = ["\n", " ", "\u3000"],
    special_result_case: Optional[List[str]] = [r"\\n"],
) -> None:
    """Regularize the JSON file list.

    Args:
        json_file (str): A JSON file path which contains several JSONs which want to be regularized.
        out_variable (bool): Defaults to False. If the output is a python varialble not write a file.
        output_path (str, optional): The path of regularized JSON. Defaults to "./".
        regularize_text (Optional[List[str]], optional): List of the special tokens. Each string in the list must be only one character (len(TOKEN) == 1).  Defaults to ["\n", " ", "\u3000"].
        special_result_case (Optional[List[str]], optional): Other special case which cannot be regularize in regularize_text. Defaults to [r"\n"].
    """

    if not os.path.exists(json_file):
        raise ValueError(
            f"Label studio file not found in {json_file}. Please input the correct path of label studio file."
        )

    def test_regularized_data(regularized_data: dict) -> bool:
        for i in range(len(regularized_data["annotations"][0]["result"])):
            start = regularized_data["annotations"][0]["result"][i]["value"]["start"]
            end = regularized_data["annotations"][0]["result"][i]["value"]["end"]
            adjusted_data = regularized_data["data"]["text"][start:end]
            true_data = regularized_data["annotations"][0]["result"][i]["value"]["text"]
            if adjusted_data != true_data:
                raise ConvertingError(
                    f"adjusted_data: {adjusted_data} is not equal to true_data: {true_data}. start:end = {start}:{end}"
                )

    for i in regularize_text:
        if len(i) != 1:
            raise ValueError("Default special token in regularize_text takes only 1 char!")

    logger.info(f"Start regularize ...")

    json_list = read_json(json_file)
    result_list = []
    for each_json in json_list:
        regularized_data = regularize_content(
            each_json, regularize_text=regularize_text, special_result_case=special_result_case
        )
        test_regularized_data(regularized_data)
        result_list.append(regularized_data)

    logger.info(f"Finish regularize data...")

    if out_variable:
        return result_list
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, "regularized_data.json"), "w", encoding="utf-8") as outfile:
            jsonString = json.dumps(result_list, ensure_ascii=False)
            outfile.write(jsonString)
