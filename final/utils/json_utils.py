import os
import json
import numpy as np
import pandas as pd
import paddle
import random
from typing import List, Optional, Dict
from paddlenlp.utils.log import logger

# TODO merge data 和 split data into train/test 時，import exceptions會有問題 （相對路徑）
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
    # TODO docstring 增加變數entity_type
    """轉換格式邏輯程式，將 label studio output 轉換成 UIE 模型所吃的格式。

    Args:
        dataset (List[dict]): label studio output的json檔。
        is_shuffle (bool, optional): 是否隨機打亂資料. Defaults to True.

    Raises:
        ValueError: 任務格式錯誤（此邏輯程式只處理 NER 任務，有 Relation 標籤無法處理）。

    Returns:
        List[dict]: 模型所吃的訓練格式。
    """

    logger.debug(f"In convert_format, data len = {len(dataset)}")
    results = []
    for data in dataset:
        uie_format = {
            output_type: {"content": data["data"]["text"], "result_list": [], "prompt": output_type}
            for output_type in entity_type
        }
        for label_result in data["annotations"][0]["result"]:
            if label_result["type"] != "labels":
                breakpoint()
                raise ValueError(
                    "Now we only deal with NER tasks, " "which means the type of label studio result is 'labels'."
                )

            uie_format[label_result["value"]["labels"][0]]["result_list"].append(
                {
                    "text": label_result["value"]["text"],
                    "start": label_result["value"]["start"],
                    "end": label_result["value"]["end"],
                }
            )
        # add_negative_samples(uie_format)# if result == []
        results.extend(uie_format.values())
    return shuffle_data(results) if is_shuffle else results


def merge_json(json_folder_path: str, output_path: Optional[str] = "./") -> None:
    """Merge JSON files.

    Args:
        json_folder_path (str): The folder where all JSON files locate.
        output_path (Optional[str], optional): Where the merged JSON file locates. Defaults to "./".

    Raises:
        ValueError: File not found.
    """

    merge_result = []
    counter = 0
    if os.path.exists(json_folder_path):
        all_json_file = [json_file for json_file in os.listdir(json_folder_path) if json_file[-5:] == ".json"]
        logger.info(f"Find the following json file: {all_json_file}.")
        if len(all_json_file) > 0:
            # read
            for each_json in all_json_file:
                logger.info(f"Merging the file {each_json}...")
                with open(json_folder_path + "/" + each_json, "r", encoding="utf-8") as infile:
                    for f in infile:
                        all_content = json.loads(f)
                        logger.info(f"Length of {each_json} is {len(all_content)}.")
                        counter += len(all_content)
                        for content in all_content:
                            merge_result.append(content)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            with open(output_path + "/merged_data.json", "w", encoding="utf-8") as outfile:
                jsonString = json.dumps(merge_result, ensure_ascii=False)
                outfile.write(jsonString)

            logger.info(f"Successful merge all file (len={counter}) to {output_path}/merged_data.json.")

        else:
            raise ValueError("Cannot found json file.")
    else:
        raise ValueError(f"Cannot found the path {json_folder_path}")


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


def is_repeat_content_exist(json_file: str) -> dict:
    """Check if there are duplicate cases in the JSON file.

    Args:
        json_file (str): JSON file locate and name. Must have the same format as label studio output and ''jid' attributes exist.

    Raises:
        ValueError: File not found.

    Returns:
        dict: The information of duplicate cases in JSON file.
    """

    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as infile:
            for f in infile:
                all_content = json.loads(f)
        all_jid = [i["data"]["jid"] for i in all_content]
        unique_data_len = len(pd.unique(all_jid))
        all_content_len = len(all_content)
        logger.info(
            f"All data: {all_content_len}. Unique data: {unique_data_len}. Repeat data: {all_content_len - unique_data_len}."
        )
        if all_content_len != unique_data_len:
            logger.warning(f"Head 5 repeat data: {pd.value_counts(all_jid)[:5]}")
        return {
            "is_repeat": not (all_content_len == unique_data_len),
            "all_data_length": all_content_len,
            "unique_data_len": unique_data_len,
            "value_counts_table": pd.value_counts(all_jid),
        }
    else:
        raise ValueError(f"Cannot found the path {json_file}")


# remove "\n" and " "
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
    # TODO update doc string
    """Regularize the JSON file list.

    Args:
        json_file (str): A JSON file which contains several JSONs which want to be regularized.
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

    json_list = read_json(json_file)
    result_list = []
    for each_json in json_list:
        regularized_data = regularize_content(
            each_json, regularize_text=regularize_text, special_result_case=special_result_case
        )
        test_regularized_data(regularized_data)
        result_list.append(regularized_data)

    logger.info(f"End regularize all data.")

    if out_variable:
        return result_list
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(output_path + "/regularized_data.json", "w", encoding="utf-8") as outfile:
            jsonString = json.dumps(result_list, ensure_ascii=False)
            outfile.write(jsonString)
