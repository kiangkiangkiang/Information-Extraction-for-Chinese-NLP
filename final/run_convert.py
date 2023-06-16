from utils.json_utils import convert_format, shuffle_data, set_seed, regularize_json_file
from config.base_config import NER_type
from paddlenlp.utils.log import logger
from typing import List, Tuple
import json
import os
from decimal import Decimal
import argparse


def do_split(
    dataset: List[dict],
    split_ratio: List[int] = [0.8, 0.1, 0.1],
    is_shuffle: bool = True,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """分割資料的邏輯程式

    Args:
        dataset (List[dict]): label studio output的json檔
        split_ratio (List[int], optional): 分割資料，train/dev/test，加總 = 1. Defaults to [0.8, 0.1, 0.1].
        is_shuffle (bool, optional): 是否隨機打亂資料. Defaults to True.

    Raises:
        ValueError: 資料集太小或分割比例太小，導致沒有training資料。

    Returns:
        Tuple[List[dict], List[dict], List[dict]]: 回傳training data，eval data，testing data．
    """

    logger.debug(f"in do_split, len(dataset)={len(dataset)}")

    if is_shuffle:
        dataset = shuffle_data(dataset)

    p1 = round(len(dataset) * split_ratio[0])
    p2 = round(len(dataset) * (split_ratio[0] + split_ratio[1]))

    if p1 <= 0:
        raise ValueError(f"Number of training data is too small {p1} <= 0")

    return (
        convert_format(dataset[:p1], NER_type, is_shuffle),
        convert_format(dataset[p1:p2], NER_type, is_shuffle),
        convert_format(dataset[p2:], NER_type, is_shuffle),
    )


# main function
def split_labelstudio(
    labelstudio_file: str,
    labelstudio_list: list = None,
    save_dir: str = "./",
    seed: int = 100,
    split_ratio: List[int] = [0.8, 0.1, 0.1],
    is_shuffle: bool = True,
) -> None:
    """主要轉換的程式，把label studio output (only json, \
        only NER (Relation Extraction: NER))轉換成模型所吃的input。

    Args:
        labelstudio_file (str): label studio output的檔案。Default 在label_data/ 內。
        save_dir (str, optional): 轉換後的training/eval/testing資料. Defaults 在information_extraction/data/內.
        seed (int, optional): 固定種子. Defaults to 100.
        split_ratio (List[int], optional): 分割資料，train/eval/test，加總 = 1. Defaults to [0.8, 0.1, 0.1].
        is_shuffle (bool, optional): 是否隨機打亂資料. Defaults to True.

    Raises:
        ValueError: 找不label studio檔案。
        ValueError: split_ratio長度不等於3，若不分割資料可用[1, 0, 0]設定。
        ValueError: split_ratio加總不等於1。
    """

    logger.info(f"Converting {os.path.basename(labelstudio_file)} into {save_dir}...")
    set_seed(seed)

    if not os.path.exists(save_dir):
        logger.warning(f"{save_dir} not found. Automatically making a directory...")
        os.makedirs(save_dir)

    if len(split_ratio) != 3:
        raise ValueError(
            f"Length Error in split_ratio: {len(split_ratio)}. Only accept len(split_ratio)==3 for splits."
        )

    if Decimal(str(split_ratio[0])) + Decimal(str(split_ratio[1])) + Decimal(str(split_ratio[2])) != Decimal("1"):
        raise ValueError("Please set correct split_ratio, sum of elements in split_ratio should be equal to 1.")

    if labelstudio_list:
        splitted_data = do_split(dataset=labelstudio_list, split_ratio=split_ratio, is_shuffle=is_shuffle)
    else:
        if not os.path.exists(labelstudio_file):
            raise ValueError(
                f"Label studio file not found in {labelstudio_file}. Please input the correct path of label studio file."
            )

        with open(labelstudio_file, "r", encoding="utf-8") as infile:
            for content in infile:
                dataset = json.loads(content)
            splitted_data = do_split(dataset=dataset, split_ratio=split_ratio, is_shuffle=is_shuffle)

    for data, data_name in zip(splitted_data, ("train.txt", "dev.txt", "test.txt")):
        logger.debug(f"len({data_name}) = {len(data)}")
        with open(save_dir + data_name, "w", encoding="utf-8") as outfile:
            for item in data:
                outline = json.dumps(item, ensure_ascii=False)
                outfile.write(outline + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labelstudio_file",
        default="./data/label_studio_data/label_studio_output.json",
        type=str,
        help="The export file path of label studio, only support the JSON format.",
    )
    parser.add_argument(
        "--save_dir", default="./data/model_input_data/", type=str, help="The path of data that you wanna save."
    )
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization")
    parser.add_argument(
        "--split_ratio",
        default=[0.8, 0.1, 0.1],
        type=float,
        nargs="*",
        help="The ratio of samples in datasets. [0.7, 0.2, 0.1] means 70% samples used for training, 20% for evaluation and 10% for test.",
    )
    parser.add_argument(
        "--is_shuffle",
        choices=["True", "False"],
        default="True",
        type=str,
        help="Whether to shuffle the labeled dataset, defaults to True.",
    )
    parser.add_argument("--is_regularize_data", default="True", type=str, help="The path of data that you wanna save.")
    args = parser.parse_args()

    regular_result = None
    if eval(args.is_regularize_data):
        regularized_result = regularize_json_file(json_file=args.labelstudio_file, out_variable=True)

    split_labelstudio(
        labelstudio_file=args.labelstudio_file,
        labelstudio_list=regular_result,
        save_dir=args.save_dir,
        seed=args.seed,
        split_ratio=args.split_ratio,
        is_shuffle=eval(args.is_shuffle),
    )