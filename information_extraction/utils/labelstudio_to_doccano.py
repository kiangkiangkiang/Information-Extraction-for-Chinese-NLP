import os
import json
import argparse
from base_utils import setup_config, get_root_dir
from exceptions import ConvertingError
from typing import List

setup_config()

from config import generate_logger

logger = generate_logger(__name__)


def do_convert(dataset: dict) -> List[dict]:
    """將label studio output對齊doccano格式的邏輯程式

    Args:
        dataset (dict): 用json.load進來的檔案（only for the following format
            1. output of label studio.
            2. json format.
            3. relation extraction task.
            ）

    Raises:
        ConvertingError: 轉換前後長度不一致

    Returns:
        List[dict]: doccano format of the output of label studio.
    """

    results = []
    outer_id = 0
    label_id = 0
    for data in dataset:
        outer_id += 1
        item = {"id": outer_id, "text": data["data"]["text"], "entities": [], "relations": []}
        for anno in data["annotations"][0]["result"]:
            if anno["type"] == "labels":
                label_id += 1
                item["entities"].append(
                    {
                        "id": label_id,
                        "label": anno["value"]["labels"][0],
                        "start_offset": anno["value"]["start"],
                        "end_offset": anno["value"]["end"],
                    }
                )
        results.append(item)
    if len(dataset) != len(results):
        raise ConvertingError("Length is not equal after convert.")
    return results


def convert_to_doccano(
    labelstudio_file: str,
    doccano_file: str = "doccano_ext.jsonl",
):
    """把labelstudio的output（only json）轉換成paddleNLP內所使用的doccano格式，並寫出檔案（doccano_file）

    Args:
        labelstudio_file (str): The export file path of label studio, only support the JSON format.
        doccano_file (str, optional): Saving path in doccano format.. Defaults to "doccano_ext.jsonl".

    Raises:
        ValueError: 找不到labelstudio_file檔案
    """

    logger.info(f"Converting {os.path.basename(labelstudio_file)} into {os.path.basename(doccano_file)}...")
    if not os.path.exists(labelstudio_file):
        raise ValueError("Label studio file not found. Please input the correct path of label studio file.")

    with open(labelstudio_file, "r", encoding="utf-8") as infile:
        for content in infile:
            dataset = json.loads(content)
        results = do_convert(dataset)

    with open(doccano_file, "w", encoding="utf-8") as outfile:
        for item in results:
            outline = json.dumps(item, ensure_ascii=False)
            outfile.write(outline + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logger.debug("Now os.getcwd()=" + os.getcwd())
    root_dir = None
    try:
        root_dir = get_root_dir(root_dir_name="Chinese-Verdict-NLP")
        default_doccano_file = root_dir + "/data/doccano/doccano_ext.jsonl"
    except:
        logger.warning("Fail to get root directory.")

    parser.add_argument(
        "--labelstudio_file",
        type=str,
        help="The export file path of label studio, only support the JSON format.",
    )
    parser.add_argument("--doccano_file", type=str, help="Saving path in doccano format.")
    args = parser.parse_args()

    if args.labelstudio_file:
        args.doccano_file = default_doccano_file if args.doccano_file is None else args.doccano_file
        convert_to_doccano(args.labelstudio_file, args.doccano_file)
    else:
        if root_dir is not None:
            data_path = root_dir + "/data/label_studio/"
            label_studio_data = os.listdir(data_path)
            logger.info(f"{len(label_studio_data)} label studio file will be convert...")
            if args.doccano_file is None:
                args.doccano_file = [os.path.splitext(data)[0] + "_doccano.jsonl" for data in label_studio_data]
                for data, output_file in zip(label_studio_data, args.doccano_file):
                    convert_to_doccano(data_path + data, root_dir + "/data/doccano/" + output_file)
            else:
                for data in label_studio_data:
                    convert_to_doccano(data_path + data, args.doccano_file)
        else:
            raise ValueError("Label studio file not found. Please input the correct path of label studio file.")
