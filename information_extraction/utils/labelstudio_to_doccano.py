import os
import json
import argparse
from base_utils import setup_config, get_default_data_path

setup_config()

from config import generate_logger

logger = generate_logger(__name__)


def do_convert(dataset):
    logger.debug("dataset len: " + str(len(dataset)))
    results = []
    outer_id = 0
    label_id = 0
    relation_id = 0
    for data in dataset:
        outer_id += 1
        item = {"id": outer_id, "text": data["data"]["text"], "entities": [], "relations": []}
        item, label_id, relation_id = append_attrs(data, item, label_id, relation_id)
        results.append(item)
    return results


def convert_to_doccano(
    labelstudio_file: str,
    doccano_file: str = "doccano_ext.jsonl",
):
    """把labelstudio的output（only json）轉換成paddleNLP內所使用的doccano格式，並且寫出檔案（doccano_file）

    Args:
        labelstudio_file (str): The export file path of label studio, only support the JSON format.
        doccano_file (str, optional): Saving path in doccano format.. Defaults to "doccano_ext.jsonl".

    Raises:
        ValueError: 找不到labelstudio_file檔案
    """

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

    parser.add_argument(
        "--labelstudio_file",
        type=str,
        help="The export file path of label studio, only support the JSON format.",
    )
    parser.add_argument("--doccano_file", type=str, default="doccano_ext.jsonl", help="Saving path in doccano format.")
    args = parser.parse_args()

    if args.labelstudio_file:
        convert_to_doccano(args.labelstudio_file, args.doccano_file)
    else:
        data_path = get_default_data_path()
        label_studio_data = os.listdir(data_path)
        for data in label_studio_data:
            convert_to_doccano(data_path + data, args.doccano_file)
