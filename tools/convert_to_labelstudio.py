import json
import argparse
import datetime
import os
import logging
import sys
import colorlog
from tqdm import tqdm
from typing import List


LOGGER_LEVEL = logging.INFO


def create_logger(level=logging.DEBUG):
    log_config = {
        "DEBUG": {"level": 10, "color": "purple"},
        "INFO": {"level": 20, "color": "green"},
        "WARNING": {"level": 30, "color": "yellow"},
        "ERROR": {"level": 40, "color": "red"},
    }
    logger = logging.getLogger("convert_to_labelstudio.log")
    logger.setLevel(level)
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s - %(name)s: %(message)s",
        log_colors={key: conf["color"] for key, conf in log_config.items()},
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    return logger


def get_labelstudio_template(path: str) -> dict:
    """Get an output of label studio JSON file, which is labeled with NER tasks.

    Args:
        path (str): An path of labeled label studio output JSON file.

    Returns:
        _type_: Dictionary of the JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        label_studio_template = json.loads(f.read())
    return label_studio_template


def read_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf8") as f:
        tmp = f.readlines()
        for contents in tmp:
            contents = json.loads(contents)
            result = [content for content in contents]
    return result


def read_uie_inference_results(path: str) -> List[dict]:
    """Get the UIE results made by run_infer.py

    Args:
        path (str): Path of UIE results.

    Returns:
        _type_: List of UIE results.
    """
    uie_result_list = []
    with open(path, "r", encoding="utf8") as f:
        result_list = json.loads(f.read())
        uie_result_list = [result for result in result_list]
    return uie_result_list


def flatten_uie_output(uie_result: dict, threshold: float = 0.5) -> List[dict]:
    """flatten hierarchical UIE result into list

    Args:
        uie_result (dict): Default UIE result
        threshold (float, optional): Filter the UIE result whose probability greater than threshold. Defaults to 0.5.

    Returns:
        _type_: List of UIE result
    """
    flatten_uie_result = []
    for key in uie_result[0]:
        for each_information in uie_result[0][key]:
            if each_information["probability"] >= threshold:
                each_information.update({"entity": key})
                flatten_uie_result.append(each_information)
    return flatten_uie_result


def uie_result_to_labelstudio(
    uie_result: dict, labelstudio_mail: str, label_studio_template: dict, task_id: int = 0
) -> dict:
    """Align the UIE result with label_studio_template format

    Args:
        uie_result (dict): Single UIE result
        labelstudio_mail (str): label studio mail
        label_studio_template (dict): An output of label studio JSON file, which is labeled with NER tasks.
        task_id (int, optional): Task id on label studio. It can be made by user. Defaults to 0.

    Returns:
        dict: UIE result with label studio format.
    """
    labelstudio_format_result = label_studio_template.copy()
    labelstudio_format_result.update(
        {
            "id": task_id,
            "data": {
                "text": uie_result["Content"],
            },
            "annotations": [
                {"id": task_id, "task": task_id, "project": 0, "completed_by": {"email": labelstudio_mail}}
            ],
        }
    )

    tmp_result = []
    uie_inference_result = flatten_uie_output(uie_result["InferenceResults"])
    for each_result in uie_inference_result:
        tmp_result.append(
            {
                "value": {
                    "start": each_result["start"],
                    "end": each_result["end"],
                    "text": each_result["text"],
                    "labels": [each_result["entity"]],
                },
                "id": str(datetime.datetime.now()),
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
            }
        )
    labelstudio_format_result.update(
        {
            "annotations": [{"result": tmp_result}],
        }
    )
    return labelstudio_format_result


if __name__ == "__main__":
    """將「run_infer.py」inference 產生的結果，轉換成 label studio 格式，使其結果能在 label studio 上呈現。

    注意：Inference 結果必須要包含 「text, start, end」，因此在 run_infer.py 時，select_key 最少要有此三項。

    Example:
        python convert_to_labelstudio.py \
            --uie_results_path ./inference_results.txt \
            --labelstudio_mail aaa1aaa@gmail.com

    Raises:
        ValueError: uie_results_path is not found.
        ValueError: labelstudio_template_path is not found.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--uie_results_path", type=str)
    parser.add_argument("--labelstudio_mail", type=str)
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--save_name", type=str, default="uie_result_for_labelstudio.json")
    parser.add_argument("--labelstudio_template_path", type=str, default="./labelstudio_template.json")
    args = parser.parse_args()

    if not os.path.exists(args.uie_results_path):
        raise ValueError(f"Path not found: {args.uie_results_path}.")

    if not os.path.exists(args.labelstudio_template_path):
        raise ValueError(f"Path not found: {args.labelstudio_template_path}.")

    if not os.path.exists(args.save_path):
        print(f"Path not found: {args.save_path}. Auto-create the path...")
        os.mkdir(args.save_path)

    uie_result_list = read_uie_inference_results(path=args.uie_results_path)
    label_studio_template = get_labelstudio_template(path=args.labelstudio_template_path)
    logger = create_logger(level=LOGGER_LEVEL)
    logger.info("Start converting...")

    label_studio_result = []
    for id, uie_result in tqdm(enumerate(uie_result_list)):
        logger.debug(f"id: {id}, len(content): {len(uie_result['Content'])}, result: {uie_result['InferenceResults']}.")
        label_studio_result.append(
            uie_result_to_labelstudio(
                uie_result=uie_result,
                labelstudio_mail=args.labelstudio_mail,
                label_studio_template=label_studio_template,
                task_id=id,
            )
        )

    logger.info(f"Finish converting. Write results to {os.path.join(args.save_path, args.save_name)}")

    with open(os.path.join(args.save_path, args.save_name), "w", encoding="utf8") as f:
        tmp = json.dumps(label_studio_result, ensure_ascii=False)
        f.write(tmp)

    logger.info("Conversion successful.")
