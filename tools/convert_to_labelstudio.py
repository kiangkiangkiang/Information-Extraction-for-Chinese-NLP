# convert the label result to label studio format
from paddlenlp.utils.log import logger
import json
import argparse
import datetime


def get_labelstudio_template(path):
    with open(path, "r", encoding="utf-8") as f:
        label_studio_template = json.loads(f.read())
    return label_studio_template


def read_json(path):
    with open(path, "r", encoding="utf8") as f:
        tmp = f.readlines()
        for contents in tmp:
            contents = json.loads(contents)
            result = [content for content in contents]
    return result


def read_uie_inference_results(path):
    uie_result_list = []
    with open(path, "r", encoding="utf8") as f:
        result_list = json.loads(f.read())
        uie_result_list = [result for result in result_list]
    return uie_result_list


def flatten_uie_output(uie_result: dict, threshold=0.5):
    flatten_uie_result = []
    for key in uie_result[0]:
        for each_information in uie_result[0][key]:
            if each_information["probability"] >= threshold:
                each_information.update({"entity": key})
                flatten_uie_result.append(each_information)
    return flatten_uie_result


def uie_result_to_labelstudio(uie_result, labelstudio_mail, task_id=0):
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


# python convert_to_labelstudio.py --uie_results_path ./inference_results.txt --labelstudio_mail aaa1aaa@gmail.com
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uie_results_path", type=str)
    parser.add_argument("--labelstudio_mail", type=str)
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--save_name", type=str, default="uie_result_for_labelstudio.json")
    parser.add_argument("--labelstudio_template_path", type=str, default="./labelstudio_template.json")
    args = parser.parse_args()

    uie_result_list = read_uie_inference_results(path=args.uie_results_path)
    label_studio_template = get_labelstudio_template(path=args.labelstudio_template_path)

    label_studio_result = []
    for id, uie_result in enumerate(uie_result_list):
        label_studio_result.append(
            uie_result_to_labelstudio(
                uie_result=uie_result,
                labelstudio_mail=args.labelstudio_mail,
                task_id=id,
            )
        )

    with open("./uie_result_verdict_8000_with_label_studio.json", "w", encoding="utf8") as f:
        tmp = json.dumps(label_studio_result, ensure_ascii=False)
        f.write(tmp)
