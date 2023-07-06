# convert the label result to label studio format
import json


def get_labelstudio_template(path="./labelstudio_template.json"):
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
    # TODO
    pass


def flatten_uie_output(uie_result: dict, threshold=0.5):
    flatten_uie_result = []
    for key in uie_result[0]:
        for each_information in uie_result[0][key]:
            if each_information["probability"] >= threshold:
                each_information.update({"entity": key})
                flatten_uie_result.append(each_information)
    return flatten_uie_result


def uie_result_to_labelstudio(uie_result, labelstudio_mail):
    # TODO
    pass


if __name__ == "__main__":
    verdict_8000 = read_uie_inference_results()

    label_studio_result = []

    label_studio_template = get_labelstudio_template()

    result_counter = 0

    for i, verdict in enumerate(verdict_8000):
        tmp = ""
        tmp = label_studio_template.copy()
        tmp.update(
            {
                "id": verdict["id"],
                "data": {
                    "jid": verdict["jid"],
                    "text": verdict["jfull_compress"],
                },
                "annotations": [{"id": i, "task": i, "project": 0, "completed_by": {"email": "aaa1aaa@gmail.com"}}],
            }
        )

        tmp_result = []
        uie_result = flatten_uie_output(verdict["uie_inference_results_with_checkpoint_9200"])
        for each_result in uie_result:
            tmp_result.append(
                {
                    "value": {
                        "start": each_result["start"],
                        "end": each_result["end"],
                        "text": each_result["text"],
                        "labels": [each_result["entity"]],
                    },
                    "id": str(result_counter),
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "origin": "manual",
                }
            )
            result_counter += 1
        tmp.update(
            {
                "annotations": [{"result": tmp_result}],
            }
        )

        label_studio_result.append(tmp)

    with open("./uie_result_verdict_8000_with_label_studio.json", "w", encoding="utf8") as f:
        tmp = json.dumps(label_studio_result, ensure_ascii=False)
        f.write(tmp)

    print("Finish...")


# {'value': {'start': 959, 'end': 961, 'text': '死亡', 'labels': ['受有傷害']}, 'id': 'EaNTJc9Kw0', 'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'manual'}
