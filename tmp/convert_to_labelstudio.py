import json
from postproces import read_json


def flatten_uie_output(uie_result: dict, threshold=0.5):
    flatten_uie_result = []
    for key in uie_result[0]:
        for each_information in uie_result[0][key]:
            if each_information["probability"] >= threshold:
                each_information.update({"entity": key})
                flatten_uie_result.append(each_information)
    return flatten_uie_result


if __name__ == "__main__":
    verdict_8000 = read_json()
    label_studio_result = []

    label_studio_template = {}
    with open("./label_studio_format.json", "r", encoding="utf-8") as f:
        label_studio_template = json.loads(f.read())

    for i, verdict in enumerate(verdict_8000):
        tmp = label_studio_template.copy()
        tmp["id"] = verdict["id"]
        tmp["data"]["jid"] = verdict["jid"]
        tmp["data"]["text"] = verdict["jfull_compress"]

        tmp["annotations"][0]["id"] = i
        tmp["annotations"][0]["task"] = i
        tmp["annotations"][0]["project"] = 0

        tmp["annotations"][0]["result"]
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
                    "id": str(i),
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "origin": "manual",
                }
            )
        tmp["annotations"][0]["result"] = tmp_result

        label_studio_result.append(tmp)

    with open("./uie_result_verdict_8000_with_label_studio.json", "w", encoding="utf8") as f:
        tmp = json.dumps(label_studio_result, ensure_ascii=False)
        f.write(tmp)

    print("Finish...")


# {'value': {'start': 959, 'end': 961, 'text': '死亡', 'labels': ['受有傷害']}, 'id': 'EaNTJc9Kw0', 'from_name': 'label', 'to_name': 'text', 'type': 'labels', 'origin': 'manual'}
