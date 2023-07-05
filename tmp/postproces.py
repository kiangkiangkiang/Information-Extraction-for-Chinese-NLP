import json
import csv

ENTITY_TYPE = ["精神慰撫金額", "醫療費用", "薪資收入"]
REMAIN_KEYS = ["text"]

KEYS_MAPPING_TO_CSV_TABLE = {
    "start": "uie_result_start_index",
    "end": "uie_result_end_index",
    "probability": "uie_result_probability",
}


def read_json(path="./verdict8000_uie_inference_result.json"):
    with open(path, "r", encoding="utf8") as f:
        tmp = f.readlines()
        for contents in tmp:
            contents = json.loads(contents)
            result = [content for content in contents]
    return result


# fill nan
def uie_result_fill_null_entity(uie_result, fill_what="nan"):
    for entity in ENTITY_TYPE:
        if not uie_result[0].get(entity):
            uie_result[0].update({entity: [{"text": fill_what, "start": -1, "end": -1, "probability": 0.0}]})
    return uie_result


# max filter
def uie_result_max_select(uie_result):
    new_result = [{}]
    for entity in uie_result[0]:
        new_result[0][entity] = [sorted(uie_result[0][entity], key=lambda x: x["probability"], reverse=True)[0]]
    return new_result


# select key
def uie_result_key_remain(uie_result):
    new_result = [{}]
    for entity in uie_result[0]:
        tmp_list = []
        for each_result_in_entity in uie_result[0][entity]:
            tmp_dict = {}
            for key in REMAIN_KEYS:
                tmp_dict.update({key: each_result_in_entity[key]})
            tmp_list.append(tmp_dict)
        new_result[0][entity] = tmp_list
    return new_result


# only work in single result
def adjust_verdict_to_csv_format(verdict, drop_keys=["jfull_compress", "uie_inference_results_with_checkpoint_9200"]):
    update_entity_result = {}
    for entity in verdict["uie_inference_results_with_checkpoint_9200"][0]:
        for key in REMAIN_KEYS:
            if key == "text":
                update_entity_result.update(
                    {entity: verdict["uie_inference_results_with_checkpoint_9200"][0][entity][0]["text"]}
                )
            else:
                update_entity_result.update(
                    {
                        f"{KEYS_MAPPING_TO_CSV_TABLE[key]}_for_{entity}": verdict[
                            "uie_inference_results_with_checkpoint_9200"
                        ][0][entity][0][key]
                    }
                )

    for drop_key in drop_keys:
        verdict.pop(drop_key)

    verdict.update(update_entity_result)
    return verdict


def write_json_list_to_csv(file_list, write_keys=None, save_dir="./verdict8000_uie_inference_result.csv"):
    header = write_keys if write_keys else list(file_list[0].keys())
    with open(save_dir, "w", encoding="utf_8_sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for file in file_list:
            data = [file[key] for key in header]
            writer.writerow(data)


if __name__ == "__main__":
    verdict_8000 = read_json()

    for i, verdict in enumerate(verdict_8000):
        verdict_8000[i]["uie_inference_results_with_checkpoint_9200"] = uie_result_fill_null_entity(
            uie_result=verdict["uie_inference_results_with_checkpoint_9200"]
        )

        verdict_8000[i]["uie_inference_results_with_checkpoint_9200"] = uie_result_max_select(
            uie_result=verdict["uie_inference_results_with_checkpoint_9200"]
        )

        verdict_8000[i]["uie_inference_results_with_checkpoint_9200"] = uie_result_key_remain(
            uie_result=verdict["uie_inference_results_with_checkpoint_9200"]
        )

        verdict_8000[i] = adjust_verdict_to_csv_format(verdict)

    write_json_list_to_csv(verdict_8000)
