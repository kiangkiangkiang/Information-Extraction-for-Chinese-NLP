import re
import os
import json
from paddlenlp.utils.log import logger
import pandas as pd


def is_filter_text_by_non_chinese(text) -> bool:
    # Remove non-chinese
    criterion = re.sub("[^\u4e00-\u9fa5]+", "", text[0] + text[-1])
    return True if len(criterion) == 0 else False


def read_data(path: str = "./Chinese-Verdict-NLP/information_extraction/data/final_data/"):
    out_data = []
    with open(path, "r", encoding="utf8") as f:
        for i in f:
            json_content = json.loads(i.strip())
            content_length = len(json_content["content"]) - 1
            repeat_list = pd.value_counts([result["text"] for result in json_content["result_list"]])
            result_mapping = dict(repeat_list)
            repeat_list = repeat_list[repeat_list > 1].keys().to_list()

            remove_list = []
            for i, result in enumerate(json_content["result_list"]):
                if result["text"] in repeat_list:
                    start = result["start"] - 1 if result["start"] - 1 >= 0 else 0
                    end = result["end"] + 1 if result["end"] + 1 <= content_length else content_length
                    if is_filter_text_by_non_chinese(text=json_content["content"][start:end]):
                        if result_mapping[result["text"]] > 1:
                            logger.debug(f"Remove text: {json_content['content'][start-8:end+8]}")
                            remove_list.append(i)
                            result_mapping[result["text"]] -= 1

            out_data.append(
                {
                    "content": json_content["content"],
                    "result_list": [
                        result for i, result in enumerate(json_content["result_list"]) if i not in remove_list
                    ],
                    "prompt": json_content["prompt"],
                }
            )
    return out_data


def write_result(data, path, file_name):
    if not os.path.exists(path):
        os.mkdir(path)

    with open(os.path.join(path, file_name), "w", encoding="utf-8") as f:
        for each_result in data:
            jsonString = json.dumps(each_result, ensure_ascii=False)
            f.write(jsonString)
            f.write("\n")


if __name__ == "__main__":
    read_path = "./Chinese-Verdict-NLP/information_extraction/data/final_data/"
    write_path = "./Chinese-Verdict-NLP/information_extraction/data/final_data_remove_repeat/"

    name_list = ["training_data.txt", "eval_data.txt", "testing_data.txt"]

    for n in name_list:
        data = read_data(path=os.path.join(read_path, n))
        write_result(data=data, path=write_path, file_name=n)
    logger.info("Finish Remove.")
