import json
from collections import defaultdict
import numpy as np
import os


def read_all_data(path):
    result = defaultdict(int)
    all_data = defaultdict(list)
    data_type = ["training_data.txt", "eval_data.txt", "testing_data.txt"]
    for i in data_type:
        with open(path + i, "r", encoding="utf8") as f:
            for u in f:
                content = json.loads(u.strip())
                if content["result_list"]:
                    result[content["prompt"]] += 1
                    all_data[content["prompt"]].append(content)
                else:
                    result["negative_sample"] += 1
                    all_data["negative_sample"].append(content)
    return result, all_data


def mean_arrange(data, length, split_type=["training_data.txt", "eval_data.txt", "testing_data.txt"]):
    # training_data
    new_data = defaultdict(list)
    remain_list = defaultdict(list)
    for k in data:
        np.random.shuffle(data[k])
        new_data["training_data"].extend(data[k][:length])
        remain_list[k] = len(data[k]) - length
    print(f"training len: {len(new_data['training_data'])}")

    # eval
    for k in data:
        tmp_len = int(remain_list[k] / 2)
        new_data["eval_data"].extend(data[k][length : (length + tmp_len)])
        remain_list[k] = length + tmp_len
    print(f"eval len: {len(new_data['eval_data'])}")

    # test
    for k in data:
        new_data["testing_data"].extend(data[k][remain_list[k] :])
    print(f"testing len: {len(new_data['testing_data'])}")

    return new_data


def mean_plus_arrange(data, length, split_type=["training_data.txt", "eval_data.txt", "testing_data.txt"]):
    # training_data
    new_data = defaultdict(list)
    remain_list = defaultdict(list)
    for k in data:
        np.random.shuffle(data[k])
        new_data["training_data"].extend(data[k][:length])
        remain_list[k] = len(data[k]) - length
    print(f"training len: {len(new_data['training_data'])}")

    # eval
    for k in data:
        tmp_len = int(remain_list[k] / 2)
        new_data["eval_data"].extend(data[k][length : (length + tmp_len)])
        remain_list[k] = length + tmp_len
    print(f"eval len: {len(new_data['eval_data'])}")

    # test
    for k in data:
        new_data["testing_data"].extend(data[k][remain_list[k] :])
    print(f"testing len: {len(new_data['testing_data'])}")

    return new_data


def write_result(data, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for k in data:
        with open(os.path.join(path, k) + ".txt", "w", encoding="utf-8") as f:
            for each_result in data[k]:
                jsonString = json.dumps(each_result, ensure_ascii=False)
                f.write(jsonString)
                f.write("\n")


def double_check(path):
    data_type = ["training_data.txt", "eval_data.txt", "testing_data.txt"]
    for i in data_type:
        with open(path + i, "r", encoding="utf8") as f:
            counter = 0
            for _ in f:
                counter += 1
            print(f"{i}: {counter}")


if __name__ == "__main__":
    read_path = "./Chinese-Verdict-NLP/information_extraction/data/final_data/"
    write_path = "./Chinese-Verdict-NLP/information_extraction/data/arrange_final_data_mean_plus/"
    type_counter, all_data = read_all_data(read_path)
    min_count = min(type_counter.values())
    new_data = mean_plus_arrange(all_data, min_count)
    # write_result(new_data, write_path)
    # double_check(write_path)
