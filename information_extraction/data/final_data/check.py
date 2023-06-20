import os
from collections import defaultdict
import json

data = ["training_data.txt", "eval_data.txt", "testing_data.txt"]

for d in data:
    all_data_true = {"精神慰撫金額": 0, "醫療費用": 0, "薪資收入": 0}
    all_data_false = all_data_true.copy()
    with open(os.path.join("./", d), "r", encoding="utf8") as f:
        for i in f:
            json_content = json.loads(i.strip())
            if json_content["result_list"]:
                all_data_true[json_content["prompt"]] += 1
            else:
                all_data_false[json_content["prompt"]] += 1
    print(f"{d}: ")
    print(f"有 result 的:   {all_data_true}")
    print(f"沒有 result 的: {all_data_false}")
    ratio = [all_data_true[k] / (all_data_true[k] + all_data_false[k]) for k in all_data_true]
    print(f"比例: {ratio}")
