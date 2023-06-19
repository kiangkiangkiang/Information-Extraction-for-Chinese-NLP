import json
from collections import defaultdict

result = defaultdict(int)
with open("./Chinese-Verdict-NLP/information_extraction/data/final_data/eval_data.txt", "r", encoding="utf8") as f:
    for i in f:
        content = json.loads(i.strip())
        if content["result_list"]:
            result[content["prompt"]] += 1
print(result)

368 + 46 + 48
596 + 80 + 76
669 + 89 + 53
