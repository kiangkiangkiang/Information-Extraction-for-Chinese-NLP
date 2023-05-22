from paddlenlp.utils.log import logger
from paddlenlp import Taskflow
import json
from typing import List
import pandas as pd

""" TODO
切文本：

方法一：
先用隨便模型的 NER 快速找到所有「錢的index」，之後後開 30字（字數設參數做調整，也許跟max_seq_len有關）
的 windows，然後無腦 concate再一起當作原文本來跑模型看看。


"""


def experiment_inference(
    data_path: str = "../information_extraction/data/testing_data.txt",
    task_path: str = "../information_extraction/results/checkpoint",
    schema: List[str] = ["精神慰撫金額", "醫療費用", "薪資收入"],
    do_compute_auc: bool = True,
    evaluation_method: str = "simple",
):
    logger.info("Inference...")
    schema = schema
    testing_path = data_path
    my_ie = Taskflow(
        "information_extraction",
        schema=schema,
        task_path=task_path,
        precision="fp16",
    )
    evaluation_result = {ner_type: {"Number of Correct": 0, "Total": 0} for ner_type in schema}
    evaluation_method = "simple"
    several_text = 0

    with open(testing_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            # logger.debug(f"data = {json_line['content'].strip()}")
            # logger.debug(f"prompt = {json_line['prompt']}")

            inference_result = my_ie(json_line["content"].strip())
            result_in_this_prompt = inference_result[0].get(json_line["prompt"])
            final_result = filter_result(result_in_this_prompt, "threshold", 0.6)

            # logger.debug(f"inference result = {final_result}")
            # logger.debug(f"ground true = {json_line['result_list']}")

            if do_compute_auc:
                evaluation_result[json_line["prompt"]]["Total"] += 1
                if evaluation_method == "simple":
                    # 只考慮每個類別都只有一個真實答案的金額
                    # 只考慮預測的text和真實的text一樣就算正確

                    # case 1 [] == []
                    if final_result == json_line["result_list"]:
                        logger.info(f"Correct in {json_line['prompt']}, [] == [].")
                        evaluation_result[json_line["prompt"]]["Number of Correct"] += 1
                    else:
                        # case 2 [...] -> [...]
                        if len(final_result) > 0 and len(json_line["result_list"]) > 0:
                            infer_text = final_result[0]["text"]

                            ground_text = pd.unique([i["text"] for i in json_line["result_list"]])
                            # TODO add number normalize function to normalize ground_text
                            if len(ground_text) > 1:
                                several_text += 1
                                evaluation_result[json_line["prompt"]]["Total"] -= 1
                                logger.debug(f"Ground true has several answers!!!! Accumulate: {several_text}")
                                logger.debug(f"Several text in {json_line['prompt']}.")
                                logger.debug(f"Several sample: {json_line['result_list']}.")
                                # if json_line["prompt"] == "精神慰撫金額":
                                #     logger.debug(f"Several sample: {json_line['content'].strip()}.")

                            else:
                                if ground_text == infer_text:
                                    logger.info(f"Correct in {json_line['prompt']}, {ground_text} == {infer_text}.")
                                    evaluation_result[json_line["prompt"]]["Number of Correct"] += 1
                                else:
                                    # wrong
                                    logger.error(
                                        f"Fail!! Incorrect in {json_line['prompt']}, {ground_text} != {infer_text}."
                                    )
                                    # logger.error(f"content={json_line['content'].strip()}")
                        else:
                            # wrong case [...] -> [] or [] -> [...]
                            logger.error(
                                f"Fail!! Incorrect in {json_line['prompt']}, True: {json_line['result_list']} != Inf: {final_result}."
                            )
                            # logger.error(f"content={json_line['content'].strip()}")

                else:
                    # not implement
                    pass

    logger.info("End Inference...")
    logger.info("Evaluation Metrics shown as following...")
    if do_compute_auc:
        logger.info(f"Total duplicate text is {several_text}.")
        total = 0
        for ner_type in evaluation_result:
            acu = evaluation_result[ner_type]["Number of Correct"] / evaluation_result[ner_type]["Total"]
            logger.info(f"{ner_type}: ")
            logger.info(f"Number of Correct: {evaluation_result[ner_type]['Number of Correct']}.")
            logger.info(f"Total: {evaluation_result[ner_type]['Total']}.")
            logger.info(f"Accurancy: {acu}.")
            total += evaluation_result[ner_type]["Total"]
        logger.info(f"Total Sample (exclude duplicate text) is {total}.")
    logger.info("Done")


def filter_result(results, method="max_prob", threshold=0.5):
    # format of results: [{'text': '200,000元', 'start': 470, 'end': 478, 'probability': 0.8655009269714355}]
    if results:
        results = sorted(results, key=lambda x: x["probability"], reverse=True)
        if method == "max_prob":
            return [results[0]] if results[0]["probability"] > threshold else []
        elif method == "threshold":
            results = list(filter(lambda x: x["probability"] > threshold, results))
            return results
    else:
        return []


if __name__ == "__main__":
    experiment_inference()

"""
a = [
    {"text": "202,700元", "start": 530, "end": 538, "probability": 0.29718875885009766},
    {"text": "100,000元", "start": 3309, "end": 3317, "probability": 0.9713973999023438},
    {"text": "30,000元", "start": 3632, "end": 3639, "probability": 0.8628144264221191},
]

for i in a:
    print(i)
"""
