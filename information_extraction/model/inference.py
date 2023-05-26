from preprocessing import *
from paddlenlp.utils.log import logger
from paddlenlp import Taskflow
import json
from typing import Optional, List, Any, Callable, Dict, Union, Tuple, Literal
import pandas as pd
from callbacks import load_model_and_tokenizer
import paddle.nn.functional as F


""" TODO
切文本：

方法一：
先用隨便模型的 NER 快速找到所有「錢的index」，之後後開 30字（字數設參數做調整，也許跟max_seq_len有關）
的 windows，然後無腦 concate再一起當作原文本來跑模型看看。


"""


from functools import partial

import paddle

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import MapDataset, load_dataset

from paddlenlp.metrics import SpanEvaluator
from paddlenlp.transformers import UIE, UIEM, AutoTokenizer
from paddlenlp.utils.log import logger


'''
def evaluate(model, metric, data_loader, multilingual=False):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        multilingual(bool): Whether is the multilingual model.
    """
    model.eval()
    metric.reset()
    for batch in data_loader:
        if multilingual:
            start_prob, end_prob = model(batch["input_ids"], batch["position_ids"])
        else:
            start_prob, end_prob = model(
                batch["input_ids"], batch["token_type_ids"], batch["position_ids"], batch["attention_mask"]
            )

        start_ids = paddle.cast(batch["start_positions"], "float32")
        end_ids = paddle.cast(batch["end_positions"], "float32")
        num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    model.train()
    return precision, recall, f1


def do_eval():
    paddle.set_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.multilingual:
        model = UIEM.from_pretrained(args.model_path)
    else:
        model = UIE.from_pretrained(args.model_path)

    test_ds = load_dataset(reader, data_path=args.test_path, max_seq_len=args.max_seq_len, lazy=False)
    class_dict = {}
    relation_data = []
    if args.debug:
        for data in test_ds:
            class_name = unify_prompt_name(data["prompt"])
            # Only positive examples are evaluated in debug mode
            if len(data["result_list"]) != 0:
                p = "的" if args.schema_lang == "ch" else " of "
                if p not in data["prompt"]:
                    class_dict.setdefault(class_name, []).append(data)
                else:
                    relation_data.append((data["prompt"], data))

        relation_type_dict = get_relation_type_dict(relation_data, schema_lang=args.schema_lang)
    else:
        class_dict["all_classes"] = test_ds

    trans_fn = partial(
        convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len, multilingual=args.multilingual
    )

    for key in class_dict.keys():
        if args.debug:
            test_ds = MapDataset(class_dict[key])
        else:
            test_ds = class_dict[key]
        test_ds = test_ds.map(trans_fn)

        data_collator = DataCollatorWithPadding(tokenizer)

        test_data_loader = create_data_loader(test_ds, mode="test", batch_size=args.batch_size, trans_fn=data_collator)

        metric = SpanEvaluator()
        precision, recall, f1 = evaluate(model, metric, test_data_loader, args.multilingual)
        logger.info("-----------------------------")
        logger.info("Class Name: %s" % key)
        logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" % (precision, recall, f1))

    if args.debug and len(relation_type_dict.keys()) != 0:
        for key in relation_type_dict.keys():
            test_ds = MapDataset(relation_type_dict[key])
            test_ds = test_ds.map(trans_fn)
            test_data_loader = create_data_loader(
                test_ds, mode="test", batch_size=args.batch_size, trans_fn=data_collator
            )

            metric = SpanEvaluator()
            precision, recall, f1 = evaluate(model, metric, test_data_loader)
            logger.info("-----------------------------")
            if args.schema_lang == "ch":
                logger.info("Class Name: X的%s" % key)
            else:
                logger.info("Class Name: %s of X" % key)
            logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" % (precision, recall, f1))
'''


def preprocess_inference_data(
    data_path: str,
    tokenizer: Any,
    read_data_method: str = "chunk",
    max_seq_len: int = 512,
    schema: List[str] = ["精神慰撫金", "醫療費", "薪資收入"],
):
    if read_data_method == "chunk":
        read_data = read_data_by_chunk
        convert_and_tokenize_function = convert_to_uie_format
        # convert_and_tokenize_function = convert_example
    else:
        read_data = read_full_data
        convert_and_tokenize_function = convert_to_full_data_format

    test_ds = load_dataset(read_data, data_path=data_path, max_seq_len=max_seq_len, lazy=False)

    # TODO convert data by schema
    convert_function = partial(
        convert_and_tokenize_function,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    test_ds = test_ds.map(convert_function)
    pass


@paddle.no_grad()
def do_inference(
    model_path: str,
    data_path: str,
    schema: List[str],
    max_seq_len: int = 512,
    batch_size: int = 16,
    device: str = "cpu",
    model_name_or_path: str = "uie-base",
    export_model_dir: Optional[str] = None,
    multilingual: Optional[bool] = False,
    read_data_method: Optional[Literal["chunk", "full"]] = "chunk",
    convert_and_tokenize_function: Optional[
        Callable[[Dict[str, str], Any, int], Dict[str, Union[str, float]]]
    ] = convert_to_uie_format,
):
    # TODO detect device type

    paddle.set_device(device)
    model, tokenizer = load_model_and_tokenizer(model_path)

    if read_data_method not in ["chunk", "full"]:
        logger.warning(
            f"read_data_method must be 'chunk' or 'full', {read_data_method} is not support. \
            Automatically change read_data_method to 'chunk'."
        )
        read_data_method = "chunk"

    test_dataset = preprocess_inference_data(
        data_path=data_path,
        tokenizer=tokenizer,
        read_data_method=read_data_method,
        max_seq_len=max_seq_len,
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    test_data_loader = create_data_loader(test_dataset, batch_size=batch_size, trans_fn=data_collator)
    for inputs in test_data_loader:
        outputs = model(**inputs)
        outputs_prob = (F.softmax(output, axis=1) for output in (outputs[0], outputs[1]))
        breakpoint()
        for l, start, end in zip(*np.where(outputs_prob > 0.0)):
            breakpoint()
            """ 
            ent_prob = entity_probs[l, start, end]
            start, end = (offset_mapping[start][0], offset_mapping[end][-1])
            ent = {
                "text": text[start:end],
                "type": label_maps["id2entity"][str(l)],
                "start_index": start,
                "probability": ent_prob,
            }
            ent_list.append(ent)
            """

        # batch_ent_results.append(ent_list)

        breakpoint()
        print(123)


"""
if __name__ == "__main__":
    # yapf: disable
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None, help="The path of test set.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU/NPU for training.")
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu", "npu"], help="Device selected for evaluate.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--debug", action='store_true', help="Precision, recall and F1 score are calculated for each class separately if this option is enabled.")
    parser.add_argument("--multilingual", action='store_true', help="Whether is the multilingual model.")
    parser.add_argument("--schema_lang", choices=["ch", "en"], default="ch", help="Select the language type for schema.")

    args = parser.parse_args()
    # yapf: enable

    do_eval()
"""


def experiment_inference(
    data_path: str = "../information_extraction/data/first_stage_data/testing_data.txt",
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
                                    logger.info(
                                        f"Correct in {json_line['prompt']}, 'True': {ground_text} == 'Infer': {infer_text}."
                                    )
                                    evaluation_result[json_line["prompt"]]["Number of Correct"] += 1
                                else:
                                    # wrong
                                    logger.error(
                                        f"Fail!! Incorrect in {json_line['prompt']}, 'True': {ground_text} != 'Infer': {infer_text}."
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
    experiment = True
    if experiment:
        experiment_inference()
    else:
        do_inference(
            device="gpu",
            model_path="../information_extraction/results/checkpoint",
            data_path="../information_extraction/data/testing_data.txt",
        )


"""
a = [
    {"text": "202,700元", "start": 530, "end": 538, "probability": 0.29718875885009766},
    {"text": "100,000元", "start": 3309, "end": 3317, "probability": 0.9713973999023438},
    {"text": "30,000元", "start": 3632, "end": 3639, "probability": 0.8628144264221191},
]

for i in a:
    print(i)
"""
