import argparse
from config.base_config import entity_type
from typing import List, Callable
from paddlenlp import Taskflow
from paddle import set_device
from paddlenlp.utils.log import logger
import os
import json
from tqdm import tqdm


class ResultProcesser:
    def __init__(
        self,
        select_strategy: str = "all",
        threshold: float = 0.5,
        select_key: List[str] = ["text", "start", "end", "probability"],
    ) -> None:

        """
        each_entity_results (example): [{'text': '22,154元', 'start': 1487, 'end': 1494, 'probability': 0.46060848236083984}, {'text': '2,954元', 'start': 3564, 'end': 3570, 'probability': 0.8074951171875}]
        """
        self.select_strategy_fun = eval("self._" + select_strategy + "_process")
        self.threshold = threshold if threshold else 0.5
        self.select_key = select_key if select_key else ["text", "start", "end", "probability"]

    def _key_filter(strategy_fun):
        def select_key(self, each_entity_results):
            each_entity_results = strategy_fun(self, each_entity_results)
            for i, each_entity_result in enumerate(each_entity_results):
                each_entity_results[i] = {key: each_entity_result[key] for key in self.select_key}
            return each_entity_results

        return select_key

    def process(self, results):
        new_result = []
        for result in results:
            tmp = [{}]
            for entity in result[0]:
                tmp[0][entity] = self.select_strategy_fun(result[0][entity])
            new_result.append(tmp)
        return new_result

    @_key_filter
    def _max_process(self, each_entity_results):
        return [sorted(each_entity_results, key=lambda x: x["probability"], reverse=True)[0]]

    @_key_filter
    def _threshold_process(self, each_entity_results):
        return list(filter(lambda x: x["probability"] > self.threshold, each_entity_results))

    @_key_filter
    def _all_process(self, each_entity_results):
        return each_entity_results


def inference(
    data_path: str,
    schema: List[str],
    device_id: int = 0,
    text_list: List[str] = None,
    precision: str = "fp32",
    batch_size: int = 1,
    model: str = "uie-base",
    task_path: str = None,
    postprocess_fun: Callable = None,
):
    if not os.path.exists(data_path) and not text_list:
        raise ValueError(f"Data not found in {data_path}. Please input the correct path of data.")

    if task_path:
        if not os.path.exists(task_path):
            raise ValueError(f"{task_path} is not a directory.")

        uie = Taskflow(
            "information_extraction",
            schema=schema,
            task_path=task_path,
            precision=precision,
            batch_size=batch_size,
            device_id=device_id,
        )
    else:
        uie = Taskflow(
            "information_extraction",
            schema=schema,
            model=model,
            precision=precision,
            batch_size=batch_size,
            device_id=device_id,
        )

    if not text_list:
        with open(data_path, "r", encoding="utf8") as f:
            text_list = [line.strip() for line in f]

    return (
        postprocess_fun([uie(text) for text in tqdm(text_list)])
        if postprocess_fun
        else [uie(text) for text in text_list]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./data/model_infer_data/example.txt",
        type=str,
        help="The path of data that you wanna inference.",
    )
    parser.add_argument(
        "--text_list",
        default=None,
        type=List[str],
        help="The path of data that you wanna inference.",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        help="The path where you wanna to save results of inference. If None, model won't write data.",
    )
    parser.add_argument(
        "--device_id",
        default=0,
        type=int,
        help="TODO edit",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp32",
        type=str,
        help="Default 'fp32', which is slower than 'fp16'. If 'fp16' is applied, make sure your CUDA>=11.2 and cuDNN>=8.1.1.",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size of model input.",
    )
    parser.add_argument(
        "--model",
        default="uie-base",
        type=str,
        help="The model you want to use.",
    )
    parser.add_argument(
        "--task_path",
        default=None,
        type=str,
        help="The checkpoint you want to use in inference.",
    )
    parser.add_argument(
        "--select_strategy",
        choices=["max", "all", "threshold"],
        default="all",
        type=str,
        help="Strategy of getting results. max: only get the max prob. result. all: get all results. threshold",
    )
    parser.add_argument(
        "--select_strategy_threshold",
        default=0.5,
        type=float,
        help="The checkpoint you want to use in inference.",
    )
    # TODO select_key還沒實作好
    parser.add_argument(
        "--select_key",
        default=["text", "start", "end", "probability"],
        nargs="+",
        help="UIE will output ['text', 'start', 'end', 'probability']. --select_key is to select which key in the list you want to remain.",
    )

    args = parser.parse_args()

    if args.precision == "fp16" and args.device_id == -1:
        logger.warning("Cannot apply fp16 on cpu. Auto-adjust to fp32.")
        args.precision = "fp32"

    result_processer = ResultProcesser(
        select_strategy=args.select_strategy, threshold=args.select_strategy_threshold, select_key=args.select_key
    )
    postprocess_fun = result_processer.process

    logger.info("Start Inference...")
    inference_result = inference(
        data_path=args.data_path,
        device_id=args.device_id,
        schema=entity_type,
        text_list=args.text_list,
        precision=args.precision,
        batch_size=args.batch_size,
        model=args.model,
        task_path=args.task_path,
        postprocess_fun=postprocess_fun,
    )

    logger.info("========== Inference Results ==========")
    for i, text_inference_result in enumerate(inference_result):
        logger.info(f"========== Content {i} Results ==========")
        logger.info(text_inference_result)
    logger.info("End Inference...")

    if args.save_dir:
        out_result = []
        if not os.path.exists(args.save_dir):
            logger.warning(f"{args.save_dir} is not found. Auto-create the dir.")
            os.makedirs(args.save_dir)

        with open(args.data_path, "r", encoding="utf8") as f:
            text_list = [line.strip() for line in f]

        with open(os.path.join(args.save_dir, "inference_results.txt"), "w", encoding="utf8") as f:
            for content, result in zip(text_list, inference_result):
                out_result.append(
                    {
                        "Content": content,
                        "InferenceResults": result,
                    }
                )
            jsonString = json.dumps(out_result, ensure_ascii=False)
            f.write(jsonString)
