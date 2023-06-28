from config.base_config import (
    logger,
    entity_type,
    InferenceDataArguments,
    InferenceStrategyArguments,
    InferenceTaskflowArguments,
)
from typing import List, Callable
from paddlenlp import Taskflow
from paddlenlp.trainer import PdArgumentParser
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
    data_file: str,
    schema: List[str],
    device_id: int = 0,
    text_list: List[str] = None,
    precision: str = "fp32",
    batch_size: int = 1,
    model: str = "uie-base",
    task_path: str = None,
    postprocess_fun: Callable = None,
):
    if not os.path.exists(data_file) and not text_list:
        raise ValueError(f"Data not found in {data_file}. Please input the correct path of data.")

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
        with open(data_file, "r", encoding="utf8") as f:
            text_list = [line.strip() for line in f]

    return (
        postprocess_fun([uie(text) for text in tqdm(text_list)])
        if postprocess_fun
        else [uie(text) for text in text_list]
    )


if __name__ == "__main__":
    parser = PdArgumentParser((InferenceDataArguments, InferenceStrategyArguments, InferenceTaskflowArguments))
    data_args, strategy_args, taskflow_args = parser.parse_args_into_dataclasses()

    if taskflow_args.precision == "fp16" and taskflow_args.device_id == -1:
        logger.warning("Cannot apply fp16 on cpu. Auto-adjust to fp32.")
        taskflow_args.precision = "fp32"

    result_processer = ResultProcesser(
        select_strategy=strategy_args.select_strategy,
        threshold=strategy_args.select_strategy_threshold,
        select_key=strategy_args.select_key,
    )
    postprocess_fun = result_processer.process

    logger.info("Start Inference...")
    inference_result = inference(
        data_file=data_args.data_file,
        device_id=taskflow_args.device_id,
        schema=entity_type,
        text_list=data_args.text_list,
        precision=taskflow_args.precision,
        batch_size=taskflow_args.batch_size,
        model=taskflow_args.model,
        task_path=taskflow_args.task_path,
        postprocess_fun=postprocess_fun,
    )

    logger.info("========== Inference Results ==========")
    for i, text_inference_result in enumerate(inference_result):
        logger.info(f"========== Content {i} Results ==========")
        logger.info(text_inference_result)
    logger.info("End Inference...")

    if data_args.save_dir:
        out_result = []
        if not os.path.exists(data_args.save_dir):
            logger.warning(f"{data_args.save_dir} is not found. Auto-create the dir.")
            os.makedirs(data_args.save_dir)

        with open(data_args.data_file, "r", encoding="utf8") as f:
            text_list = [line.strip() for line in f]

        with open(os.path.join(data_args.save_dir, "inference_results.txt"), "w", encoding="utf8") as f:
            for content, result in zip(text_list, inference_result):
                out_result.append(
                    {
                        "Content": content,
                        "InferenceResults": result,
                    }
                )
            jsonString = json.dumps(out_result, ensure_ascii=False)
            f.write(jsonString)
