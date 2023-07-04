from config.base_config import logger, entity_type, EvaluationArguments
from functools import partial
import paddle
from utils.data_utils import read_data_by_chunk, convert_to_uie_format, create_data_loader
from utils.exceptions import DataError
import os
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import SpanEvaluator
from paddlenlp.transformers import UIE, AutoTokenizer
from paddlenlp.trainer import PdArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_min_word_in_entity_type(entity_type):
    min_word = np.min([len(i) for i in entity_type])
    ner_type = [i[:min_word] for i in entity_type]
    if len(pd.unique(ner_type)) != len(entity_type):
        raise ValueError(f"The first word in ner_type is repeat. Please adjust it.")
    return min_word


def get_eval_group(min_word, inputs, tokenizer):
    group = tokenizer.convert_ids_to_tokens(np.array(inputs[:, 1 : (min_word + 1)]).flatten())
    group = [
        "".join(group)[start:end]
        for start, end in zip(
            range(0, len(group), min_word),
            range(min_word, len(group) + min_word, min_word),
        )
    ]
    return np.array(group)


@paddle.no_grad()
def evaluate_loop_by_class(model, data_loader, entity_type, tokenizer):
    metric = {entity: SpanEvaluator() for entity in entity_type + ["total"]}
    min_word = get_min_word_in_entity_type(entity_type)
    name_mapping = {entity[:min_word]: entity for entity in entity_type}
    model.eval()
    for batch in tqdm(data_loader):
        #
        start_ids = np.array(batch.pop("start_positions"))
        end_ids = np.array(batch.pop("end_positions"))
        start_prob, end_prob = model(**batch)
        eval_group = get_eval_group(min_word, batch["input_ids"], tokenizer)
        unique_group = pd.unique(eval_group)
        for each_group in unique_group:
            if name_mapping[each_group] not in entity_type:
                raise DataError(
                    f"Cannot map {name_mapping[each_group]} to {entity_type}, check if the entity type is modified or data is not preprocessed to UIE-input-format."
                )
            selected_group = eval_group == each_group
            num_correct, num_infer, num_label = metric[name_mapping[each_group]].compute(
                np.array(start_prob)[selected_group, :],
                np.array(end_prob)[selected_group, :],
                start_ids[selected_group, :],
                end_ids[selected_group, :],
            )
            metric[name_mapping[each_group]].update(num_correct, num_infer, num_label)
        num_correct, num_infer, num_label = metric["total"].compute(start_prob, end_prob, start_ids, end_ids)
        metric["total"].update(num_correct, num_infer, num_label)
    for entity in entity_type + ["total"]:
        precision, recall, f1 = metric[entity].accumulate()
        logger.info(f"-----------------{entity}-----------------")
        logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" % (precision, recall, f1))
    model.train()


@paddle.no_grad()
def evaluate_loop(model, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        multilingual(bool): Whether is the multilingual model.
    """
    metric = SpanEvaluator()
    model.eval()
    metric.reset()
    for batch in tqdm(data_loader):
        start_ids = paddle.cast(batch.pop("start_positions"), "float32")
        end_ids = paddle.cast(batch.pop("end_positions"), "float32")
        start_prob, end_prob = model(**batch)
        num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    model.train()
    return precision, recall, f1


def evaluate(
    dev_file: str,
    device: str = "gpu",
    model_name_or_path: str = "uie-base",
    max_seq_len: int = 512,
    batch_size: int = 16,
    is_eval_by_class: bool = False,
):
    if not os.path.exists(dev_file):
        raise ValueError(f"Data not found in {dev_file}. Please input the correct path of data.")

    paddle.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = UIE.from_pretrained(model_name_or_path)

    test_ds = load_dataset(
        read_data_by_chunk,
        data_path=dev_file,
        max_seq_len=max_seq_len,
        lazy=False,
    )

    convert_function = partial(
        convert_to_uie_format,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    test_ds = test_ds.map(convert_function)

    data_collator = DataCollatorWithPadding(tokenizer)
    test_data_loader = create_data_loader(test_ds, mode="test", batch_size=batch_size, trans_fn=data_collator)
    logger.info("Start Evaluation Loop...")
    if is_eval_by_class:
        evaluate_loop_by_class(model, test_data_loader, entity_type, tokenizer)
    else:
        precision, recall, f1 = evaluate_loop(model, test_data_loader)
        logger.info("-----------------------------")
        logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" % (precision, recall, f1))


if __name__ == "__main__":
    parser = PdArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]

    evaluate(
        model_name_or_path=args.model_name_or_path,
        dev_file=args.dev_file,
        batch_size=args.batch_size,
        device=args.device,
        is_eval_by_class=args.is_eval_by_class,
        max_seq_len=args.max_seq_len,
    )
