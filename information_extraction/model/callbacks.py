import os
import sys
from typing import Optional

# from paddlenlp.metrics import SpanEvaluator

from test_metric import SpanEvaluator

from paddlenlp.trainer.trainer_utils import EvalPrediction
from paddle import cast, nn
from paddlenlp.transformers import AutoTokenizer, ErnieTokenizer, XLNetTokenizer, RoFormerTokenizer, UIEX

from modeling import UIE, IE_XLNet, IE_Ernie, IE_Roformer, IE_Bert, IE_Electra

import pandas as pd
import numpy as np

from paddlenlp.transformers import ErnieModel, UIEM


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from paddlenlp.utils.log import logger

loss_func = nn.BCELoss()


def uie_loss_func(outputs, labels, group=None, mlflow_key=None, mlflow_step=None) -> float:
    # TODO add lambda for each group
    loss_func = nn.BCELoss()
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = cast(start_ids, "float32")
    end_ids = cast(end_ids, "float32")
    loss_start = loss_func(start_prob, start_ids)
    loss_end = loss_func(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    logger.debug(f"{mlflow_key} step: {mlflow_step}, loss = {np.round(loss, 5)[0]}")
    return loss


def uie_loss_func_by_group(outputs, labels, group=None, mlflow_key=None, mlflow_step=None) -> float:
    # TODO add lambda for each group
    breakpoint()
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = cast(start_ids, "float32")
    end_ids = cast(end_ids, "float32")
    loss_start = loss_func(start_prob, start_ids)
    loss_end = loss_func(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    logger.debug(f"{mlflow_key} step: {mlflow_step}, loss = {np.round(loss, 5)[0]}")
    return loss


# metrics calculator
def SpanEvaluator_metrics(result):

    metric = SpanEvaluator()

    def compute_metrics(predictions, label_ids, descriptions=""):
        # test metrics
        logger.debug(f"Now eval: {descriptions}")
        start_prob, end_prob = predictions
        start_ids, end_ids = label_ids
        metric.reset()
        num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label)
        precision, recall, f1 = metric.accumulate()
        metric.reset()
        return {descriptions + "precision": precision, descriptions + "recall": recall, descriptions + "f1": f1}

    if result.__class__.__name__ == "IEEvalPrediction":
        ner_type = pd.unique(result.eval_group)
        num_type = len(ner_type)
        if num_type < 1:
            raise ValueError("Cannot compute metrics by eval_group when length of ner_type < 1.")
        else:
            metric_result = {}
            # compute by group
            for each_type in ner_type:
                selected_group = result.eval_group == each_type
                # this_data = np.array(result.dataloader.dataset.data)[selected_group]
                metric_result.update(
                    compute_metrics(
                        predictions=tuple(result.predictions[i][selected_group, :] for i in (0, 1)),
                        label_ids=tuple(result.label_ids[i][selected_group, :] for i in (0, 1)),
                        descriptions=each_type + "_",
                    )
                )
            # compute total
            Total_Precision, Total_Recall, Total_F1 = 0, 0, 0
            tmp_metrics = list(metric_result.values())
            runs = int(len(tmp_metrics) / num_type)
            for i in range(0, len(tmp_metrics), runs):
                Total_Precision += tmp_metrics[i]
                Total_Recall += tmp_metrics[i + 1]
                Total_F1 += tmp_metrics[i + 2]
            metric_result.update(
                {
                    "precision": Total_Precision / num_type,
                    "recall": Total_Recall / num_type,
                    "f1": Total_F1 / num_type,
                }
            )
            return metric_result
    else:
        return compute_metrics(predictions=result.predictions, label_ids=result.label_ids)


# load model and tokenizer
def load_model_and_tokenizer(model_name_or_path: str):

    # main model: UIE
    # from pretrained tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # from pretrained model
    model = UIE.from_pretrained(model_name_or_path)

    # test for xlnet (fail)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = IE_XLNet.from_pretrained(model_name_or_path)
    
    """

    # test for bigbird
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = IE_BigBird.from_pretrained(model_name_or_path)
    breakpoint()
    """

    """
    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-tiny-mini-v1-zh")
    model = ErnieModel.from_pretrained("ernie-3.0-tiny-mini-v1-zh")
    """

    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # from pretrained model
    model = UIEX.from_pretrained(model_name_or_path)
    """

    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # from pretrained model
    model = IE_Ernie.from_pretrained(model_name_or_path)
    """
    """

    model = IE_XLNet.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    """

    """
    model = IE_Roformer.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    """
    """

    model = IE_Bert.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    """
    """

    model = IE_Electra.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    """
    return model, tokenizer
