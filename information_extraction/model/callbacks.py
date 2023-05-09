import os
import sys
from typing import Optional
from paddlenlp.metrics import SpanEvaluator
from paddlenlp.trainer.trainer_utils import EvalPrediction
from paddle import cast, nn
from paddlenlp.transformers import AutoTokenizer
from modeling import UIE

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from paddlenlp.utils.log import logger


# Loss function
def uie_loss_func(outputs, labels) -> float:
    # TODO MLFLOW
    loss_func = nn.BCELoss()
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = cast(start_ids, "float32")
    end_ids = cast(end_ids, "float32")
    loss_start = loss_func(start_prob, start_ids)
    loss_end = loss_func(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    return loss


# metrics calculator
# TODO 增加AUC (取出機率最高的錢，然後看他跟真實資料的錢的差異)
def SpanEvaluator_metrics(result: EvalPrediction):
    metric = SpanEvaluator()
    start_prob, end_prob = result.predictions
    start_ids, end_ids = result.label_ids
    metric.reset()
    num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
    metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    metric.reset()
    return {"precision": precision, "recall": recall, "f1": f1}


# load model and tokenizer
def load_model_and_tokenizer(model_name_or_path: str):
    # TODO add example for customize
    # from pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # from pretrained model
    model = UIE.from_pretrained(model_name_or_path)
    return model, tokenizer
