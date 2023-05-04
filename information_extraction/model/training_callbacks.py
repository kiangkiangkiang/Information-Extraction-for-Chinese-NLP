from paddle import nn, squeeze, load, cast, Tensor
import os
import sys
from paddlenlp.transformers import AutoModel
from typing import Optional

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from base_logger import generate_logger

logger = generate_logger(__name__)

# setup loss function and scale output dimension
def uie_loss_func(outputs, labels) -> float:
    # TODO MLFLOW
    loss_func = nn.BCELoss()
    logger.debug(f"In uie_loss_func, log type(outputs): {type(outputs)}, type(labels): {type(labels)}")
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = cast(start_ids, "float32")
    end_ids = cast(end_ids, "float32")
    # logger.debug(f"start_prob: {start_prob}, start_ids: {start_ids}")
    loss_start = loss_func(start_prob, start_ids)
    loss_end = loss_func(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    return loss


class model_scaler(nn.Layer):
    def __init__(self, model_name_or_path=None, name_scope=None, dtype="float32", in_features=768, out_features=1):
        super().__init__(name_scope, dtype)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.linear_start = nn.Linear(in_features, out_features)
        self.linear_end = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid()
        self._load_linear_param()

    def _load_linear_param(self):
        linear_start_param = load("params/linear_start.pdparams")
        linear_end_param = load("params/linear_start.pdparams")
        self.linear_start.set_state_dict(linear_start_param)
        self.linear_end.set_state_dict(linear_end_param)

    def forward(self, **inputs):
        outputs = self.model(**inputs)

        # start linear scale
        start_logits = self.linear_start(outputs)
        start_logits = squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(outputs)
        end_logits = squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob
