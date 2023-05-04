import os
import sys
from typing import Optional

from paddle import Tensor, cast, load, nn, squeeze
from paddlenlp.transformers import AutoModel, PretrainedModel

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from base_logger import generate_logger

logger = generate_logger(__name__)

# setup loss function and scale output dimension
def uie_loss_func(outputs, labels) -> float:
    # TODO MLFLOW
    loss_func = nn.BCELoss()
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = cast(start_ids, "float32")
    end_ids = cast(end_ids, "float32")
    # logger.debug(f"start_prob: {start_prob}, start_ids: {start_ids}")
    loss_start = loss_func(start_prob, start_ids)
    loss_end = loss_func(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    return loss


class model_scaler(PretrainedModel):
    def __init__(self, model_name_or_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.model.config
        self.linear_start = nn.Linear(self.config.hidden_size, 1)
        self.linear_end = nn.Linear(self.config.hidden_size, 1)
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
