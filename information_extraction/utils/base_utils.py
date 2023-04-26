import os
import sys
import numpy as np
import paddle
import random
from typing import List

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from config import generate_logger, BaseConfig

logger = generate_logger(__name__)
base_config = BaseConfig()


def set_seed(seed: int) -> None:
    """設定種子

    Args:
        seed (int): 固定種子
    """
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def shuffle_data(data: list) -> list:
    """shuffle data"""
    indexes = np.random.permutation(len(data))
    return [data[i] for i in indexes]
