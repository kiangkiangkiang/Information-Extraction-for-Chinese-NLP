import os
import sys
import numpy as np
import paddle
import random

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from config import generate_logger

logger = generate_logger(__name__)


def get_root_dir(root_dir_name: str = "Chinese-Verdict-NLP", limits: int = 10) -> str:
    """
    找到根目錄root_dir_name的完整路徑

    Args:
        root_dir_name (str, optional): 根目錄資料夾名稱. Defaults to "Chinese-Verdict-NLP".
        limits (int, optional): 找根目錄的上限次數. Defaults to 10.

    Raises:
        ValueError: 找不到 root_dir_name 的路徑。

    Returns:
        str: root dir of root_dir_name, if it found, else raise ValueError.
    """

    # setup root dir
    now_folder = os.path.dirname(os.path.realpath(__file__))
    now_folder_name = os.path.basename(now_folder)
    for _ in range(limits):
        if now_folder_name != root_dir_name:
            now_folder = os.path.dirname(now_folder)
            now_folder_name = os.path.basename(now_folder)
        else:
            os.chdir(now_folder)
            return now_folder

    # if root_dir_name not found
    raise ValueError(
        f"{root_dir_name} not found or path error. \
            Please make sure {root_dir_name} is the parent folder of {os.path.basename(__file__)}."
    )


def set_seed(seed: int) -> None:
    """設定種子

    Args:
        seed (int): 固定種子
    """
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# 這沒用到 準備刪掉他
def get_default_data_path(root_dir_name: str = "Chinese-Verdict-NLP") -> str:
    """取得 label studio 的 data 資料夾的路徑

    Args:
        root_dir_name (str, optional): 根目錄的名稱. Defaults to "Chinese-Verdict-NLP".

    Raises:
        ValueError: 根目錄名稱找不到。
        ValueError: data資料夾找不到

    Returns:
        str: data資料夾的路徑
    """

    root_dir = get_root_dir(root_dir_name=root_dir_name)
    result = root_dir + "/data/label_studio/"
    if root_dir == "Fail":
        raise ValueError(
            f"root dir not found. Please make sure {root_dir_name} is the parent folder of {os.path.basename(__file__)}."
        )
    if not os.path.exists(result):
        raise ValueError(f"Data path not found. Please make sure {result} is exist")
    return result
