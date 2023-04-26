from base_utils import get_root_dir, set_seed
from typing import List
import argparse
from paddlenlp.trainer.argparser import strtobool
from config import generate_logger, BaseConfig
import os
from decimal import Decimal

logger = generate_logger(name=__name__)


def split_doccano(
    doccano_file: str,
    save_dir: str = "./",
    seed: int = 100,
    split_ratio: List[int] = [0.8, 0.1, 0.1],
    is_shuffle: bool = True,
) -> None:

    set_seed(seed)

    if not os.path.exists(doccano_file):
        raise ValueError("Please input the correct path of doccano file.")

    if not os.path.exists(save_dir):
        logger.warning(f"{save_dir} not found. Automatically making a directory...")
        os.makedirs(save_dir)

    if len(split_ratio) != 3:
        raise ValueError(
            f"Length Error in split_ratio: {len(split_ratio)}. Only accept len(split_ratio)==3 for splits."
        )

    if Decimal(str(split_ratio[0])) + Decimal(str(split_ratio[1])) + Decimal(str(split_ratio[2])) != Decimal("1"):
        raise ValueError("Please set correct split_ratio, sum of elements in split_ratio should be equal to 1.")

    with open(doccano_file, "r", encoding="utf-8") as f:
        raw_examples = f.readlines()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    try:
        root_dir = get_root_dir()
        default_deccano_file = root_dir + BaseConfig.doccano_data_path
        default_save_dir = root_dir + BaseConfig.experiment_data_path
    except:
        logger.error("Fail to get root directory.")
        default_deccano_file = None
        default_save_dir = "./"

    parser.add_argument(
        "--doccano_file",
        type=str,
        default=default_deccano_file,
        help="The doccano file exported from doccano platform.",
    )
    parser.add_argument("--save_dir", default=default_save_dir, type=str, help="The path of data that you wanna save.")
    parser.add_argument(
        "--split_ratio",
        default=[0.8, 0.1, 0.1],
        type=float,
        nargs="*",
        help="The ratio of samples in datasets. [0.7, 0.2, 0.1] means 70% samples used for training, 20% for evaluation and 10% for test.",
    )
    parser.add_argument(
        "--is_shuffle", default="True", type=strtobool, help="Whether to shuffle the labeled dataset, defaults to True."
    )
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization")
    args = parser.parse_args()

    # TODO deal with muliple file in the directory

    split_doccano(
        doccano_file=args.doccano_file,
        save_dir=args.save_dir,
        seed=args.seed,
        split_ratio=args.splits,
        is_shuffle=eval(args.is_shuffle),
    )
