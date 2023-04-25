from base_utils import get_root_dir
from typing import List
import argparse
from paddlenlp.trainer.argparser import strtobool
from config import generate_logger

logger = generate_logger(name=__name__)


def split_doccano(
    doccano_file: str,
    save_dir: str,
    seed: int,
    split_ratio: List[int] = [0.8, 0.1, 0.1],
    is_shuffle: bool = True,
):

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--doccano_file",
        default="./data/doccano.json",
        type=str,
        help="The doccano file exported from doccano platform.",
    )
    parser.add_argument("--save_dir", default="./data", type=str, help="The path of data that you wanna save.")
    parser.add_argument(
        "--splits",
        default=[0.8, 0.1, 0.1],
        type=float,
        nargs="*",
        help="The ratio of samples in datasets. [0.6, 0.2, 0.2] means 60% samples used for training, 20% for evaluation and 20% for test.",
    )
    parser.add_argument(
        "--is_shuffle", default="True", type=strtobool, help="Whether to shuffle the labeled dataset, defaults to True."
    )
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization")
    args = parser.parse_args()

    split_doccano(
        doccano_file=args.doccano_file,
        save_dir=args.save_dir,
        seed=args.seed,
        split_ratio=args.splits,
        is_shuffle=eval(args.is_shuffle),
    )
