from config import DefaultConfig, generate_logger
from base_utils import set_seed
from typing import List
import json
import os
from decimal import Decimal

logger = generate_logger(name=__name__)


def do_split(dataset: dict) -> dict:

    pass


def split_labelstudio(
    labelstudio_file: str,
    save_dir: str = "./",
    seed: int = 100,
    split_ratio: List[int] = [0.8, 0.1, 0.1],
    is_shuffle: bool = True,
) -> None:

    logger.info(f"Converting {os.path.basename(labelstudio_file)} into {save_dir}...")
    set_seed(seed)

    if not os.path.exists(labelstudio_file):
        raise ValueError("Label studio file not found. Please input the correct path of label studio file.")

    if not os.path.exists(save_dir):
        logger.warning(f"{save_dir} not found. Automatically making a directory...")
        os.makedirs(save_dir)

    if len(split_ratio) != 3:
        raise ValueError(
            f"Length Error in split_ratio: {len(split_ratio)}. Only accept len(split_ratio)==3 for splits."
        )

    if Decimal(str(split_ratio[0])) + Decimal(str(split_ratio[1])) + Decimal(str(split_ratio[2])) != Decimal("1"):
        raise ValueError("Please set correct split_ratio, sum of elements in split_ratio should be equal to 1.")

    with open(labelstudio_file, "r", encoding="utf-8") as infile:
        for content in infile:
            dataset = json.loads(content)
        results = do_split(dataset)

    # ======================
    with open(doccano_file, "w", encoding="utf-8") as outfile:
        for item in results:
            outline = json.dumps(item, ensure_ascii=False)
            outfile.write(outline + "\n")
    pass


if __name__ == "__main__":
    pass
