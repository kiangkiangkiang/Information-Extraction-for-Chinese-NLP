from base_logger import generate_logger
from typing import List
from dataclasses import dataclass, field
import os


logger = generate_logger(name=__name__)

# default config in this project
@dataclass()
class BaseConfig:
    ner_type: List[str] = field(default_factory=lambda: ["精神慰撫金額", "醫療費用", "薪資收入"])

    label_studio_data_path: str = field(
        default="/label_data/",
        metadata={"help": "The label studio output data path after root directory."},
    )

    experiment_data_path: str = field(
        default="/information_extraction/data/",
        metadata={"help": "The train/dev/test data path (splitted from deccano) after root directory."},
    )

    train_result_path: str = field(
        default="/information_extraction/results/checkpoint/",
        metadata={"help": "The train/dev/test data path (splitted from deccano) after root directory."},
    )

    root_dir_name: str = field(
        default="Chinese-Verdict-NLP", metadata={"help": "The root directory name of this project."}
    )

    root_dir: str = field(default="", metadata={"help": "root dir of root_dir_name"}, init=False)

    finding_limits_of_root: int = field(default=10, metadata={"help": "找根目錄的上限次數. Defaults to 10."})

    def __post_init__(self):
        self.root_dir = self._get_root_dir()

    def _get_root_dir(self) -> str:
        """
        找到根目錄root_dir_name的完整路徑

        Args:
            root_dir_name (str, optional): 根目錄資料夾名稱. Defaults to base_config.root_dir.
            limits (int, optional): 找根目錄的上限次數. Defaults to 10.

        Returns:
            str: root dir of root_dir_name, if it found, else raise ValueError.
        """

        now_folder = os.path.dirname(os.path.realpath(__file__))
        now_folder_name = os.path.basename(now_folder)
        for _ in range(self.finding_limits_of_root):
            if now_folder_name != self.root_dir_name:
                now_folder = os.path.dirname(now_folder)
                now_folder_name = os.path.basename(now_folder)
            else:
                return now_folder

        logger.error("Fail to get root directory.")
        return None
