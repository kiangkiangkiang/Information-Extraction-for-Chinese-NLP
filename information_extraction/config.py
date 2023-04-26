from colorama import Back, Fore, Style
import sys
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import os

# logging
class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args, colors: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""
        super().__init__(*args, **kwargs)
        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""
        record.color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL
        return super().format(record)


def generate_logger(name: str = __name__) -> logging.Logger:
    """產生基本的logger

    Args:
        name (str, optional): logger的名字. Defaults to __name__.

    Returns:
        logging.Logger: logger
    """

    formatter = ColoredFormatter(
        "{asctime} |{color} {levelname:8} {reset}| {name} | {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        colors={
            "DEBUG": Fore.CYAN,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
        },
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.handlers[:] = []
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


logger = generate_logger()

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
