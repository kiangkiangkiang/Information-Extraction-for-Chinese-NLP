from colorama import Back, Fore, Style
import sys
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field


# default config in this project
@dataclass
class DefaultConfig:
    root_dir: str = field(default="Chinese-Verdict-NLP", metadata={"help": "The root directory name of this project."})
    label_studio_data_path: str = field(
        default="/label_data/label_studio/",
        metadata={"help": "The label studio output data path after root directory."},
    )
    doccano_data_path: str = field(
        default="/label_data/doccano/",
        metadata={"help": "The doccano data path (converted from label studio output) after root directory."},
    )
    experiment_data_path: str = field(
        default="/information_extraction/data/",
        metadata={"help": "The train/dev/test data path (splitted from deccano) after root directory."},
    )


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
