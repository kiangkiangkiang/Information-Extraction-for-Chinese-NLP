from colorama import Back, Fore, Style
import sys
import logging
from typing import Dict, Optional

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


# label studio data name
label_studio_default_data = []
