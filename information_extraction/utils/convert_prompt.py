# from ..config import get_logger
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
print(parent)
sys.path.append(parent)
from config import get_logger

if __name__ == "__main__":
    logger = get_logger()

    logger.debug("dfijaerpijfaefijp")
