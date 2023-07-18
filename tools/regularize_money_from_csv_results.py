import logging
import sys
import os
import colorlog
import cn2an
import opencc
from typing import List
import pandas as pd
import argparse
import re


ENTITY_TYPE = ["精神慰撫金額", "醫療費用", "薪資收入"]
LOGGER_LEVEL = logging.INFO


def create_logger(level=logging.DEBUG):
    log_config = {
        "DEBUG": {"level": 10, "color": "purple"},
        "INFO": {"level": 20, "color": "green"},
        "WARNING": {"level": 30, "color": "yellow"},
        "ERROR": {"level": 40, "color": "red"},
    }
    logger = logging.getLogger("convert_to_labelstudio.log")
    logger.setLevel(level)
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s - %(name)s: %(message)s",
        log_colors={key: conf["color"] for key, conf in log_config.items()},
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    return logger


logger = create_logger(level=LOGGER_LEVEL)


class ArabicNumbersFormatter(object):
    """數值格式化工具"""

    def __init__(self) -> None:
        self.converter_t2s = opencc.OpenCC("t2s.json")
        self.converter_s2t = opencc.OpenCC("s2t.json")
        self.convert_chinese_to_number = lambda x: cn2an.cn2an(self.converter_t2s.convert(x), "smart")

    def __add_zero_for_missing_unit(self, money: str) -> str:
        """將「單位為萬以內」的缺失單位補零。

        Ex.:
            六百二十五元 -> 零六百二十五元
            八十八元 -> 零八十八元

        Args:
            money (str): 萬元以內的金錢字串

        Returns:
            str: 補零後的金錢字串
        """
        result = re.sub("零", "", self.converter_s2t.convert(money))
        current_index = []
        for index, unit in enumerate(["十", "百", "千"]):
            if unit in result:
                current_index.append(index)

        # Add zero in all possible cases
        if len(current_index) == 3:  # [0, 1, 2]
            return result
        elif len(current_index) == 0 or current_index[-1] != 2:  # [], [0, 1], [0], [1]
            if current_index == [1]:
                index = [i for i in range(len(result)) if result[i] == "百"][0]
                if index != len(result) - 1:
                    result = result[: index + 1] + "零" + result[index + 1 :]
                return "零" + result
            else:
                return "零" + result
        elif current_index != [1, 2]:  # [0, 2], [2]
            index = [i for i in range(len(result)) if result[i] == "千"][0]
            if index != len(result) - 1:
                result = result[: index + 1] + "零" + result[index + 1 :]
            return result
        else:  # [1, 2]
            index = [i for i in range(len(result)) if result[i] == "百"][0]
            if index != len(result) - 1:
                result = result[: index + 1] + "零" + result[index + 1 :]
            return result

    def __format_arabic_numbers(self, money: str) -> str:
        """格式化數值：將金錢統一轉成中文。

        Args:
            money (str): Mix of Chinese and Arabic numbers

        Returns:
            str: If successfully convert, return the money with Arabic numbers only. Else, return the original input money.
        """
        money = money + "元" if money[-1] != "元" else money
        found_chinese = re.finditer("[\u4e00-\u9fa5]+", money)
        last_start = 0
        final_money = []
        try:
            for each_found in found_chinese:
                start, end = each_found.span()
                number = []
                for index in range(last_start, start):
                    number.append(money[index])
                number = cn2an.an2cn("".join(number)) if number[0] != "0" else "零" + cn2an.an2cn("".join(number))
                for index in range(start, end):
                    number += money[index]
                final_money.append(number)
                last_start = end
            final_money[-1] = self.__add_zero_for_missing_unit(final_money[-1])
            final_money = int(self.convert_chinese_to_number("".join(final_money)))
            logger.debug(f"Stage 2 success!!!, Before: {money}, After: {final_money}")
            return final_money
        except:
            return money

    def chinese_to_number(
        self, money_list: List[str], remain_outlier: bool = False, outlier_representation="nan"
    ) -> List[str]:
        """Convert a list of mix of Chinese and Arabic numbers into Arabic numbers only.

        Args:
            money_list (List[str]): List of mix of Chinese and Arabic numbers, like ['一萬五千元', '三千500'].

        Returns:
            List[str]: If successfully convert, return the money with Arabic numbers only. Else, return the original input money.
        """

        regularized_money_list = []
        fail_cases = []
        for money in money_list:
            if str(money) != "nan":
                money = "".join(filter(str.isalnum, re.sub("餘", "", money)))
                try:
                    regularized_money = int(self.convert_chinese_to_number(money))
                except:
                    regularized_money = self.__format_arabic_numbers(money)
                if regularized_money == money:
                    fail_cases.append(money)
                    if not remain_outlier:
                        regularized_money = outlier_representation
            regularized_money_list.append(regularized_money)
        if fail_cases:
            logger.error(f"Fail Cases: {fail_cases}")
            if not remain_outlier:
                logger.info(f"Replace all the fail cases into {outlier_representation}.")
        logger.info(
            f"Error Rate of Converting: {len(fail_cases)}/{len(money_list)} = {len(fail_cases)/len(money_list):.4f}."
        )
        return regularized_money_list


# python regularize_money_from_csv_results.py --csv_results_path ./verdict8000_uie_inference_result.csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_results_path", type=str)
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--save_name", type=str, default="regularized_result.csv")
    args = parser.parse_args()

    if not os.path.exists(args.csv_results_path):
        raise ValueError(f"Path not found: {args.csv_results_path}.")

    if not os.path.exists(args.save_path):
        print(f"Path not found: {args.save_path}. Auto-create the path...")
        os.mkdir(args.save_path)

    csv_results = pd.read_csv(args.csv_results_path)

    formatter = ArabicNumbersFormatter()

    logger.info("Start Converting...")
    for each_entity in ENTITY_TYPE:
        logger.info(f"==========Arabic Numbers Converting: {each_entity}==========")
        regularized_money_list = formatter.chinese_to_number(money_list=csv_results.loc[:, each_entity].tolist())
        csv_results.loc[:, each_entity] = regularized_money_list
    logger.info("Finish Converting...")

    logger.info(f"Write the results into {os.path.join(args.save_path, args.save_name)}")
    csv_results.to_csv(os.path.join(args.save_path, args.save_name), header=True, index=False, encoding="utf_8_sig")
    logger.info("Finish.")
