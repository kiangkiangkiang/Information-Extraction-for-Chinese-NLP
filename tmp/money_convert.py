import cn2an
import opencc
import pandas as pd
import re
from paddlenlp.utils.log import logger

converter_t2s = opencc.OpenCC("t2s.json")
converter_s2t = opencc.OpenCC("s2t.json")
convert_chinese_to_number = lambda x: cn2an.cn2an(converter_t2s.convert(x), "smart")


df = pd.read_csv("./verdict8000_uie_inference_result.csv")


def format_number(money):
    result = re.sub("零", "", converter_s2t.convert(money))
    # '十二萬九百一十二元' -> '十二萬零九百一十二元'
    current_index = []
    for index, unit in enumerate(["十", "百", "千"]):
        if unit in result:
            current_index.append(index)

    if len(current_index) == 3:
        return result
    elif len(current_index) == 0 or current_index[-1] != 2:
        # 五萬零八百零一
        if current_index == [1]:
            index = [i for i in range(len(result)) if result[i] == "百"][0]
            if index != len(result) - 1:
                result = result[: index + 1] + "零" + result[index + 1 :]
            return "零" + result
        else:
            return "零" + result
    elif current_index != [1, 2]:
        # 加在千後面
        index = [i for i in range(len(result)) if result[i] == "千"][0]
        if index != len(result) - 1:
            result = result[: index + 1] + "零" + result[index + 1 :]
        return result
    else:
        # 加在百後面
        index = [i for i in range(len(result)) if result[i] == "百"][0]
        if index != len(result) - 1:
            result = result[: index + 1] + "零" + result[index + 1 :]
        return result


def postprocess(money):
    if money[-1] != "元":
        money = money + "元"

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

        final_money[-1] = format_number(final_money[-1])

        final_money = convert_chinese_to_number("".join(final_money))
        logger.info(f"Success!!!, Before: {money}, After: {final_money}")
    except:
        logger.error(f"Cannot convert this case: {money} -> {final_money} -> Fail.")
        final_money = money

    return final_money


# TODO 逗點無法處理：200,000元
number_of_money = []
for i in (16, 17, 18):
    money = list(df.iloc[:, i])
    for u in money:
        if str(u) != "nan":
            try:
                new_u = "".join(filter(str.isalnum, u))
                new_u = re.sub("餘", "", new_u)
                number_of_money.append(convert_chinese_to_number(new_u))
            except:
                number_of_money.append(postprocess(new_u))
                # breakpoint()
        else:
            number_of_money.append("nan")
    print("finish")
breakpoint()
