######### json_utils.py, not use
def merge_json(json_folder_path: str, output_path: Optional[str] = "./") -> None:
    """Merge JSON files.

    Args:
        json_folder_path (str): The folder where all JSON files locate.
        output_path (Optional[str], optional): Where the merged JSON file locates. Defaults to "./".

    Raises:
        ValueError: File not found.
    """

    merge_result = []
    counter = 0
    if os.path.exists(json_folder_path):
        all_json_file = [json_file for json_file in os.listdir(json_folder_path) if json_file[-5:] == ".json"]
        logger.info(f"Find the following json file: {all_json_file}.")
        if len(all_json_file) > 0:
            # read
            for each_json in all_json_file:
                logger.info(f"Merging the file {each_json}...")
                with open(os.path.join(json_folder_path, each_json), "r", encoding="utf-8") as infile:
                    for f in infile:
                        all_content = json.loads(f)
                        logger.info(f"Length of {each_json} is {len(all_content)}.")
                        counter += len(all_content)
                        for content in all_content:
                            merge_result.append(content)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            with open(os.path.join(output_path, "merged_data.json"), "w", encoding="utf-8") as outfile:
                jsonString = json.dumps(merge_result, ensure_ascii=False)
                outfile.write(jsonString)

            logger.info(f"Successful merge all file (len={counter}) to {output_path}/merged_data.json.")

        else:
            raise ValueError("Cannot found json file.")
    else:
        raise ValueError(f"Cannot found the path {json_folder_path}")


def is_repeat_content_exist(json_file: str) -> dict:
    """Check if there are duplicate cases in the JSON file.

    Args:
        json_file (str): JSON file locate and name. Must have the same format as label studio output and ''jid' attributes exist.

    Raises:
        ValueError: File not found.

    Returns:
        dict: The information of duplicate cases in JSON file.
    """

    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as infile:
            for f in infile:
                all_content = json.loads(f)
        all_jid = [i["data"]["jid"] for i in all_content]
        unique_data_len = len(pd.unique(all_jid))
        all_content_len = len(all_content)
        logger.info(
            f"All data: {all_content_len}. Unique data: {unique_data_len}. Repeat data: {all_content_len - unique_data_len}."
        )
        if all_content_len != unique_data_len:
            logger.warning(f"Head 5 repeat data: {pd.value_counts(all_jid)[:5]}")
        return {
            "is_repeat": not (all_content_len == unique_data_len),
            "all_data_length": all_content_len,
            "unique_data_len": unique_data_len,
            "value_counts_table": pd.value_counts(all_jid),
        }
    else:
        raise ValueError(f"Cannot found the path {json_file}")


######### data_utils.py, not use
def random_choose(prob: float = 0.5) -> bool:
    """根據機率隨機產生 True/False
    Args:
        prob (float, optional): 產生 True 的機率值. Defaults to 0.5.

    Returns: bool.
    """
    criterion = np.random.uniform(0, 1, 1)[0]
    return True if criterion < prob else False
