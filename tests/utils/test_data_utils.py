from utils.data_utils import *
import pytest


def test_read_data_by_chunk_when_path_not_found_then_raise_error():
    # given
    example_error_path = "./not/a/real/path"

    # when
    with pytest.raises(ValueError) as error:
        _ = next(read_data_by_chunk(data_path=example_error_path))

    # then
    expected_error = f"Path not found {example_error_path}."
    assert str(error.value) == expected_error


def test_read_data_by_chunk_when_max_seq_len_too_small_then_raise_error():
    # given
    example_model_input_data_path = "./tests/data/example_model_input_data.txt"

    # when
    with pytest.raises(ValueError) as error:
        _ = next(read_data_by_chunk(data_path=example_model_input_data_path, max_seq_len=1))

    # then
    expected_error = "The value of max_seq_len is too small. Please set a larger value."
    assert str(error.value) == expected_error


def test_read_data_by_chunk_when_start_ids_larger_than_end_ids_raise_error():
    # given
    example_model_input_data_path = "./tests/data/example_model_input_error_data.txt"

    # when
    with pytest.raises(DataError) as error:
        _ = next(read_data_by_chunk(data_path=example_model_input_data_path))

    # then
    expected_error = f"Error in result list. Invalid start or end location (start: 19, end: 13). Please check the data in {example_model_input_data_path}."
    assert str(error.value) == expected_error


def test_read_data_by_chunk_successful(example_model_input_content):
    # given
    example_model_input_data_path = "./tests/data/example_model_input_data.txt"
    max_seq_len = 512
    prompt = "醫療費用"

    # when
    result = next(read_data_by_chunk(data_path=example_model_input_data_path, max_seq_len=max_seq_len))
    result_content = result["content"]

    # then
    expected_result_content = example_model_input_content[0][: (max_seq_len - len(prompt) - 3)]  # 3: [CLS] [SEP] [SEP]
    assert expected_result_content == result_content
