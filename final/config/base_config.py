from paddle.static import InputSpec

UIE_input_spec = [
    InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
    InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
    InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
    InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
]

NER_type = ["精神慰撫金額", "醫療費用", "薪資收入"]
