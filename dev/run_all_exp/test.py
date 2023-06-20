import os

opts = ["Adagrad", "Adamax", "Adadelta", "Momentum", "Lamb", "SGD", "AdamW"]
# 6/19
for opt in opts:
    os.system(
        f"python3 ../../information_extraction/model/finetune.py  \
        --device gpu:0 \
        --logging_steps 10 \
        --save_steps 750 \
        --eval_steps 750 \
        --seed 11 \
        --model_name_or_path uie-base  \
        --train_path ../../information_extraction/data/toy_data/training_data.txt \
        --dev_path ../../information_extraction/data/toy_data/eval_data.txt  \
        --test_path ../../information_extraction/data/toy_data/testing_data.txt  \
        --max_seq_len 768  \
        --read_data_method chunk \
        --per_device_eval_batch_size 8 \
        --per_device_train_batch_size 8 \
        --multilingual True \
        --num_train_epochs 1 \
        --learning_rate 2e-5 \
        --label_names 'start_positions' 'end_positions' \
        --do_train \
        --do_eval \
        --do_export \
        --optimizer {opt} \
        --output_dir ../../information_extraction/results/ckp_toy_data_768_epochs5_seed_11_lr2e-5_opt{opt}\
        --overwrite_output_dir \
        --disable_tqdm True \
        --metric_for_best_model eval_f1 \
        --load_best_model_at_end  True \
        --save_total_limit 1"
    )
