import os

# 6/19
for seed in range(10, 17):
    os.system(
        f"python3 ../../information_extraction/model/finetune.py  \
        --device gpu:0 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --seed {seed} \
        --model_name_or_path uie-base  \
        --train_path ../../information_extraction/data/final_data/training_data.txt \
        --dev_path ../../information_extraction/data/final_data/eval_data.txt  \
        --test_path ../../information_extraction/data/final_data/testing_data.txt  \
        --max_seq_len 768  \
        --read_data_method chunk \
        --per_device_eval_batch_size 8 \
        --per_device_train_batch_size 8 \
        --multilingual True \
        --num_train_epochs 5 \
        --learning_rate 2e-5 \
        --label_names 'start_positions' 'end_positions' \
        --do_train \
        --do_eval \
        --do_export \
        --output_dir ../../information_extraction/results/ckp_final_data_768_epochs5_seed_{seed}_lr2e-5\
        --overwrite_output_dir \
        --disable_tqdm True \
        --metric_for_best_model eval_f1 \
        --load_best_model_at_end  True \
        --save_total_limit 1"
    )
