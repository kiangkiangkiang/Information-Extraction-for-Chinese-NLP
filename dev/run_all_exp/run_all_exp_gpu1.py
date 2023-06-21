import os

# 6/19
"""
learning_rate = 5e-05
for i in range(6):
    os.system(
        f"python3 ../../information_extraction/model/finetune.py  \
        --device gpu:1 \
        --logging_steps 10 \
        --save_steps 600 \
        --eval_steps 600 \
        --seed 1000 \
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
        --learning_rate {learning_rate} \
        --label_names 'start_positions' 'end_positions' \
        --do_train \
        --do_eval \
        --do_export \
        --output_dir ../../information_extraction/results/ckp_final_data_768_epochs5_seed_1000_lr{learning_rate}\
        --overwrite_output_dir \
        --disable_tqdm True \
        --metric_for_best_model eval_f1 \
        --load_best_model_at_end  True \
        --save_total_limit 1"
    )
    learning_rate = learning_rate / 2
"""
loss_weights = ["1 3 6", "1 2 3"]
for lw in loss_weights:
    os.system(
        f"python3 ../../information_extraction/model/finetune_loss_by_group.py  \
        --device gpu:1 \
        --logging_steps 10 \
        --save_steps 750 \
        --eval_steps 750 \
        --seed 1000 \
        --model_name_or_path uie-base  \
        --train_path ../../information_extraction/data/final_data/training_data.txt \
        --dev_path ../../information_extraction/data/final_data/eval_data.txt  \
        --test_path ../../information_extraction/data/final_data/testing_data.txt  \
        --max_seq_len 768  \
        --read_data_method chunk \
        --per_device_eval_batch_size 8 \
        --per_device_train_batch_size 8 \
        --multilingual True \
        --num_train_epochs 3 \
        --learning_rate 1.25e-05 \
        --label_names 'start_positions' 'end_positions' \
        --do_train \
        --do_eval \
        --do_export \
        --output_dir ../../information_extraction/results/ckp_final_data_768_epochs4_lr1.25e-05_loss{lw.replace(' ', '-')}\
        --overwrite_output_dir \
        --disable_tqdm True \
        --metric_for_best_model eval_f1 \
        --load_best_model_at_end  True \
        --save_total_limit 1 \
        --loss_weight {lw}"
    )
