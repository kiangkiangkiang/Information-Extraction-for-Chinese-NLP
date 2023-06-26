import os

# 6/19 6/20
"""
seed_list = [2023]
for seed in seed_list:
    os.system(
        f"python3 ../../information_extraction/model/finetune.py  \
        --device gpu:0 \
        --logging_steps 10 \
        --save_steps 750 \
        --eval_steps 750 \
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
        --num_train_epochs 3 \
        --learning_rate 1.3e-5 \
        --label_names 'start_positions' 'end_positions' \
        --do_train \
        --do_eval \
        --do_export \
        --output_dir ../../information_extraction/results/ckp_final_data_768_epochs4_seed_{seed}_lr1.3e-5\
        --overwrite_output_dir \
        --disable_tqdm True \
        --metric_for_best_model eval_f1 \
        --load_best_model_at_end  True \
        --save_total_limit 1"
    )
"""

# ckp_final_data_768_epochs4_seed_11_lr125e-5_optAdamax
os.system(
    f"python3 ../../information_extraction/model/finetune.py  \
        --device gpu:0 \
        --logging_steps 10 \
        --save_steps 1000 \
        --eval_steps 1000 \
        --seed 11 \
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
        --learning_rate 2e-6 \
        --label_names 'start_positions' 'end_positions' \
        --do_train \
        --do_eval \
        --do_export \
        --output_dir ../../information_extraction/results/retrain_ckp__final_data_768_epochs4_seed_1_retrain_3e-6\
        --overwrite_output_dir \
        --disable_tqdm True \
        --metric_for_best_model eval_f1 \
        --load_best_model_at_end  True \
        --save_total_limit 1 \
        --resume_from_checkpoint ../../information_extraction/results/retrain_ckp__final_data_768_epochs4_seed_1_retrain_3e-6/checkpoint-12750"
)
