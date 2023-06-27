import os

# 6/19
"""
learning_rate = 5e-05
for i in range(6):
    os.system(
        f"python3 ../../information_extraction/model/finetune.py  \
        --device gpu:2 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --seed 1000 \
        --model_name_or_path uie-base  \
        --train_path ../../information_extraction/data/arrange_final_data_mean/training_data.txt \
        --dev_path ../../information_extraction/data/arrange_final_data_mean/eval_data.txt  \
        --test_path ../../information_extraction/data/arrange_final_data_mean/testing_data.txt  \
        --max_seq_len 512  \
        --read_data_method chunk \
        --per_device_eval_batch_size 16 \
        --per_device_train_batch_size  16 \
        --multilingual True \
        --num_train_epochs 5 \
        --learning_rate {learning_rate} \
        --label_names 'start_positions' 'end_positions' \
        --do_train \
        --do_eval \
        --do_export \
        --output_dir ../../information_extraction/results/ckp_arrange_final_data_mean_512_epochs5_seed_1000_lr{learning_rate}\
        --overwrite_output_dir \
        --disable_tqdm True \
        --metric_for_best_model eval_f1 \
        --load_best_model_at_end  True \
        --save_total_limit 1"
    )
    learning_rate = learning_rate / 2
"""

# 6/20
learning_rate = [1e-05]
max_seq_len = [(768, 8)]
for lr in learning_rate:
    for msl in max_seq_len:
        os.system(
            f"python3 ../../information_extraction/model/finetune.py  \
            --device gpu:2 \
            --logging_steps 10 \
            --save_steps 500 \
            --eval_steps 500 \
            --seed 11 \
            --model_name_or_path uie-base  \
            --train_path ../../information_extraction/data/final_data_remove_repeat/training_data.txt \
            --dev_path ../../information_extraction/data/final_data_remove_repeat/eval_data.txt  \
            --test_path ../../information_extraction/data/final_data_remove_repeat/testing_data.txt  \
            --max_seq_len {msl[0]}  \
            --read_data_method chunk \
            --per_device_eval_batch_size {msl[1]} \
            --per_device_train_batch_size  {msl[1]} \
            --multilingual True \
            --num_train_epochs 3 \
            --learning_rate {lr} \
            --label_names 'start_positions' 'end_positions' \
            --do_train \
            --do_eval \
            --do_export \
            --output_dir ../../information_extraction/results/ckp_final_data_remove_repeat_{msl[0]}_epochs4_seed_11_lr{lr}\
            --overwrite_output_dir \
            --disable_tqdm True \
            --metric_for_best_model eval_f1 \
            --load_best_model_at_end  True \
            --save_total_limit 1"
        )


"""
python3 ../../information_extraction/model/finetune.py  \
            --device gpu:2 \
            --logging_steps 10 \
            --save_steps 500 \
            --eval_steps 500 \
            --seed 11 \
            --model_name_or_path uie-base  \
            --train_path ../../information_extraction/data/final_data_remove_repeat/training_data.txt \
            --dev_path ../../information_extraction/data/final_data_remove_repeat/eval_data.txt  \
            --test_path ../../information_extraction/data/final_data_remove_repeat/testing_data.txt  \
            --max_seq_len 768  \
            --read_data_method chunk \
            --per_device_eval_batch_size 8 \
            --per_device_train_batch_size  8 \
            --multilingual True \
            --num_train_epochs 2.7348 \
            --learning_rate 1.25e-09 \
            --label_names 'start_positions' 'end_positions' \
            --do_train \
            --do_eval \
            --do_export False\
            --output_dir ../../information_extraction/results/test \
            --overwrite_output_dir \
            --disable_tqdm True \
            --metric_for_best_model eval_f1 \
            --load_best_model_at_end  True \
            --save_total_limit 1 \
            --resume_from_checkpoint ../../information_extraction/results/ckp_final_data_768_epochs4_lr1.25e-05_loss1-2-4/checkpoint-7500
"""


"""
python ../../information_extraction/model/run_eval.py \
    --model_path ../../information_extraction/results/ckp_final_data_768_epochs5_seed_1000_lr1.25e-05 \
    --test_path ../../information_extraction/data/final_data/testing_data.txt \
    --device gpu:0 \
    --is_eval_by_class True \
    --max_seq_len 768 \
    --batch_size 32     
"""
