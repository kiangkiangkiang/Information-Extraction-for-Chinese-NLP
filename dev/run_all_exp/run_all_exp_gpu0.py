import os

# gpu:0 for arrange data
learning_rate = 3e-05
for i in range(5):
    os.system(
        f"python3 ../../information_extraction/model/finetune.py  \
        --device gpu:0 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --seed 1000 \
        --model_name_or_path uie-base  \
        --train_path ../../information_extraction/data/arrange_final_data/training_data.txt \
        --dev_path ../../information_extraction/data/arrange_final_data/eval_data.txt  \
        --test_path .../../information_extraction/data/arrange_final_data/testing_data.txt  \
        --max_seq_len 512  \
        --read_data_method chunk \
        --per_device_eval_batch_size 16 \
        --per_device_train_batch_size  16 \
        --multilingual True \
        --num_train_epochs 4 \
        --learning_rate {learning_rate} \
        --label_names 'start_positions' 'end_positions' \
        --do_train \
        --do_eval \
        --do_export \
        --output_dir ../../information_extraction/results/ckp_arrange_data_512_seed1000_{learning_rate}/ \
        --overwrite_output_dir \
        --disable_tqdm True \
        --metric_for_best_model eval_f1 \
        --load_best_model_at_end  True \
        --save_total_limit 1"
    )
    learning_rate = learning_rate / 2
