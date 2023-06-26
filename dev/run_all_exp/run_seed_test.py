import os

my_seed = 123

os.system(
    f"python3 ../../information_extraction/model/finetune.py  \
    --device gpu:3 \
    --logging_steps 10 \
    --save_steps 750 \
    --eval_steps 750 \
    --seed {my_seed} \
    --model_name_or_path uie-base \
    --train_path ../../information_extraction/data/final_data_seed{my_seed}/training_data.txt \
    --dev_path ../../information_extraction/data/final_data_seed{my_seed}/eval_data.txt  \
    --test_path ../../information_extraction/data/final_data_seed{my_seed}/testing_data.txt  \
    --max_seq_len 768  \
    --read_data_method chunk \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size 8 \
    --multilingual True \
    --num_train_epochs 3 \
    --learning_rate 1e-05 \
    --label_names 'start_positions' 'end_positions' \
    --do_train \
    --do_eval \
    --do_export \
    --output_dir ../../information_extraction/results/ckp_final_data_seed{my_seed}\
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1"
)
