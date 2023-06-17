# Usage

## Convert

``` python
python run_convert.py 
```
- 預設讀取 data/label_studio_data/label_studio_output.json
- 預設 train/dev/test 比例為 0.8/0.1/0.1
- 預設會做正規化（若在 label 時就已經把特殊字元例如\n, 空白等等特殊字元去除，則設定 `--is_regularize_data False`）

## Train

``` python
python run_train.py \
    --device gpu \
    --logging_steps 10 \
    --save_steps 10 \
    --eval_steps 10 \
    --seed 42 \
    --model_name_or_path uie-base  \
    --dataset_path ./data/model_input_data/ \
    --train_file train.txt \
    --dev_file dev.txt  \
    --test_file test.txt  \
    --max_seq_len 512  \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size  16 \
    --num_train_epochs 0.01 \
    --learning_rate 1e-5 \
    --label_names 'start_positions' 'end_positions' \
    --do_train \
    --do_eval \
    --do_predict \
    --do_export \
    --output_dir ./results/checkpoint/model_best \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1
```

## Evaluation

``` python
python run_eval.py \
    --model_path ./results/checkpoint/model_best \
    --test_path ./data/model_input_data/test.txt \
    --device gpu \
    --is_eval_by_class True \
    --max_seq_len 512 \
    --batch_size 16 
```

## Inference

``` python
python run_infer.py \
    --data_path ./data/model_infer_data/example.txt \
    --save_dir ./results/inference_results/ \
    --precision fp16 \
    --batch_size 16 \
    --task_path ./results/checkpoint/model_best \
    --select_key text probability \
    --select_strategy max 
```

## Prediction

``` python
python main_app.py ...
```



# TODO
1. inference
2. predict
3. config
4. mlflow (check train eval by class)
5. unit test