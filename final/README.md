# Usage

## Convert Function

將 label studio 針對 UIE 任務所標記完的資料匯出後，透過 run_convert.py 轉換成模型所吃的 .txt 檔案，並依照比例切割成訓練資料集、驗證資料集、測試資料集後匯出。

``` python
python run_convert.py 
```
- 預設讀取 data/label_studio_data/label_studio_output.json
- 預設 train/dev/test 比例為 0.8/0.1/0.1
- 預設會做正規化（若在 label 時就已經把特殊字元例如\n, 空白等等特殊字元去除，則設定 `--is_regularize_data False`）

### 參數設定
## Training Function

微調模型的主要運行程式。
### 單卡訓練
``` python
python run_train.py \
    --device gpu \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --seed 11 \
    --model_name_or_path uie-base  \
    --dataset_path ./data/model_input_data/ \
    --train_file train.txt \
    --dev_file dev.txt  \
    --test_file test.txt  \
    --max_seq_len 768  \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size  8 \
    --num_train_epochs 5 \
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

### 多卡訓練

``` python
python -u -m paddle.distributed.launch --gpus "0,1,2,3" run_train.py \
    --device gpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 11 \
    --model_name_or_path uie-base  \
    --dataset_path ./data/model_input_data/ \
    --train_file train.txt \
    --dev_file dev.txt  \
    --test_file test.txt  \
    --max_seq_len 768  \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size  8 \
    --num_train_epochs 18 \
    --learning_rate 7e-7 \
    --label_names 'start_positions' 'end_positions' \
    --do_train \
    --do_eval \
    --do_predict \
    --do_export \
    --weight_decay 0.0005 \
    --output_dir ./results/checkpoint/model_best \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --resume_from_checkpoint ./results/checkpoint/model_best
```

### 參數設定
## Evaluation Function

驗證的主要運行程式。

``` python
python run_eval.py \
    --model_name_or_path ./results/checkpoint/model_best \
    --dev_file ./data/model_input_data/test.txt \
    --device gpu \
    --is_eval_by_class True \
    --max_seq_len 768 \
    --batch_size 8
```

``` python
# test
python run_eval.py \
    --model_name_or_path ./results/checkpoint/model_best/checkpoint-6000 \
    --dev_file ./data/model_input_data/test.txt \
    --device gpu:3 \
    --is_eval_by_class True \
    --max_seq_len 512 \
    --batch_size 16
```

### 參數設定
## Inference Function

預測的主要運行程式。

``` python
python run_infer.py \
    --data_file ./data/model_infer_data/example.txt \
    --save_dir ./results/inference_results/ \
    --precision fp32 \
    --batch_size 16 \
    --task_path ./results/checkpoint/model_best \
    --select_key all \
    --select_strategy threshold \
    --select_strategy_threshold 0.5
```

``` python
# test
python run_infer.py \
    --data_file ./data/model_input_data/example.txt \
    --save_dir ./results/inference_results/ \
    --precision fp32 \
    --batch_size 16 \
    --task_path ./results/checkpoint/model_best \
    --select_key all w\
    --select_strategy threshold \
    --select_strategy_threshold 0.5
```

### 參數設定
# 已完成

1. utils 們
2. run_train.py
3. run_eval.py
4. run_infer.py
5. logger.set_level -> ok
6. 整理 config

# 未完成

1. infer.py regular tokens
2. 整理所有 function docstring 和程式碼命名等等
3. test case
4. 看能不能實驗的時候 也能看所有指標
5. mlflow



