# 基於機器學習之中文司法判決書資訊抽取分析

--- 

## Introduction

資訊抽取（Information Extraction）主要針對特定場景萃取重要資訊。在給定文本之下，資訊抽取任務能有效針對此文字進一步對文本做出相應資訊之萃取，本文實作 [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop) 官方所提供之 [UIE](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie) 模型，針對[臺灣司法院判決書資料](https://opendata.judicial.gov.tw/dataset?categoryTheme4Sys%5B0%5D=051&sort.publishedDate.order=desc&page=1)進行分析（詳細資料取得過程可參考[這裡](https://github.com/kiangkiangkiang/NLLP/tree/main/data)）。

由於在司法判決書資料文本過長，資訊繁瑣，格式不一，因此針對內文需花大量人力進行分析，因此在 NLP 任務當中亦有許多處理司法相關之技術 (NLLP, Natural Legal Language Processing)。而**本文主要是針對判決書進行資訊萃取**，在給定使用者所感興趣之資訊（prompt）之下，對判決書進行抽取，將相關資訊呈現。

## Environment

- python >= 3.7
- paddlenlp >= 2.5.2
- paddlepaddle-gpu >= 2.5.0 (if GPU used) 
## Experiment & Results

對於資訊抽取任務，本篇使用多種模型進行實驗比較，包括 UIE, BERT, xlnet 等等，並且在處理判決書等長文本問題當中，由於任務特性無法使用 [Two-Stage Method](https://github.com/kiangkiangkiang/Two-Stage-Method-For-Chinese-NLP) 解決，因此透過其他常見文本處理方式（chunking）。

模型實驗結果如下表：

| **Model**         |  **Precision** | **Recall** |  **F1** |
|:-----------------:|:--------------:|:----------:|:-------:|
|      UIE (zero shot)     |  0.22  |  0.38  |   0.28     |
|  UIE (chunking)    | **0.91** |  **0.87**  |  **0.89**   |
|  Mac-BERT      | 0.57 |  0.80   |  0.66     |
| electra, xlnet, roformer |  0  |  0  |  0   |
| gpt 3.5 |  0.26  |  0.06  |  0.1   |

由於 UIE 已預訓練在此 Prompt Tuning 等中文資訊抽取任務上，因此表現明顯較好。其他模型包含 xlnet, roformer 對於中文資料仍有相關問題（可參考 [issue](https://github.com/PaddlePaddle/PaddleNLP/issues?q=is%3Aissue+is%3Aopen+out+of+range+)）而無法有效訓練。

## Quick Start

### Convert Function

將 label studio 針對 UIE 任務所標記完的資料匯出後，透過 run_convert.py 轉換成模型所吃的 .txt 檔案，並依照比例切割成訓練資料集、驗證資料集、測試資料集後匯出。

``` python
python run_convert.py 
```
#### 重要參數

- `--labelstudio_file`: 預設`./data/label_studio_data/label_studio_output.json`，label studio 標記完後匯出的 JSON 檔案。
- `--save_dir`: 預設`./data/model_input_data/`，轉換後的 txt 檔案。
- `--split_ratio`: 預設`[0.8, 0.1, 0.1]`，訓練資料集、驗證資料集、測試資料集各個佔比。
- `--is_regularize_data`: 預設`True`，是否在轉換前清除特殊字元，ex. "\n"。
### Training Function

微調模型的主要運行程式。
#### 單卡訓練
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

#### 多卡訓練

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
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
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
    --save_total_limit 1 
```

#### 重要參數

- `--device`: 預設`gpu`，選擇用何種裝置訓練模型，可使用`cpu`或是指定 gpu ，例如：`gpu:0`。
- `--model_name_or_path`: 預設`uie-base`，訓練時所使用的模型或是模型 checkpoint 路徑。
- `--max_seq_len`: 預設`512`，模型在每個 batch 所吃的最大文本長度。
- `--per_device_train_batch_size`: 預設`16`，模型在每個裝置訓練所使用的批次資料數量。
- `--per_device_eval_batch_size`: 預設`16`，模型在每個裝置驗證所使用的批次資料數量。
- `--dataset_path`: 預設`./data/model_input_data/`，主要存放資料集的位置。
- `--train_file`: 預設`train.txt`，訓練資料集檔名。
- `--dev_file`: 預設`dev.txt`，驗證資料集檔名。
- `--test_file`: 預設`test.txt`，測試資料集檔名。
- `--eval_steps`: 預設與`--logging_steps`相同，指模型在每幾個訓練步驟時要做驗證。
- `--output_dir`: **必須**，模型訓練產生的 checkpoint 檔案位置。
- `--metric_for_best_model`: 預設`loss`，訓練過程中，選擇最好模型的依據。



### Evaluation Function

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

#### 重要參數

- `--device`: 預設`gpu`，選擇用何種裝置訓練模型，可使用`cpu`或是指定 gpu ，例如：`gpu:0`。
- `--model_name_or_path`: 預設`uie-base`，訓練時所使用的模型或是模型 checkpoint 路徑。
- `--max_seq_len`: 預設`512`，模型在每個 batch 所吃的最大文本長度。
- `--dev_file`: 預設`./data/model_input_data/test.txt`，驗證資料集的檔案路徑。
- `--batch_size`: 預設`16`，模型所使用的批次資料數量。
- `--is_eval_by_class`: 預設`False`，是否根據不同類別算出各自指標。

### Inference Function

預測的主要運行程式。

``` python
python run_infer.py \
    --data_file ./data/model_infer_data/example.txt \
    --save_dir ./results/inference_results/ \
    --precision fp32 \
    --batch_size 16 \
    --task_path ./results/checkpoint/model_best \
    --select_key text probability \
    --select_strategy threshold \
    --select_strategy_threshold 0.5
```

#### 重要參數

- `--data_file`: 預設`dev.txt`，驗證資料集檔名。
- `--save_dir`: **必須**，模型訓練產生的 checkpoint 檔案位置。
- `--is_regularize_data`: 預設`False`，是否在轉換前清除特殊字元，ex. "\n"。
- `--precision`: 預設`fp32`，模型推論時的精確度，可使用`fp16` (only for gpu) 或`fp32`，其中`fp16`較快，使用`fp16`需注意CUDA>=11.2，cuDNN>=8.1.1，初次使用需按照提示安装相關依賴（`pip install onnxruntime-gpu onnx onnxconverter-common`）。
- `--batch_size`: 預設`16`，模型所使用的批次資料數量。
- `--taskpath`: 用來推論所使用的 checkpoint 檔案位置。
- `select_strategy`: 預設`all`，模型推論完後，保留推論結果的策略，`all`表示所有推論結果皆保留。其他可選`max`，表示保留機率最高的推論結果。`threshold`表示推論結果機率值高於`select_strategy_threshold`的結果皆保留。
- `select_strategy_threshold`: 預設`0.5`，表示當`select_strategy=threshold`時的門檻值。
- `select_key`: 預設`text start end probability`，表示最終推論保留的值。僅保留文字及機率可設`text probability`。
