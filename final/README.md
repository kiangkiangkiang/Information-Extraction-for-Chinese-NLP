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
python run_train.py ...
```

## Evaluation

``` python
python run_eval.py ...
```

## Inference

``` python
python run_infer.py ...
```

## Prediction

``` python
python main_app.py ...
```