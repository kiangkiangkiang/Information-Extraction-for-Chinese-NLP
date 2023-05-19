# 模型是怎麼evaluation?

## 前情提要

模型主要是針對input的每個token都做預測，看哪個token屬於「start token」，哪個token屬於「end token」。

因此假設最初設定input最大長度為512個token，模型則是針對此512 token都做預測。因此每個token都有一個機率值。


## 透過paddlenlp.metrics.SpanEvaluator模組


### 1. 取出機率高於threshold的值

在init SpanEvaluator時，可以設「limit」參數，預測機率高於此參數的token取出，預設0.5。

### 2. get span

取出所有機率高於limit的token後，透過get_span()內部演算法，將這些token重組成 (start, end)格式的span，代表最後的預測答案。

    - Note: get_span() 演算法的概念基本上就是把相近的token組合起來。


### 3. 統計正確答案

取的所有 (start, end) span後，便可將此預測的span，和真實的span做評估，其中：

    - num_correct: 兩個span完全相等。
    - num_infer: 預測的span總數
    - num_label: 真實答案的span總數

### 4. Evaluation

- precision: num_correct/num_infer
  -  代表所有預測的值當中，正確的比例。
  -  低表示預測的不准：可能答案是A卻預測B
  
- recall: num_correct/num_label
  - 代表在所有真實答案中，正確抓出多少比例。
  - 低表示沒抓出正確答案，可能答案有A，模型卻沒output出A。

- f1: 2 * precision * recall / (precision + recall)

### QA

1. Ｑ：怎麼判斷有正負樣本差異大造成的影像？
    - Ａ：透過evaluation metrics，當**recall**太低，代表模型傾向不output出東西，可能表示負樣本太高，導致模型學習過程中不易output。

