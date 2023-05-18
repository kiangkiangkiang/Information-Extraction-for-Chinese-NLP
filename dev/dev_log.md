1. 先把finetune流程搞定
2. 把轉換相關的功能搞定
3. pipeline, inference搞定
4. 模型相關的開發搞定
5. 


Note:
1. 模型相關的test case可能寫死一個input data，確認output維度啥的就好了
2. example最後全都搞定了再說


待做：
two_shot
three_shot
five_shot
more_data


test first build

test sce com

MLFLOW目前埋在trainer和finetune裏面



1. token & 切資料
   1. 先token後切資料: 目前做的，問題：1. [CLS], [prompt], [SEP] 這些應該都需要每筆資料都有
   2. 先切資料後token: 和3一樣，好像沒必要特別把token放到後面（想不出這麼做的意義）
   3. 先切資料先token: 預設16 & 2048）過模型前再處理 （無腦concate)，問題: 16 by 2048的16是16篇文本，不是同一篇
   4. 後切資料後token: ＮＯＷ TRY (不能完全不痛直接餵)
   5. 結論：不能把token擺在切資料前，不然會有1的問題 