from uie import *

""" TODO
切文本：

方法一：
先用隨便模型的 NER 快速找到所有「錢的index」，之後後開 30字（字數設參數做調整，也許跟max_seq_len有關）
的 windows，然後無腦 concate再一起當作原文本來跑模型看看。


"""
