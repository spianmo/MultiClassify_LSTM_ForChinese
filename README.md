# LSTM中文文本进行情感多分类
使用LSTM神经网络来对中文文本进行情感分类，包含八个类别（like, disgust, happiness, sadness, anger, surprise, fear, none）


## 1 清洗数据
**data/**
清洗数据，去掉特殊符号，只保留汉字**code/dataset.py**

## 2 分词
jieba分词，见**code/dataset.py**

## 3 Word2Vec
词语嵌入(编码)到一个高维空间(向量)，利用python 的gensim库,见**code/word2vec.py**

## 4 LSTM构建
**code/lstm.py**

## 5 训练Train
**code/train.py**

## 6 推理Infer
**code/infer.py**
**model/**
