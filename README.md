# Bi-LSTM中文文本情感多分类
该子项目归属于TrackerDesktop IM舆情分析。使用Bi-LSTM神经网络来对中文文本进行情感分类，包含八个类别（like, disgust, happiness, sadness, anger, surprise, fear, none）。

项目涉及的八分类情感语料库对应的文本标注工具及其后端:

[SentimentMarkTool: 一个简易的基于Vuetify的八分类中文情绪标注工具 (github.com)](https://github.com/spianmo/SentimentMarkTool)

[SentimentMarkServer: 八分类中文情绪标注工具后端，Springboot+MongoDB (github.com)](https://github.com/spianmo/SentimentMarkServer)

**simplifyweibo_8_moods八分类多情感数据集**

汇总emotion_corpus_microblog、simplifyweibo_5_moods、Nlpcc2014Train 的八分类情感分类数据集

> total：83790 items
> anger：6422
> disgust：8149
> happiness：12802
> like：8947
> sadness：16465
> fear：952
> surprise：1817
> none：28236

## 语料库

![image-20211202100725237](http://oss.cache.ren/img/image-20211202100725237.png)

![image-20211202100747826](http://oss.cache.ren/img/image-20211202100747826.png)

![image-20211202101546695](http://oss.cache.ren/img/image-20211202101546695.png)

![image-20211202101521766](http://oss.cache.ren/img/image-20211202101521766.png)

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

## 情感分析语料库统计
### 😇Weibo Emotion Corpus 七分类多情感数据集

2016 The Hong Kong Polytechnic University	微博语料，标注了7 emotions: like, disgust, happiness, sadness, anger, surprise, fear。 
- 数据条目：39661 items
- 论文地址：http://www.lrec-conf.org/proceedings/lrec2016/pdf/515_Paper.pdf
- 数据地址：https://github.com/hehuihui1994/emotion_corpus_weibo/blob/master/emotion_corpus_microblog.txt
- 文件名称：emotion_corpus_microblog.txt
- 文件大小：5.50MB
- 来源：香港理工大学

> happiness：9959
> 
> disgust：4876
> 
> like：4540
> 
> fear：661
> 
> sadness：14052
> 
> anger：4562
> 
> surprise：1011

![image-20211201214015915](http://oss.cache.ren/img/image-20211201214015915.png)

### 🍀simplifyweibo_8_moods 八分类多情感数据集

八分类情感分类数据集，详细带None
- 大小：26462条数据
- 文件名称：simplifyweibo_8_moods.txt
- 文件大小：2.20MB
- 来源：CSDN

> happiness：1456
> 
> disgust：2008
> 
> like：2446
> 
> fear：230
> 
> sadness：1676
> 
> anger：1436
> 
> surprise：620
> 
> none：16590

![image-20211201214102091](http://oss.cache.ren/img/image-20211201214102091.png)

### 🤩Nlpcc2014Train 八分类多情感数据集

Nlpcc2014八分类情感分类数据集，详细带None，包含2013
- 大小：48875条数据
- 文件名称：Nlpcc2014Train.txt
- 文件大小：4.12MB

> happiness：3192
> 
> disgust：3433
> 
> like：4921
> 
> fear：332
> 
> sadness：2787
> 
> anger：2138
> 
> surprise：901
> 
> none：31171

![image-20211201214127830](http://oss.cache.ren/img/image-20211201214127830.png)

### 🐳simplifyweibo_5_moods五分类多情感数据集

微博五分类情感分类数据集
- 数据条目：14306 items
- 文件名称：simplifyweibo_5_moods.txt
- 文件大小：1.17MB
- Author：Finger🌖

> anger:1860
> 
> disgust:3073
> 
> happiness:2872
> 
> like:4106
> 
> sadnass:2395

![image-20211202084724806](http://oss.cache.ren/img/image-20211202084724806.png)


### 🐲simplifyweibo_8_moods八分类多情感数据集

汇总emotion_corpus_microblog、simplifyweibo_5_moods、Nlpcc2014Train 的八分类情感分类数据集
- 数据条目：83790 items
- 文件名称：simplifyweibo_7_moods.csv
- 文件大小：9.29MB


> anger：6422
> 
> disgust：8149
> 
> happiness：12802
> 
> like：8947
> 
> sadness：16465
> 
> fear：952
> 
> surprise：1817
> 
> none：28236

![image-20211202101546695](http://oss.cache.ren/img/image-20211202101546695.png)

![image-20211202101521766](http://oss.cache.ren/img/image-20211202101521766.png)

![image-20211202100725237](http://oss.cache.ren/img/image-20211202100725237.png)

![image-20211202100747826](http://oss.cache.ren/img/image-20211202100747826.png)

### 😈simplifyweibo_4_moods 四分类情感数据集！！极不准确！！

36 万多条，带情感标注 新浪微博，包含 4 种情感，
其中喜悦happiness约 20 万条，愤怒anger、厌恶disgust、低落各约 5 万条 
> 微博数目（总体）：361744
> 
> 微博数目（喜悦happiness）：199496
> 
> 微博数目（愤怒anger）：51714
> 
> 微博数目（厌恶disgust）：55267
> 
> 微博数目（低落）：55267
- 字段说明 label	0 喜悦happiness，1 愤怒anger，2 厌恶disgust，3 低落
- 详细说明：https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb
- 文件名称：simplifyweibo_4_moods.csv
- 文件大小：68.0MB

### 🧐weibo_senti_100k 极性分析

带情感标注 新浪微博，正负向评论约各 5 万条 
评论数目（总体）：119988
评论数目（正向）：59993
评论数目（负向）：59995
- 字段说明：label	1 表示正向评论，0 表示负向评论
- 详细说明：https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb
- 文件大小：weibo_senti_100k.csv
- 文件大小：18.7MB
