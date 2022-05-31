import re

import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
from keras.preprocessing import sequence

if __name__ == '__main__':
    voc_dim = 150
    ######

    model_word = Word2Vec.load('../model/Word2Vec.pkl')

    input_dim = len(model_word.wv.key_to_index) + 1
    embedding_weights = np.zeros((input_dim, voc_dim))
    w2dic = {}

    for i in range(len(model_word.wv.key_to_index)):
        embedding_weights[i + 1, :] = model_word.wv[list(model_word.wv.key_to_index)[i]]
        w2dic[list(model_word.wv.key_to_index)[i]] = i + 1

    #model = load_model('../model/lstm_sentiment_classification_8_total.h5')
    model = load_model('../model/simplifyweibo_5_moods.h5')

    pchinese = re.compile('([\u4e00-\u9fa5]+)+?')

    # ['anger', 'disgust', 'happiness', 'like', 'sadness', 'fear', 'surprise']
    label = {0: "生气", 1: "厌恶", 2: "快乐", 3: "喜爱", 4: "悲伤", 5: "恐惧", 6: "惊讶", 7: "中性"}
    #label = {0: "生气", 1: "厌恶", 2: "快乐", 3: "喜爱", 4: "悲伤"}
    print("请输入：")
    while True:
        in_str = input()
        in_stc = ''.join(pchinese.findall(in_str))

        in_stc = list(jieba.cut(in_stc, cut_all=True, HMM=False))

        new_txt = []

        data = []
        for word in in_stc:
            try:
                new_txt.append(w2dic[word])
            except:
                new_txt.append(0)
        data.append(new_txt)
        print(data)
        data = sequence.pad_sequences(data, maxlen=voc_dim)
        print(data)
        pre = model.predict(data)[0].tolist()
        print(pre)
        print("输入：")
        print("  ", in_str)
        print("        ")
        print("输出:")
        print("  ", label[pre.index(max(pre))])
        print("  ", label[pre.index(max(pre))])
