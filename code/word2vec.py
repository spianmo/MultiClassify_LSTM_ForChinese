import multiprocessing

import numpy as np
from gensim.models.word2vec import Word2Vec

voc_dim = 150  # word的向量维度
min_out = 4  # 单词出现频率数
window_size = 7  #
cpu_count = multiprocessing.cpu_count()


def word2vec_train(X_Vec):
    model_word = Word2Vec(vector_size=voc_dim,
                          min_count=min_out,
                          window=window_size,
                          workers=cpu_count,
                          epochs=100)
    model_word.build_vocab(X_Vec)
    model_word.train(X_Vec, total_examples=model_word.corpus_count, epochs=model_word.epochs)
    model_word.save('../model/Word2Vec.pkl')

    print(len(model_word.wv))
    input_dim = len(model_word.wv.key_to_index) + 1  # 频数小于阈值的词语统统放一起，编码为0
    embedding_weights = np.zeros((input_dim, voc_dim))
    w2dic = {}
    for i in range(len(model_word.wv.key_to_index)):
        embedding_weights[i + 1, :] = model_word.wv[list(model_word.wv.key_to_index)[i]]
        w2dic[list(model_word.wv.key_to_index)[i]] = i + 1
    return input_dim, embedding_weights, w2dic
