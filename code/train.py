import os
import sys

import tensorflow.keras as keras
import numpy as np
import yaml

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard

from dataset import loadfile, clean_data
from lstm import lstm
from word2vec import word2vec_train
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.config.experimental.list_physical_devices('GPU')
np.random.seed()
# 参数

voc_dim = 150  # word的向量维度
lstm_input = 150  # lstm输入维度
epoch_time = 10  # epoch
batch_size = 32  # batch


def data2inx(w2indx, X_Vec):
    data = []
    for sentence in X_Vec:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)

        data.append(new_txt)
    return data


def train_lstm(model, x_train, y_train, x_test, y_test):
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',  # hinge
                  optimizer='adam', metrics=['mae', 'acc'])

    print("Train...")
    tbCallBack = TensorBoard(log_dir="../model/log", write_images=1, histogram_freq=1, write_grads=True)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_time, verbose=1, callbacks=[tbCallBack])

    print("Evaluate...")
    print(model.predict(x_test))
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)

    json_string = model.to_json()
    with open('../model/lstm_sentiment_classification_5_total.json', 'w') as outfile:
        outfile.write(json_string)
    model.save('../model/lstm_sentiment_classification_5_total.h5')
    print('Test score:', score)


if __name__ == '__main__':
    mood_list = ['anger', 'disgust', 'happiness', 'like', 'sadness']
    print("开始清洗数据................")
    if not os.path.exists('../data/clean/'):
        os.makedirs('../data/clean/')
    for mood in mood_list:
        clean_data('../data/' + mood + '.txt', '../data/clean/' + mood + '_clean.txt')
    print("清洗数据完成................")
    print("开始下载数据................")
    X_Vec, y = loadfile(mood_list)
    print("下载数据完成................")
    print("开始构建词向量................")
    input_dim, embedding_weights, w2dic = word2vec_train(X_Vec)
    print("构建词向量完成................")

    index = data2inx(w2dic, X_Vec)
    index2 = sequence.pad_sequences(index, maxlen=voc_dim)
    x_train, x_test, y_train, y_test = train_test_split(index2, y, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train, num_classes=len(mood_list))
    y_test = keras.utils.to_categorical(y_test, num_classes=len(mood_list))

    model = lstm(input_dim, embedding_weights, len(mood_list))
    train_lstm(model, x_train, y_train, x_test, y_test)
