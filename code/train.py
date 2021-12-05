import os
import random
import warnings

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping

from dataset import loadfile, clean_data
from gen_dict import generate_dict
from keras_to_tensorflow import h5_to_pb
from lstm import lstm
from word2vec import word2vec_train

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.config.experimental.list_physical_devices('GPU')
np.random.seed(512)
random.seed(512)
tf.random.set_seed(512)
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


class CustomEarlyStopping(Callback):
    def __init__(self, ratio=0.0,
                 patience=0, verbose=0):
        super(CustomEarlyStopping, self).__init__()

        self.ratio = ratio
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get('val_loss')
        current_train = logs.get('loss')
        if current_val is None:
            warnings.warn('Early stopping requires %s available!' %
                          self.monitor, RuntimeWarning)

        # If ratio current_loss / current_val_loss > self.ratio
        if self.monitor_op(np.divide(current_train, current_val), self.ratio):
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % self.stopped_epoch)


def train_lstm(model, x_train, y_train, x_test, y_test, output_tag: str):
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',  # hinge
                  optimizer='adam', metrics=['mae', 'acc'])

    print("Train...")
    tbCallBack = TensorBoard(log_dir="../model/log", write_images=1, histogram_freq=1, write_grads=True)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_time, validation_split=0.2,
                        verbose=1,
                        callbacks=[tbCallBack,
                                   ModelCheckpoint(filepath='../model/tmp/weights_{epoch:02d}-{val_loss:.4f}-{'
                                                            'val_acc:.4f}.hdf5',
                                                   monitor='val_loss', verbose=0, save_best_only=False,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=1),
                                   CustomEarlyStopping(ratio=0.5, patience=2, verbose=1)
                                   ]
                        )

    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('../model/model-accuracy.png')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('../model/model-loss.png')
    plt.show()

    print("Evaluate...")
    print(model.predict(x_test))
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)

    json_string = model.to_json()
    with open('../model/' + output_tag + '.json', 'w') as outfile:
        outfile.write(json_string)
    model.save('../model/' + output_tag + '.h5')
    plot_model(model, to_file='../model/model.png')
    print('Test score:', score)


if __name__ == '__main__':
    dataset_name = 'simplifyweibo_5_moods'
    # mood_list = ['anger', 'disgust', 'happiness', 'like', 'sadness', 'fear', 'surprise']
    mood_list = ['anger', 'disgust', 'happiness', 'like', 'sadness']
    print("开始清洗数据................")
    if not os.path.exists('../data/clean/'):
        os.makedirs('../data/clean/')
    clean_data('../data/' + dataset_name + '.txt', '../data/clean/' + dataset_name + '.txt')

    print("清洗数据完成................")
    print("开始下载数据................")
    X_Vec, y = loadfile('../data/clean/' + dataset_name + '.txt', mood_list)
    print("下载数据完成................")
    print("开始构建词向量................")
    input_dim, embedding_weights, w2dic = word2vec_train(X_Vec)
    print("构建词向量完成................")

    index = data2inx(w2dic, X_Vec)
    index2 = sequence.pad_sequences(index, maxlen=voc_dim)
    x_train, x_test, y_train, y_test = train_test_split(index2, y, test_size=0.2, shuffle=True)
    y_train = keras.utils.to_categorical(y_train, num_classes=len(mood_list))
    y_test = keras.utils.to_categorical(y_test, num_classes=len(mood_list))

    model = lstm(input_dim, embedding_weights, len(mood_list))
    train_lstm(model, x_train, y_train, x_test, y_test, dataset_name)

    h5_to_pb(dataset_name)
    generate_dict(dataset_name)
