from keras.layers import Bidirectional
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam

lstm_input = 150  # lstm输入维度
voc_dim = 150  # word的向量维度


def bi_lstm(input_dim, embedding_weights, dense_num):
    model = Sequential()
    model.add(Embedding(output_dim=voc_dim,
                        input_dim=input_dim,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=lstm_input,
                        trainable=False))
    model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
    model.add(LSTM(units=16, return_sequences=False))
    model.add(Dropout(0.5))
    # Dense=>全连接层,输出维度=情感分类的数量
    model.add(Dense(dense_num, activation='softmax'))
    print('Compiling the Model...')
    # 使用adam以0.001的learning rate进行优化
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['mae', 'acc'])
    model.summary()
    return model


def lstm(input_dim, embedding_weights, dense_num):
    model = Sequential()
    model.add(Embedding(output_dim=voc_dim,
                        input_dim=input_dim,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=lstm_input))
    model.add(LSTM(256, activation='softsign'))
    model.add(Dropout(0.5))
    # Dense=>全连接层,输出维度=情感分类的数量
    model.add(Dense(dense_num, activation='softmax'))
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',  # hinge
                  optimizer=Adam(lr=1e-3), metrics=['mae', 'acc'])
    model.summary()
    return model
