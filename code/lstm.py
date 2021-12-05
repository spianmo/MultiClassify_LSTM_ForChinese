from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential

lstm_input = 150  # lstm输入维度
voc_dim = 150  # word的向量维度


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
    model.add(Dense(dense_num))
    # Bi-LSTM
    model.add(Activation('softmax'))
    return model
