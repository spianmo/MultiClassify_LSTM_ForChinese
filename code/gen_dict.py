import os

from gensim.models.word2vec import Word2Vec

if __name__ == '__main__':
    model_word = Word2Vec.load('../model/Word2Vec.pkl')

    if os.path.exists("../model/sentiment_classification_5_dict.txt"):
        os.remove("../model/sentiment_classification_5_dict.txt")
    with open("../model/sentiment_classification_5_dict.txt", "w", encoding="utf-8") as f:
        for i in range(len(model_word.wv.key_to_index)):
            b = list(model_word.wv.key_to_index)[i] + ":" + str(i + 1)
            b += "\n"
            f.write(b)
