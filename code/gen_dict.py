import os

from gensim.models.word2vec import Word2Vec


def generate_dict(name: str):
    model_word = Word2Vec.load('../model/Word2Vec.pkl')

    if os.path.exists("../model/" + name + "_dict.txt"):
        os.remove("../model/" + name + "_dict.txt")
    with open("../model/" + name + "_dict.txt", "w", encoding="utf-8") as f:
        for i in range(len(model_word.wv.key_to_index)):
            b = list(model_word.wv.key_to_index)[i] + ":" + str(i + 1)
            b += "\n"
            f.write(b)


if __name__ == '__main__':
    generate_dict('sentiment_classification_5_dict')
