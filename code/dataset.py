import re

import jieba
import numpy as np


# 数据清洗，去掉特殊符号（标点符号，数字，空格等）只保留汉字
def clean_data(rpath, wpath):
    # coding=utf-8
    pchinese = re.compile('([\u4e00-\u9fa5]+)+?')
    f = open(rpath, encoding="UTF-8")
    fw = open(wpath, "w", encoding="UTF-8")
    for line in f.readlines():
        m = pchinese.findall(str(line))
        if m:
            str1 = ''.join(m)
            str2 = str(str1)
            fw.write(str2)
            fw.write("\n")
    f.close()
    fw.close()


def loadfile(mood_list: list):
    mood_list_Vec = []
    for mood in mood_list:
        with open('../data/clean/' + mood + '_clean.txt', encoding='UTF-8') as f:
            mood_item = []
            for line in f.readlines():
                mood_item.append(list(jieba.cut(line, cut_all=False, HMM=True))[:-1])
            f.close()
            mood_list_Vec.append(mood_item)
    X_Vec = np.concatenate(mood_list_Vec)

    mood_list_Vec_deal = []
    count = 0
    for mood_Vec_item in mood_list_Vec:
        if count == 0:
            mood_list_Vec_deal.append(np.zeros(len(mood_Vec_item), dtype=int))
        else:
            mood_list_Vec_deal.append(count * np.ones(len(mood_Vec_item), dtype=int))
        count += 1
    y = np.concatenate(mood_list_Vec_deal)
    return X_Vec, y
