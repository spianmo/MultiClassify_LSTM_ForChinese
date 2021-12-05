import re

import jieba
import numpy as np


# 数据清洗，去掉特殊符号（标点符号，数字，空格等）只保留汉字
from data.dataset.summary import TS_transform


def clean_data(rpath, wpath):
    # coding=utf-8
    pchinese = re.compile('([\u4e00-\u9fa5]+)+?')
    f = open(rpath, encoding="UTF-8")
    fw = open(wpath, "w", encoding="UTF-8")
    for line in f.readlines():
        if line.startswith('#'):
            continue
        m = pchinese.findall(str(line))
        if m:
            str1 = ''.join(m)
            str2 = str(str1)
            fw.write(line.split(':', 1)[0] + ':' + str2)
            fw.write("\n")
    f.close()
    fw.close()


def loadfile(path: str, mood_list: list):
    mood_list_Vec = []
    with open(path, encoding='UTF-8') as f:
        mood_item_map = {}
        for mood in mood_list:
            mood_item_map[mood] = []
        for line in f.readlines():
            items = line.split(':', 1)
            sentence = TS_transform.T2S_NCP(items[1])
            if sentence == '':
                continue
            mood_item_map[items[0]].append(list(jieba.cut(sentence, cut_all=False, HMM=True))[:-1])
        f.close()
        for mood in mood_list:
            mood_list_Vec.append(mood_item_map[mood])
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
