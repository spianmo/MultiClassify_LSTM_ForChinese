from datetime import datetime

import TS_transform

if __name__ == '__main__':
    dataset = {}
    file_list = ["emotion_corpus_microblog.txt", "Nlpcc2014Train.txt", "simplifyweibo_8_moods.txt", "simplifyweibo_5_moods.txt"]
    mood_list = ['anger', 'disgust', 'happiness', 'like', 'sadness', 'fear', 'surprise', 'none']
    count = 0

    for mood in mood_list:
        dataset[mood] = []

    for file_name in file_list:
        with open(file_name, encoding='UTF-8') as f:
            for line in f.readlines():
                items = line.split(':', 1)
                sentence = TS_transform.T2S_NCP(items[1])
                if sentence == '':
                    # print("非法语句: " + items[1])
                    continue
                dataset[items[0]].append(items[1])
                count += 1
            f.close()

    count_dist = 0
    for key in dataset.keys():
        map_tmp = {}
        for dataset_item in dataset[key]:
            map_tmp[TS_transform.T2S_NCP(dataset_item)] = dataset_item
        dataset[key] = list(map_tmp.values())
        count_dist += len(dataset[key])

    fw1 = open('weibo82813_8_classify.txt', "w", encoding="UTF-8")
    fw1.write("#" * 100 + '\n')
    fw1.write("##  语料库名称: weibo82813_8_classify\n")
    fw1.write("##  语料规模: {} totals\n".format(count_dist))
    fw1.write("##  输出时间: {} \n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    fw1.write("##  描述: 八分类语料，['anger', 'disgust', 'happiness', 'like', 'sadness', 'fear', 'surprise', 'none']\n")
    fw1.write("#" * 100 + '\n')
    for mood in mood_list:
        fw1.write('##  {}: {:.4f}% {} items\n'.format(mood + ' ' * (10 - len(mood)), 100 * (len(dataset[mood]) / count_dist), len(dataset[mood])))
    fw1.write("#" * 100 + '\n')
    for key in dataset.keys():
        for dataset_item in dataset[key]:
            fw1.write("{},{}".format(key, dataset_item))
    fw1.close()

    fw2 = open('weibo82813_7_classify.txt', "w", encoding="UTF-8")
    fw2.write("#" * 100 + '\n')
    fw2.write("##  语料库名称: weibo82813_7_classify\n")
    fw2.write("##  语料规模: {} totals\n".format(count_dist - len(dataset['none'])))
    fw2.write("##  输出时间: {} \n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    fw2.write("##  描述: 七分类语料，['anger', 'disgust', 'happiness', 'like', 'sadness', 'fear', 'surprise']\n")
    fw2.write("#" * 100 + '\n')
    for mood in mood_list:
        if mood == 'none':
            continue
        fw2.write('##  {}: {:.4f}% {} items\n'.format(mood + ' ' * (10 - len(mood)), 100 * (len(dataset[mood]) / (count_dist - len(dataset['none']))), len(dataset[mood])))
    fw2.write("#" * 100 + '\n')
    for key in dataset.keys():
        if key == 'none':
            continue
        for dataset_item in dataset[key]:
            fw2.write("{},{}".format(key, dataset_item))
    fw2.close()

    print('done!')
