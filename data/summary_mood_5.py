if __name__ == '__main__':
    data_set = {}
    ml = ['anger', 'disgust', 'happiness', 'like', 'sadness']
    for m in ml:
        data_set[m] = []
        with open(m + ".txt", encoding='UTF-8') as f:
            for line in f.readlines():
                data_set[m].append(line)
        f.close()

    fw = open("./dataset/simplifyweibo_5_moods.csv", 'w', encoding="UTF-8")
    for key in data_set.keys():
        for dataset_item in data_set[key]:
            fw.write("{},{}".format(key, dataset_item))
    fw.close()