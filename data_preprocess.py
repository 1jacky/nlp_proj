# -*- coding: utf-8 -*-

"""
数据预处理
"""

import json


def load_json(data_path):
    with open(data_path, encoding="utf-8") as f:
        return json.loads(f.read())


def dump_json(project, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False)


def preprocess(train_data_path, label2idx_path, max_len_ratio=0.9):
    """
    :param train_data_path:
    :param label2idx_path:
    :param max_len_ratio:
    :return:
    """
    labels = []
    text_length = []
    with open(train_data_path, encoding="utf-8") as f:
        datas = json.load(f)
        for data in datas:
            # data = json.load(data)
            text_length.append(len(data["text"]))
            labels.extend(data["label"])
    labels = list(set(labels))
    label2idx = {label: idx for idx, label in enumerate(labels)}

    dump_json(label2idx, label2idx_path)

    text_length.sort()

    print("当设置max_len={}时，可覆盖{}的文本".format(text_length[int(len(text_length)*max_len_ratio)], max_len_ratio))

#写入json文件
def write_json(file_path,param):
    with open(file_path,'w',encoding='utf8') as f:
        json.dump(param,f,ensure_ascii=False)


"""
本方法主要实现指定原始json文件的根据id值的大小分别写入
train_json 文件和dev_json文件"""
def make_dataset(src_json_data_path,train_json_data_path,dev_json_data_path):
    data_train=[]
    data_dev = []
    with open(src_json_data_path,encoding='utf8') as f:
        result = json.load(f)
        for i in result:
            if i.get('id')>142:
                data_train.append(i)
            else:
                data_dev.append(i)
    write_json(train_json_data_path,data_train)
    write_json(dev_json_data_path,data_dev)


src_data ='./data/admin.json'
train_json_data_path='./data/train.json'
dev_json_data_path='./data/dev.json'

if __name__ == '__main__':
    # pass
    #把原始数据分成训练集和验证集
    # make_dataset(src_data,train_json_data_path,dev_json_data_path)
    preprocess("./data/train.json", "./data/label2idx.json")
    # data_train=[]
    # data_dev = []
    # with open(src_data,encoding='utf-8') as f:
    #     result = json.load(f)
    #     for i in result:
    #         if i.get('id')>142:
    #             data_train.append(i)
    #         else:
    #             data_dev.append(i)


    #         data_ids.append(i.get('id'))
    #
    # list_id = sorted(data_ids,reverse=True)
    # print(list_id)
    # print(len(list_id))
    # print(len(list_id)*0.7)
    # print('选择70%作为训练集30%作为测试集') #大于142为训练集
