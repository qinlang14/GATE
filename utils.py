import json, pickle
import logging
from time import gmtime, strftime
import sys
import os
import json5
import numpy as np


def absolute_path(path):
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(PROJECT_ROOT, path)

    return path

def mkdir(path):
    """
    创建文件夹
    """
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)
        # print(path + ' 创建成功')
        return True
    else:
        # print(path + ' 目录已存在')
        return False


def read_dir_file_name(path, suffix='json'):
    """
    读取文件夹下的所有文件名，并返回特定后缀的文件名
    """
    files_names = os.listdir(path)
    new_file_names = []
    for file_name in files_names:
        if file_name.split('.')[-1] == suffix:
            new_file_names.append(file_name)

    return new_file_names


def read_numpy(path):
    """
    读取npy文件
    """
    path = absolute_path(path)
    data = np.load(path, allow_pickle=True)
    return data


def write_numpy(path, data):
    """
    读取npy文件
    """
    path = absolute_path(path)
    np.save(file=path, arr=data)
    print('已写入数据至文件{}，数据量：{}'.format(path, data.shape[0]))


def read_json(path):
    """
    读取json文件
    """
    path = absolute_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    """
    写入数据至json文件
    """
    path = absolute_path(path)
    with open(path, 'w', encoding='utf8') as f_write:
        json.dump(data, f_write, indent=2, ensure_ascii=False)

    print('File path: {}, data size: {}'.format(path, len(data)))


def read_txt(path):
    path = absolute_path(path)
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        file = f.readlines()
    for line in file:
        lines.append(line.strip('\n'))
    return lines


def write_txt(data, path):
    path = absolute_path(path)
    lines = []
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line)
            f.write('\n')
    return lines


def read_pickle(path):
    """
    写入数据至pickle文件
    data = {"input_ids": input_ids_all, "token_type_ids": token_type_ids_all, "input_masks": input_mask_all, "labels": label_all}
    """
    path = absolute_path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def write_pickle(data, path):
    """
    写入数据至pickle文件
    data = {"input_ids": input_ids_all, "token_type_ids": token_type_ids_all, "input_masks": input_mask_all, "labels": label_all}
    """
    path = absolute_path(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)

    print('已写入数据至文件{}'.format(path))


def create_logger(name, silent=False, to_disk=False, log_file=None):
    """Logger wrapper"""
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
    )
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = (
            log_file
            if log_file is not None
            else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        )
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log


class Config(object):
    """Config load from json file
    """

    def __init__(self, config=None, config_file=None):
        if config_file:
            with open(config_file, 'r') as fin:
                config = json5.load(fin)

        self.dict = config
        if config:
            self._update(config)

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, item):
        return item in self.dict

    def items(self):
        return self.dict.items()

    def add(self, key, value):
        """Add key value pair
        """
        self.__dict__[key] = value

    def _update(self, config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(config[key])

            if isinstance(config[key], list):
                config[key] = [Config(x) if isinstance(x, dict) else x for x in
                               config[key]]

        self.__dict__.update(config)


def flatten_list(nested_list):
    """
    将嵌套列表中的所有元素提取为一个新的平面列表
    :param nested_list: 可能包含嵌套列表的列表
    :return: 包含所有元素的平面列表
    """
    flattened_list = []

    for element in nested_list:
        if isinstance(element, list):
            # 如果元素是一个列表，则递归调用
            flattened_list.extend(flatten_list(element))
        else:
            # 如果元素不是列表，直接添加到平面列表中
            flattened_list.append(element)

    return flattened_list


def merge_dict(dict1, dict2):
    """
    合并两个字典，按照键加和值
    :param dict1: 第一个字典
    :param dict2: 第二个字典
    :return: 新的字典，键相同的值相加
    """
    # 创建一个新的字典，用于存储结果
    result = {}

    # 获取两个字典的所有键
    all_keys = set(dict1.keys()).union(dict2.keys())

    # 遍历所有键
    for key in all_keys:
        # 获取第一个字典中的值，默认为0
        value1 = dict1.get(key, 0)
        # 获取第二个字典中的值，默认为0
        value2 = dict2.get(key, 0)
        # 相同键的值相加
        result[key] = value1 + value2

    return result


def data_split_and_save(dataset, splits, filepaths):
    assert len(splits) == len(filepaths) - 1, "Splits and filenames do not match."

    previous_split = 0
    for split, filepath in zip(splits + [1.0], filepaths):
        current_split = int(split * len(dataset))
        subset = dataset[previous_split:current_split]
        write_json({"data": subset}, filepath)
        previous_split = current_split


def data_name_to_path(data_name):
    path_list = [f"Topic_split/{s}" for s in ["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"]]
    if data_name == "OpendialKG":
        path_list = [f"Normal_split/{s}" for s in ["train", "valid", "test"]] + path_list
    path_list = [f"Data/{data_name}/Data_RL/{s}.json" for s in path_list]

    embed_path_list = [f"Topic_split/Embedding/{s}" for s in ["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"]]
    if data_name == "OpendialKG":
        embed_path_list = [f"Normal_split/Embedding/{s}" for s in ["train", "valid", "test"]] + embed_path_list
    embed_path_list = [f"Data/{data_name}/Data_RL/{s}_embedding.pth" for s in embed_path_list]

    return zip(path_list, embed_path_list)

