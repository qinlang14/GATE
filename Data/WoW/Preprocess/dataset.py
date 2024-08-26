from Data.WoW.Preprocess.single_sample import compile_one_dialog
import random
from tqdm import tqdm
from keybert import KeyBERT
from utils import read_json, write_json


class WoWDataset():
    def __init__(self):
        self.keyword_model = KeyBERT()
        self.raw = {"train": read_json("WoW/OriginalData/train.json"),
                    "valid_seen": read_json("WoW/OriginalData/valid_random_split.json"),
                    "valid_unseen": read_json("WoW/OriginalData/valid_topic_split.json"),
                    "test_seen": read_json("WoW/OriginalData/test_random_split.json"),
                    "test_unseen": read_json("WoW/OriginalData/test_topic_split.json")}

    def get_data(self):
        print("Building WoW Dataset.")
        for split, data in tqdm(self.raw.items()):
            dataset = []
            for example in data:
                example_temp_list = compile_one_dialog(example, self.keyword_model)
                dataset += example_temp_list
            if split == "train":
                random.seed(42)
                random.shuffle(dataset)
            write_json({"data": dataset}, f"WoW/Data_RL/Topic_split/{split}.json")