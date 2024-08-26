import random, json
from tqdm import tqdm
from keybert import KeyBERT
from Data.OpendialKG.Preprocess.single_sample import OpenDialKG_Compile
from Data.OpendialKG.Preprocess.opendialkg_utils import load_kg, topic_split
from utils import data_split_and_save, read_dir_file_name, read_json


class OpendialkgDataset():
    def __init__(self):

        self.keyword_model = KeyBERT()
        self.g = load_kg("OpendialKG/OriginalData/opendialkg_triples.txt")
        with open('OpendialKG/Preprocess/Intermediate/opendialkg.json', 'r', encoding='utf-8') as f:
            raw = [line.strip() for line in f.readlines()]
        self.raw = [json.loads(i) for i in raw]
        self.compiler = OpenDialKG_Compile()

    def get_data(self, origin=True, k_hop=2):
        existed_data = read_dir_file_name("OpendialKG/Data_RL/Normal_split")

        if origin or len(existed_data) < 3:
            print("Building OpenDialKG Dataset (Normal Split).")
            dataset = []
            for example in tqdm(self.raw):
                example_temp = self.compiler.compile_one_dialog(example, self.g, k_hop, self.keyword_model)
                if example_temp is not None:
                    dataset.append(example_temp)

            self.compiler.monitor() # Fail node: 3857 (path=2: 2700)

            random.seed(42)
            random.shuffle(dataset)

            paths = [f"OpendialKG/Data_RL/Normal_split/{split}.json" for split in ["train", "valid", "test"]]
            data_split_and_save(dataset, splits=[0.7, 0.85], filepaths=paths)

        else:
            print("Building OpenDialKG Dataset (Topic Split).")
            data = []
            for split in ["train", "valid", "test"]:
                data += read_json(f"OpendialKG/Data_RL/Normal_split/{split}.json")["data"]
            seen_samples, unseen_samples = topic_split(data)

            # train:70%, seen(valid+test):15%
            paths = [f"OpendialKG/Data_RL/Topic_split/{split}.json" for split in ["train", "valid_seen", "test_seen"]]
            data_split_and_save(seen_samples, splits=[0.7/0.85, 0.775/0.85], filepaths=paths)

            # unseen(valid+test):15%
            paths = [f"OpendialKG/Data_RL/Topic_split/{split}.json" for split in ["valid_unseen", "test_unseen"]]
            data_split_and_save(unseen_samples, splits=[0.5], filepaths=paths)