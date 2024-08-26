import torch
from typing import List, Dict
from tqdm import tqdm
from utils import read_json, write_json, data_name_to_path, absolute_path
from sentence_transformers import SentenceTransformer


class DataProcessor():
    def __init__(self):
        self.sentence_encoder = SentenceTransformer('all-mpnet-base-v2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_sentence_representations(self, sentence):
        sentence_embedding = self.sentence_encoder.encode(sentence, device=self.device, convert_to_tensor=True)
        return sentence_embedding

    def get_knowledge_representations(self, knowledge: Dict):
        knowledge_embedding = {}
        for key in tqdm(list(knowledge.keys())):
            if knowledge[key]:
                embedding = self.get_sentence_representations(knowledge[key])
                pooling = torch.nn.functional.avg_pool1d(input=embedding.t(), kernel_size=embedding.shape[0])
                knowledge_embedding[key] = {"avg_pool": pooling.squeeze(), "embedding": embedding}
            else:
                knowledge_embedding[key] = None

        return knowledge_embedding

    def get_dialogue_representations(self, dialogue: List[Dict]):
        dialogue_embedding = []
        for example in tqdm(dialogue):
            inp = [example["History"], example["Utterance"], example["Keywords"]]
            out = self.get_sentence_representations(inp)
            dialogue_embedding.append(out)

        return dialogue_embedding

    def embedding(self, data_name):
        print(f"Static embedding for {data_name}.")
        data_path = data_name_to_path(data_name)

        prefix = f"Data/{data_name}/Preprocess/Intermediate/"
        knowledge = read_json(prefix + "knowledge_base.json")
        # knowledge = {"A": ["i love you", "i love you too"]}
        knowledge_embedding = self.get_knowledge_representations(knowledge)
        torch.save(knowledge_embedding, absolute_path(prefix+"knowledge_embedding.pth"))

        for json_path, embed_path in data_path:
            dialogue = read_json(json_path)["data"]
            dialogue_embedding = self.get_dialogue_representations(dialogue)
            torch.save(dialogue_embedding, absolute_path(embed_path))

    def padding(self, inp, pad=0):
        new = torch.nn.utils.rnn.pad_sequence(inp, batch_first=True, padding_value=pad).to(self.device)
        mask_matrix = torch.where(torch.sum(new,dim=-1)==0, 1, 0).to(self.device)

        return new, mask_matrix

    def smooth_labels(self, input, smoothing_rate=0.20):
        max_values, _ = torch.max(input, dim=0, keepdim=True)
        smooth_values = max_values * smoothing_rate
        smooth_labels = input * (1 - smoothing_rate)
        smooth_labels = smooth_labels + (smooth_values / (input.shape[0] - 1))
        normalized_labels = smooth_labels / smooth_labels.sum(dim=0, keepdim=True)

        if True in torch.isnan(normalized_labels) or True in torch.isinf(normalized_labels):
            print("Nan in labels!")

        return normalized_labels

    def add_label(self, data_name):
        knowledge_embedding = torch.load(f"{data_name}/Preprocess/Intermediate/knowledge_embedding.pth",
                                         map_location=torch.device('cpu'))
        print(f"Add nodes label for {data_name}.")
        data_path = data_name_to_path(data_name)

        for json_path, embed_path in data_path:
            dialogue = read_json(json_path)["data"]
            response = [i["Response"] for i in dialogue]
            response_embedding = self.get_sentence_representations(response)

            nodes = [i["Nodes"] for i in dialogue]
            fail = 0
            for idx, n in enumerate(tqdm(nodes)):
                nodes_embedding = torch.stack([knowledge_embedding[i]["avg_pool"] for i in n]).to(self.device)
                tmp = torch.sum(nodes_embedding * response_embedding[idx] / torch.sqrt(torch.tensor(768)), dim=-1).squeeze()
                label = dialogue[idx]["Gold_Node"]
                score = torch.softmax(tmp, dim=-1)
                if label in n:
                    score[n.index(label)] = len(score)/2
                else:
                    fail+=1
                score = self.smooth_labels(score)
                dialogue[idx]["Node_Label"] = torch.tensor(score, dtype=torch.float16).tolist()

            print("Fail samples of Node Label(%):", fail/len(dialogue))
            write_json({"data": dialogue}, json_path)