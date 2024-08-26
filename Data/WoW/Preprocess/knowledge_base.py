from tqdm import tqdm
from utils import read_json, write_json


def kb():
    print("Building WoW Knowledge Base.")
    raw = read_json("WoW/OriginalData/train.json") + \
          read_json("WoW/OriginalData/valid_random_split.json") + \
          read_json("WoW/OriginalData/valid_topic_split.json") + \
          read_json("WoW/OriginalData/test_random_split.json") + \
          read_json("WoW/OriginalData/test_topic_split.json")

    knowledge_base = {}
    def update(key, value):
        if key not in list(knowledge_base.keys()):
            knowledge_base[key] = value
        else:
            if len(value) > len(knowledge_base[key]):
                knowledge_base[key] = value

    for ITEM in tqdm(raw):
        TOPIC = ITEM['chosen_topic'].replace("amp;", "")
        DIALOGS = ITEM['dialog']
        TOPIC_KNOWLEDGES = ITEM['chosen_topic_passage']
        for i in range(len(TOPIC_KNOWLEDGES)):
            TOPIC_KNOWLEDGES[i] = TOPIC + ": " + TOPIC_KNOWLEDGES[i]
        update(TOPIC, TOPIC_KNOWLEDGES)

        for ind, DIALOG in enumerate(DIALOGS):
                retrieved_passages = DIALOG['retrieved_passages']
                for passage in retrieved_passages:
                    key = list(passage.keys())[0]
                    # 存储知识（资源节点）
                    knowledge = []
                    for k in passage[key]:
                        knowledge.append(key.replace("amp;", "") + ": " + k)
                    update(key.replace("amp;", ""), knowledge)

    knowledge_base["no_passages_used"] = ["no_passages_used: no_passages_used"]

    write_json(knowledge_base, "WoW/Preprocess/Intermediate/knowledge_base.json")
