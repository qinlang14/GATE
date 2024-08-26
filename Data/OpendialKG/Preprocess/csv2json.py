import csv
import json
from typing import Any, Dict, Iterable, Tuple
from tqdm import tqdm
import spacy


def _tokenize(sent: str) -> str:
    nlp = spacy.load("en_core_web_sm")
    return " ".join([tok.text for tok in nlp(sent)])


def read_csv(data_file: str) -> Iterable[Tuple[str, int]]:
    with open(data_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)  # skip header row
        dialog_id = 0
        for i, row in enumerate(reader):
            dialog_id += 1
            # 丢弃UserRating与AssistantRating
            dialogue, _, _ = row[0].strip(), row[1].strip(), row[2].strip()

            yield dialogue, dialog_id


def parse_message(dialogue: str, dialog_id: int) -> Iterable[Dict[str, Any]]:
    json_dialog = json.loads(dialogue)
    history = []
    metadata = {}

    for i, turn in enumerate(json_dialog):
        if i == 0:
            # 每一会话的首句均直接作为对话历史（的一部分）
            if "message" in turn:
                # history.append(_tokenize(turn["message"]))
                history.append(turn["message"])
        else:
            if "metadata" in turn:
                if "path" in turn["metadata"]:
                    # 数据集内metadata项与message项处于同一级目录下，代表下一回复应参考的知识
                    metadata = {
                        # 丢弃PathRating
                        "rating": turn["metadata"]["path"][0],
                        "paths": turn["metadata"]["path"][1],   # paths为知识（三元组）列表
                        "render": turn["metadata"]["path"][2],  # render为三元组的表现形式，例如：["Iron Man", "starred_actors", "Robert Downey Jr."] → "Iron Man is starring Robert Downey Jr.
                    }

            else:
                # response = _tokenize(turn["message"])
                response = turn["message"]
                yield {
                    "history": history,
                    "response": response,
                    "speaker": turn["sender"],
                    "knowledge_base": metadata,
                    "dialogue_id": dialog_id,
                }

                # metadata置空，因生成回复仅需要最近的metadata项
                metadata = {}
                # 对话历史保留，一轮对话可形成多个样本
                history.append(response)


def convert(data_file: str, out_file: str):
    print("Convert OpenDialKG from csv to json.")
    with open(out_file, "w", encoding='utf-8') as out:
        for dialogue, dialog_id in tqdm(read_csv(data_file)):
            for utterance in parse_message(dialogue, dialog_id):
                out.write(json.dumps(utterance, ensure_ascii=False) + "\n")
