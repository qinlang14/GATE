import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import re
from utils import write_json


def get_render():
    print("Get the \'render\' of paths on OpenDialKG.")
    with open('OpendialKG/Preprocess/Intermediate/opendialkg.json', 'r', encoding='utf-8') as f:
        raw = [line.strip() for line in f.readlines()]
    raw = [json.loads(i) for i in raw]

    path_render = {}
    valid_relations = set()
    for example in tqdm(raw):
        example_key = list(example.keys())
        if "knowledge_base" not in example_key:
            continue

        knowledge = example["knowledge_base"]
        keys = list(knowledge.keys())
        if "paths" not in keys or "render" not in keys:
            continue

        checked_path = []
        path_list = knowledge['paths']

        for p in path_list:
            checked_path.append("|".join(p))
            if p[1][0] == "~":
                valid_relations.add(p[1][1:])
            else:
                valid_relations.add(p[1])

        render = knowledge["render"]
        if len(checked_path) > 1:
            # To resolve issues like 'He also directed Batman R.I.P.. Have you seen that one?'
            render = re.sub(r"(\w+\.)\.\s", r"\1 . ", render)
            render = re.sub(r"(\.\w\.)\.\s", r"\1 . ", render)
            # To resolve issues like 'I like Neil Brown Jr..' and 'I think he plays for Real Madrid C.F..'
            if re.match(r".*\w+\.\.$", render) or re.match(r".*\.\w\.\.$", render):
                render = render[:-1] + " ."

            render_split = render.split(".")
            if len(render_split) != len(checked_path):
                render_split = sent_tokenize(render)
                if len(render_split) != len(checked_path):
                    continue
            render = render_split
        else:
            render = [render]

        for i in range(len(render)):
            if ": `" in render[i]:
                tmp = render[i].split(":")
                render[i] = ": ".join(t.strip() for t in tmp)

        for i in range(len(checked_path)):
            if checked_path[i] not in list(path_render.keys()):
                path_render[checked_path[i]] = render[i]

    write_json(path_render, "OpendialKG/Preprocess/Intermediate/path_render.json")