import networkx as nx
from Data.OpendialKG.Preprocess.opendialkg_utils import load_kg
from utils import read_json, write_json
from tqdm import tqdm
import random


def get_knowledge_base():
    print("Building OpenDialKG Knowledge Base.")
    g = load_kg("OpendialKG/OriginalData/opendialkg_triples.txt")
    knowledge_base = {}
    path_render = read_json("OpendialKG/Preprocess/Intermediate/path_render.json")

    G2 = nx.MultiDiGraph()
    for e in tqdm(g.edges):
        head, head_degree = e[0], g.degree(e[0])
        tail, tail_degree = e[1], g.degree(e[1])
        relation = e[2]
        if head_degree > tail_degree:
            continue
        if not G2.has_edge(head, tail, key=relation):
            G2.add_edge(head, tail, key=relation)

    for n in tqdm(G2.nodes):
        edges = G2.edges(n, keys=True)
        knowledge = []
        for e in edges:
            triple = "|".join([e[0], e[2], e[1]])
            if triple in list(path_render.keys()):
                render = e[0]+": "+path_render[triple]
            else:
                render = e[0]+": "+": ".join([e[0], "`"+e[2]+"`", e[1]])
            knowledge.append(render)

        if len(knowledge) > 0:
            knowledge_base[n] = knowledge

    write_json(knowledge_base, "OpendialKG/Preprocess/Intermediate/knowledge_base.json")