from collections import OrderedDict
from typing import TypedDict, List, Dict
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from utils import read_json

class OpenDialKGExample4RL(TypedDict):
    History: str
    Utterance: str
    Keywords: str
    Gold_Node: str
    Gold_Knowledge: List[str]
    Response: str
    Root: List[str]
    Nodes: List[str]
    Adj_Matrix: Dict[str, List[int]]

class OpenDialKG_Compile():
    def __init__(self):
        self.path_render = read_json("OpendialKG/Preprocess/Intermediate/path_render.json")
        self.knowledgebase = read_json("OpendialKG/Preprocess/Intermediate/knowledge_base.json")
        self.valid_nodes = list(self.knowledgebase.keys())
        self.relation = read_json("OpendialKG/Preprocess/Intermediate/relation_count.json")
        self.max_n = 64

        # monitor
        self.fail_node = 0
        self.fail_node_path_2 = 0
        self.fail_knowledge = 0

    def monitor(self):
        print("Fail node:", self.fail_node)
        print("Fail node (path=2):", self.fail_node_path_2)
        print("Fail knowledge:", self.fail_knowledge)

    def node_sort(self, nodes, degrees):
        values = np.array(degrees)
        name = np.array(nodes)
        sorted_indices = np.argsort(values)
        name = name[sorted_indices[0:self.max_n]]
        return name.tolist()

    def khop_subgraph(self, g, k, nodes):
        num = 1+len(nodes[1])
        root = [nodes[0][0]]
        if num>1:
            k_root = dict(zip(nodes[1], [g.degree(r) for r in nodes[1]]))
            k_root = nodes[0] + self.node_sort(list(k_root.keys()), list(k_root.values()))
        else:
            k_root = [nodes[0][0]]
        candidate_nodes = k_root
        for hop in range(k):
            k_neighbors = []
            for n in k_root:
                neighbor = g.edges(n, keys=True)
                neighbor = self.node_sort([r[1] for r in neighbor], [self.relation[r[2]] for r in neighbor])
                if hop == k-1:
                    father_degree = g.degree(n)
                    neighbor = [t for t in neighbor if (g.degree(t) <= father_degree)]
                k_neighbors.extend(neighbor)
                if len(k_neighbors) > self.max_n:
                    k_neighbors = list(OrderedDict.fromkeys(k_neighbors).keys())
                    if len(k_neighbors) > self.max_n:
                        break
            k_root = self.node_sort(k_neighbors, [g.degree(r) for r in k_neighbors])
            candidate_nodes.extend(k_root[::-1])
            if len(candidate_nodes) > 1.2*self.max_n:
                break

        candidate_nodes = [n for n in candidate_nodes if n in self.valid_nodes]
        candidate_nodes = candidate_nodes[0:self.max_n]

        sub_graph = nx.Graph(nx.freeze(nx.subgraph(g, candidate_nodes)))
        isolated_nodes = list(set(list(nx.isolates(sub_graph))))
        if isolated_nodes:
            if root[0] in sub_graph.nodes:
                sub_graph.add_edges_from(zip(len(isolated_nodes)*root, isolated_nodes))
            else:
                sub_graph.remove_nodes_from(isolated_nodes)

        return sub_graph

    def get_gold(self, triple):
        if triple[0] in self.valid_nodes:
            if "|".join(triple) in list(self.path_render.keys()):
                gold_k = triple[0] + ": " + self.path_render["|".join(triple)]
            else:
                gold_k = triple[0] + ": " + ": ".join([triple[0], "`" + triple[1] + "`", triple[2]])
        else:
            return None
        if gold_k not in self.knowledgebase[triple[0]]:
            gold_k = None
        return gold_k


    def compile_one_dialog(self, ITEM: Dict, g, knowledge_k_hop, keyword_model):
        knowledge = ITEM["knowledge_base"]
        if not knowledge:
            return None
        path_list = knowledge['paths']
        for p in path_list:
            if "" in p:
                return None

        dialogue = ITEM['history']

        num_utterance = len(dialogue) + 1
        response = ITEM['response']

        speaker = ["user", "assistant"]
        last_speaker = ITEM['speaker']
        speaker.remove(last_speaker)
        if num_utterance%2 == 1:
            speaker = [last_speaker, *speaker]
        else:
            speaker = [*speaker, last_speaker]

        utterances = []
        for i in range(len(dialogue)):
            utterances.append(speaker[i%2] + ": " + dialogue[i])
        history = "|".join(utterances)
        utterance = utterances[-1]

        gold_knowledge = []
        gold_root = [path_list[0][0]]
        if gold_root[0] not in self.valid_nodes:
            return None

        for p in path_list:
            tmp = self.get_gold(p)

            if tmp is None:
                p[0], p[2] = p[2], p[0]
                if p[1][0] == "~":
                    p[1] = p[1][1:]
                else:
                    p[1] = "~"+p[1]
                tmp = self.get_gold(p)
            if tmp is None:
                tmp = "None"
                self.fail_knowledge += 1
            gold_knowledge.append(tmp)

        gold_node = p[0]    # 路径上最后一个节点为正确节点
        if gold_node not in self.valid_nodes:
            return None

        subgraph = self.khop_subgraph(g, knowledge_k_hop, (gold_root, []))
        nodes = list(subgraph.nodes)
        if len(nodes) == 0:
            return None
        if gold_node not in nodes:
            self.fail_node += 1
            if len(path_list)>1:
                self.fail_node_path_2 += 1

        adj_matrix = nx.adjacency_matrix(subgraph).todense()
        diag = np.identity(adj_matrix.shape[0], dtype=int)
        adj_matrix += diag
        adj_matrix[adj_matrix==2]=1
        sparse_matrix = csr_matrix(adj_matrix)

        # 创建一个字典，将稀疏矩阵存储为值
        sparse_dict = {"indptr": sparse_matrix.indptr.tolist(),
                       "indices": sparse_matrix.indices.tolist()}

        keywords = "|".join([i[0] for i in keyword_model.extract_keywords(" ".join(dialogue), keyphrase_ngram_range=(1, 2))])

        return OpenDialKGExample4RL(History=history,
                                    Utterance=utterance,
                                    Keywords=keywords,
                                    Gold_Node=gold_node,
                                    Gold_Knowledge=gold_knowledge,
                                    Response=response,
                                    Root=gold_root,
                                    Nodes=nodes,
                                    Adj_Matrix=sparse_dict)