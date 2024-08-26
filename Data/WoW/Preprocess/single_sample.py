# https://github.com/qq1263632494/MyWizardOfWikipedia/blob/master/DataUtils.py
# 数据处理的相关工具
from typing import TypedDict, List, Dict
import networkx as nx
import numpy as np
import random
from scipy.sparse import csr_matrix


class WizardOfWikipediaExample4RL(TypedDict):
    History: str
    Utterance: str
    Keywords: str
    Gold_Node: str
    Gold_Knowledge: List[str]
    Response: str
    Root: str
    Nodes: List[str]
    Adj_Matrix: Dict[str, List[int]]


def matrix_adjust(node_sequence, adj_matrix):
    # 节点与行列的对应关系
    node_to_idx = {node: idx for idx, node in enumerate(node_sequence)}

    # 打乱节点序列
    random.seed(42)
    random.shuffle(node_sequence)

    # 更新邻接矩阵
    new_adj_matrix = np.zeros_like(adj_matrix)
    for i in range(len(node_sequence)):
        node1 = node_sequence[i]
        for j in range(len(node_sequence)):
            node2 = node_sequence[j]
            if adj_matrix[node_to_idx[node1], node_to_idx[node2]] == 1:
                # 如果原邻接矩阵中节点1和节点2之间有边，则在更新后的邻接矩阵中也要保留这条边
                new_adj_matrix[i][j] = 1

    return node_sequence, new_adj_matrix


def compile_one_dialog(ITEM: Dict, KeyWordModel) -> List[WizardOfWikipediaExample4RL]:
    """
        将一段对话转换为若干个WizardOfWikipediaExample对象
        e.g. data = read_json(数据路径)
             examples = compile_one_dialog(data[0])
        :param ITEM: 从json读取出的数据中的某一个
        :return: 为处理后的WizardOfWikipediaExample对象的列表
        :rtype List[WizardOfWikipediaExample]
    """
    TOPIC = ITEM['chosen_topic']
    DIALOGS = ITEM['dialog']

    HISTORY = ''
    UTTER = ''
    RAW = ''
    datas = []
    last2turn_retrieved = []
    for ind, DIALOG in enumerate(DIALOGS):
        if DIALOG['speaker'][2:] == 'Apprentice':
            UTTER = 'Apprentice: ' + DIALOG['text']
            HISTORY += 'Apprentice: ' + DIALOG['text'] + '|'
            RAW += DIALOG['text']
            last2turn_retrieved.append([list(i.keys())[0].replace("amp;", "") for i in DIALOG['retrieved_passages']])

        elif DIALOG['speaker'][2:] == 'Wizard':
            last2turn_retrieved.append([list(i.keys())[0].replace("amp;", "") for i in DIALOG['retrieved_passages']])

            if ind == 0 and DIALOG['speaker'] == '0_Wizard':
                HISTORY += 'Wizard: ' + DIALOG['text'] + '|'
                RAW += DIALOG['text']
                continue

            response = DIALOG['text']

            golden_sentence = DIALOG['checked_sentence']
            keys = list(golden_sentence.keys())

            if len(keys) != 0:
                golden_sentence = golden_sentence[keys[0]]
            # else:
            #     #针对checked_sentence为空，实际上却将正确文档index与句子放在checked_passage里面的情况 (弃用，无法找到正确句子)
            else:
                HISTORY += 'Wizard: ' + DIALOG['text'] + '|'
                RAW += DIALOG['text']
                continue

            golden_title = DIALOG['checked_passage']
            keys = list(golden_title.keys())
            if len(keys) != 0:
                golden_title = golden_title[keys[0]]
            else:
                # 数据集实际情况：未标注出正确篇章时，实际的正确篇章名位于正确句子的键中
                golden_title = " ".join(list(DIALOG['checked_sentence'].keys())[0].split("_")[1:-1])

            if golden_sentence == "no_passages_used" and golden_title!=golden_sentence:
                #数据集实际情况：使用了知识但未给出句子，且在checked_passage中有两个键值对
                HISTORY += 'Wizard: ' + DIALOG['text'] + '|'
                RAW += DIALOG['text']
                continue

            gold_knowledge = [golden_title + ": " + golden_sentence]

            knowledge_graph = nx.Graph()

            #根节点
            knowledge_graph.add_node(TOPIC)

            topics_tmp = sum(last2turn_retrieved[-1:-1-min(len(last2turn_retrieved),2):-1],[])
            retrieved_topics = [TOPIC] + topics_tmp + ["no_passages_used"]
            retrieved_topics = list(set(retrieved_topics))

            for rt in retrieved_topics:
                if rt not in knowledge_graph and rt != "No Passages Retrieved":
                    # 新建点并与根节点连接（拓扑节点）
                    knowledge_graph.add_edge(TOPIC, rt)

            #仅记录拓扑关系，方便存储
            nodes = list(knowledge_graph.nodes)
            adj_matrix = nx.adjacency_matrix(knowledge_graph).todense()
            diag = np.identity(adj_matrix.shape[0], dtype=int)
            adj_matrix += diag
            nodes, adj_matrix = matrix_adjust(nodes, adj_matrix)
            sparse_matrix = csr_matrix(adj_matrix)

            # 创建一个字典，将稀疏矩阵存储为值
            sparse_dict = {"indptr": sparse_matrix.indptr.tolist(),
                           "indices": sparse_matrix.indices.tolist()}

            keywords = "|".join([i[0] for i in KeyWordModel.extract_keywords(RAW, keyphrase_ngram_range=(1, 2))])

            datas.append(WizardOfWikipediaExample4RL(History=HISTORY[0:-1],
                                                     Utterance=UTTER,
                                                     Response=response,
                                                     Nodes=nodes,
                                                     Adj_Matrix=sparse_dict,
                                                     Gold_Node=golden_title,
                                                     Gold_Knowledge=gold_knowledge,
                                                     Keywords=keywords,
                                                     Root=TOPIC))
            HISTORY += 'Wizard: ' + DIALOG['text'] + '|'
            RAW += DIALOG['text']

    return datas
