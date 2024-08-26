import networkx as nx
import random


def load_kg(freebase_file: str) -> nx.Graph:
    G = nx.MultiDiGraph()

    with open(freebase_file, "r", encoding='utf-8') as f:
        for line in f.readlines():
            if len(line.strip().split("\t")) < 3:
                continue
            head, edge, tail = line.strip().split("\t")

            if not G.has_edge(head, tail, key=edge):
                G.add_edge(head, tail, key=edge)

    return G


def topic_split(data):
    root = {}
    for sample in data:
        start_e = sample["Root"]
        for e in start_e:
            if e not in root.keys():
                root[e] = 1
            else:
                root[e] += 1

    total_sum = sum(root.values())
    target_sum = total_sum * 0.85  # 目标和为总和的15%
    current_sum1, current_sum2 = 0, 0
    seen = {}
    unseen = {}

    # 按值从大到小排序字典
    sorted_items = sorted(root.items(), key=lambda x: x[1], reverse=True)

    # 将键值对分配到两个组中
    for key, value in sorted_items:
        # 计算当前两个组的总和与目标和的差值
        diff1 = abs(current_sum1 - target_sum) / target_sum
        diff2 = abs((total_sum - current_sum2) - target_sum) / (total_sum - target_sum)

        # 选择差值最小的组进行分配
        if diff1 > diff2 or current_sum2 + value > total_sum - target_sum:
            seen[key] = value
            current_sum1 += value
        else:
            unseen[key] = value
            current_sum2 += value

    random.seed(42)
    random.shuffle(data)

    seen_topics = list(seen.keys())
    seen_samples = []
    unseen_samples = []
    for sample in data:
        start_e = sample["Root"][0]
        if start_e in seen_topics:
            seen_samples.append(sample)
        else:
            unseen_samples.append(sample)

    return seen_samples, unseen_samples