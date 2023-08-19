from util import multirank_cuda, process_tweet
import csv
import os
from sklearn import metrics
import numpy as np
import pickle
from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning, EncodingTree
from torch_scatter import scatter_sum
import torch
import networkx as nx


def process(filename):
    path = filename + '/' + filename + '_f.csv'
    if not os.path.exists(path):
        process_tweet(filename)
    fp = open(path, 'r', encoding='utf-8')
    datas = csv.reader(fp)
    id = []
    label = []
    ff = []
    types = []  # original,retweet,comment
    infs = []  # 原创推文的评论点赞转发数总
    line = 0
    for data in datas:
        line += 1
        if line == 1:
            continue
        id.append(data[0])
        label.append(int(float(data[1])))
        infs.append(float(data[2]))
        ff.append(float(data[3]))
        types.append([float(data[4]), float(data[5]), float(data[6])])
    num = line - 1
    print('{}:{}'.format(filename, num))
    return num, id, label, ff, types, infs


def test(filename):
    num, id, label, ff, types, infs = process(filename)
    id_block = np.array(id)
    label_block = np.array(label)
    ff_block = np.array(ff)
    types_block = np.array(types)
    infs_block = np.array(infs)
    print('num:{},bot:{}'.format(num, np.sum(label_block)))

    # 构图
    graph = np.zeros((3, num, num))
    edge = np.zeros(3)

    diff_ff = abs(ff_block.reshape(-1, 1) - ff_block.reshape(1, -1))
    max_ff = np.maximum(ff_block.reshape(-1, 1), ff_block.reshape(1, -1))
    min_ff = np.minimum(ff_block.reshape(-1, 1), ff_block.reshape(1, -1))
    mask_ff = (diff_ff < (max_ff * 0.1)) & (min_ff >= 3)
    graph[0][mask_ff] = 1
    graph[0] -= np.diag(np.diag(graph[0]))
    diff_types = abs(types_block[:, 0].reshape(-1, 1) - types_block[:, 0].reshape(1, -1)) + \
                 abs(types_block[:, 1].reshape(-1, 1) - types_block[:, 1].reshape(1, -1)) + \
                 abs(types_block[:, 2].reshape(-1, 1) - types_block[:, 2].reshape(1, -1))
    mask_types = diff_types < 0.1
    graph[1][mask_types] = 1
    graph[1] -= np.diag(np.diag(graph[1]))
    diff_infs = abs(infs_block.reshape(-1, 1) - infs_block.reshape(1, -1))
    max_infs = np.maximum(infs_block.reshape(-1, 1), infs_block.reshape(1, -1))
    mask_infs = diff_infs < (max_infs * 0.1)
    graph[2][mask_infs] = 1
    graph[2] -= np.diag(np.diag(graph[2]))

    max_ff[~mask_ff] = 1
    max_infs[~mask_infs] = 1
    adj_matrix = np.zeros((num, num))
    adj_matrix += (1 - (diff_ff / max_ff)) * mask_ff + (1 - diff_types) * mask_types + (
            1 - (diff_infs / max_infs)) * mask_infs
    # adj_matrix += (diff_ff / max_ff)*mask_ff + diff_types*mask_types + (diff_infs / max_infs)*mask_infs
    adj_matrix -= np.diag(np.diag(adj_matrix))
    G = nx.Graph(adj_matrix)


    index = np.where(np.sum(adj_matrix, axis=1) == 0)
    for i in index:
        adj_matrix[i, i] = 0.01

    # 计算mulrirank
    x, y = multirank_cuda(graph)

    # 编码树
    edges = np.array(adj_matrix.nonzero())  # [2, E]
    ew = adj_matrix[edges[0, :], edges[1, :]]
    ew, edges = torch.Tensor(ew), torch.tensor(edges).t()
    dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])  # dist/2=di
    dist = dist / (2 * ew.sum())  # ew.sum()=vol(G) dist=di/vol(G)

    g = GraphSparse(edges, ew, dist)
    optim = OperatorPropagation(Partitioning(g, None))
    optim.perform(p=0.15)
    division = optim.enc.node_id
    SE2d = optim.enc.structural_entropy(reduction='sum', norm=True)
    module_se = optim.enc.structural_entropy(reduction='module', norm=True)
    totol_comm = torch.max(division) + 1
    comms = {}
    for i in range(totol_comm):
        idx = division == i
        if idx.any():
            comms[i] = idx.nonzero().squeeze(1)

    bot_rate = []
    pre_bot = np.zeros(num)

    # 社区划分
    num_bot_comm = 0
    for i in comms.keys():
        comm = comms[i]
        n_bots = 0
        n_nodes = 0
        n_x = 0
        for node in comm:
            n_bots += label_block[node]
            n_nodes += 1
            n_x += x[node]
        comm_SE = module_se[i]
        # n_beta = (n_x / n_nodes) / (1 / num) * 0.9 + comm_SE / (SE2d / num * n_nodes) * 0.1  # TWT:0.5 0.5; others:0.9 0.1
        # n_beta = (n_x / n_nodes) / (1 / num) * 0.9 + comm_SE / (sum(module_se) / num * n_nodes) * 0.1
        n_beta = (n_x / n_nodes) / (1 / num) * 0.9 + comm_SE / (sum(module_se) / totol_comm) * 0.1
        if n_beta >= 1:
            num_bot_comm += 1
            for node in comm:
                pre_bot[node] = 1
        bot_rate.append([n_bots / n_nodes, n_bots, n_nodes, n_x / n_nodes, comm_SE, n_beta])

    # 准确率计算
    acc = metrics.accuracy_score(label_block, pre_bot)
    precision = metrics.precision_score(label_block, pre_bot)
    recall = metrics.recall_score(label_block, pre_bot)
    f1 = metrics.f1_score(label_block, pre_bot)
    print('acc:{}'.format(acc))
    print('Precision:{}'.format(precision))
    print('Recall:{}'.format(recall))
    print('F1:{}'.format(f1))

    for b_data in bot_rate:
        if b_data[5] >= 1:
            print(b_data)
    print('')
    for b_data in bot_rate:
        if b_data[5] < 1:
            print(b_data)

if __name__ == "__main__":
    test('botwiki-2019')
