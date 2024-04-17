import csv
import os
import pickle
from sklearn import metrics
from util import multirank, load_user,multirank_cuda
import numpy as np
import networkx as nx
from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning, EncodingTree
from torch_scatter import scatter_sum
import torch
import time
import matplotlib.pyplot as plt
import math


def load_tweet(filename, account):
    print("is loading tweet from {}...".format(filename))
    tweet_file = open(filename, 'r', encoding='utf-8')
    datas = csv.reader(tweet_file)
    n = 0
    num = len(account)
    types = np.zeros((num, 3))  # original,retweet,comment
    inf = np.zeros(num)
    for data in datas:
        n = n + 1
        if n == 1:
            continue
        else:
            t = 0
            while t < 6 and t < len(data):
                if data[t].isdigit() and (data[t] in account):
                    break
                else:
                    t += 1
            if t == 6 or t == len(data):
                continue
            x = account.index(data[t])
            if data[2].startswith('RT @'):
                types[x][1] += 1
            elif data[2].startswith('@'):
                types[x][2] += 1
            else:
                types[x][0] += 1
                if data[-7].isdigit():
                    inf[x] += int(data[-7])
                if data[-6].isdigit():
                    inf[x] += int(data[-6])
                if data[-5].isdigit():
                    inf[x] += int(data[-5])
    print('')
    for i in range(num):
        if types[i][0] != 0:
            inf[i] = inf[i] / types[i][0]
        sum_t = sum(types[i])
        if sum_t != 0:
            types[i] = types[i] / sum_t
    return types, inf


def load_tweet1(filename, account):
    print("is loading tweet from {}...".format(filename))
    tweet_file = open(filename, 'r', encoding='utf-8')
    datas = csv.reader(tweet_file)
    n = 0
    num = len(account)
    types = np.zeros((num, 3))
    inf = np.zeros(num)
    for data in datas:
        n = n + 1
        if n == 1:
            continue
        else:
            t = 0 
            while t < 5 and t < len(data):
                if data[t].isdigit() and (data[t] in account):
                    break
                else:
                    t += 1
            if t == 5 or t == len(data):
                continue
            x = account.index(data[t])
            if data[1].startswith('RT @'):
                types[x][1] += 1
            elif data[1].startswith('@'):
                types[x][2] += 1
            else:
                types[x][0] += 1
                if data[-4].isdigit():
                    inf[x] += int(data[-4])
                if data[-3].isdigit():
                    inf[x] += int(data[-3])
                if data[-2].isdigit():
                    inf[x] += int(data[-2])
    print(n)
    for i in range(num):
        if types[i][0] != 0:
            inf[i] = inf[i] / types[i][0]
        sum_t = sum(types[i])
        if sum_t != 0:
            types[i] = types[i] / sum_t
    return types, inf


# 导入tweet并处理(social_bot)
def load_tweet2(filename, account):
    print("is loading tweet from {}...".format(filename))
    tweet_file = open(filename, 'r', encoding='utf-8')
    datas = csv.reader(tweet_file)
    n = 0
    num = len(account)
    types = np.zeros((num, 3))  # original,retweet,comment
    inf = np.zeros(num)
    for data in datas:
        n = n + 1
        if n == 1:
            continue
        else:
            t = 0 
            while t < 7 and t < len(data):
                if data[t].isdigit() and (data[t] in account):
                    break
                else:
                    t += 1
            if t == 7 or t == len(data):
                continue
            x = account.index(data[t])
            if data[1].startswith('RT @'):
                types[x][1] += 1
            elif data[1].startswith('@'):
                types[x][2] += 1
            else:
                types[x][0] += 1
                if data[- 11].isdigit():
                    inf[x] += int(data[- 11])
                if data[- 12].isdigit():
                    inf[x] += int(data[- 12])
                if data[- 13].isdigit():
                    inf[x] += int(data[- 13]) 
    print(n)
    for i in range(num):
        if types[i][0] != 0:
            inf[i] = inf[i] / types[i][0]
        sum_t = sum(types[i]) 
        if sum_t != 0:
            types[i] = types[i] / sum_t
    return types, inf


def process(dataset, filename, b, ft=0):
    path = dataset + '/' + filename + '_f.csv'
    if os.path.exists(path):
        id = []
        label = []
        ff = []
        types = []  # original,retweet,comment
        infs = []
        line = 0
        file0 = open(path, 'r', encoding='utf-8')
        datas = csv.reader(file0)
        for data in datas:
            line += 1
            if line == 1:
                continue
            else:
                id.append(data[0])
                label.append(int(float(data[1])))
                infs.append(float(data[3]))
                ff.append(float(data[2]))
                types.append([float(data[4]), float(data[5]), float(data[6])])
            num = line - 1
        print('{}:{}'.format(filename, num))
        return num, id, label, ff, types, infs
    else:
        file1 = dataset + '/' + filename + '/users.csv'
        user, real_label, ff = load_user(file1, b)
        num1 = len(user)
        file2 = dataset + '/' + filename + '/tweets1.csv'
        if ft == 1:
            types, infs = load_tweet1(file2, user)
        elif ft == 2:
            types, infs = load_tweet2(file2, user)
        else:
            types, infs = load_tweet(file2, user)
        print('{}:{}'.format(filename, num1))
        with open(path, "w+", newline='') as csv_file:
            writer = csv.writer(csv_file)
            header = ["userid", "label", "friends/follower", "inf", "type1", "type2", "type3", "times"]
            writer.writerow(header)
            for i in range(num1):
                datarow = [user[i], real_label[i], ff[i], infs[i]]
                datarow.extend(types[i])
                writer.writerow(datarow)  # 写入csv
        return num1, user, real_label, ff, types, infs

def test():
    num1, id1, label1, ff1, types1, infs1 = process('cresci-2017', 'human', 0)
    num2, id2, label2, ff2, types2, infs2 = process('cresci-2017', 'bot', 1)

    num = num1 + num2
    id_block = np.array(id1+id2)
    label_block = np.array(label1+label2)
    ff_block = np.array(ff1+ff2)
    types_block = np.array(types1+types2)
    infs_block = np.array(infs1+infs2)
    print('num:{},bot:{}'.format(num, np.sum(label_block)))

    graph = np.zeros((3, num, num))
    edge = np.zeros(3)

    diff_ff = abs(ff_block.reshape(-1, 1) - ff_block.reshape(1, -1))
    max_ff = np.maximum(ff_block.reshape(-1, 1), ff_block.reshape(1, -1))
    min_ff = np.minimum(ff_block.reshape(-1, 1), ff_block.reshape(1, -1))
    mask_ff = (diff_ff < (max_ff * 0.1)) & (min_ff >= 3)
    graph[0][mask_ff] = 1
    graph[0] -= np.diag(np.diag(graph[0]))
    #print('U-F-U:{}'.format(np.count_nonzero(graph[0])/2))
    #max_ff[max_ff==0]=1
    #Simi_F=1-diff_ff/max_ff
    #print('Simi-F:{}'.format(np.mean(Simi_F)))

    diff_types = abs(types_block[:, 0].reshape(-1, 1) - types_block[:, 0].reshape(1, -1)) + \
                abs(types_block[:, 1].reshape(-1, 1) - types_block[:, 1].reshape(1, -1)) + \
                abs(types_block[:, 2].reshape(-1, 1) - types_block[:, 2].reshape(1, -1))
    mask_types = diff_types < 0.1
    graph[1][mask_types] = 1
    graph[1] -= np.diag(np.diag(graph[1]))
    #print('U-T-U:{}'.format(np.count_nonzero(graph[1])/2))
    #Simi_T=1-diff_types
    #print('Simi-T:{}'.format(np.mean(Simi_T)))

    diff_infs = abs(infs_block.reshape(-1, 1) - infs_block.reshape(1, -1))
    max_infs = np.maximum(infs_block.reshape(-1, 1),  infs_block.reshape(1, -1))
    min_infs = np.minimum(infs_block.reshape(-1, 1),  infs_block.reshape(1, -1))
    #mask_infs = (diff_infs < (max_infs * 0.1)) & (min_infs >= 3)
    mask_infs = (diff_infs < (max_infs * 0.1))
    graph[2][mask_infs] = 1
    graph[2] -= np.diag(np.diag(graph[2]))
    #print('U-I-U:{}'.format(np.count_nonzero(graph[2])/2))
    #max_infs[max_infs==0]=1
    #Simi_I=1-diff_infs/max_infs
    #print('Simi-I:{}'.format(np.mean(Simi_I)))

    max_ff[~mask_ff] = 1
    max_infs[~mask_infs] = 1
    adj_matrix = np.zeros((num, num))
    adj_matrix += (1 - (diff_ff / max_ff)) * mask_ff + (1 - diff_types) * mask_types + (1 - (diff_infs / max_infs)) * mask_infs
    #adj_matrix += (1-(diff_ff / max_ff))*mask_ff + (1-diff_types)*mask_types #FT 0,1
    #adj_matrix += (1-(diff_ff / max_ff))*mask_ff + (1-(diff_infs / max_infs))*mask_infs #FI 0,2
    #adj_matrix += (1-diff_types)*mask_types + (1-(diff_infs / max_infs))*mask_infs #TI 1,2

    adj_matrix -= np.diag(np.diag(adj_matrix))
    G = nx.Graph(adj_matrix)
    index=np.where(np.sum(adj_matrix,axis=1)==0)
    for i in index:
        adj_matrix[i,i]=0.01

    x, y = multirank_cuda(graph)
    #x, y = multirank_cuda(graph[[0,1]])

    edges = np.array(adj_matrix.nonzero())  # [2, E]
    ew = adj_matrix[edges[0, :], edges[1, :]]
    devices = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ew, edges = torch.tensor(ew, device=devices), torch.tensor(edges, device=devices).t()
    #ew, edges = torch.tensor(ew), torch.tensor(edges).t()
    dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])
    dist = dist / (2 * ew.sum())
    print('construct encoding tree...')
    g = GraphSparse(edges, ew, dist)
    optim = OperatorPropagation(Partitioning(g, None))
    optim.perform(p=0.15)
    print('construct encoding tree done')
    division = optim.enc.node_id
    SE2d = optim.enc.structural_entropy(reduction='sum', norm=True)
    module_se = optim.enc.structural_entropy(reduction='module', norm=True)
    totol_comm = torch.max(division) + 1
    print('totol_comm:{}'.format(totol_comm))
    comms = {}
    for i in range(totol_comm):
        idx = division == i
        if idx.any():
            comms[i] = idx.nonzero().squeeze(1)

    bot_rate = []
    pre_bot = np.zeros(num)
    value = np.zeros(num)
    num_bot_comm=0
    bot_list=[]
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
        #n_beta = (n_x / n_nodes) / (1 / num) * 0.9 + comm_SE / (sum(module_se) / num * n_nodes) * 0.1
        n_beta = (n_x / n_nodes) / (1 / num) * 0.6 + comm_SE / (sum(module_se) / totol_comm) * 0.4
        if n_beta >= 1:
            num_bot_comm += 1
            for node in comm:
                pre_bot[node] = 1
                value[node] = n_beta
        else:
            for node in comm:
                bot_list.append(node)
                value[node] = n_beta
        bot_rate.append([n_bots / n_nodes, n_bots, n_nodes, n_x / n_nodes, comm_SE, n_beta])

    acc = metrics.accuracy_score(label_block, pre_bot)
    precision = metrics.precision_score(label_block, pre_bot)
    recall = metrics.recall_score(label_block, pre_bot)
    f1 = metrics.f1_score(label_block, pre_bot)
    fpr, tpr, thresholds = metrics.roc_curve(label_block, value)
    auc_score = metrics.roc_auc_score(label_block, value)
    print('acc:{}'.format(acc))
    print('Precision:{}'.format(precision))
    print('Recall:{}'.format(recall))
    print('F1:{}'.format(f1))
    print('AUC:{}'.format(auc_score))

if __name__ == "__main__":
    test()


