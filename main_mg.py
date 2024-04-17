from util import multirank, process_tweet,multirank_cuda
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
import time
import matplotlib.pyplot as plt
import math
import networkx as nx


def process_bot(filename):
    path = filename + '/' + filename + '_mg.csv'
    if not os.path.exists(path):
        process_tweet(filename)
    fp = open(path, 'r', encoding='utf-8')
    datas = csv.reader(fp)
    id = []
    label = []
    tweets = []
    friends = [] 
    favourites = []
    line = 0
    for data in datas:
        line += 1
        if line == 1:
            continue
        id.append(data[0])
        label.append(1)
        tweets.append(int(data[2]))
        friends.append(int(data[3]))
        favourites.append(int(data[4]))
    num = line - 1
    print('{}:{}'.format(filename, num))
    return num, id, label, tweets, friends, favourites

def process_human(dataset, filename):
    path = dataset + '/' + filename + '/users1.csv'
    if os.path.exists(path):
        id = []
        label = []
        tweets = []
        friends = []
        favourites = []
        line = 0
        file0 = open(path, 'r', encoding='utf-8')
        datas = csv.reader(file0)
        for data in datas:
            line += 1
            if line == 1:
                continue
            id.append(data[0])
            label.append(0)
            tweets.append(int(data[3]))
            friends.append(int(data[5]))
            favourites.append(float(data[6]))
        num = line - 1
        print('{}:{}'.format(filename, num))
        return num, id, label, tweets, friends, favourites


def test(filename):
    num0, id0, label0, tweets0, friends0, favourites0 = process_bot(filename)
    num1, id1, label1, tweets1, friends1, favourites1 = process_human('cresci-2015', 'E13')  #botwiki:TFP,pronbots:E13
    # 合并人类账户和机器账户
    num = num0 + num1
    id_block = np.array(id0+id1)
    label_block = np.array(label0+label1)
    twe_block = np.array(tweets0+tweets1)
    fri_block = np.array(friends0+friends1)
    favo_block = np.array(favourites0+favourites1)
    print('num:{},bot:{}'.format(num,np.sum(label_block)))

    # 构图
    graph = np.zeros((3, num, num))
    edge = np.zeros(3)
    #tweets
    diff_twe = abs(twe_block.reshape(-1, 1) - twe_block.reshape(1, -1))
    max_twe = np.maximum(twe_block.reshape(-1, 1), twe_block.reshape(1, -1))
    min_twe = np.minimum(twe_block.reshape(-1, 1),  twe_block.reshape(1, -1))
    mask_twe = (diff_twe < (max_twe * 0.01))
    graph[0][mask_twe] = 1
    graph[0] -= np.diag(np.diag(graph[0]))
    print('U-twe-U:{}'.format(np.count_nonzero(graph[0])/2))
    #friends
    diff_fri = abs(fri_block.reshape(-1, 1) - fri_block.reshape(1, -1))
    max_fri = np.maximum(fri_block.reshape(-1, 1), fri_block.reshape(1, -1))
    min_fri = np.minimum(fri_block.reshape(-1, 1),  fri_block.reshape(1, -1))
    mask_fri = (diff_fri < (max_fri * 0.01))
    graph[1][mask_fri] = 1
    graph[1] -= np.diag(np.diag(graph[1]))
    print('U-fri-U:{}'.format(np.count_nonzero(graph[1])/2))
    #favourites
    diff_favo = abs(favo_block.reshape(-1, 1) - favo_block.reshape(1, -1))
    max_favo = np.maximum(favo_block.reshape(-1, 1), favo_block.reshape(1, -1))
    min_favo = np.minimum(favo_block.reshape(-1, 1),  favo_block.reshape(1, -1))
    mask_favo = (diff_favo < (max_favo * 0.01))
    graph[2][mask_favo] = 1
    graph[2] -= np.diag(np.diag(graph[2]))
    print('U-fav-U:{}'.format(np.count_nonzero(graph[2])/2))
    #max_favo[max_favo==0]=1
    #Simi_favo=1-diff_favo/max_favo
    #print('Simi-favo:{}'.format(np.sum(Simi_favo-np.diag(np.diag(Simi_favo)))/(num*num-num)))
    #print('Simi-favo:{}'.format(np.mean(Simi_favo)))
    
    max_twe[~mask_twe] = 1
    max_fri[~mask_fri] = 1
    max_favo[~mask_favo] = 1
    adj_matrix = np.zeros((num, num))
    adj_matrix += (1 - (diff_twe / max_twe)) * mask_twe  + (1 - (diff_fri / max_fri)) * mask_fri + (1 - (diff_favo / max_favo)) * mask_favo
    #adj_matrix += (1-(diff_ff / max_ff))*mask_ff + (1-diff_types)*mask_types #FT 0,1
    #adj_matrix += (1-(diff_ff / max_ff))*mask_ff + (1-(diff_infs / max_infs))*mask_infs #FI 0,2
    #adj_matrix += (1-diff_types)*mask_types + (1-(diff_infs / max_infs))*mask_infs #TI 1,2
    
    adj_matrix -= np.diag(np.diag(adj_matrix))
    G = nx.Graph(adj_matrix)
    index=np.where(np.sum(adj_matrix,axis=1)==0)
    for i in index:
        adj_matrix[i,i]=0.01   
    
    
    #计算mulrirank
    x, y = multirank_cuda(graph)
    #x, y = multirank_cuda(graph[[1,2]])

    #编码树
    edges = np.array(adj_matrix.nonzero())  # [2, E]
    ew = adj_matrix[edges[0, :], edges[1, :]]
    devices = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #devices = 'cpu'
    ew, edges = torch.tensor(ew, device=devices), torch.tensor(edges, device=devices).t()
    #ew, edges = torch.tensor(ew), torch.tensor(edges).t()
    dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])  # dist/2=di
    dist = dist / (2 * ew.sum())  # ew.sum()=vol(G) dist=di/vol(G)
    print('construct encoding tree...')
    g = GraphSparse(edges, ew, dist)
    optim = OperatorPropagation(Partitioning(g, None))
    optim.perform(p=0.05)
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
    num_bot_comm =0
    bot_list=[]
    #社区划分
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
        if n_beta >= 0.6:  #botwiki:0.55; pronbots:0.6
            num_bot_comm += 1
            for node in comm:
                pre_bot[node] = 1
        else:
            for node in comm:
                bot_list.append(node)
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
        if b_data[5] >= 0.6:
            print(b_data)
    print('')
    for b_data in bot_rate:
        if b_data[5] < 0.6:
            print(b_data)
    

if __name__ == "__main__":
    start_time=time.time()
    for i in range(1):
        test('pronbots-2019')
    end_time=time.time()
    print('running time:{}s'.format((end_time-start_time)/1))
