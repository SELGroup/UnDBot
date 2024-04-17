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
    infs = []
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

def process_human(dataset, filename):
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
            id.append(data[0])
            label.append(int(float(data[1])))
            infs.append(float(data[3]))
            ff.append(float(data[2]))
            types.append([float(data[4]), float(data[5]), float(data[6])])
        num = line - 1
        print('{}:{}'.format(filename, num))
        return num, id, label, ff, types, infs


def test(filename,fig_network=False,fig_tree=False):
    num0, id0, label0, ff0, types0, infs0 = process(filename)
    num1, id1, label1, ff1, types1, infs1 = process_human('cresci-2015', 'E13')  #botwiki:TFP[:65],pronbots::E13
    num = num0 + num1
    id_block = np.array(id0+id1)
    label_block = np.array(label0+label1)
    ff_block = np.array(ff0+ff1)
    types_block = np.array(types0+types1)
    infs_block = np.array(infs0+infs1)
    print('num:{},bot:{}'.format(num,np.sum(label_block)))
    

    graph = np.zeros((3, num, num))
    edge = np.zeros(3)

    diff_ff = abs(ff_block.reshape(-1, 1) - ff_block.reshape(1, -1))
    max_ff = np.maximum(ff_block.reshape(-1, 1), ff_block.reshape(1, -1))
    min_ff = np.minimum(ff_block.reshape(-1, 1), ff_block.reshape(1, -1))
    mask_ff = (diff_ff < (max_ff * 0.01)) & (min_ff >= 3)
    graph[0][mask_ff] = 1
    graph[0] -= np.diag(np.diag(graph[0]))
    #print('U-F-U:{}'.format(np.count_nonzero(graph[0])/2))
    #max_ff[max_ff==0]=1
    #Simi_F=1-diff_ff/max_ff
    #print('Simi-F:{}'.format(np.sum(Simi_F-np.diag(np.diag(Simi_F)))/(num*num-num)))
    #print('Simi-F:{}'.format(np.mean(Simi_F)))

    diff_types = abs(types_block[:, 0].reshape(-1, 1) - types_block[:, 0].reshape(1, -1)) + \
                 abs(types_block[:, 1].reshape(-1, 1) - types_block[:, 1].reshape(1, -1)) + \
                 abs(types_block[:, 2].reshape(-1, 1) - types_block[:, 2].reshape(1, -1))
    mask_types = diff_types < 0.01
    graph[1][mask_types] = 1
    graph[1] -= np.diag(np.diag(graph[1]))
    #print('U-T-U:{}'.format(np.count_nonzero(graph[1])/2))
    #Simi_T=1-diff_types
    #print('Simi-T:{}'.format(np.sum(Simi_T-np.diag(np.diag(Simi_T)))/(num*num-num)))
    #print('Simi-T:{}'.format(np.mean(Simi_T)))

    diff_infs = abs(infs_block.reshape(-1, 1) - infs_block.reshape(1, -1))
    max_infs = np.maximum(infs_block.reshape(-1, 1), infs_block.reshape(1, -1))
    min_infs = np.minimum(infs_block.reshape(-1, 1),  infs_block.reshape(1, -1))
    mask_infs = (diff_infs < (max_infs * 0.01))  #pronbots:0.01 others:0.1
    graph[2][mask_infs] = 1
    graph[2] -= np.diag(np.diag(graph[2]))
    #print('U-I-U:{}'.format(np.count_nonzero(graph[2])/2))
    #max_infs[max_infs==0]=1
    #Simi_I=1-diff_infs/max_infs
    #print('Simi-I:{}'.format(np.sum(Simi_I-np.diag(np.diag(Simi_I)))/(num*num-num)))
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
    
    
    #x, y = multirank_cuda(graph)
    x, y = multirank(graph)
    #x, y = multirank_cuda(graph[[1,2]])

    edges = np.array(adj_matrix.nonzero())  # [2, E]
    ew = adj_matrix[edges[0, :], edges[1, :]]
    devices = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    devices = 'cpu'
    ew, edges = torch.tensor(ew, device=devices), torch.tensor(edges, device=devices).t()
    #ew, edges = torch.tensor(ew), torch.tensor(edges).t()
    dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])  # dist/2=di
    dist = dist / (2 * ew.sum())  # ew.sum()=vol(G) dist=di/vol(G)
    print('construct encoding tree...')
    g = GraphSparse(edges, ew, dist)
    optim = OperatorPropagation(Partitioning(g, None))
    optim.perform(p=0.05)#pronbots:0.05 others:0.15
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
    num_bot_comm =0
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
        if n_beta >= 0.60:  #botwiki:0.55; pronbots:0.6
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
    start_time=time.time()
    for i in range(1):
        test('pronbots-2019', fig_network=False, fig_tree=False)
    end_time=time.time()
    print('running time:{}s'.format((end_time-start_time)/1))
