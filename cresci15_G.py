import csv
import os
from sklearn import metrics
from util import multirank,load_user,load_tweet,multirank_cuda
import numpy as np
import pickle
from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning, EncodingTree
from torch_scatter import scatter_sum
import torch


print('cresci-2015')
file=["E13","FSF","INT","TFP","TWT"]
b=[0,1,1,0,1]
id=[]
label=[]
print('load user id and label...')
for dataset in file:
    line = 0
    f=open('./cresci-2015/'+dataset+'/users1.csv', 'r', encoding='utf-8')
    datas = csv.reader(f)
    for data in datas:
        line += 1
        if line == 1:
            continue
        id.append(data[0])
        label.append(b[file.index(dataset)])
print('load user id and label done')
num=len(id)
print('num:{},bot:{}'.format(num,sum(label)))
if os.path.exists('./graph/cresci15_f.adj'):
    print('load adj_matrix...')
    adj_matrix = pickle.load(open('./graph/cresci15_f.adj', 'rb+'))
    print('load adj_matrix done')
else:
    print('construct adj_matrix...')
    num=len(id)
    adj_matrix=np.zeros((num,num))
    for dataset in file:
        print(dataset)
        line = 0
        f=open('./cresci-2015/'+dataset+'/followers.csv', 'r', encoding='utf-8')
        datas=csv.reader(f)
        for data in datas:
            line += 1
            if line == 1:
                continue
            if not (data[0] in id and data[1] in id):
                continue
            x,y=id.index(data[0]),id.index(data[1])
            adj_matrix[x, y]=1
            adj_matrix[y, x] = 1
        line=0
        f = open('./cresci-2015/' + dataset + '/friends.csv', 'r', encoding='utf-8')
        datas = csv.reader(f)
        for data in datas:
            line += 1
            if line == 1:
                continue
            if not (data[0] in id and data[1] in id):
                continue
            x, y = id.index(data[0]), id.index(data[1])
            adj_matrix[x, y] = 1
            adj_matrix[y, x] = 1

    with open('./graph/cresci15_f.adj', 'wb+') as f:
        pickle.dump(adj_matrix, f)
        f.close()
    print('construct adj_matrix done')
print('num of edges:',sum(sum(adj_matrix)))
#计算mulrirank
x, y = multirank_cuda([adj_matrix])
index=np.where(np.sum(adj_matrix,axis=1)==0)
for i in index:
    adj_matrix[i,i]=0.01

#编码树
edges = np.array(adj_matrix.nonzero()) # [2, E]
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
    idx = division==i
    if idx.any():
        comms[i] = idx.nonzero().squeeze(1)

# 社区二分类
bot_rate = []
pre_bot = np.zeros(num)
for i in comms.keys():
    comm = comms[i]
    n_bots = 0
    n_nodes = 0
    n_x = 0
    for node in comm:
        n_bots += label[node]
        n_nodes += 1
        n_x += x[node]
    comm_SE = module_se[i]
    #n_beta = (n_x / n_nodes) / (1 / num) * 0.9 + comm_SE / (sum(module_se) / num * n_nodes) * 0.1  
    n_beta = (n_x / n_nodes) / (1 / num) * 0.6 + comm_SE / (sum(module_se) / totol_comm) * 0.4
    if n_beta >= 1:
        for node in comm:
            pre_bot[node] = 1
    bot_rate.append([n_bots / n_nodes, n_bots, n_nodes, n_x / n_nodes, comm_SE, n_beta])

# 准确率计算
acc = metrics.accuracy_score(label, pre_bot)
precision = metrics.precision_score(label, pre_bot)
recall = metrics.recall_score(label, pre_bot)
f1 = metrics.f1_score(label, pre_bot)
print('acc:{}'.format(acc))
print('Precision:{}'.format(precision))
print('Recall:{}'.format(recall))
print('F1:{}'.format(f1))
exit()
for b_data in bot_rate:
    if b_data[5] >= 1:
        print(b_data)
print('')
for b_data in bot_rate:
    if b_data[5] < 1:
        print(b_data)
