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
from sklearn.metrics.pairwise import cosine_similarity


def test(filename):
    device = "cpu"
    # num*13 true/false
    category_properties = torch.load("./baseline/GAE/{}/processed_data/cat_properties_tensor.pt".format(filename)).to(device)
    category_properties1 = torch.load("./baseline/GAE/{}/processed_data/cat_properties_tensor.pt".format('cresci-2015')).to(device)
    #category_properties=torch.cat((category_properties[:,:1],category_properties1[1481:1481+65,:]),0)
    category_properties=torch.cat((category_properties[:,:1],category_properties1[:1481,:]),0)
    # num*768 description
    des = torch.load("./baseline/GAE/{}/processed_data/des_tensor.pt".format(filename)).to(device)
    des1 = torch.load("./baseline/GAE/{}/processed_data/des_tensor.pt".format('cresci-2015')).to(device)
    #des=torch.cat((des,des1[1481:1481+65,:]),0)
    des=torch.cat((des,des1[:1481,:]),0)
    # num*7  followers_count, friends_count, listed_count, favourites_count, statuses_count, active_days, screen_name_length
    num_properties = torch.load("./baseline/GAE/{}/processed_data/num_properties_tensor.pt".format(filename)).to(device)
    num_properties1 = torch.load("./baseline/GAE/{}/processed_data/num_properties_tensor.pt".format('cresci-2015')).to(device)
    #num_properties=torch.cat((num_properties[:,:5],num_properties1[1481:1481+65,:]),0)
    num_properties=torch.cat((num_properties[:,:5],num_properties1[:1481,:]),0)
    # num*768 tweets
    tweets = torch.load("./baseline/GAE/{}/processed_data/tweets_tensor.pt".format(filename)).to(device)
    tweets1 = torch.load("./baseline/GAE/{}/processed_data/tweets_tensor.pt".format('cresci-2015')).to(device)
    #tweets=torch.cat((tweets,tweets1[1481:1481+65,:]),0)
    tweets=torch.cat((tweets,tweets1[:1481,:]),0)

    true_label = torch.load("./baseline/GAE/{}/processed_data/label.pt".format(filename)).to(device)
    true_label1 = torch.load("./baseline/GAE/{}/processed_data/label.pt".format('cresci-2015')).to(device)
    #true_label=torch.cat((true_label,true_label1[1481:1481+65]),0)
    true_label=torch.cat((true_label,true_label1[:1481]),0)

    X = torch.cat([category_properties, num_properties], dim=1)
    num = X.size(0)
    print('num:', num)
    Simi_cos = cosine_similarity(X, X)
    mask = Simi_cos > 0.9
    adj_matrix = Simi_cos * mask
    adj_matrix -= np.diag(np.diag(adj_matrix))

    #计算mulrirank
    mask_graph=(adj_matrix>0)
    graph=np.zeros((1,num,num))
    graph[0][mask_graph]=1
    print('num of edges:',sum(sum(sum(graph))))
    x, y = multirank_cuda(graph)
    index=np.where(np.sum(adj_matrix,axis=1)==0)
    for i in index:
        adj_matrix[i,i]=0.01

    #编码树
    edges = np.array(adj_matrix.nonzero())  # [2, E]
    ew = adj_matrix[edges[0, :], edges[1, :]]
    devices = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ew, edges = torch.tensor(ew, device=devices), torch.tensor(edges, device=devices).t()
    #ew, edges = torch.tensor(ew), torch.tensor(edges).t()
    dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])  # dist/2=di
    dist = dist / (2 * ew.sum())  # ew.sum()=vol(G) dist=di/vol(G)
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

    #社区划分
    for i in comms.keys():
        comm = comms[i]
        n_bots = 0
        n_nodes = 0
        n_x = 0
        for node in comm:
            n_bots += true_label[node]
            n_nodes += 1
            n_x += x[node]
        comm_SE = module_se[i]
        #n_beta = (n_x / n_nodes) / (1 / num) * 0.9 + comm_SE / (sum(module_se) / num * n_nodes) * 0.1
        n_beta = (n_x / n_nodes) / (1 / num) * 0.4 + comm_SE / (sum(module_se) / totol_comm) * 0.6
        if n_beta >= 0.8:
            for node in comm:
                pre_bot[node] = 1
        bot_rate.append([n_bots / n_nodes, n_bots, n_nodes, n_x / n_nodes, comm_SE, n_beta])
    
    # 准确率计算
    acc = metrics.accuracy_score(true_label, pre_bot)
    precision = metrics.precision_score(true_label, pre_bot)
    recall = metrics.recall_score(true_label, pre_bot)
    f1 = metrics.f1_score(true_label, pre_bot)
    print('acc:{}'.format(acc))
    print('Precision:{}'.format(precision))
    print('Recall:{}'.format(recall))
    print('F1:{}'.format(f1))
    
    for b_data in bot_rate:
        if b_data[5] >= 0.8:
            print(b_data)
    print('')
    for b_data in bot_rate:
        if b_data[5] < 0.8:
            print(b_data)


if __name__ == "__main__":
    test('pronbots-2019')


    