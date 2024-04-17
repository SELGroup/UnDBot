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
import time
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
import seaborn as sns

# 导入加工过的tweet特征
def process(dataset, filename,b):
    path = dataset + '/' + filename + '_f.csv'
    if os.path.exists(path):
        id = []
        label = []
        ff = []
        types = []  # original,retweet,comment
        infs = []  # 原创推文的评论点赞转发数总
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
    else:
        file1 = dataset + '/' + filename + '/users1.csv'
        user, real_label, ff = load_user(file1, b)
        num1 = len(user)
        file2 = dataset + '/' + filename + '/tweets1.csv'
        types, infs = load_tweet(file2, user)
        print('{}:{}'.format(filename, num1))
        with open(path, "w+", newline='') as csv_file:  # 新建csv格式文件
            writer = csv.writer(csv_file)  # 对象化
            header = ["userid", "label", "ff", "inf", "type1", "type2", "type3"]  # 构造表头
            writer.writerow(header)  # 写入表头
            for i in range(num1):
                datarow = [user[i], real_label[i], ff[i], infs[i]]
                datarow.extend(types[i])
                writer.writerow(datarow)  # 写入csv
        return num1, user, real_label, ff, types, infs

def test(fig_network=False,fig_tree=False,fig_statistic=False):
    num1, id1, label1, ff1, types1, infs1 = process('cresci-2015', 'human', 0)
    num2, id2, label2, ff2, types2, infs2 = process('cresci-2015', 'bot', 1)

    # 合并人类账户和机器账户

    num = num1 + num2
    id_block = np.array(id1+id2)
    label_block = np.array(label1+label2)
    ff_block = np.array(ff1+ff2)
    types_block = np.array(types1+types2)
    infs_block = np.array(infs1+infs2)
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
    #print('U-F-U:{}'.format(np.count_nonzero(graph[0])/2))
    #max_ff[max_ff==0]=1
    #Simi_F=1-diff_ff/max_ff
    #print('Simi-F:{}'.format(np.mean(Simi_F)))

    diff_types = abs(types_block[:, 0].reshape(-1, 1) - types_block[:, 0].reshape(1, -1))+ \
                abs(types_block[:, 1].reshape(-1, 1) - types_block[:, 1].reshape(1, -1))+\
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

    adj_matrix += (1-(diff_ff / max_ff))*mask_ff + (1-diff_types)*mask_types + (1-(diff_infs / max_infs))*mask_infs
    #adj_matrix += (1-(diff_ff / max_ff))*mask_ff + (1-diff_types)*mask_types #FT 0,1
    #adj_matrix += (1-(diff_ff / max_ff))*mask_ff + (1-(diff_infs / max_infs))*mask_infs #FI 0,2
    #adj_matrix += (1-diff_types)*mask_types + (1-(diff_infs / max_infs))*mask_infs #TI 1,2

    adj_matrix -= np.diag(np.diag(adj_matrix))
    G = nx.Graph(adj_matrix)
    index=np.where(np.sum(adj_matrix,axis=1)==0)
    for i in index:
        adj_matrix[i,i]=0.01

    #计算mulrirank

    #x, y = multirank_cuda(graph[[1,2]])
    x, y = multirank_cuda(graph)

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

    # 准确率计算
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

    fsize=25
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', linewidth=5)
    # 随即分类器没有分类能力，其FPR=TPR。随机分类器的性能通常表示为ROC曲线上的对角线
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
    plt.xlabel('False Positive Rate', fontsize=fsize)
    plt.ylabel('True Positive Rate', fontsize=fsize)
    plt.title('ROC Curve', fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.grid(True)
    plt.legend(loc='lower right',fontsize=fsize)
    plt.savefig('./figure/ROC_{}_1.png'.format('Cresci-2015'), bbox_inches='tight')                
    plt.savefig('./figure/ROC_{}_1.pdf'.format('Cresci-2015'), bbox_inches='tight')
    
    for b_data in bot_rate:
        if b_data[5] >= 1:
            print(b_data)
    print('')
    for b_data in bot_rate:
        if b_data[5] < 1:
            print(b_data)
    
    if fig_statistic == True:
        label2 = np.vstack((pre_bot,label_block))
        index11=np.intersect1d(np.where(pre_bot==1), np.where(label_block==1))
        inf_Tbot=infs_block[index11]
        index10=np.intersect1d(np.where(pre_bot==1), np.where(label_block==0))
        inf_Fbot=infs_block[index10]
        index01=np.intersect1d(np.where(pre_bot==0), np.where(label_block==1))
        inf_Fhuman=infs_block[index01]
        index00=np.intersect1d(np.where(pre_bot==0), np.where(label_block==0))
        inf_Thuman=infs_block[index00]

        
        '''
        bins_type=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        hist1,_,_=np.histogram2d(type_bot[:,0],type_bot[:,1],bins=(bins_type,bins_type))
        hist0,_,_=np.histogram2d(type_human[:,0],type_human[:,1],bins=(bins_type,bins_type))
        print(hist1,sum(sum(hist1)))
        print(hist0,sum(sum(hist0)))
        df_cm=pd.DataFrame(hist0/sum(sum(hist0))*100,index=bins_type[1:],columns=bins_type[1:])
        mask=np.zeros_like(df_cm)
        for i in range(len(mask)):
            for j in range(i+1,len(mask)):
                mask[i][j]=True
        plt.figure(figsize=(12,10))
        ax=sns.heatmap(df_cm,vmin=0,vmax=10,mask=mask,cmap='GnBu',annot=True,linewidths=0.05,annot_kws={'size':14,'color':'k'},fmt='.2f')
        #调用Colorbar类：
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        plt.xlabel('original tweet',fontsize=20)
        plt.ylabel('retweet',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        '''
        #bins_ff=[0,2,4,6,8,10,20,30,40,60,80,100,float('inf')]
        bins_inf=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,float('inf')]
        
        hist11,_=np.histogram(inf_Tbot,bins=bins_inf)
        hist10,_=np.histogram(inf_Fbot,bins=bins_inf)
        hist00,_=np.histogram(inf_Thuman,bins=bins_inf)
        hist01,_=np.histogram(inf_Fhuman,bins=bins_inf)
        sum1=sum(hist11)+sum(hist10)
        sum2=sum(hist00)+sum(hist01)
        print(sum1,sum2)
        hist11=hist11/sum1*100
        hist10=hist10/sum1*100
        hist00=hist00/sum2*100
        hist01=hist01/sum2*100
        x=np.arange(len(bins_inf)-1)
        width=0.8/2
        plt.figure(figsize=(10,6))
        plt.bar(x,hist11,width=width,label='bot',color='darkred')
        for i in range(len(x)):
            plt.text(x[i],hist10[i]+hist11[i]+1,str("%.2f" % hist11[i])+'+'+str("%.2f" % hist10[i]),ha='center',va='bottom',fontsize=14,rotation=90)
        plt.bar(x,hist10,bottom=hist11,width=width,label='misjudge',color='gray')
        #for i,j in zip(x,hist10):
        #    plt.text(i,j-1,"%.1f"%j,ha='center',va='bottom',fontsize=15,rotation=90)
        plt.bar(x+width,hist00,width=width,label='human')
        for i in range(len(x)):
            plt.text(x[i]+width,hist00[i]+hist01[i]+1,str("%.2f" % hist00[i])+'+'+str("%.2f" % hist01[i]),ha='center',va='bottom',fontsize=14,rotation=90)
        plt.bar(x+width,hist01,bottom=hist00,width=width,color='gray')
        #for i,j in zip(x,hist01):
        #    plt.text(i+width,j-1,"%.1f"%j,ha='center',va='bottom',fontsize=15,rotation=90)
        plt.legend(fontsize=18)
        plt.xlabel('posting influence',fontsize=18)
        plt.ylabel('Proportion',fontsize=18)
        plt.ylim(0,80)
        plt.xticks(x-width*0.75,bins_inf[:-1],fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.savefig('./figure/statistic_cresci-2015_inf.pdf', bbox_inches='tight')
        plt.savefig('./figure/statistic_cresci-2015_inf.png', bbox_inches='tight')
    
    # 画网络图
    if fig_network == True:
        k=0.2
        print('calculating pos...')
        pos = nx.spring_layout(G, k=k, iterations=20, seed=1721)
        fig, ax = plt.subplots(figsize=(10,10))
        #ax.axis("off")
        plt.xlim(-1.05,1.05)
        plt.ylim(-1.05,1.05)
        color = []
        for item in pre_bot:
            if item == 1:
                color.append('darkred')
            else:
                color.append('royalblue')
        print('drawing figure...')
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, alpha=1, node_color=color, node_size=15)
        plt.scatter([], [], marker='o', c='darkred', s=30, alpha=1, label='bot')
        plt.scatter([], [], marker='o', c='royalblue', s=30, alpha=1, label='human')
        plt.legend(scatterpoints=1, fontsize=18, loc='upper left')
        plt.savefig('./figure/network_pred_cresci-2015_k{}.pdf'.format(k), bbox_inches='tight')
        plt.savefig('./figure/network_pred_cresci-2015_k{}.png'.format(k), bbox_inches='tight')

    # 画树
    if fig_tree == True:
        size = [10, 15, 20, 35, 50, 60, 85, 110, 135, 160]  # <=1,9,19,49,99,199,499,999,1999,>=2000
        fig = plt.figure(figsize=(10, 4))
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴

        # root
        x0 = 0.5
        y0 = 1
        y1 = 0.8
        s = size[9]
        plt.scatter(x0, y0, marker='s', c='gray', edgecolor='k', linewidth='0.5', s=s + 10, zorder=2)
        plt.text(x0 - 0.05, y0, 'Root', ha='center', va='center', fontsize=9, color='k')
        #
        num_f_bot = []
        num_f_human = []
        num_t_bot = []
        num_t_human = []
        for i in range(totol_comm):
            num_t_bot.append(bot_rate[i][1])
            num_t_human.append(bot_rate[i][2] - bot_rate[i][1])
            num_f_bot.append(math.ceil(bot_rate[i][1] / 100))
            num_f_human.append(math.ceil((bot_rate[i][2] - bot_rate[i][1]) / 100))
        print(num_f_bot)
        print(num_f_human)
        index = np.argsort(-(np.array(num_f_bot) + np.array(num_f_human)))
        print(index)
        sum_f = sum(num_f_bot) + sum(num_f_human)
        dx = 1 / (num_bot_comm + sum_f + 1)
        x2 = 0
        y2 = 0.6
        plt.text(0.1, y1, 'Comm', ha='center', va='center', fontsize=9, color='k')
        for i in index:
            x2 += dx
            x1 = x2 + (num_f_bot[i] + num_f_human[i] - 1) * dx / 2
            if num_t_bot[i] + num_t_human[i] >= 2000:
                s = size[9]
            elif num_t_bot[i] + num_t_human[i] > 999:
                s = size[8]
            elif num_t_bot[i] + num_t_human[i] > 499:
                s = size[7]
            elif num_t_bot[i] + num_t_human[i] > 199:
                s = size[6]
            elif num_t_bot[i] + num_t_human[i] > 99:
                s = size[5]
            elif num_t_bot[i] + num_t_human[i] > 49:
                s = size[4]
            elif num_t_bot[i] + num_t_human[i] > 19:
                s = size[3]
            elif num_t_bot[i] + num_t_human[i] > 9:
                s = size[2]
            elif num_t_bot[i] + num_t_human[i] > 1:
                s = size[1]
            else:
                s = size[0]
            plt.plot([x0, x1], [y0, y1], color='lightgrey', linewidth=0.5, zorder=1)
            if bot_rate[i][5] >= 1:
                color = 'darkred'
            else:
                color = 'royalblue'
            plt.scatter(x1, y1, marker='s', c=color, edgecolor='k', linewidth='0.5', s=s + 10, zorder=2)
            plt.text(x1, y1 + 0.03, '{}'.format(num_t_bot[i] + num_t_human[i]), ha='center', va='center', fontsize=9,
                     color='k')
            if num_f_bot[i] + num_f_human[i] != 1:
                for j in range(num_f_bot[i]):
                    if j != num_f_bot[i] - 1 or num_t_bot[i] % 100 == 0:
                        s = size[5]
                    elif num_t_bot[i] % 100 > 49:
                        s = size[4]
                    elif num_t_bot[i] % 100 > 19:
                        s = size[3]
                    elif num_t_bot[i] % 100 > 9:
                        s = size[2]
                    elif num_t_bot[i] % 100 > 1:
                        s = size[1]
                    else:
                        s = size[0]
                    plt.plot([x1, x2], [y1, y2], color='lightgrey', linewidth=0.5, zorder=1)
                    plt.scatter(x2, y2, marker='o', c='indianred', edgecolor='k', linewidth='0.5', s=s, zorder=2)
                    x2 += dx
                for j in range(num_f_human[i]):
                    if j != num_f_human[i] - 1 or num_t_human[i] % 100 == 0:
                        s = size[5]
                    elif num_t_human[i] % 100 > 49:
                        s = size[4]
                    elif num_t_human[i] % 100 > 19:
                        s = size[3]
                    elif num_t_human[i] % 100 > 9:
                        s = size[2]
                    elif num_t_human[i] % 100 > 1:
                        s = size[1]
                    else:
                        s = size[0]
                    plt.plot([x1, x2], [y1, y2], color='lightgrey', linewidth=0.5, zorder=1)
                    plt.scatter(x2, y2, marker='o', c='cornflowerblue', edgecolor='k', linewidth='0.5', s=s, zorder=2)
                    x2 += dx
            else:
                plt.plot([x1, x2], [y1, y2], color='lightgrey', linewidth=0.5, zorder=1)
                if num_t_bot[i]+num_t_human[i] % 100 == 0:
                    s = size[5]
                elif num_t_human[i] % 100 > 49:
                    s = size[4]
                elif num_t_human[i] % 100 > 19:
                    s = size[3]
                elif num_t_human[i] % 100 > 9:
                    s = size[2]
                elif num_t_human[i] % 100 > 1:
                    s = size[1]
                else:
                    s = size[0]
                if num_f_bot[i]==1:
                    plt.scatter(x2, y2, marker='o', c='indianred', edgecolor='k',linewidth='0.5', s=s, zorder=2)
                else:
                    plt.scatter(x2, y2, marker='o', c='cornflowerblue', edgecolor='k',linewidth='0.5', s=s, zorder=2)
        plt.scatter([], [], marker='s', c='darkred', edgecolor='k', linewidth='0.5', s=35, label='bot community')
        plt.scatter([], [], marker='s', c='royalblue', edgecolor='k', linewidth='0.5', s=35, label='human community')
        plt.scatter([], [], marker='o', c='indianred', edgecolor='k', linewidth='0.5', s=35, label='bots')
        plt.scatter([], [], marker='o', c='cornflowerblue', edgecolor='k', linewidth='0.5', s=35, label='humans')
        plt.legend(scatterpoints=1, fontsize=10, loc='upper left')
        plt.savefig('./figure/tree_cresci-2015_revise.pdf', bbox_inches='tight')

if __name__ == "__main__":
    test(fig_tree=False)
    #start_time=time.time()
    #for i in range(5):
    #    test()
    #end_time=time.time()
    #print('running time:{}s'.format((end_time-start_time)/5))
