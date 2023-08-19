import ijson
import numpy as np
import csv
import os
import torch

# 导入用户
def load_user(filename, b):
    print("is loading user from {}...".format(filename))
    name_file = open(filename, 'r', encoding='utf-8')
    datas = csv.reader(name_file)
    account = []
    real_bot = []
    ff = []
    n = 0
    for data in datas:
        n = n + 1
        if n == 1:
            continue
        else:
            account.append(data[0])
            real_bot.append(b)
            ff.append((int(data[5]) + 1) / (int(data[4]) + 1))
    num = n - 1
    print('node:', num)
    return account, real_bot, ff

#从.tsv导入标签
def load_label(filename):
    fp = open(filename + '/' + filename + '.tsv', 'r', encoding='utf8')
    csv_file = open(filename + "/label.csv", "w+", newline='')  # 新建csv格式文件
    writer = csv.writer(csv_file)  # 对象化
    header = ["userid", "label"]  # 构造表头
    writer.writerow(header)  # 写入表头
    for line in fp:
        line = line.strip('\n').split('\t')
        id = line[0]
        if line[1] == 'human':
            label = 0
        else:
            label = 1
        writer.writerow([id, label])

# 导入tweet并处理(cresci15以及cresci17中fake follower)
def load_tweet(filename, account):
    print("is loading tweet from {}...".format(filename))
    tweet_file = open(filename, 'r', encoding='utf-8')
    datas = csv.reader(tweet_file)
    n = 0
    num = len(account)
    types = np.zeros((num, 3))  # original,retweet,comment
    inf = np.zeros(num)  # 原创推文的评论点赞转发数总
    for data in datas:
        n = n + 1
        if n == 1:
            continue
        else:
            t = 0  # 找user_id在的列
            while t < 6 and t < len(data):
                if data[t].isdigit() and (data[t] in account):
                    break
                else:
                    t += 1
            if t == 6 or t == len(data):
                continue
            x = account.index(data[t])

            # 统计三种类型推文
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
                    inf[x] += int(data[-5])  # 原创推文的评论点赞转发数总和
    print('')
    for i in range(num):
        if types[i][0] != 0:
            inf[i] = inf[i] / types[i][0]
        sum_t = sum(types[i])  # 总推文数
        if sum_t != 0:
            types[i] = types[i] / sum_t
    return types, inf


def process_json(filename):
    csv_file = open(filename + "/" + filename + ".csv", "w+", newline='')  # 新建csv格式文件
    writer = csv.writer(csv_file)  # 对象化
    header = ["userid", "screenname", "follower", "friend"]  # 构造表头
    writer.writerow(header)  # 写入表头
    fp = open(filename + '/' + filename + '_tweets.json', 'r', encoding='utf8')
    for record in ijson.items(fp, 'item'):
        id = '\'' + str(record["user_id"])
        name = record["screen_name"]
        follower = record["followers_count"]
        friend = record["friends_count"]
        '''
        user=record["user"]
        print(user)
        print(json.dumps(record,indent=4))
        id='\''+user["id_str"]
        name=user["screen_name"]
        follower=user["followers_count"]
        friend=user["friends_count"]
        '''
        writer.writerow([id, name, follower, friend])


def process_tweet(filename):
    path_label = filename + '/label.csv'
    path_tweets = filename + '/tweet.csv'
    path_followers = filename + "/" + filename + ".csv"
    if not os.path.exists(path_label):
        load_label(filename)
    id = []
    # 关注粉丝比
    ff = []
    fp_ff = open(path_followers, 'r', encoding='utf-8')
    ffs = csv.reader(fp_ff)
    line = 0
    for data in ffs:
        line += 1
        if line == 1:
            continue
        id.append(data[0].replace('\'', ''))
        ff.append((int(data[3]) + 1) / (int(data[2]) + 1))
    # 真实标签
    num = len(id)
    label = np.zeros(num)
    fp_label = open(path_label, 'r', encoding='utf8')
    labels = csv.reader(fp_label)
    line = 0
    for data in labels:
        line += 1
        if line == 1:
            continue
        if data[0] in id:
            x = id.index(data[0])
            label[x] = data[1]

    print(len(id), len(label), len(ff))
    # 发帖类型分布，发帖影响力
    types = np.zeros((num, 3))  # original,retweet,comment
    infs = np.zeros(num)  # 推文的评论点赞转发数总
    fp_tweet = open(path_tweets, 'r', encoding='utf8')
    '''
    fp_tweet1 = fp_tweet.read()
    fp_tweet2 = fp_tweet1.replace('\x00', '?')
    tweets = csv.reader(StringIO(fp_tweet2))
    '''
    tweets = csv.reader(fp_tweet)
    for data in tweets:
        if not (data[1] in id):
            continue
        x = id.index(data[1])
        types[x][int(data[-1]) - 1] += 1
        i = -4
        flag = 0
        if int(data[-1]) != 1:
            continue
        while flag != 3:
            if data[i].isdigit():
                infs[x] += int(data[i])
                i -= 1
                flag += 1
            else:
                text = data[i].split('?')
                infs[x] += int(text[-1])
                flag += 1
    # 写入文件中
    with open(filename + '/' + filename + '_f.csv', "w+", newline='') as out_file:  # 新建csv格式文件
        writer = csv.writer(out_file)  # 对象化
        header = ["userid", "label", "infs", "ff", "type_o", "type_r", "type_c"]  # 构造表头
        writer.writerow(header)  # 写入表头
        for i in range(num):
            if sum(types[i]) != 0:
                infs[i] = infs[i] / sum(types[i])
                types[i] = types[i] / sum(types[i])
            datarow = [id[i], label[i], infs[i], ff[i]]
            datarow.extend(types[i])
            writer.writerow(datarow)  # 写入csv


def multirank_cuda(multi_graph):
    # multi_graph: [r, m, m]
    # print('----------------------------------------------')
    print('is calculating multirank...')
    devices = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    multi_graph = torch.tensor(multi_graph, device=devices)
    m = len(multi_graph[0])
    r = len(multi_graph)

    sum_i1 = torch.sum(multi_graph, dim=1)
    sum_j = torch.sum(multi_graph, dim=0)
    O = multi_graph / sum_i1.reshape(r, 1, m)  # m, m, r
    R = multi_graph / sum_j.reshape(1, m, m)  # r, m, m
    O[torch.tile(sum_i1.reshape(r, 1, m) == 0, (1, m, 1))] = 1 / m
    O = O.permute(1, 2, 0)
    R[torch.tile(sum_j.reshape(1, m, m) == 0, (r, 1, 1))] = 1 / r

    x_old = torch.ones(m, device=devices, dtype=torch.float64)
    x_new = torch.ones(m, device=devices, dtype=torch.float64)
    x_old = x_old * (1 / m)
    y_old = torch.ones(r, device=devices, dtype=torch.float64)
    y_new = torch.ones(r, device=devices, dtype=torch.float64)
    y_old = y_old * (1 / r)
    x_new[:] = x_old[:]
    y_new[:] = y_old[:]
    while True:
        x_old[:] = x_new[:]
        y_old[:] = y_new[:]
        for i in range(m):
            x_new[i] = x_old @ O[i] @ y_old
        for j in range(r):
            y_new[j] = x_old @ R[j] @ x_old
        z = torch.sqrt(sum((x_old - x_new) ** 2)) + torch.sqrt(sum((y_old - y_new) ** 2))
        if z < 0.004:
            break
    print('y_final:', y_new)
    return x_new.cpu().detach().numpy(), y_new.cpu().detach().numpy()

def multirank(multi_graph):
    # multi_graph: [r, m, m]
    # print('----------------------------------------------')
    print('is calculating multirank...')
    m = len(multi_graph[0])
    r = len(multi_graph)

    sum_i1 = np.sum(multi_graph, axis=1)
    sum_j = np.sum(multi_graph, axis=0)
    O = multi_graph / sum_i1.reshape(r, 1, m)
    R = multi_graph / sum_j.reshape(1, m, m)
    O[np.tile(sum_i1.reshape(r, 1, m) == 0, (1, m, 1))] = 1 / m
    O = np.transpose(O, (1, 2, 0))
    R[np.tile(sum_j.reshape(1, m, m) == 0, (r, 1, 1))] = 1 / r

    x_old = np.ones(m)
    x_new = np.ones(m)
    x_old = x_old * (1 / m)
    y_old = np.ones(r)
    y_new = np.ones(r)
    y_old = y_old * (1 / r)
    x_new[:] = x_old[:]
    y_new[:] = y_old[:]
    while True:
        x_old[:] = x_new[:]
        y_old[:] = y_new[:]
        for i in range(m):
            x_new[i] = np.dot(np.dot(x_old, O[i]), y_old.T)
        for j in range(r):
            y_new[j] = np.dot(np.dot(x_old, R[j]), x_old.T)
        z = np.sqrt(sum((x_old - x_new) ** 2)) + np.sqrt(sum((y_old - y_new) ** 2))
        if z < 0.004:
            break
    print('y_final:', y_new)
    return x_new, y_new

def multirank_old(multi_graph):
    # print('----------------------------------------------')
    print('is calculating multirank...')
    m = len(multi_graph[0])
    r = len(multi_graph)
    O = np.zeros((m, m, r))
    R = np.zeros((r, m, m))
    sum_j = np.zeros((m, m))
    sum_i1 = np.zeros((r, m))
    for i1 in range(m):
        for i2 in range(m):
            for j in range(r):
                sum_i1[j][i2] += multi_graph[j][i1][i2]
                sum_j[i1][i2] += multi_graph[j][i1][i2]
    for i1 in range(m):
        for i2 in range(m):
            for j in range(r):
                O[i1][i2][j] = multi_graph[j][i1][i2] / sum_i1[j][i2] if sum_i1[j][i2] != 0 else 1 / m
                R[j][i1][i2] = multi_graph[j][i1][i2] / sum_j[i1][i2] if sum_j[i1][i2] != 0 else 1 / r
    x_old = np.ones(m)
    x_new = np.ones(m)
    x_old = x_old * (1 / m)
    y_old = np.ones(r)
    y_new = np.ones(r)
    y_old = y_old * (1 / r)
    x_new[:] = x_old[:]
    y_new[:] = y_old[:]
    while True:
        x_old[:] = x_new[:]
        y_old[:] = y_new[:]
        for i in range(m):
            x_new[i] = np.dot(np.dot(x_old, O[i]), y_old.T)
        for j in range(r):
            y_new[j] = np.dot(np.dot(x_old, R[j]), x_old.T)
        z = np.sqrt(sum((x_old - x_new) ** 2)) + np.sqrt(sum((y_old - y_new) ** 2))
        if z < 0.004:
            break
    print('y_final:', y_new)
    return x_new, y_new




