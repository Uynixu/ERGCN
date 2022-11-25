import numpy as np
import os
import dgl
import torch
import collections
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
from torch.nn import functional as F


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def load_quadruples(inPath, fileName, time_sort=False):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            time = int(line_split[3])
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        if time_sort == True:
            time_id_dict = get_tim_id_dict(times)
            times_new = set()
            quad_new = []
            for data in quadrupleList:
                t = int(data[3])
                data[3] = int(time_id_dict[t])
                quad_new.append(data)
                times_new.add(int(time_id_dict[t]))
            times = times_new
            quadrupleList = quad_new

    return np.array(quadrupleList), np.sort(list(times))


def get_graph(data, num_rels):
    data = np.asarray(data)
    src = data[:, 2]
    rel = data[:, 1]
    dst = data[:, 0]
    uniq_v = np.unique((src, dst))
    frm, twrd = np.concatenate((src, dst)), np.concatenate((dst, src))
    g = dgl.graph((frm, twrd))
    rel_o = np.concatenate((rel + num_rels, rel))
    rel_s = np.concatenate((rel, rel + num_rels))
    g.edata['rel_s'] = torch.from_numpy(rel_s).view(-1).long()
    g.edata['rel_o'] = torch.from_numpy(rel_o).view(-1).long()
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.tensor(list(range(1 + max(uniq_v)))).long().view(-1, 1),
                    'norm': norm.view(-1, 1)})
    g.ids = {}
    idx = 0
    for ids in list(range(1 + max(uniq_v))):
        g.ids[ids] = idx
        idx += 1
    return g


def comp_deg_norm(g):
    in_deg = g.in_degrees().type(torch.float64)
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


def soft_cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax()
    # pred = pred.type('torch.DoubleTensor').cuda()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def get_subgraph(graph, nodes):
    if len(nodes) != 0:
        relabel_nodes = []
        for node in nodes:
            try:
                relabel_nodes.append(graph.ids[int(node)])
            except KeyError:
                # print(node)
                pass
        nodes = np.array(relabel_nodes)
        sub_g = dgl.node_subgraph(graph, nodes)
    else:
        sub_g = []
    return sub_g


def get_history(data, s_his_dict, time_list, tim_id_dict, his_len=10, layers=1, reverse=False):
    if reverse:
        s = int(data[2])
        r = data[1]
        o = data[0]
        t = data[3]
    else:
        s = int(data[0])
        r = data[1]
        o = data[2]
        t = data[3]
    g_id = []
    sub_g_id = set()
    node_id = []
    idx = int(tim_id_dict[t])

    tim = time_list[:idx]
    for tt in tim[::-1]:
        if len(g_id) >= his_len:
            break

        sub_g_id.add(s)
        nodes = []
        if len(s_his_dict[tt][s]) != 0:
            g_id.append(tt)
            for _ in range(layers):
                for sb in list(sub_g_id):
                    if len(s_his_dict[tt][sb]) != 0:
                        s_his = s_his_dict[tt][sb]
                        ob = s_his[:, 1]
                        nodes += ob.tolist()
                sub_g_id = sub_g_id | set(nodes)

            node_id.append(list(sub_g_id))
            sub_g_id = set()

    len_hist = len(g_id)

    hist = [s, g_id[::-1], node_id[::-1], r]
    return hist, o, len_hist


def get_tim_id_dict(time_list):
    tim_id_dict = {}
    times = np.sort(time_list)
    for i, t in enumerate(times):
        tim_id_dict[t] = i
    return tim_id_dict


def get_history_batch(datas, s_his_dict, time_list, tim_id_dict, his_len=10, layers=1, reverse=False):
    results = []
    labels = []
    len_hist = []

    for data in datas:
        hist, o, lens = get_history(data, s_his_dict, time_list, tim_id_dict, his_len=his_len, layers=layers,
                                    reverse=reverse)
        results.append(hist)
        labels.append(o)
        len_hist.append(lens)

    return results, np.array(labels), len_hist


def make_batch(datas, batch_size):
    for i in range(0, len(datas), batch_size):
        yield datas[i:i + batch_size]



def his_pred(s, t, s_his_dict, time_list, tim_id_dict, his_len=10):
    g_id = []
    sub_g_id = []

    tim = time_list[:int(tim_id_dict[t])]
    for tt in reversed(tim):

        if len(s_his_dict[tt][s]) != 0:
            s_his = s_his_dict[tt][s]
            nodes = np.concatenate([np.array([s]), s_his[:, 1]])
            g_id.append(tt)
            sub_g_id.append(nodes)
            if len(g_id) >= his_len:
                break
    hist = [s, g_id, sub_g_id]

    return hist


def his_pred_batch(s, t, s_his_dict, time_list, tim_id_dict):
    results = []
    for ss in s:
        hist = his_pred(ss, t, s_his_dict, time_list, tim_id_dict)
        results.append(hist)
    return results


def get_order(x, valid=0):
    index_raw = np.argsort(x)
    order_raw = np.sort(x)
    index = index_raw[::-1]
    order = order_raw[::-1]
    if valid == 0:
        order[order == 0] = 1
    else:
        for i, k in enumerate(order):
            if k == 0:
                break
        index = index[:i]
        order = order[:i]
    return order, index


def get_batch_idx(history, graph_dict):
    g_list = []
    target_s = []
    g_idx_dict = {}
    graphs = []

    for data in history:
        g_list += data[1]

    g_idx_list, g_list_idx = np.unique(g_list, return_inverse=True)  # graph unique ids for training in dgl

    for k, v in enumerate(g_idx_list):
        g_idx_dict[v] = k

    nodes_t_list = [set() for _ in range(len(g_idx_list))]

    for data in history:
        if len(data[1]) > 0:
            for g_id, nodes in zip(data[1], data[2]):
                key_g = g_idx_dict[g_id]
                nodes_t_list[key_g] = set(list(nodes)) | nodes_t_list[key_g]

    nodes_idx_list = [dict() for _ in range(len(g_idx_list))]
    g_start = 0
    g_index = []
    # sum_node = 0
    for idx, nodes_t in enumerate(nodes_t_list):
        g_index.append(g_start)
        nodes = list(nodes_t)
        g_start += len(nodes)
        for i, j in enumerate(nodes):
            nodes_idx_list[idx][j] = i
        graphs.append(get_subgraph(graph_dict[g_idx_list[idx]], nodes))

    for data in history:
        s = data[0]
        if len(data[1]) != 0:
            for t in data[1]:
                g_idx = g_idx_dict[t]
                g_loc = g_index[g_idx]
                nodes_idx = nodes_idx_list[g_idx][s]
                s_loc = g_loc + nodes_idx
                target_s.append(int(s_loc))
    return graphs, np.array(target_s), g_list_idx


def get_hidden_split_pad(feat, len_s):
    info = torch.split(feat, len_s)
    dim = int(feat.size(-1))
    hidden = []
    len_new = []
    for x, i in zip(info, len_s):
        if i == 0:
            xx = torch.zeros(1, dim, device=feat.device)
            ii = 1
            hidden.append(xx)
        else:
            ii = i
            hidden.append(x)
        len_new.append(ii)

    return hidden, len_new


def dicttensor(dicts, device='cpu'):
    values = torch.tensor([], device=device)
    for x in dicts.values():
        values = torch.cat((values, x.to(device)), dim=0)
    lens = len(values)

    return values.view(lens, -1)


def get_true_distribution(train_data, num_s):
    true_s = np.zeros(num_s)
    true_o = np.zeros(num_s)
    true_prob_s = None
    true_prob_o = None
    current_t = 0
    for triple in train_data:
        s = triple[0]
        o = triple[2]
        t = triple[3]
        true_s[s] += 1
        true_o[o] += 1
        if current_t != t:

            true_s = true_s / np.sum(true_s)
            true_o = true_o / np.sum(true_o)

            if true_prob_s is None:
                true_prob_s = true_s.reshape(1, num_s)
                true_prob_o = true_o.reshape(1, num_s)
            else:
                true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_s)), axis=0)
                true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_s)), axis=0)

            true_s = np.zeros(num_s)
            true_o = np.zeros(num_s)
            current_t = t

    true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_s)), axis=0)
    true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_s)), axis=0)

    return true_prob_s, true_prob_o


def get_history_for_predict(s, r, t, s_his_dict, time_list, tim_id_dict, his_len=10, layers=1):
    g_id = []
    sub_g_id = set()
    node_id = []
    idx = int(tim_id_dict[t])

    tim = time_list[:idx]
    for tt in tim[::-1]:
        if len(g_id) >= his_len:
            break

        sub_g_id.add(s)
        nodes = []
        if len(s_his_dict[tt][s]) != 0:
            g_id.append(tt)
            for _ in range(layers):
                for sb in list(sub_g_id):
                    if len(s_his_dict[tt][sb]) != 0:
                        s_his = s_his_dict[tt][sb]
                        ob = s_his[:, 1]
                        nodes += ob.tolist()
                sub_g_id = sub_g_id | set(nodes)

            node_id.append(list(sub_g_id))
            sub_g_id = set()

    len_hist = len(g_id)

    hist = [s, g_id[::-1], node_id[::-1], r]
    return hist, len_hist


def get_history_for_predict_batch(ss, rr, t, s_his_dict, time_list, tim_id_dict, his_len=10, layers=1):
    results = []

    len_hist = []
    for s, r, in zip(ss, rr):
        hist, lens = get_history_for_predict(s, r, t, s_his_dict, time_list, tim_id_dict, his_len=his_len,
                                             layers=layers)
        results.append(hist)
        len_hist.append(lens)

    return results, len_hist
