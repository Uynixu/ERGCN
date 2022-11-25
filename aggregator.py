from RGCN import RGCNBlockLayer as RGCNLayers
from GCN import ERGCNLayer
import torch.nn.functional as F
import torch
import dgl
import torch.nn as nn
import utils
import numpy as np


class RGCNaggregator(nn.Module):
    def __init__(self, h_dim, num_e, num_r, num_bases=100, device='cuda:0', order=1):
        super(RGCNaggregator, self).__init__()
        self.h_dim = h_dim
        self.device = device
        self.num_rels = num_r
        self.num_e = num_e
        self.order = order
        self.fc = nn.Linear(2 * h_dim, h_dim, bias=False)
        self.layer1 = RGCNLayers(self.h_dim, self.h_dim, 2 * self.num_rels, num_bases=num_bases, activation=F.relu)
        self.layer2 = ERGCNLayer(self.h_dim, self.h_dim, 2 * self.num_rels, 1)
        self.layer3 = ERGCNLayer(self.h_dim, self.h_dim, 2 * self.num_rels, 1)
        self.layer = self.layer2
        self.layers = ['h', 'f', 's', 'r', 'fr', 'fv', 'sx']

    def forward(self, history, ent_embeds, rel_embeds, graph_dict, global_emb_dict, reverse=False):
        e_s, e_r, h_s, g_t, len_s, data_idx = self.get_features(history, ent_embeds, rel_embeds, graph_dict,
                                                                global_emb_dict,
                                                                reverse)
        packed_s = []
        packed_r = []
        for h, g, i in zip(h_s, g_t, len_s):
            inp = torch.cat((h, g), dim=-1)

            packed_s.append(inp)
            packed_r.append(inp)
        packed_r = nn.utils.rnn.pad_sequence(packed_r, batch_first=True)
        packed_s = nn.utils.rnn.pad_sequence(packed_s, batch_first=True)
        packed_s = nn.utils.rnn.pack_padded_sequence(packed_s, len_s, batch_first=True, enforce_sorted=False)
        packed_r = nn.utils.rnn.pack_padded_sequence(packed_r, len_s, batch_first=True, enforce_sorted=False)
        return packed_s, packed_r, data_idx

    def get_features(self, history, ent_embeds, rel_embeds, graph_dict, global_emb_dict, reverse=False):
        # history: [s,[g_t],[neighbor nodes in each t],r]
        # output: es er hs g len-s
        len_s = []
        s_list = []
        r_list = []
        data_idx = []
        history_sorted = []

        global_emb = utils.dicttensor(global_emb_dict, device=self.device)

        for i, data in enumerate(history):
            data_idx.append(i)
            s_list.append(int(data[0]))
            r_list.append(int(data[3]))
            len_s.append(len(data[1]))
            history_sorted.append(data)

        s_list = np.array(s_list)
        r_list = np.array(r_list)

        graphs, target_s, g_list_idx = utils.get_batch_idx(history_sorted, graph_dict)
        graphs = [x.to('cpu') for x in graphs]
        if len(graphs) > 0:
            batched_graph = dgl.batch(graphs)
            batched_graph = batched_graph.to(self.device)
            batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id'].long()].view(-1, self.h_dim)
            if reverse:
                batched_graph.edata['e'] = rel_embeds[batched_graph.edata['rel_o'].long()].view(-1, self.h_dim)
            else:
                batched_graph.edata['e'] = rel_embeds[batched_graph.edata['rel_s'].long()].view(-1, self.h_dim)

            self.layer(batched_graph, reverse)

            graph_info = torch.empty(len(batched_graph.ndata['h']), 0, device=self.device)
            for x in range(self.order):
                key = self.layers[x]
                feat = batched_graph.ndata[key]
                graph_info = torch.cat((graph_info, feat), dim=-1)
                batched_graph.ndata.pop(key)
            batched_graph.edata.pop('e')
            graph_info = graph_info[target_s]
            g_emb = global_emb[g_list_idx]
        else:
            k = self.order
            graph_info = torch.zeros(len(history), k * self.h_dim, device=self.device)
            g_emb = torch.zeros(len(history), k * self.h_dim, device=self.device)
            len_s = [1] * len(history)
        ###
        hidden_s, len_split = utils.get_hidden_split_pad(graph_info, len_s)
        g_split, _ = utils.get_hidden_split_pad(g_emb, len_s)
        e_s = ent_embeds[s_list]
        e_r = rel_embeds[r_list]

        return e_s, e_r, hidden_s, g_split, len_split, data_idx


########################################################################################################

class RGCNAggregator_global(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, seq_len=10, maxpool=0):
        super(RGCNAggregator_global, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.num_heads = 1
        self.maxpool = maxpool
        self.device = 'cuda:0'

        self.layer1 = RGCNLayers(self.h_dim, self.h_dim, 2 * self.num_rels)
        self.layer2 = ERGCNLayer(self.h_dim, self.h_dim, 2 * self.num_rels, self.num_heads)
        self.layer3 = ERGCNLayer(self.h_dim, self.h_dim, 2 * self.num_rels, self.num_heads)
        self.layer = self.layer2

    def forward(self, t_list, ent_embeds, rel_embeds, graph_dict, reverse=False):
        times = np.sort(list(graph_dict.keys()))
        time_list = []

        len_non_zero = []

        for i, tim in enumerate(t_list):
            if i == 0:
                time_list.append(torch.LongTensor(times[:1]))
                len_non_zero.append(1)

            elif i <= self.seq_len:
                time_list.append(torch.LongTensor(times[:i]))
                len_non_zero.append(i)
            else:
                time_list.append(torch.LongTensor(times[i - self.seq_len:i]))
                len_non_zero.append(self.seq_len)

        unique_t = torch.unique(torch.cat(time_list))
        time_to_idx = dict()
        g_list = []
        idx = 0
        for tim in unique_t:
            time_to_idx[tim.item()] = idx
            idx += 1
            g_list.append(graph_dict[tim.item()])

        batched_graph = dgl.batch(g_list)
        batched_graph = batched_graph.to(self.device)
        ent_embeds = ent_embeds.cuda()
        batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id']].view(-1, self.h_dim).cuda()
        if reverse == False:
            batched_graph.edata['e'] = rel_embeds[batched_graph.edata['rel_s'].long()].view(-1, self.h_dim)
        else:
            batched_graph.edata['e'] = rel_embeds[batched_graph.edata['rel_o'].long()].view(-1, self.h_dim)

        self.layer(batched_graph, reverse)
        self.layer2(batched_graph, reverse)

        global_info = dgl.max_nodes(batched_graph, 'h')
        batched_graph.ndata.pop('h')

        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, self.h_dim).cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                embed_seq_tensor[i, j, :] = global_info[time_to_idx[t.item()]]

        embed_seq_tensor = self.dropout(embed_seq_tensor)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               len_non_zero,
                                                               batch_first=True, enforce_sorted=False)

        return packed_input

    def predict(self, t, ent_embeds, rel_embeds, graph_dict, reverse=False):

        times = np.sort(list(graph_dict.keys()))
        id = 0
        for tt in times:
            if tt >= t:
                break
            id += 1

        if self.seq_len <= id:
            timess = torch.LongTensor(times[id - self.seq_len:id])
        else:
            timess = torch.LongTensor(times[:id])

        g_list = []

        for tim in timess:
            graph_dict[tim.item()] = graph_dict[tim.item()].to('cpu')
            g_list.append(graph_dict[tim.item()])

        batched_graph = dgl.batch(g_list)
        batched_graph = batched_graph.to(self.device)
        ent_embeds = ent_embeds.cuda()
        batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id']].view(-1, ent_embeds.shape[1])
        if reverse:

            batched_graph.edata['e'] = rel_embeds[batched_graph.edata['rel_s'].long()].view(-1, self.h_dim)
        else:
            batched_graph.edata['e'] = rel_embeds[batched_graph.edata['rel_o'].long()].view(-1, self.h_dim)
        self.layer(batched_graph, reverse)
        self.layer2(batched_graph, reverse)
        global_info = dgl.max_nodes(batched_graph, 'h')
        batched_graph.ndata.pop('h')

        return global_info
