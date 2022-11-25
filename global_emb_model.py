import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from aggregator import RGCNAggregator_global
import utils
import time


class RENet_global(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, dropout=0.2, model=0, seq_len=10, num_k=10, maxpool=1, pretrained=0):
        super(RENet_global, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.model = model
        self.seq_len = seq_len
        self.num_k = num_k
        self.dropout = nn.Dropout(dropout)
        self.pretrained = pretrained
        self.encoder_global = nn.GRU(h_dim, h_dim, batch_first=True)

        self.aggregator = RGCNAggregator_global(h_dim, dropout, in_dim, 1 * self.num_rels, seq_len=self.seq_len)

        self.linear_s = nn.Linear(h_dim, in_dim)
        self.linear_o = nn.Linear(h_dim, in_dim)
        self.global_emb = None
        self.time_id_dict = None
        self.dicts = None
        self.distribution_s = None
        self.distribution_o = None

        self.ent_embeds = nn.Embedding(1 * self.in_dim, h_dim, max_norm=1.0).weight
        self.rel_embed_s = nn.Embedding(1 * self.num_rels, h_dim, max_norm=1.0).weight
        self.rel_embed_o = nn.Embedding(1 * self.num_rels, h_dim, max_norm=1.0).weight

    def forward(self, t_list, reverse=False):
        if reverse == False:
            linear = self.linear_s
            true_prob = self.distribution_s
            rel_embeds = torch.cat((self.rel_embed_s, self.rel_embed_o))
        else:
            linear = self.linear_o
            true_prob = self.distribution_o
            rel_embeds = torch.cat((self.rel_embed_o, self.rel_embed_s))
        graph_dict = self.dicts
        sorted_t = []
        idx = []
        for t in t_list:
            sorted_t.append(int(t))
            idx.append(int(self.time_id_dict[int(t)]))

        packed_input = self.aggregator(sorted_t, self.ent_embeds, rel_embeds, graph_dict, reverse=reverse)

        tt, s_q = self.encoder_global(packed_input)
        s_q = s_q.squeeze()
        s_q = torch.cat((s_q, torch.zeros(len(t_list) - len(s_q), self.h_dim).cuda()), dim=0)
        pred = linear(s_q)
        loss = utils.soft_cross_entropy(pred, true_prob[idx])

        return loss

    def get_graph_emb(self, t_list, graph_dict):
        global_emb = dict()
        times = list(graph_dict.keys())
        time_unit = times[1] - times[0]
        if t_list[0] == 0:
            prev_t = 0
        else:
            prev_t = t_list[0]
        for t in t_list[1:]:
            emb, _, _ = self.predict(t, graph_dict)
            global_emb[prev_t] = emb.detach_()
            prev_t = t

        global_emb[t_list[-1]], _, _ = self.predict(t_list[-1] + int(time_unit), graph_dict)
        global_emb[t_list[-1]].detach_()
        return global_emb

    """
    Prediction function in testing
    """

    def predict(self, t, graph_dict, reverse=False):  # Predict s at time t, so <= t-1 graphs are used.
        if reverse:
            linear = self.linear_o
            rel_embeds = torch.cat((self.rel_embed_o, self.rel_embed_s))
        else:
            linear = self.linear_s
            rel_embeds = torch.cat((self.rel_embed_s, self.rel_embed_o))
        rnn_inp = self.aggregator.predict(t, self.ent_embeds, rel_embeds, graph_dict, reverse=reverse)
        tt, s_q = self.encoder_global(rnn_inp.view(1, -1, self.h_dim))
        sub = linear(s_q)
        prob_sub = torch.softmax(sub.view(-1), dim=0)
        return s_q, sub, prob_sub

    def update_global_emb(self, t, graph_dict):
        pass
