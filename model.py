import torch
import utils
from aggregator import RGCNaggregator
import numpy as np
import torch.nn as nn


def get_history_t(triple, num_e, reverse=False):
    valid_s_his_cache = [[] for _ in range(num_e)]
    for data in triple:
        if reverse:
            s = data[2]
            o = data[0]
        else:
            s = data[0]
            o = data[2]
        r = data[1]
        if len(valid_s_his_cache[s]) == 0:
            valid_s_his_cache[s] = np.array([[r, o]])
        else:
            valid_s_his_cache[s] = np.concatenate((valid_s_his_cache[s], [[r, o]]), axis=0)
    return valid_s_his_cache


class ergcn(nn.Module):
    def __init__(self, num_e, num_r, h_dim, seq_len, dropout=0.0, device='cuda:0', pretrained=False):
        super(ergcn, self).__init__()
        self.h_dim = h_dim
        self.seq_len = seq_len
        self.num_e = num_e
        self.num_rels = num_r
        self.num_r = num_r
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.aggregator = RGCNaggregator(h_dim=self.h_dim,
                                         num_e=self.num_e,
                                         num_r=1 * self.num_rels,
                                         num_bases=100,
                                         device=self.device)

        self.rel_embed_s = nn.Embedding(1 * self.num_rels, h_dim, max_norm=1.0).weight
        self.rel_embed_o = nn.Embedding(1 * self.num_rels, h_dim, max_norm=1.0).weight
        self.ent_embeds = nn.Embedding(self.num_e, h_dim, max_norm=1.0).weight
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.s_dict = {}
        self.o_dict = {}
        self.dicts = {}
        self.s_distribution = None
        self.o_distribution = None
        self.encoder_s = nn.GRU(2 * h_dim, 1 * h_dim, batch_first=True)
        self.encoder_o = nn.GRU(2 * h_dim, 1 * h_dim, batch_first=True)
        self.encoder_r = nn.GRU(2 * h_dim, 1 * h_dim, batch_first=True)
        self.linear_s = nn.Linear(3 * self.h_dim, num_e)
        self.linear_o = nn.Linear(3 * self.h_dim, num_e)
        self.linear_r = nn.Linear(2 * self.h_dim, num_r)
        self.fc = nn.Linear(2 * self.h_dim, h_dim)
        self.time_list = None
        self.tim_id_dict = None
        self.global_model = None
        self.layers = 1
        self.global_emb_dict = None
        self.alpha = 0.1
        self.k = 10

        self.classifier = None

    def forward(self, datas, reverse=False):
        if reverse:
            hist_dict = self.o_dict
            rel_embeds = torch.cat((self.rel_embed_o, self.rel_embed_s))
            es = self.ent_embeds[datas[:, 2]]
            encoder = self.encoder_o
            decoder = self.linear_o
        else:
            hist_dict = self.s_dict
            rel_embeds = torch.cat((self.rel_embed_s, self.rel_embed_o))
            es = self.ent_embeds[datas[:, 0]]
            encoder = self.encoder_s
            decoder = self.linear_s
        er = rel_embeds[datas[:, 1]]
        history, labels, lens_raw = utils.get_history_batch(datas, hist_dict, self.time_list, self.tim_id_dict,
                                                            his_len=self.seq_len, layers=self.layers, reverse=reverse)
        inp_s, inp_r, idx = self.aggregator(history, self.ent_embeds, rel_embeds, self.dicts, self.global_emb_dict,
                                            reverse=reverse)
        inp_es = es[idx]
        inp_er = er[idx]
        _, hidden_inp_s = encoder(inp_s)

        inp_s = self.dropout(torch.cat((inp_es, inp_er, hidden_inp_s.squeeze()), dim=1))
        inp_r = self.dropout(torch.cat((inp_es, hidden_inp_s.squeeze()), dim=1))
        o_pred = decoder(inp_s)
        r_pred = self.linear_r(inp_r)
        o_labels = labels[idx]
        r_labels = datas[:, 1][idx]

        loss_s = self.criterion(o_pred, torch.LongTensor(o_labels).to(self.device))
        loss_r_s = self.criterion(r_pred, torch.LongTensor(r_labels).to(self.device))
        loss = loss_s + self.alpha * loss_r_s
        return loss

    def evaluate(self, data, o_pred_list, reverse=False):
        if reverse:
            o_labels = data[:, 0]
        else:
            o_labels = data[:, 2]
        rank_list = []
        o_pred = self.softmax(o_pred_list)
        o_pred = o_pred.cpu().numpy()

        for o_p, o in zip(o_pred, o_labels):
            o = int(o)
            ob_pred_comp1 = np.sum((o_p > o_p[o]))
            ob_pred_comp2 = np.sum((o_p == o_p[o]))

            rank_o = ob_pred_comp1 + ((ob_pred_comp2 - 1) / 2) + 1
            rank_list.append(rank_o)
        return np.array(rank_list)

    def evaluate_filter(self, data, o_pred_list, reverse=False):
        if reverse:
            k = 0
        else:
            k = 2
        rank_list = []
        o_pred = self.softmax(o_pred_list)
        o_pred = o_pred.cpu().numpy()

        for o_p, quad in zip(o_pred, data):
            t = quad[3]
            o = quad[2]
            r = quad[1]
            s = quad[0]
            o = int(o)
            ground = o_p[o]
            s_id = data[data[:, 0] == s]
            r_id = s_id[s_id[:, 1] == r]
            t_id = r_id[r_id[:, 3] == t]
            o_idx = t_id[:, k]
            o_p[o_idx.astype(int)] = 0
            o_p[o] = ground

            ob_pred_comp1 = np.sum((o_p > ground))
            ob_pred_comp2 = np.sum((o_p == ground))

            rank_o = ob_pred_comp1 + ((ob_pred_comp2 - 1) / 2) + 1
            rank_list.append(rank_o)
        return np.array(rank_list)

    def predict(self, datas, history_dict, reverse=False):
        if reverse:
            hist_dict = history_dict
            rel_embeds = torch.cat((self.rel_embed_o, self.rel_embed_s))
            es = self.ent_embeds[datas[:, 2]]
            encoder = self.encoder_o
            decoder = self.linear_o
            k = 0
        else:
            hist_dict = history_dict
            rel_embeds = torch.cat((self.rel_embed_s, self.rel_embed_o))
            es = self.ent_embeds[datas[:, 0]]
            encoder = self.encoder_s
            decoder = self.linear_s
            k = 2
        er = rel_embeds[datas[:, 1]]

        history, labels, lens_raw = utils.get_history_batch(datas, hist_dict, self.time_list, self.tim_id_dict,
                                                            his_len=self.seq_len, layers=self.layers,
                                                            reverse=reverse)
        inp_s, inp_r, idx = self.aggregator(history, self.ent_embeds, rel_embeds, self.dicts, self.global_emb_dict,
                                            reverse=reverse)
        _, hidden_inp_s = encoder(inp_s)
        inp_es = es
        inp_er = er
        inp_s = self.dropout(torch.cat((inp_es, inp_er, hidden_inp_s.view(-1, self.h_dim)), dim=1))
        o_pred = decoder(inp_s)
        o_labels = datas

        loss_s = self.criterion(o_pred, torch.LongTensor(o_labels[:, k]).to(self.device))
        loss = loss_s

        return o_pred, o_labels, loss
