import torch
import torch.nn as nn
import dgl.function as fn
import dgl
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=True, activation=None,
                 self_loop=False, dropout=0.0):
        super(GCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g, reverse):
        raise NotImplementedError

    def forward(self, g, reverse):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g, reverse)
        # self.apply_func(g.dstdata['h'])
        # apply bias and activation

        node_repr = g.ndata['h']
        # self.apply_func(node_repr) #
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        return g


class RGCNBlockLayer(GCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=100, bias=None,
                 activation=None, self_loop=False):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop
                                             )
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat

        if in_feat // self.num_bases == 0:
            self.submat_in = 1
            self.submat_out = 1
        else:
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases
        # assuming in_feat and out_feat are both divisible by num_bases
        # if self.num_rels == 2:
        #     self.in_feat = in_feat
        #     self.weight = nn.Parameter(torch.Tensor(
        #         self.num_rels, in_feat, out_feat))
        # else:
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges, reverse):
        if reverse:
            weight = self.weight.index_select(0, edges.data['rel_o']).view(
                -1, self.submat_in, self.submat_out)
        else:
            weight = self.weight.index_select(0, edges.data['rel_s']).view(
                -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        # msg = msg + edges.dst['h']
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}  ####

    def propagate(self, g, reverse):
        g.update_all(lambda x: self.msg_func(x, reverse), fn.sum(msg='msg', out='h'))
        g.apply_nodes(lambda nodes: self.apply_func(nodes))  #### add


class ERGCNLayer(GCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_heads, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(ERGCNLayer, self).__init__(in_feat, out_feat, bias=bias,
                                         activation=activation, self_loop=self_loop,
                                         dropout=dropout)
        self.num_rels = num_rels
        self.num_heads = num_heads
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        self.attention = nn.Parameter(torch.Tensor(
            self.num_rels, self.out_feat))
        nn.init.xavier_uniform_(self.attention, gain=nn.init.calculate_gain('relu'))
        self.activation = activation
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)

        self.fc = nn.Linear(
            in_feat, out_feat * num_heads, bias=bias)
        nn.init.xavier_uniform_(self.attention, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges, reverse):
        if reverse:
            weight = self.weight.index_select(0, edges.data['rel_o']).view(
                -1, self.out_feat)
            attn = self.attention.index_select(0, edges.data['rel_o']).view(
                -1, self.out_feat)
        else:
            weight = self.weight.index_select(0, edges.data['rel_s']).view(
                -1, self.out_feat)
            attn = self.attention.index_select(0, edges.data['rel_s']).view(
                -1, self.out_feat)

        edge = edges.data['e'].view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)
        feat = torch.mul(node, weight) + torch.mul(edge, attn)

        return {'msg': feat}

    def reduce_func(self, nodes):
        h = nodes.data['h'] + torch.sum(nodes.mailbox['msg'], dim=1)
        return {'h': h}

    def propagate(self, g, reverse):
        g.update_all(lambda x: self.msg_func(x, reverse), self.reduce_func)

