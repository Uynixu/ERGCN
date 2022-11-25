import torch
import torch.nn as nn
import dgl.function as fn
import dgl
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
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


class RGCNBlockLayer(RGCNLayer):
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
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}  ####

    def propagate(self, g, reverse):
        g.update_all(lambda x: self.msg_func(x, reverse), fn.sum(msg='msg', out='h'))
        g.apply_nodes(lambda nodes: self.apply_func(nodes))  #### add


class RGCNAttLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_heads, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNAttLayer, self).__init__(in_feat, out_feat, bias,
                                           activation, self_loop=self_loop,
                                           dropout=dropout)
        self.num_rels = num_rels
        self.num_heads = num_heads

        self.in_feat = in_feat
        self.out_feat = out_feat

        # assuming in_feat and out_feat are both divisible by num_bases
        # if self.num_rels == 2:
        #     self.in_feat = in_feat
        #     self.weight = nn.Parameter(torch.Tensor(
        #         self.num_rels, in_feat, out_feat))
        # else:

        self.activation = activation
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.attn = nn.Parameter(torch.FloatTensor(size=(self.num_rels, num_heads, out_feat)))  #
        self.fc = nn.Linear(
            in_feat, out_feat * num_heads, bias=bias)
        nn.init.xavier_uniform_(self.attn, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges, reverse):

        if reverse:
            weight = self.attn.index_select(0, edges.data['rel_o']).view(
                -1, self.num_heads, self.out_feat)
        else:
            weight = self.attn.index_select(0, edges.data['rel_s']).view(
                -1, self.num_heads, self.out_feat)

        edge = edges.data['e']
        node = edges.src['h'].view(-1, self.out_feat)
        feat = self.leaky_relu(self.attn_drop(weight * self.fc(edge).view(-1, self.num_heads, self.out_feat))).view(
            -1, self.num_heads, self.out_feat)
        # softmax = torch.softmax(feat, dim=1)
        # feat = torch.sum(softmax * feat, dim=1)
        feat = torch.sum(feat, dim=1).view(-1, self.out_feat)
        return {'msg': node + feat}

    def propagate(self, g, reverse):
        g.update_all(lambda x: self.msg_func(x, reverse), fn.sum('msg', 'h'))


class RGCNedgeLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_heads, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNedgeLayer, self).__init__(in_feat, out_feat, bias,
                                            activation, self_loop=self_loop,
                                            dropout=dropout)
        self.num_rels = num_rels
        self.num_heads = num_heads
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))

    def msg_func(self, edges, reverse):
        edge = edges.data['e'].view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)
        if reverse:
            weight = self.weight.index_select(0, edges.data['rel_o']).view(-1,
                                                                           self.out_feat)
        else:
            weight = self.weight.index_select(0, edges.data['rel_s']).view(-1,
                                                                           self.out_feat)
        feat = torch.mul(weight, edge).view(-1, self.out_feat)
        return {'msg': feat}

    def propagate(self, g, reverse):
        g.update_all(lambda x: self.msg_func(x, reverse), fn.sum('msg', 'h'))


class RGCNAtt2Layer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_heads=3, bias=None,
                 activation=None, self_loop=False, dropout=0.2, num_bases=100):
        super(RGCNAtt2Layer, self).__init__(in_feat, out_feat, bias,
                                            activation, self_loop=self_loop,
                                            dropout=dropout)
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

        self.activation = activation
        self.num_heads = num_heads
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.attn = nn.Parameter(torch.FloatTensor(size=(self.num_rels, num_heads, out_feat)))
        self.fc = nn.Linear(
            in_feat, out_feat * num_heads, bias=bias)
        nn.init.xavier_uniform_(self.attn, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges, reverse):
        if reverse:
            weight = self.weight.index_select(0, edges.data['rel_o']).view(
                -1, self.submat_in, self.submat_out)
            attn = self.attn.index_select(0, edges.data['rel_o']).view(
                -1, self.num_heads, self.out_feat)
        else:
            weight = self.weight.index_select(0, edges.data['rel_s']).view(
                -1, self.submat_in, self.submat_out)
            attn = self.attn.index_select(0, edges.data['rel_s']).view(
                -1, self.num_heads, self.out_feat)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        edge = edges.data['e']
        feat = self.leaky_relu(self.attn_drop(attn * self.fc(edge).view(-1, self.num_heads, self.out_feat))).view(
            -1, self.num_heads, self.out_feat)
        feat = torch.sum(feat, dim=1).view(-1, self.out_feat)

        return {'msg': msg + feat}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}  ####

    def propagate(self, g, reverse):
        g.update_all(lambda x: self.msg_func(x, reverse), fn.sum(msg='msg', out='h'))
        g.apply_nodes(lambda nodes: self.apply_func(nodes))  #### add


class ERGCNLayer(RGCNLayer):
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

    def propagate(self, g, reverse):
        g.update_all(lambda x: self.msg_func(x, reverse), fn.sum('msg', 'h'))


class mlplayer(nn.Module):
    def __init__(self, input, hidden, output):
        super(mlplayer, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EGCNlayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_heads, bias=None,
                 activation=None, self_loop=False, dropout=0.0, num_bases=100):
        super(EGCNlayer, self).__init__(in_feat, out_feat, bias,
                                        activation, self_loop=self_loop,
                                        dropout=dropout)

        #########
        self.num_rels = num_rels
        self.num_heads = num_heads
        self.num_bases = num_bases
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.mlp = mlplayer(2 * in_feat, int(1.5 * in_feat), out_feat)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges, reverse):

        if reverse:
            weight = self.weight.index_select(0, edges.data['rel_o']).view(
                -1, self.out_feat)

        else:
            weight = self.weight.index_select(0, edges.data['rel_s']).view(
                -1, self.out_feat)

        node = edges.src['h'].view(-1, self.out_feat)
        edge = edges.data['e'].view(-1, self.out_feat)
        feat = torch.cat((node, edge), dim=1)
        feat = self.mlp(feat.view(-1, 2 * self.in_feat))
        msg = torch.mul(feat, weight).view(-1, self.out_feat)

        return {'msg': msg}

    def propagate(self, g, reverse):
        g.update_all(lambda x: self.msg_func(x, reverse), fn.sum('msg', 'h'))
