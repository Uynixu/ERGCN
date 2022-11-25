import argparse
import numpy as np
import torch
import utils
import pickle
import os
from model import ergcn
from global_emb_model import RENet_global
import time


def test(args):
    rs = False
    num_e, num_r = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    torch.set_default_dtype(torch.float64)
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_state_file = './models/' + args.dataset + '/model-len-' + str(args.seq_len) + '-method-' + str(
        args.name) + '.pth'
    graph_dict = {}
    s_dict = {}
    o_dict = {}

    if use_cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
        torch.cuda.manual_seed_all(999)

    if args.dataset == 'ICEWS4':
        train_data, train_time_list = utils.load_quadruples('./data/' + args.dataset, 'train.txt', rs)
        test_data, test_time_list = utils.load_quadruples('./data/' + args.dataset, 'test.txt', rs)
        with open('./data/' + args.dataset + '/train_graph_dict.txt', 'rb') as f:
            train_graph_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/train_s_his_dict.txt', 'rb') as f:
            s_his_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/train_o_his_dict.txt', 'rb') as f:
            o_his_dict = pickle.load(f)
        s_dict.update(s_his_dict)
        o_dict.update(o_his_dict)
        graph_dict.update(train_graph_dict)
        time_list = np.concatenate((train_time_list, test_time_list))
    else:
        train_data, train_time_list = utils.load_quadruples('./data/' + args.dataset, 'train.txt', rs)
        valid_data, valid_time_list = utils.load_quadruples('./data/' + args.dataset, 'valid.txt', rs)
        test_data, test_time_list = utils.load_quadruples('./data/' + args.dataset, 'test.txt', rs)

        with open('./data/' + args.dataset + '/train_graph_dict.txt', 'rb') as f:
            train_graph_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/train_s_his_dict.txt', 'rb') as f:
            train_s_his_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/train_o_his_dict.txt', 'rb') as f:
            train_o_his_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/valid_graph_dict.txt', 'rb') as f:
            valid_graph_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/valid_s_his_dict.txt', 'rb') as f:
            valid_s_his_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/valid_o_his_dict.txt', 'rb') as f:
            valid_o_his_dict = pickle.load(f)

        s_dict.update(train_s_his_dict)
        s_dict.update(valid_s_his_dict)

        o_dict.update(valid_o_his_dict)
        o_dict.update(train_o_his_dict)

        graph_dict.update(train_graph_dict)
        graph_dict.update(valid_graph_dict)

        time_list = np.concatenate((train_time_list, valid_time_list, test_time_list))
    with open('./data/' + args.dataset + '/test_graph_dict.txt', 'rb') as f:
        test_graph_dict = pickle.load(f)
    with open('./data/' + args.dataset + '/test_s_his_dict.txt', 'rb') as f:
        test_s_his_dict = pickle.load(f)
    with open('./data/' + args.dataset + '/test_o_his_dict.txt', 'rb') as f:
        test_o_his_dict = pickle.load(f)
    s_dict.update(test_s_his_dict)
    o_dict.update(test_o_his_dict)
    graph_dict.update(test_graph_dict)
    tim_id_dict = utils.get_tim_id_dict(time_list)

    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model_state_global_file = './premodels/' + args.dataset + '/premodel-' + str(args.h_dim) + '.pth'
    checkpoint_global = torch.load(model_state_global_file, map_location=lambda storage, loc: storage)
    global_models = RENet_global(num_e, args.h_dim, 1 * num_r)
    global_models.load_state_dict(checkpoint_global['model'])
    global_emb_dict = checkpoint_global['global_emb']

    reverse = args.reverse
    model = ergcn(num_e=num_e,
                  num_r=num_r,
                  h_dim=args.h_dim,
                  seq_len=args.seq_len,
                  dropout=args.dropout,
                  device=device)

    model.load_state_dict(checkpoint['model'])
    print('starting test....')
    if use_cuda:
        model.cuda()
        global_models.cuda()
    model.global_emb_dict = global_emb_dict
    model.dicts = graph_dict
    model.s_dict = s_dict
    model.o_dict = o_dict
    model.time_list = time_list
    model.tim_id_dict = tim_id_dict
    model.device = device
    print(args.dataset, 'using best epoch: {}/{}'.format(checkpoint['best_epoch'], checkpoint['epoch']))
    model.eval()
    total_loss = 0

    o_list = torch.tensor([])
    o_true = np.empty((0, 4))
    t1 = time.time()
    with torch.no_grad():
        if reverse:
            history_dict = model.o_dict
            history_dict.update(test_o_his_dict)
        else:
            history_dict = model.s_dict
            history_dict.update(test_s_his_dict)
        for batch_valid in utils.make_batch(test_data, args.batch_size):
            o_pred, o_labels, loss = model.predict(batch_valid, history_dict, reverse=reverse)
            o_list = torch.cat((o_list, o_pred.cpu()))
            o_true = np.concatenate((o_true, o_labels))
            total_loss += loss.item()
        if args.filter == 1:
            eval = model.evaluate_filter
        else:
            eval = model.evaluate
        ranks = eval(o_true, o_list)
        total_ranks = ranks

    mrr = np.mean(1.0 / total_ranks)
    t2 = time.time()
    hits = []

    for hit in [1, 3, 10]:
        avg_count = np.mean((total_ranks <= hit))
        hits.append(avg_count)
        print("test Hits @ {}: {:.6f}".format(hit, avg_count))
    print("test MRR : {:.6f}".format(mrr))
    print("test Loss: {:.6f}".format(loss / (len(test_data) / args.batch_size)))
    print("time cost:{:.1f}".format(t2 - t1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ERGCN')
    parser.add_argument('--h_dim', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-d', '--dataset', type=str, default='YAGO')
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--reverse', type=bool, default=False)
    parser.add_argument('--type', type=str, default='all')
    parser.add_argument('--name', type=int, default=1)
    parser.add_argument('--filter', type=int, default=0)
    parser.add_argument('--split', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    test(args)
