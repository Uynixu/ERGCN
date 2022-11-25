import argparse
import numpy as np
import time
import torch
import os
import utils
from global_emb_model import RENet_global
import pickle


def train(args):

    num_e, num_r = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    graph_dict = {}
    if os.path.exists('./data/' + args.dataset + '/valid.txt'):
        train_data, train_time_list = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
        valid_data, valid_time_list = utils.load_quadruples('./data/' + args.dataset, 'valid.txt')
        test_data, test_time_list = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        time_list = np.concatenate((train_time_list, valid_time_list, test_time_list))

        with open('./data/' + args.dataset + '/train_graph_dict.txt', 'rb') as f:
            train_graph_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/valid_graph_dict.txt', 'rb') as f:
            valid_graph_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/test_graph_dict.txt', 'rb') as f:
            test_graph_dict = pickle.load(f)
        graph_dict.update(train_graph_dict)
        graph_dict.update(valid_graph_dict)
        graph_dict.update(test_graph_dict)
        s_dist, o_dist = utils.get_true_distribution(np.concatenate((train_data, valid_data)), num_e)

    else:
        train_data, time_list = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
        valid_start_id = int(len(time_list) * 0.9)  # loc
        train_time_list = time_list[:valid_start_id]
        test_data, test_time_list = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        with open('./data/' + args.dataset + '/train_graph_dict.txt', 'rb') as f:
            train_graph_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/test_graph_dict.txt', 'rb') as f:
            test_graph_dict = pickle.load(f)
        graph_dict.update(train_graph_dict)
        graph_dict.update(test_graph_dict)
        time_list = np.concatenate((time_list, test_time_list))
        s_dist, o_dist = utils.get_true_distribution(train_data, num_e)

    tim_id_dict = utils.get_tim_id_dict(time_list)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    torch.set_default_dtype(torch.float64)
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_time_list = torch.from_numpy(train_time_list)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        print('using CUDA')
        device = 'cuda:0'
        train_time_list = train_time_list.cuda()
        s_dist = torch.from_numpy(s_dist).cuda()
        o_dist = torch.from_numpy(o_dist).cuda()
    else:
        print('using cpu')
        s_dist = torch.from_numpy(s_dist)
        o_dist = torch.from_numpy(o_dist)
        device = 'cpu'
    os.makedirs('premodels', exist_ok=True)
    os.makedirs('premodels/' + args.dataset, exist_ok=True)
    model_state_file = './premodels/' + args.dataset + '/premodel-' + str(args.h_dim) + '.pth'

    model = RENet_global(num_e, args.h_dim, num_r)
    model.time_id_dict = tim_id_dict
    model.dicts = graph_dict
    model.distribution_s = s_dist
    model.distribution_o = o_dist
    if use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
    print('starting training...')
    epoch = 0
    best_result = 9999999999
    best_epoch = 0

    while True:
        model.train()
        if epoch == args.max_epochs:
            break
        epoch += 1
        loss_epoch = 0
        t0 = time.time()
        optimizer.zero_grad()

        for times in utils.make_batch(train_time_list, args.batch_size):
            Loss_s = model(times, reverse=False)
            Loss_o = model(times, reverse=True)
            Loss = Loss_s + Loss_o
            Loss.backward()
            loss_epoch += Loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        t1 = time.time()
        if epoch % 10 == 0:
            print('epoch {}/{}: {:.6f}/ time cost: {:.1f}'.format(epoch, args.max_epochs,
                                                                  loss_epoch / len(time_list) * args.batch_size,
                                                                  t1 - t0))
        if epoch >= int(args.max_epochs * 0.9):
            if loss_epoch < best_result:
                best_epoch = epoch
                with torch.no_grad():
                    model.eval()
                    global_emb = model.get_graph_emb(time_list, graph_dict)
                torch.save({'model': model.state_dict(), 'best_epoch': best_epoch,
                            'global_emb': global_emb},
                           model_state_file)
                best_result = loss_epoch
    print('Best result is {:.6f} at {}/{}'.format(best_result / len(time_list) * args.batch_size, best_epoch,
                                                  args.max_epochs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--h_dim', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-d', '--dataset', type=str, default='YAGO')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--maxpool', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=96)
    args = parser.parse_args()
    print(args)
    train(args)
