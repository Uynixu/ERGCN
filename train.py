import argparse
import numpy as np
import time
import torch
import os
import utils
from model import ergcn
from sklearn.utils import shuffle
import pickle
from global_emb_model import RENet_global


def train(args):
    initial = torch.cuda.memory_allocated()
    rs = False
    num_e, num_r = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    torch.set_default_dtype(torch.float64)
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        print('using CUDA')
        device = 'cuda:0'
    else:
        print('using cpu')
        device = 'cpu'

    os.makedirs('models', exist_ok=True)
    os.makedirs('models/' + args.dataset, exist_ok=True)
    model_state_file = './models/' + args.dataset + '/model-len-' + str(args.seq_len) + '-method-' + str(
        args.name) + '.pth'

    # demo: YAGO 5000-triples
    if args.dataset == 'ICEWS4':
        train_data, train_time_list = utils.load_quadruples('./data/' + args.dataset, 'train.txt', rs)
        valid_start_id = int(len(train_time_list) * 0.1)  # loc
        valid_start = train_time_list[-valid_start_id]

        all_data = train_data
        train_data = all_data[all_data[:, 3] < valid_start]
        valid_data = all_data[all_data[:, 3] >= valid_start]
        time_list = np.array(train_time_list)
        with open('./data/' + args.dataset + '/train_graph_dict.txt', 'rb') as f:
            graph_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/train_s_his_dict.txt', 'rb') as f:
            his_dict = pickle.load(f)
        with open('./data/' + args.dataset + '/train_o_his_dict.txt', 'rb') as f:
            o_his_dict = pickle.load(f)
        valid_graph_dict = {}
        train_graph_dict = {}
        train_s_his_dict = {}
        train_o_his_dict = {}
        valid_s_his_dict = {}
        valid_o_his_dict = {}
        for t in train_time_list:
            if t >= valid_start:
                valid_graph_dict[t] = graph_dict[t]
                valid_s_his_dict[t] = his_dict[t]
                valid_o_his_dict[t] = o_his_dict[t]
            else:
                train_graph_dict[t] = graph_dict[t]
                train_s_his_dict[t] = his_dict[t]
                train_o_his_dict[t] = o_his_dict[t]
        tim_id_dict = utils.get_tim_id_dict(time_list)
    else:
        train_data, train_time_list = utils.load_quadruples('./data/' + args.dataset, 'train.txt', rs)
        valid_data, valid_time_list = utils.load_quadruples('./data/' + args.dataset, 'valid.txt', rs)

        time_list = np.concatenate((train_time_list, valid_time_list))
        tim_id_dict = utils.get_tim_id_dict(time_list)
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

    graph_dict = {}
    s_dict = {}
    o_dict = {}
    graph_dict.update(train_graph_dict)
    graph_dict.update(valid_graph_dict)
    s_dict.update(train_s_his_dict)
    s_dict.update(valid_s_his_dict)
    o_dict.update(train_o_his_dict)
    o_dict.update(valid_o_his_dict)

    # loading pre-trained global embedding
    model_state_global_file = './premodels/' + args.dataset + '/premodel-' + str(args.h_dim) + '.pth'
    checkpoint_global = torch.load(model_state_global_file, map_location=lambda storage, loc: storage)
    global_emb_dict = checkpoint_global['global_emb']
    # model

    global_models = RENet_global(num_e, args.h_dim, 1 * num_r)
    global_models.load_state_dict(checkpoint_global['model'])
    model = ergcn(num_e=num_e, num_r=num_r, h_dim=args.h_dim, seq_len=args.seq_len, dropout=args.dropout, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)

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
    model.alpha = args.alpha
    model.layers = args.layers
    train_data = shuffle(train_data)
    valid_data = shuffle(valid_data)
    reverse = args.reverse
    print('starting training...')
    alpha = 1.0
    epoch = 0
    best_mrr = 0
    best_hit = 0
    best_epoch = 0
    es = 0
    model_usage = torch.cuda.memory_allocated()
    # print('the usage of GPU is {:.4f} GB'.format((model_usage - initial) / 1024 / 1024 / 1024))
    while True:
        initial_train = torch.cuda.memory_allocated()
        train_usage = 0
        model.train()
        if epoch == args.max_epochs:
            break
        epoch += 1
        loss_epoch = 0
        t0 = time.time()
        for batch_data in utils.make_batch(train_data, args.batch_size):
            optimizer.zero_grad()
            o_true = np.empty((0, 4))
            loss_s = model(batch_data, False)
            loss_o = model(batch_data, True)
            Loss = loss_s + loss_o
            Loss.backward()
            train_usage += torch.cuda.max_memory_allocated() - initial_train
            loss_epoch += Loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        training_usage = train_usage
        t1 = time.time()
        print(
            'Epoch {:04d} | loss {:.4f}  | time {:.4f} s | GPU {:.4f}'.format(
                epoch,
                loss_epoch / (len(train_data) / args.batch_size),
                t1 - t0,
                training_usage / 1024 / 1024 / 1024))

        if epoch >= int(args.max_epochs * 0.2):
            t2 = time.time()
            model.eval()
            total_loss = 0
            o_list = torch.tensor([])
            with torch.no_grad():
                model.dicts.update(graph_dict)
                if reverse:
                    history_dict = o_dict
                else:
                    history_dict = s_dict
                for batch_valid in utils.make_batch(valid_data, args.batch_size):
                    o_pred, o_labels, loss = model.predict(batch_valid, history_dict, reverse=reverse)
                    o_list = torch.cat((o_list, o_pred.cpu()))
                    o_true = np.concatenate((o_true, o_labels))
                    total_loss += loss.item()
                ranks = model.evaluate_filter(o_true, o_list)
                total_ranks = ranks
            t3 = time.time()
            mrr = np.mean(1 / total_ranks)
            hits = []
            for hit in [1, 3, 10]:
                avg_count = np.mean((total_ranks <= hit))
                hits.append(avg_count)
                print("valid Hits @ {}: {:.6f}".format(hit, avg_count))
            print("valid MRR : {:.6f}".format(mrr))
            print("valid Loss: {:.6f}".format(total_loss / (len(valid_data) / args.batch_size)))
            print('time cost: {:.4f} s'.format(t3 - t2))
            # early stop
            if mrr > best_mrr:
                best_epoch = epoch
                es = 0
                if hits[1] > best_hit:
                    torch.save({'model': model.state_dict(), 'epoch': args.max_epochs, 'best_epoch': best_epoch,
                                'valid_loss': total_loss / (len(valid_data)),
                                'hits': hits, 'mrr': mrr, 'seq_len': args.seq_len},
                               model_state_file)
                best_mrr = mrr
                best_hit = hits[1]
            else:
                es += 1
                if es > 10:
                    print('Early stopping with best validation is at {}/{}'.format(best_epoch, args.max_epochs))
                    break
    print('best validation is at: {}/{}'.format(best_epoch, args.max_epochs))
    print('training done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ERGCN')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--h_dim', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-d', '--dataset', type=str, default='YAGO')
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--reverse', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--name', type=int, default=1)
    parser.add_argument('--type', type=str, default='all')
    args = parser.parse_args()
    print(args)
    train(args)
