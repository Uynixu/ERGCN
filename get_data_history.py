import numpy as np
import pickle
import argparse
import utils
import os


def main(args):
    path = './data/' + args.dataset
    if args.sorted:
        rs = 1
    else:
        rs = 0
    with open(path + '/if_sorted.txt', 'wb') as f:
        pickle.dump(rs, f)
    train_data, train_time_list = utils.load_quadruples(path, 'train.txt', args.sorted)

    num_e, num_r = utils.get_total_number(path, 'stat.txt')

    graph_dict_train = {}
    s_his_dict = {}
    s_his_cache = [[] for _ in range(num_e)]
    o_his_dict = {}
    o_his_cache = [[] for _ in range(num_e)]

    for t in train_time_list:
        data = train_data[train_data[:, 3] == t]
        graph_dict_train[t] = utils.get_graph(data, num_r)
        for d in data:
            s = d[0]
            r = d[1]
            o = d[2]
            if len(s_his_cache[s]) == 0:
                s_his_cache[s] = np.array([[r, o]])
            else:
                s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
            if len(o_his_cache[o]) == 0:
                o_his_cache[o] = np.array([[r, s]])
            else:
                o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)

        s_his_dict[t] = s_his_cache
        o_his_dict[t] = o_his_cache
        s_his_cache = [[] for _ in range(num_e)]
        o_his_cache = [[] for _ in range(num_e)]
    with open(path + '/train_graph_dict.txt', 'wb') as f:
        pickle.dump(graph_dict_train, f)

    with open(path + '/train_s_his_dict.txt', 'wb') as f:
        pickle.dump(s_his_dict, f)
    with open(path + '/train_o_his_dict.txt', 'wb') as f:
        pickle.dump(o_his_dict, f)

    if os.path.exists(path + '/valid.txt'):
        valid_data, valid_time_list = utils.load_quadruples(path, 'valid.txt', args.sorted)
        datas = valid_data
        graph_dict_valid = {}
        valid_s_his_dict = {}
        valid_s_his_cache = [[] for _ in range(num_e)]
        valid_o_his_dict = {}
        valid_o_his_cache = [[] for _ in range(num_e)]
        for t in valid_time_list:
            data = datas[datas[:, 3] == t]
            graph_dict_valid[t] = utils.get_graph(data, num_r)
            for d in data:
                s = d[0]
                r = d[1]
                o = d[2]
                if len(valid_s_his_cache[s]) == 0:
                    valid_s_his_cache[s] = np.array([[r, o]])
                else:
                    valid_s_his_cache[s] = np.concatenate((valid_s_his_cache[s], [[r, o]]), axis=0)

                if len(valid_o_his_cache[o]) == 0:
                    valid_o_his_cache[o] = np.array([[r, s]])
                else:
                    valid_o_his_cache[o] = np.concatenate((valid_o_his_cache[o], [[r, s]]), axis=0)
            valid_s_his_dict[t] = valid_s_his_cache
            valid_o_his_dict[t] = valid_o_his_cache
            valid_s_his_cache = [[] for _ in range(num_e)]
            valid_o_his_cache = [[] for _ in range(num_e)]
        with open(path + '/valid_graph_dict.txt', 'wb') as f:
            pickle.dump(graph_dict_valid, f)

        with open(path + '/valid_s_his_dict.txt', 'wb') as f:
            pickle.dump(valid_s_his_dict, f)
        with open(path + '/valid_o_his_dict.txt', 'wb') as f:
            pickle.dump(valid_o_his_dict, f)

    if os.path.exists(path + '/test.txt'):
        test_data, test_time_list = utils.load_quadruples(path, 'test.txt', args.sorted)

        graph_dict_test = {}
        test_s_his_dict = {}
        test_s_his_cache = [[] for _ in range(num_e)]
        test_o_his_dict = {}
        test_o_his_cache = [[] for _ in range(num_e)]
        for t in test_time_list:
            data = test_data[test_data[:, 3] == t]
            graph_dict_test[t] = utils.get_graph(data, num_r)
            for d in data:
                s = d[0]
                r = d[1]
                o = d[2]
                if len(test_s_his_cache[s]) == 0:
                    test_s_his_cache[s] = np.array([[r, o]])
                else:
                    test_s_his_cache[s] = np.concatenate((test_s_his_cache[s], [[r, o]]), axis=0)

                if len(test_o_his_cache[o]) == 0:
                    test_o_his_cache[o] = np.array([[r, s]])
                else:
                    test_o_his_cache[o] = np.concatenate((test_o_his_cache[o], [[r, s]]), axis=0)
            test_s_his_dict[t] = test_s_his_cache
            test_o_his_dict[t] = test_o_his_cache
            test_s_his_cache = [[] for _ in range(num_e)]
            test_o_his_cache = [[] for _ in range(num_e)]
        with open(path + '/test_graph_dict.txt', 'wb') as f:
            pickle.dump(graph_dict_test, f)
        with open(path + '/test_s_his_dict.txt', 'wb') as f:
            pickle.dump(test_s_his_dict, f)
        with open(path + '/test_o_his_dict.txt', 'wb') as f:
            pickle.dump(test_o_his_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS18')
    parser.add_argument("--sorted", type=bool, default=False)
    args = parser.parse_args()
    main(args)
    print(args)
    print('Done')

###################################################################################
