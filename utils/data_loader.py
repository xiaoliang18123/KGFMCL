import logging
import os

import numpy as np

import scipy.sparse as sp
import heapq
import random
from time import time
from collections import defaultdict
import warnings

import torch

warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
n_train = 0
n_test = 0


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + entities
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    tans_triplets = triplets.copy()
    tans_triplets[:, 1] = triplets[:, 2]
    tans_triplets[:, 2] = triplets[:, 1]

    return tans_triplets


# adj_mat_list 每种关系对应节点构成的稀疏矩阵列表，norm_mat_list norm归一化，mean-mat_list 均值归一化
def build_sparse_neibor_matrix(train_data, user_top_k_neibor, item_top_k_neibor, dirname):
    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    print("Begin to build sparse adj matrix ...")
    np_mat = train_data
    cf = np_mat.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    vals = [1.] * len(cf)
    adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_users, n_users + n_items))
    # interaction: user->item, [n_users, n_items]
    adj = adj.tocsr()[:, n_users:]

    print(" build sparse user_neibor  matrix ...")
    user_neibor = user_adj(dirname, adj, user_top_k_neibor)
    norm_user_neibor = _si_norm_lap(user_neibor)
    print(" build sparse item_neibor  matrix ...")
    item_neibor = item_adj(dirname, adj, item_top_k_neibor)
    norm_item_neibor = _si_norm_lap(item_neibor)

    print(" Get the normalized interaction matrix of users and items ...")
    # build adj matrix
    A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    cf = np_mat.copy()
    data_dict = dict(zip(zip(cf[:, 0], cf[:, 1] + n_users), vals))
    data_dict.update(dict(zip(zip(cf[:, 1] + n_users, cf[:, 0]), vals)))
    A._update(data_dict)
    # norm adj matrix
    sumArr = (A > 0).sum(axis=1)
    deg = np.array(sumArr.flatten())[0]
    # np.savetxt(dirname + '/deg.txt', deg, fmt='%d')
    # add epsilon to avoid divide by zero Warning
    diag = np.array(sumArr.flatten())[0] + 1e-7
    diag = np.power(diag, -0.5)
    D = sp.diags(diag)
    L = D @ A @ D
    # covert norm_adj matrix to tensor
    L = sp.coo_matrix(L)
    row = L.row
    col = L.col
    i = torch.LongTensor([row, col])
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
    return norm_user_neibor, norm_item_neibor, SparseL, deg


def gridsearch_build_sparse_neibor_matrix(train_data, user_top_k_neibor, item_top_k_neibor, dirname):
    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    np_mat = train_data
    cf = np_mat.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    vals = [1.] * len(cf)
    adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_users, n_users + n_items))
    # interaction: user->item, [n_users, n_items]
    adj = adj.tocsr()[:, n_users:]
    print(" build sparse user_neibor  matrix ...")
    user_neibor = user_adj(dirname, adj, user_top_k_neibor)
    norm_user_neibor = _si_norm_lap(user_neibor)
    print(" build sparse item_neibor  matrix ...")
    item_neibor = item_adj(dirname, adj, item_top_k_neibor)
    norm_item_neibor = _si_norm_lap(item_neibor)
    return norm_user_neibor, norm_item_neibor


def gridsearch_build_sparse_adj_matrix(train_data):
    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    print("Begin to build sparse adj matrix ...")
    np_mat = train_data
    cf = np_mat.copy()
    vals = [1.] * len(cf)
    print(" Get the normalized interaction matrix of users and items ...")
    # build adj matrix
    A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    data_dict = dict(zip(zip(cf[:, 0], cf[:, 1] + n_users), vals))
    data_dict.update(dict(zip(zip(cf[:, 1] + n_users, cf[:, 0]), vals)))
    A._update(data_dict)
    # norm adj matrix
    sumArr = (A > 0).sum(axis=1)
    deg = np.array(sumArr.flatten())[0]
    # np.savetxt(dirname + '/deg.txt', deg, fmt='%d')
    # add epsilon to avoid divide by zero Warning
    diag = np.array(sumArr.flatten())[0] + 1e-7
    diag = np.power(diag, -0.5)
    D = sp.diags(diag)
    L = D @ A @ D
    # covert norm_adj matrix to tensor
    L = sp.coo_matrix(L)
    row = L.row
    col = L.col
    i = torch.LongTensor([row, col])
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
    return SparseL, deg


def gridsearch_load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    global n_train, n_test
    n_train = len(train_cf)
    n_test = len(test_cf)
    remap_item(train_cf, test_cf)  # 构建字典

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(directory + 'kg_final.txt')

    print('building the adj mat ...')
    SparseL, deg = gridsearch_build_sparse_adj_matrix(train_cf)
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None
    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations),
        'n_train': n_train,
        'n_test': n_test,
        'pretrain_data': pretrain_data
    }

    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set,
    }

    return train_cf, test_cf, user_dict, n_params, triplets, SparseL, deg


def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    global n_train, n_test
    n_train = len(train_cf)
    n_test = len(test_cf)
    remap_item(train_cf, test_cf)  # 构建字典

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(directory + 'kg_final.txt')

    print('building the adj mat ...')
    norm_user_neibor, norm_item_neibor, SparseL, deg = build_sparse_neibor_matrix(train_cf, args.user_neibor_size,
                                                                                  args.item_neibor_size,
                                                                                  directory)

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None
    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations),
        'n_train': n_train,
        'n_test': n_test,
        'pretrain_data': pretrain_data
    }

    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set,
    }

    return train_cf, test_cf, user_dict, n_params, triplets, \
           [norm_user_neibor, norm_item_neibor, SparseL, deg]


def user_adj(dirname, adj, top_K, user_neibor=True):
    """
    :param dirname: dir eg.data/last-fm
    :param adj: sparse.csr.csr_matrix size=(user_nuber,item_number)
    :param top_K: neibor
    :return: sp.coo_matrix top_k neibor matrix size=(user_nuber,user_nuber)
    """
    nums = adj.shape[0]
    if user_neibor:
        dir_path = dirname + 'prepreccess/user_{}_neibor/'.format(top_K)
    else:
        dir_path = dirname + 'prepreccess/item_{}_neibor/'.format(top_K)
    if os.path.exists(dir_path):
        index_file_name = dir_path + 'index.txt'
        data_file_name = dir_path + 'data.txt'
        index = np.loadtxt(index_file_name, dtype=np.int32)
        data = np.loadtxt(data_file_name, dtype=float)
        adj = sp.coo_matrix((data, (index[:, 0], index[:, 1])), shape=(nums, nums))
        return adj
    else:
        ensureDir(dir_path)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        index = list()

        data = []

        adj = adj.dot(adj.T)
        adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        row, col = adj.nonzero()

        score = {}

        k = 0

        def addneibor():
            K_max_item_score = heapq.nlargest(top_K + 1, score, key=score.get)
            if len(K_max_item_score) == 1:
                index.append([k, k])
                data.append(1.)
            else:
                for i in K_max_item_score:
                    if i != k:
                        index.append([k, i])
                        data.append(score[i])

        for i in range(len(row)):
            if row[i] == k:
                score[col[i]] = adj[row[i], col[i]]
            else:
                addneibor()
                score.clear()
                while k < row[i]:
                    k += 1
                score[col[i]] = adj[row[i], col[i]]
        addneibor()
        index = np.array(index)
        data = np.array(data)
        index_file_name = dir_path + 'index.txt'
        data_file_name = dir_path + 'data.txt'
        np.savetxt(index_file_name, index, fmt='%d')
        np.savetxt(data_file_name, data, fmt='%.3f')

        adj = sp.coo_matrix((data, (index[:, 0], index[:, 1])), shape=(nums, nums))
        return adj


def item_adj(dirname, adj, top_K):
    """

    :param adj: sparse.csr.csr_matrix size=(user_nuber,item_number)
    :param top_K: neibor number
    :return: sp.coo_matrix top_k neibor matrix size=(item_number,item_number)
    """
    return user_adj(dirname, adj.T, top_K, user_neibor=False)


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def load_pretrained_data(args):
    pre_model = 'mf'
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, pre_model)
    if args.pretrain == -1:
        try:
            pretrain_data = np.load(pretrain_path)
            logging.info('load the pretrained bprmf model parameters.')
        except Exception:
            pretrain_data = None
    return pretrain_data
