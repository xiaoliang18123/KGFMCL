'''
Created on dec 15, 2022
PyTorch Implementation of KGCA
@author: Heguangliang
'''
__author__ = "Heguangliang "

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from utils.data_loader import read_cf
from utils.helper import ensureDir


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users, n_items,
                 n_relations, norm_user_neibor, norm_item_neibor, norm_adj_mat, log_deg, gamma, node_dropout_rate=0.5,
                 mess_dropout_rate=0.1):  # channel enbeding size
        super(GraphConv, self).__init__()
        self.norm_user_neibor = norm_user_neibor
        self.norm_item_neibor = norm_item_neibor
        self.norm_adj_mat = norm_adj_mat

        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.hyper_layer = 2
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.log_deg = log_deg
        self.gamma = gamma
        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]#weight 关系的embeding
        self.n_layers = n_hops

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * (1 - rate)), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    # ?
    def __dropout_x(self, x, node_dropout_rate):
        keep_prob = 1 - node_dropout_rate
        size = x.size()
        index = x._indices().t()
        values = x._values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def forward(self, all_embed, entity_emb, edge_index, edge_type, mess_dropout=True,
                node_dropout=False):

        """node dropout"""
        g_droped = self.norm_adj_mat
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            g_droped = self.__dropout_x(g_droped, self.node_dropout_rate)

        all_embeddings = all_embed
        embeddings_list = [all_embeddings]
        embeddings_y = []
        embeddings_z = []
        item_kg_emb_list = [entity_emb[:self.n_items]]
        n_entities = entity_emb.shape[0]
        for layer_idx in range(1, self.n_layers + 1):
            all_embeddings1 = torch.sparse.mm(g_droped, all_embeddings)

            user_collaborate_emb = torch.sparse.mm(self.norm_user_neibor, all_embeddings[:self.n_users])
            item_collaborate_emb = torch.sparse.mm(self.norm_item_neibor, all_embeddings[self.n_users:])
            # all_embeddings2 = torch.sparse.mm(self.norm_myadj_mat, all_embeddings)
            all_embeddings2 = torch.cat((user_collaborate_emb, item_collaborate_emb), dim=0)
            embeddings_y.append(all_embeddings1)
            embeddings_z.append(all_embeddings2)

            a_1 = torch.nn.functional.normalize(all_embeddings1, p=2, dim=1)
            b_1 = torch.nn.functional.normalize(all_embeddings1 + all_embeddings2, p=2, dim=1)
            sim_1 = torch.mul(a_1, b_1).sum(dim=1).reshape(-1, 1)
            sim_1 = torch.clamp(sim_1, min=0.0)
            beta = self.gamma / (layer_idx + sim_1 * self.log_deg)
            all_embeddings2 = torch.mul(all_embeddings2, beta)
            all_embeddings = all_embeddings1 + all_embeddings2

            """KG aggregate"""

            head, tail = edge_index
            edge_relation_emb = self.weight[
                edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
            neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
            entity_emb = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

            if mess_dropout:
                all_embeddings = self.dropout(all_embeddings)
                entity_emb = self.dropout(entity_emb)
            embeddings_list.append(all_embeddings)
            item_kg_emb_list.append(entity_emb[:self.n_items, :])

        hyper_layer_embedding = embeddings_list[self.hyper_layer]

        user_all_embeddings_x, item_all_embeddings_x = torch.split(hyper_layer_embedding, [self.n_users, self.n_items])

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        lightgcn_all_embeddings_y = torch.stack(embeddings_y, dim=1)
        lightgcn_all_embeddings_y = torch.mean(lightgcn_all_embeddings_y, dim=1)

        lightgcn_all_embeddings_z = torch.stack(embeddings_z, dim=1)
        lightgcn_all_embeddings_z = torch.mean(lightgcn_all_embeddings_z, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        user_all_embeddings_y, item_all_embeddings_y = torch.split(lightgcn_all_embeddings_y,
                                                                   [self.n_users, self.n_items])
        user_all_embeddings_z, item_all_embeddings_z = torch.split(lightgcn_all_embeddings_z,
                                                                   [self.n_users, self.n_items])

        item_kg_embeddings = torch.stack(item_kg_emb_list, dim=1)
        item_kg_embeddings = torch.mean(item_kg_embeddings, dim=1)

        item_final_agg = (item_kg_embeddings + item_all_embeddings)/2

        return user_all_embeddings, item_final_agg, user_all_embeddings_x, item_all_embeddings_x, \
               user_all_embeddings_y, item_all_embeddings_y, user_all_embeddings_z, item_all_embeddings_z, item_kg_embeddings


class Recommender(nn.Module):
    def __init__(self, data_config, user_dict, args_config, graph,
                 norm_user_neibor, norm_item_neibor, SparseL, deg, device):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.test_user_dict = user_dict['test_user_set']
        self.train_user_dict = user_dict['train_user_set']
        self.n_train = data_config['n_train']
        self.n_test = data_config['n_test']
        self.path = args_config.data_path + args_config.dataset
        self.fold = args_config.fold
        self.pretrain_data = data_config['pretrain_data']

        self.norm_user_neibor = norm_user_neibor
        self.norm_item_neibor = norm_item_neibor

        self.device = device
        self.norm_adj_mat, self.deg = SparseL, deg
        self.norm_adj_mat = self.norm_adj_mat.to(self.device)
        self.deg[self.deg == 0] = 1
        log_deg = np.log(self.deg)
        mean_deg = np.mean(log_deg)
        log_deg = log_deg / mean_deg
        self.log_deg = torch.tensor(log_deg).reshape(-1, 1).to(self.device).to(torch.float32)

        self.decay = args_config.l2

        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops

        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        self.graph = graph

        self.edge_index, self.edge_type = self._get_edges(graph)  # slow
        self.gamma = args_config.gamma
        self.ssl_temp = args_config.ssl_temp
        self.ssl_reg = args_config.ssl_reg
        self.info_temp = args_config.ssl_temp
        self.info_reg = args_config.info_reg
        self.kg_temp = args_config.ssl_temp
        self.kg_reg = args_config.kg_reg
        self.alpha = args_config.alpha
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.entities_emb = nn.Parameter(self.entities_emb)
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        if self.pretrain_data is None:
            self.all_embed = initializer(torch.empty(self.n_users + self.n_items, self.emb_size))
            self.entities_emb = initializer(torch.empty(self.n_entities, self.emb_size))
        else:
            user_emb = torch.tensor(self.pretrain_data['user_embed'], requires_grad=True)
            item_emb = torch.tensor(self.pretrain_data['item_embed'], requires_grad=True)
            self.all_embed = torch.cat((user_emb, item_emb), dim=0)
            self.entities_emb = initializer(torch.empty(self.n_entities, self.emb_size))
        # [n_users, n_entities]
        self.norm_user_neibor = self._convert_sp_mat_to_sp_tensor(self.norm_user_neibor).to(self.device)
        self.norm_item_neibor = self._convert_sp_mat_to_sp_tensor(self.norm_item_neibor).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items=self.n_items,
                         n_relations=self.n_relations,
                         norm_user_neibor=self.norm_user_neibor,
                         norm_item_neibor=self.norm_item_neibor,
                         norm_adj_mat=self.norm_adj_mat,
                         log_deg=self.log_deg,
                         gamma=self.gamma,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate
                         )

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_edges(self, graph_tensor):
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_all_embeddings, item_all_embeddings, \
        user_all_embeddings_x, item_all_embeddings_x, \
        user_all_embeddings_y, item_all_embeddings_y, \
        user_all_embeddings_z, item_all_embeddings_z, item_kg_embeddings = self.gcn(
            self.all_embed, self.entities_emb, self.edge_index, self.edge_type,
            mess_dropout=self.mess_dropout, node_dropout=self.node_dropout)

        u_e = user_all_embeddings[user]
        pos_e, neg_e = item_all_embeddings[pos_item], item_all_embeddings[neg_item]
        totall_loss, mf_loss, emb_loss = self.create_bpr_loss(u_e, pos_e, neg_e)
        ssl_loss, info_loss, kg_sslloss = self.creat_sslloss(user, pos_item, user_all_embeddings, item_all_embeddings,
                                                             user_all_embeddings_x, item_all_embeddings_x,
                                                             user_all_embeddings_y, item_all_embeddings_y,
                                                             user_all_embeddings_z, item_all_embeddings_z,
                                                             item_kg_embeddings)
        ssl_totall_loss = (ssl_loss + info_loss + kg_sslloss)
        return totall_loss + ssl_totall_loss, mf_loss, ssl_totall_loss, emb_loss

    def creat_infoloss(self, users, a_embeddings, b_embeddings,
                       ssl_temp, ssl_reg):
        # argument settings
        batch_users = torch.unique(users)
        user_emb1 = torch.nn.functional.normalize(a_embeddings[batch_users], p=2, dim=1)
        user_emb2 = torch.nn.functional.normalize(b_embeddings[batch_users], p=2, dim=1)
        # user
        user_pos_score = torch.mul(user_emb1, user_emb2).sum(dim=1)
        user_ttl_score = torch.matmul(user_emb1, user_emb2.t())
        user_pos_score = torch.exp(user_pos_score / ssl_temp)
        user_ttl_score = torch.exp(user_ttl_score / ssl_temp).sum(dim=1)
        user_ssl_loss = -torch.log(user_pos_score / user_ttl_score).sum()
        ssl_loss = ssl_reg * user_ssl_loss

        return ssl_loss

    def creat_sslloss(self, user, pos_item, user_all_embeddings, item_all_embeddings, user_all_embeddings_x,
                      item_all_embeddings_x, user_all_embeddings_y, item_all_embeddings_y, user_all_embeddings_z,
                      item_all_embeddings_z, item_kg_embeddings):
        user_ssl = self.creat_infoloss(user, user_all_embeddings, user_all_embeddings_x, self.ssl_temp, self.ssl_reg)

        item_ssl = self.creat_infoloss(pos_item, item_all_embeddings, item_all_embeddings_x, self.ssl_temp,
                                       self.ssl_reg)

        user_infoy = self.creat_infoloss(user, user_all_embeddings_y, user_all_embeddings, self.info_temp,
                                         self.info_reg)
        item_infoy = self.creat_infoloss(pos_item, item_all_embeddings_y, item_all_embeddings, self.info_temp,
                                         self.info_reg)

        user_infoz = self.creat_infoloss(user, user_all_embeddings_z, user_all_embeddings, self.info_temp,
                                         self.info_reg)
        item_infoz = self.creat_infoloss(pos_item, item_all_embeddings_z, item_all_embeddings, self.info_temp,
                                         self.info_reg)
        kg_sslloss = self.creat_infoloss(pos_item, item_kg_embeddings, item_all_embeddings, self.kg_temp,
                                         self.kg_reg)
        return user_ssl + item_ssl, user_infoy + item_infoy + self.alpha * (user_infoz + item_infoz), kg_sslloss

    def generate(self):

        uuser_all_embeddings, item_all_embeddings, \
        user_all_embeddings_x, item_all_embeddings_x, \
        user_all_embeddings_y, item_all_embeddings_y, \
        user_all_embeddings_z, item_all_embeddings_z, item_kg_embeddings = self.gcn(
            self.all_embed, self.entities_emb, self.edge_index, self.edge_type,
            mess_dropout=False, node_dropout=False)

        return uuser_all_embeddings, item_all_embeddings

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')
            f.close()
        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_user_dict[uid]
            test_iids = self.test_user_dict[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            elif idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state

    def get_popularity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/' + str(self.fold) + '_user_subset' + '/popularity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get {}_user_subset popularity split.'.format(self.fold))

        except Exception:
            split_uids, split_state = self.create_popularity_split()
            path = self.path + '/' + str(self.fold) + '_user_subset' + '/popularity.split'
            ensureDir(path)
            f = open(path, 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create {}_user_subset popularity split.'.format(self.fold))
            f.close()

        return split_uids, split_state

    def create_popularity_split(self):
        def statistic_popularity(users_list):
            item_list = []
            for uid in users_list:
                item_list = np.concatenate((item_list, self.train_user_dict[uid]))
            item_list = list(item_list.astype(np.int32))
            average_popularity = int(np.mean(item_popularity[item_list]))
            max_popularity = np.max(item_popularity[item_list])
            min_popularity = np.min(item_popularity[item_list])
            return average_popularity, max_popularity, min_popularity

        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = dict()
        train_cf = read_cf(self.path + '/train.txt')

        vals = [1.] * len(train_cf)
        adj = sp.coo_matrix((vals, (train_cf[:, 0], train_cf[:, 1])), shape=(self.n_users, self.n_items))
        item_popularity = np.array(adj.sum(0)).squeeze(axis=0)

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_user_dict[uid]
            # test_iids = self.test_user_dict[uid]

            average_popularity = int(np.mean(item_popularity[train_iids]))

            if average_popularity not in user_n_iid.keys():
                user_n_iid[average_popularity] = [uid]
            else:
                user_n_iid[average_popularity].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []

        fold = self.fold
        rate = 1 / fold
        n_count = self.n_users
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += len(user_n_iid[n_iids])
            n_count -= len(user_n_iid[n_iids])

            if n_rates >= rate * (self.n_users):
                split_uids.append(temp)
                average_popularity, max_popularity, min_popularity = statistic_popularity(temp)
                state = '# average popularity of items: [%d], min popularity of items: [%d], max popularity of items: ' \
                        '[%d], #users=[%d]' % (
                            average_popularity, min_popularity, max_popularity, n_rates)
                split_state.append(state)
                print(state)
                temp = []
                n_rates = 0
            elif idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)
                average_popularity, max_popularity, min_popularity = statistic_popularity(temp)
                state = '# average popularity of items: [%d], min popularity of items: [%d], max popularity of items: ' \
                        '[%d], #users=[%d]' % (
                            average_popularity, min_popularity, max_popularity, n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state
