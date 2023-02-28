import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MMDLoss, WassersteinLoss, CORAL, OT, sgw_gpu


class BasicModel(nn.Module):
    def __init__(self, args, dataset):
        super(BasicModel, self).__init__()
        self.device = args.device
        self.n_user, self.m_item = dataset.n_user, dataset.m_item
        self.hidden_dim = args.dim
        self.user_embedding = nn.Embedding(self.n_user, self.hidden_dim)
        self.item_embedding = nn.Embedding(self.m_item, self.hidden_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def get_rating(self, users):
        raise NotImplementedError

    def bpr_loss(self, users, pos_items, neg_items):
        raise NotImplementedError


class VanillaMF(BasicModel):
    def __init__(self, args, dataset):
        super(VanillaMF, self).__init__(args, dataset)
        self.act = nn.Sigmoid()

        self.mask_dim = 2
        self.feature_a = torch.ones(self.hidden_dim).to(self.device)
        self.mmd = MMDLoss(args, 'sinkhorn')
        self.coral = CORAL()
        self.ot = OT(args)

    def get_rating(self, users):
        users = users.long()
        users_emb = self.user_embedding(users)
        items_emb = self.item_embedding.weight
        ratings = self.act(torch.matmul(users_emb, items_emb.t()))
        return ratings

    def bpr_loss(self, users, pos_items, neg_items):
        users_emb = self.user_embedding(users.long())
        pos_emb = self.item_embedding(pos_items.long())
        neg_emb = self.item_embedding(neg_items.long())
        pos_ratings = torch.sum(users_emb * pos_emb, dim=1)
        neg_ratings = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_ratings - pos_ratings))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss, users_emb * pos_emb

    def coral_loss(self, pos_inter, source_inter):
        feature_a = torch.unsqueeze(self.feature_a, 0)
        idx = np.arange(source_inter.shape[0])
        np.random.shuffle(idx)
        source_inter = source_inter[idx]
        source_inter = source_inter[:1024]

        mask_pos_inter = pos_inter * feature_a
        mask_source_inter = source_inter * feature_a
        coral_loss = self.coral(mask_source_inter, mask_pos_inter)
        return coral_loss

    def mmd_loss(self, pos_inter, source_inter):
        feature_a = torch.unsqueeze(self.feature_a, 0)
        idx = np.arange(source_inter.shape[0])
        np.random.shuffle(idx)
        source_inter = source_inter[idx]
        source_inter = source_inter[:1024]

        mask_pos_inter = pos_inter * feature_a
        mask_source_inter = source_inter * feature_a
        mmd_loss = self.mmd(mask_source_inter, mask_pos_inter)
        return mmd_loss
        
    def ot_loss(self, pos_inter, source_inter):
        idx = np.arange(source_inter.shape[0])
        np.random.shuffle(idx)
        source_inter = source_inter[idx]
        source_inter = source_inter[:pos_inter.shape[0]]

        mask_pos_inter = pos_inter
        mask_source_inter = source_inter
        ot_loss = self.ot(mask_source_inter, mask_pos_inter)
        return ot_loss

    def updata_feature_a(self, pos_inter, source_inter):
        full_set = set(range(self.hidden_dim))
        feature_a_tensor = torch.zeros(self.hidden_dim)
        idx = np.arange(source_inter.shape[0])
        np.random.shuffle(idx)
        source_inter = source_inter[idx]
        source_inter = source_inter[:1024]

        for i in range(self.hidden_dim):
            lack_1_index = sorted(list(full_set-{i}))
            feature_a_tensor[i] = self.mmd(source_inter[:, lack_1_index], pos_inter.detach()[:, lack_1_index])
        new_feature_a = torch.ones(self.hidden_dim).to(self.device)
        _, index = torch.topk(feature_a_tensor, self.mask_dim, -1, False)
        new_feature_a[index] = 0.
        self.feature_a = new_feature_a
        return new_feature_a

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.user_embedding(users)
        items_emb = self.item_embedding(items)
        inter = users_emb * items_emb
        ratings = self.act(torch.sum(inter, dim=1))
        return ratings, inter


class NeuMF(BasicModel):
    def __init__(self, args, dataset):
        super(NeuMF, self).__init__(args, dataset)
        self.act = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()

        self.mlp_user_embedding = nn.Embedding(self.n_user, self.hidden_dim)
        self.mlp_item_embedding = nn.Embedding(self.m_item, self.hidden_dim)
        nn.init.xavier_normal_(self.mlp_user_embedding.weight)
        nn.init.xavier_normal_(self.mlp_item_embedding.weight)
        self.mlp = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * self.hidden_dim, 1 * self.hidden_dim), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(1 * self.hidden_dim, 1 * self.hidden_dim), nn.ReLU())
        self.combined = nn.Sequential(nn.Linear(2 * self.hidden_dim, 1), nn.Sigmoid())

    def get_rating(self, users):
        users_emb = self.user_embedding(users.long())
        items_emb = self.item_embedding.weight
        users_emb = users_emb.unsqueeze(1).repeat(1, self.m_item, 1)
        items_emb = items_emb.unsqueeze(0).repeat(users.shape[0], 1, 1)

        mlp_users_emb = self.mlp_user_embedding(users.long())
        mlp_items_emb = self.mlp_item_embedding.weight
        mlp_users_emb = mlp_users_emb.unsqueeze(1).repeat(1, self.m_item, 1)
        mlp_items_emb = mlp_items_emb.unsqueeze(0).repeat(users.shape[0], 1, 1)

        cos = users_emb * items_emb
        mlp_in = torch.cat((mlp_users_emb, mlp_items_emb), dim=2)
        mlp_out = self.mlp(mlp_in)
        f_in = torch.cat((cos, mlp_out), dim=2)
        f_out = self.combined(f_in)
        ratings = f_out.squeeze()
        return ratings

    def forward(self, users, items, labels):
        users = users.long()
        items = items.long()
        users_emb = self.user_embedding(users)
        items_emb = self.item_embedding(items)
        mlp_users_emb = self.mlp_user_embedding(users)
        mlp_items_emb = self.mlp_item_embedding(items)

        cos = users_emb * items_emb
        mlp_in = torch.cat((mlp_users_emb, mlp_items_emb), dim=1)
        mlp_out = self.mlp(mlp_in)
        f_in = torch.cat((cos, mlp_out), dim=1)
        f_out = self.combined(f_in)
        ratings = f_out.squeeze()

        loss = self.loss_func(ratings, labels.float())
        return loss

    def loss_func(self, preds, labels):
        return self.bce_loss(preds, labels)


class LightGCN(BasicModel):
    def __init__(self, args, dataset):
        super(LightGCN, self).__init__(args, dataset)
        self.layers = args.layers
        self.dropout = args.dropout
        self.act = nn.Sigmoid()
        self.Graph = dataset.get_sparse_graph()
        self.mode = 'train'

        self.mask_dim = 2
        self.feature_a = torch.ones(self.hidden_dim).to(self.device)
        self.mmd = MMDLoss(args, 'rbf')
        self.coral = CORAL()
        self.ot = OT(args)

    def __graph_dropout(self, x, dropout):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + (1 - dropout)
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / (1 - dropout)
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __message_passing(self):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.mode == 'train':
            g = self.__graph_dropout(self.Graph, self.dropout)
        else:
            g = self.Graph
        for layer in range(self.layers):
            all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)
        layer_u_emb, layer_i_emb = [], []
        for layer in range(len(embs)):
            u_emb, i_emb = torch.split(embs[layer], [self.n_user, self.m_item])
            layer_u_emb.append(u_emb)
            layer_i_emb.append(i_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_user, self.m_item])
        return users, items, layer_u_emb, layer_i_emb

    def get_rating(self, users):
        users_emb, items_emb, _, _ = self.__message_passing()
        users_emb = users_emb[users.long()]
        ratings = self.act(torch.matmul(users_emb, items_emb.t()))
        return ratings

    def get_embedding(self, users, pos_items, neg_items):
        users_emb, items_emb, layer_u_emb, layer_i_emb = self.__message_passing()
        users_emb = users_emb[users.long()]
        pos_items_emb = items_emb[pos_items.long()]
        neg_items_emb = items_emb[neg_items.long()]
        users_emb_layers, pos_items_emb_layers, neg_items_emb_layers = [], [], []
        for i in range(len(layer_u_emb)):
            users_emb_layers.append(layer_u_emb[i][users.long()])
            pos_items_emb_layers.append(layer_i_emb[i][pos_items.long()])
            neg_items_emb_layers.append(layer_i_emb[i][neg_items.long()])
        return users_emb, pos_items_emb, neg_items_emb, \
               users_emb_layers, pos_items_emb_layers, neg_items_emb_layers

    def bpr_loss(self, users, pos_items, neg_items):
        users_emb, pos_items_emb, neg_items_emb, \
        users_emb_layers, pos_items_emb_layers, neg_items_emb_layers = self.get_embedding(users, pos_items, neg_items)

        pos_inter = users_emb * pos_items_emb
        pos_ratings = torch.sum(pos_inter, dim=1)
        neg_inter = users_emb * neg_items_emb
        neg_ratings = torch.sum(neg_inter, dim=1)

        users_emb_ego = users_emb_layers[0]
        pos_items_emb_ego = pos_items_emb_layers[0]
        neg_items_emb_ego = neg_items_emb_layers[0]

        pos_inter_layers = []
        for i in range(len(users_emb_layers)):
            pos_inter_layers.append(users_emb_layers[i] * pos_items_emb_layers[i])

        loss = torch.mean(nn.functional.softplus(neg_ratings - pos_ratings))
        reg_loss = (1/2)*(users_emb_ego.norm(2).pow(2) +
                          pos_items_emb_ego.norm(2).pow(2) +
                          neg_items_emb_ego.norm(2).pow(2)) / float(len(users))

        return loss, reg_loss, pos_inter, pos_inter_layers

    def mmd_loss(self, pos_inter, source_inter):
        feature_a = torch.unsqueeze(self.feature_a, 0)
        idx = np.arange(source_inter.shape[0])
        np.random.shuffle(idx)
        source_inter = source_inter[idx]
        source_inter = source_inter[:1024]

        mask_pos_inter = pos_inter * feature_a
        mask_source_inter = source_inter * feature_a
        mmd_loss = self.mmd(mask_source_inter, mask_pos_inter)
        return mmd_loss

    def coral_loss(self, pos_inter, source_inter):
        feature_a = torch.unsqueeze(self.feature_a2, 0)
        idx = np.arange(source_inter.shape[0])
        np.random.shuffle(idx)
        source_inter = source_inter[idx]
        source_inter = source_inter[:1024]

        mask_pos_inter = pos_inter * feature_a
        mask_source_inter = source_inter * feature_a
        coral_loss = self.coral(mask_source_inter, mask_pos_inter)
        return coral_loss

    def ot_loss(self, pos_inter, source_inter):
        idx = np.arange(source_inter.shape[0])
        np.random.shuffle(idx)
        source_inter = source_inter[idx]
        source_inter = source_inter[:pos_inter.shape[0]]

        mask_pos_inter = pos_inter
        mask_source_inter = source_inter
        ot_loss = self.ot(mask_source_inter, mask_pos_inter)
        return ot_loss

    def wd_loss(self, pos_inter, source_inter):
        feature_a = torch.unsqueeze(self.feature_a, 0)
        mask_pos_inter = pos_inter * feature_a
        mask_source_inter = source_inter * feature_a
        wd_loss, gp = self.wd(mask_source_inter, mask_pos_inter)
        return wd_loss, gp

    def updata_feature_a(self, pos_inter, source_inter):
        full_set = set(range(self.hidden_dim))
        feature_a_tensor = torch.zeros(self.hidden_dim)
        idx = np.arange(source_inter.shape[0])
        np.random.shuffle(idx)
        source_inter = source_inter[idx]
        source_inter = source_inter[:1024]
        for i in range(self.hidden_dim):
            lack_1_index = sorted(list(full_set-{i}))
            feature_a_tensor[i] = self.mmd(source_inter[:, lack_1_index], pos_inter.detach()[:, lack_1_index])
        new_feature_a = torch.ones(self.hidden_dim).to(self.device)
        _, index = torch.topk(feature_a_tensor, self.mask_dim, -1, False)
        new_feature_a[index] = 0.
        self.feature_a = new_feature_a
        return new_feature_a

    def forward(self, users, items):
        users_emb, items_emb, layer_u_emb, layer_i_emb = self.__message_passing()
        users_emb = users_emb[users.long()]
        items_emb = items_emb[items.long()]
        inter = users_emb * items_emb
        inter_layers = []
        for i in range(len(layer_u_emb)):
            users_emb_layer = layer_u_emb[i][users.long()]
            items_emb_layer = layer_i_emb[i][items.long()]
            inter_layers.append(users_emb_layer * items_emb_layer)

        ratings = self.act(torch.sum(inter, dim=1))
        return ratings, inter, inter_layers


class NGCF(BasicModel):
    def __init__(self, args, dataset):
        super(NGCF, self).__init__(args, dataset)
        self.dropout = args.dropout
        self.layers = args.layers
        self.act = nn.Sigmoid()
        self.mode = 'train'

        self.Graph = dataset.get_sparse_graph()

        self.weight = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.layers)])
        self.bi_weight = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.layers)])

    def __graph_dropout(self, x, dropout):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + (1 - dropout)
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / (1 - dropout)
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __message_passing(self):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.mode == 'train':
            g = self.__graph_dropout(self.Graph, self.dropout)
        else:
            g = self.Graph
        for layer in range(self.layers):
            side_emb = torch.sparse.mm(g, all_emb)
            sum_emb = self.weight[layer](side_emb)
            bi_emb = torch.mul(all_emb, side_emb)
            bi_emb = self.bi_weight[layer](bi_emb)
            all_emb = nn.LeakyReLU(negative_slope=0.2)(sum_emb + bi_emb)
            all_emb = nn.Dropout(self.dropout)(all_emb)
            all_emb = F.normalize(all_emb, p=2, dim=1)
            embs.append(all_emb)
        layer_u_emb, layer_i_emb = [], []
        for layer in range(len(embs)):
            u_emb, i_emb = torch.split(embs[layer], [self.n_user, self.m_item])
            layer_u_emb.append(u_emb)
            layer_i_emb.append(i_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_user, self.m_item])
        return users, items, layer_u_emb, layer_i_emb

    def get_embedding(self, users, pos_items, neg_items):
        users_emb, items_emb, layer_u_emb, layer_i_emb = self.__message_passing()
        users_emb = users_emb[users.long()]
        pos_items_emb = items_emb[pos_items.long()]
        neg_items_emb = items_emb[neg_items.long()]
        users_emb_layers, pos_items_emb_layers, neg_items_emb_layers = [], [], []
        for i in range(len(layer_u_emb)):
            users_emb_layers.append(layer_u_emb[i][users.long()])
            pos_items_emb_layers.append(layer_i_emb[i][pos_items.long()])
            neg_items_emb_layers.append(layer_i_emb[i][neg_items.long()])
        return users_emb, pos_items_emb, neg_items_emb, \
               users_emb_layers, pos_items_emb_layers, neg_items_emb_layers

    def bpr_loss(self, users, pos_items, neg_items):
        users_emb, pos_items_emb, neg_items_emb, \
        users_emb_layers, pos_items_emb_layers, neg_items_emb_layers = self.get_embedding(users, pos_items, neg_items)

        pos_inter = users_emb * pos_items_emb
        pos_ratings = torch.sum(pos_inter, dim=1)
        neg_inter = users_emb * neg_items_emb
        neg_ratings = torch.sum(neg_inter, dim=1)

        users_emb_ego = users_emb_layers[0]
        pos_items_emb_ego = pos_items_emb_layers[0]
        neg_items_emb_ego = neg_items_emb_layers[0]

        pos_inter_layers = []
        for i in range(len(users_emb_layers)):
            pos_inter_layers.append(users_emb_layers[i] * pos_items_emb_layers[i])

        loss = torch.mean(nn.functional.softplus(neg_ratings - pos_ratings))
        reg_loss = (1/2)*(users_emb_ego.norm(2).pow(2) +
                          pos_items_emb_ego.norm(2).pow(2) +
                          neg_items_emb_ego.norm(2).pow(2)) / float(len(users))

        return loss, reg_loss, pos_inter, pos_inter_layers

    def get_rating(self, users):
        users_emb, items_emb, _, _ = self.__message_passing()
        users_emb = users_emb[users.long()]
        ratings = self.act(torch.matmul(users_emb, items_emb.t()))
        return ratings

    def forward(self, users, items):
        users_emb, items_emb, layer_u_emb, layer_i_emb = self.__message_passing()
        users_emb = users_emb[users.long()]
        items_emb = items_emb[items.long()]
        inter = users_emb * items_emb
        inter_layers = []
        for i in range(len(layer_u_emb)):
            users_emb_layer = layer_u_emb[i][users.long()]
            items_emb_layer = layer_i_emb[i][items.long()]
            inter_layers.append(users_emb_layer * items_emb_layer)

        ratings = self.act(torch.sum(inter, dim=1))
        return ratings, inter, inter_layers

