import pandas as pd
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp


# from cppimport import imp_from_filepath
# path = "sources/sampling.cpp"
# sampling = imp_from_filepath(path)
# sampling.seed(42)


class BipartiteGraphDataset(Dataset):
    def __init__(self, args, dataset):
        super(BipartiteGraphDataset, self).__init__()
        self.device = args.device
        self.dataset = dataset
        self.neg_num = args.neg_num
        self.train_data = pd.read_csv('datasets/' + dataset + '/train.txt', sep='\t',
                                      names=['userid', 'itemid', 'times'],
                                      dtype={0: np.int64, 1: np.int64, 2: np.int64})
        self.test_data = pd.read_csv('datasets/' + dataset + '/test.txt', sep='\t',
                                     names=['userid', 'itemid', 'times'],
                                     dtype={0: np.int64, 1: np.int64, 2: np.int64})
        # self.train_data = pd.read_csv('datasets/' + dataset + '/train.txt', sep='\t',
        #                               names=['userid', 'itemid'],
        #                               dtype={0: np.int64, 1: np.int64})
        # self.test_data = pd.read_csv('datasets/' + dataset + '/test.txt', sep='\t',
        #                              names=['userid', 'itemid'],
        #                              dtype={0: np.int64, 1: np.int64})
        self.n_user = max(self.train_data['userid'].max(), self.test_data['userid'].max()) + 1
        self.m_item = max(self.train_data['itemid'].max(), self.test_data['itemid'].max()) + 1
        self.trainDataSize, self.testDataSize = self.train_data.shape[0], self.test_data.shape[0]

        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.train_data['userid'], self.train_data['itemid'])),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self.__allPos = self.get_user_pos_items(list(range(self.n_user)))
        self.testDict = self.__build_test()
        self.Graph = None
        self.get_sparse_graph()
        self.S = None
        self.uniform_sampling()

    def uniform_sampling(self):
        # S = sampling.sample_negative(self.n_user, self.m_item,
        #                              self.trainDataSize, self.__allPos, self.neg_num)
        # shuffle_indices = np.arange(len(S))
        # np.random.shuffle(shuffle_indices)
        # self.S = S[shuffle_indices]
        users = np.random.randint(0, self.n_user, self.trainDataSize)
        allPos = self.__allPos
        S = []
        for i, user in enumerate(users):
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, self.m_item)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
        self.S = torch.LongTensor(S)

    def negative_sampling(self):
        users = np.random.randint(0, self.n_user, self.trainDataSize)
        allPos = self.__allPos
        S = []
        for i, user in enumerate(users):
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            S.append([user, positem, 1])
            negitems = np.random.randint(0, self.m_item, self.neg_num)
            for negitem in negitems:
                while negitem in posForUser:
                    negitem = np.random.randint(0, self.m_item)
                S.append([user, negitem, 0])
        self.S = torch.LongTensor(S)

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __build_test(self):
        td = {}
        for idx, row in self.test_data.iterrows():
            user, item = row[0], row[1]
            td[user] = td.get(user, [])
            td[user].append(item)
        return td

    def __convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float64)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz('datasets/' + self.dataset + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float64)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz('datasets/' + self.dataset + '/s_pre_adj_mat.npz', norm_adj)

            self.Graph = self.__convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(self.device)
        return self.Graph

    def __getitem__(self, idx):
        return self.S[idx]

    def __len__(self):
        return len(self.S)


