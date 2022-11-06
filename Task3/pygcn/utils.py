import numpy as np
import scipy.sparse as sp
import torch
import random

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append("..") 
from preprocessing.EDA_lg import EDA



def load_data_house():
    print('Loading house dataset...')
    DATA_DIR = '../data/'
    
    filepath_test = DATA_DIR + 'task3_test.csv'
    df_test = pd.read_csv(filepath_test)
    filepath_train = DATA_DIR + 'task3_train.csv'
    df_train = pd.read_csv(filepath_train)
    eda = EDA(df_train.copy(), df_test.copy())
    eda.setup4()
    df_train = eda.df
    df_test = eda.df_test
    df_train_y = df_train['price']
    df_train_X = df_train.drop(columns=['price', 'first_mrt', 'first_dis', 'sec_mrt', 'sec_dis', 'thr_mrt', 'thr_dis'])
    df_test = df_test.drop(columns=['first_mrt', 'first_dis', 'sec_mrt', 'sec_dis', 'thr_mrt', 'thr_dis'])
    X_train = df_train_X.to_numpy(dtype = 'float32')
    y_train = df_train_y.to_numpy(dtype = 'float32')
    X_test = df_test.to_numpy(dtype = 'float32')

    X = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)

    min_max_scalers = []
    for i in range(X.shape[1]):
        min_max_scaler = MinMaxScaler()
        col_train = X[:, i]
        col_train = col_train.reshape(-1, 1)
        col_train = min_max_scaler.fit_transform(col_train)
        X[:, i] = col_train.reshape(-1)

        # col_test = X_test[:, i]
        # col_test = col_test.reshape(-1, 1)
        # col_test = min_max_scaler.transform(col_test)
        # X_test[:, i] = col_test.reshape(-1)

        min_max_scalers.append(min_max_scaler)
    
    X_train, y_train = X[:, :-1], X[:, -1]
    y_train = y_train.reshape(-1, 1)
    print('Preprocessing ok.')
    #######################################################################################
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_house2mrt = df_train[['first_mrt', 'first_dis', 'sec_mrt', 'sec_dis', 'thr_mrt', 'thr_dis']].copy()
    df_edge = pd.read_csv(DATA_DIR+'task3_con.csv')
    num_h = len(X_train)
    adj_mat = np.zeros((num_h, num_h), dtype='float32')
    mrt_name = df_edge['from'].unique().tolist()
    close_house = [[] for i in range(len(mrt_name))]
    mrt2house = dict(zip(mrt_name, close_house))
    for index, row in df_house2mrt.iterrows():
        mrt2house[row['first_mrt']].append(index)
    for index, row in df_edge.iterrows():
        nodes_from = mrt2house[row['from']]
        nodes_to = mrt2house[row['to']]
    #     adj_mat[row['distance']
        if len(nodes_from) > 0 and len(nodes_to) > 0:
            nodes_from = np.array(nodes_from).reshape(1, -1)
            nodes_to = np.array(nodes_to).reshape(-1, 1)
            adj_mat[nodes_from, nodes_to] = 1.
    print('Construct adj ok.')
    
    adj_mat = sp.csr_matrix(adj_mat)
    # adj_mat = normalize(adj_mat + sp.eye(adj_mat.shape[0])) 
    adj_mat = normalize(sp.eye(adj_mat.shape[0])) # only self-connection!!!!!!!!!!!!!!

    five_fold_idx = int(len(X_train) * 0.8)
    idx_train = range(five_fold_idx)
    idx_val = range(five_fold_idx, len(X_train))
    # idx_test = range(500, 1500)
    features = X_train
    labels = y_train

    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    
    # idx_test = torch.LongTensor(idx_test)

    return adj_mat, features, labels, idx_train, idx_val, None, min_max_scalers


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
