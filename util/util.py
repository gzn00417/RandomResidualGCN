import os
import torch
import numpy as np
import scipy.sparse as sp

from config.args import args


# ----------------------------For Dataset----------------------------

def read_id(filename):
    entity2id = {}
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        origin, index = line.strip().split()
        entity2id[origin] = int(index)
    return entity2id


def generate_adj(data, num_entity):
    rows = []
    cols = []
    for edge in data:
        rows.append(edge['h'])
        cols.append(edge['t'])
    adj = sp.coo_matrix(
        (torch.ones(len(data)), (rows, cols)),
        shape=(num_entity, num_entity),
        dtype=np.float32,
    )  # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
    # build symmetric adjacency matrix   论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵
    adj = normalize(adj + sp.eye(adj.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
    # 对应公式A~=A+IN
    adj = sparse_mx_to_torch_sparse_tensor(adj)  # 邻接矩阵转为tensor处理
    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.0  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):  # 把一个sparse matrix转为torch稀疏张量
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 不懂的可以去看看COO性稀疏矩阵的结构
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ----------------------------For Others----------------------------

def get_head_relation_tail(triples, values, type: str in ['positive', 'negative']):
    h, r, t = triples
    if type == 'positive':
        idx = values.ge(0)
    elif type == 'negative':
        idx = values.le(0)
    else:
        raise Exception
    h, r, t = h[idx], r[idx], t[idx]
    return (h, r, t), len(h)


def align(triples, src_len, tgt_len):
    h, r, t = triples
    h = torch.cat((h.repeat((tgt_len // src_len), 1), h[:tgt_len % src_len]))
    r = torch.cat((r.repeat((tgt_len // src_len), 1), r[:tgt_len % src_len]))
    t = torch.cat((t.repeat((tgt_len // src_len), 1), t[:tgt_len % src_len]))
    return (h, r, t)


def find_last_file(target_dir):
    lists = os.listdir(target_dir)
    lists.sort(key=lambda fn: os.path.getmtime(target_dir + "\\" + fn))
    last_file = os.path.join(target_dir, lists[-1])
    return last_file


def get_last_log_folder():
    return find_last_file(os.path.join('.', 'log'))


def get_checkpoint(version: int = -1):
    log_folder = get_last_log_folder() if version < 0 else os.path.join('.', 'log', 'version_' + str(version))
    return find_last_file(os.path.join(log_folder, 'checkpoints'))
