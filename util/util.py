import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from config.args import args


# ----------------------------For Losses----------------------------

def calc_trans_e_mse_loss(triples, batch_size, embedding_dim):
    h, r, t = triples
    trans_e_diff = h + r - t
    mse_loss = F.mse_loss(trans_e_diff.cpu(), torch.zeros(batch_size, embedding_dim))
    return mse_loss


def calc_tran_e_margin_ranking_loss(triples, values, margin):
    h, r, t = triples
    pos = values.ge(0)
    neg = values.le(0)

    # pos
    pos_h = h[pos]
    pos_r = r[pos]
    pos_t = t[pos]

    # neg
    neg_h = h[neg]
    neg_r = r[neg]
    neg_t = t[neg]

    # alignment
    pos_len = len(pos_h)
    neg_len = len(neg_h)
    if pos_len > neg_len:
        neg_h = torch.cat((neg_h.repeat((pos_len // neg_len), 1), neg_h[:pos_len % neg_len]))
        neg_r = torch.cat((neg_r.repeat((pos_len // neg_len), 1), neg_r[:pos_len % neg_len]))
        neg_t = torch.cat((neg_t.repeat((pos_len // neg_len), 1), neg_t[:pos_len % neg_len]))
    elif neg_len > pos_len:
        pos_h = torch.cat((pos_h.repeat((neg_len // pos_len), 1), pos_h[:neg_len % pos_len]))
        pos_r = torch.cat((pos_r.repeat((neg_len // pos_len), 1), pos_r[:neg_len % pos_len]))
        pos_t = torch.cat((pos_t.repeat((neg_len // pos_len), 1), pos_t[:neg_len % pos_len]))

    # norm
    pos_norm = torch.norm(pos_h + pos_r - pos_t, p=1, dim=1)
    neg_norm = torch.norm(neg_h + neg_r - neg_t, p=1, dim=1)

    # margin ranking loss
    margin_ranking_loss = F.margin_ranking_loss(
        pos_norm.cpu(),
        neg_norm.cpu(),
        torch.ones(len(pos_norm), requires_grad=True),
        margin=margin
    )
    return margin_ranking_loss


def calc_conv_loss(model, triples, conv_output, batch_size, lmbda: float = args.lmbda):
    h, r, t = triples
    l2_reg = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
    for W in model.conv_layer.parameters():
        l2_reg = l2_reg + W.norm(2)
    for W in model.fc_layer.parameters():
        l2_reg = l2_reg + W.norm(2)
    return torch.mean(model.criterion(conv_output.view(-1) * torch.ones(batch_size))) + lmbda * l2_reg


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
