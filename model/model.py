import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

from model.layer import GCN
from config.args import args


class RandomResidualGCN(nn.Module):
    def __init__(self, num_entities, num_relations, init_emb_e=None, init_emb_rel=None):
        super().__init__()
        self.num_entity = num_entities
        self.num_relation = num_relations

        self.gcn1 = GCN(args.embedding_dim, args.embedding_dim)
        self.gcn2 = GCN(args.embedding_dim, args.embedding_dim)
        self.gcn3 = GCN(args.embedding_dim, args.embedding_dim)
        self.gcn4 = GCN(args.embedding_dim, args.embedding_dim)
        self.gcn5 = GCN(args.embedding_dim, args.embedding_dim)
        self.layers = [self.gcn1, self.gcn2, self.gcn3, self.gcn4, self.gcn5]
        self.layer_num = len(self.layers)
        # self.dropout = nn.Dropout(p=args.dropout)
        self.outputs = []
        self.skip_to = []
        self.ac_func = []

        self.build_structure()

        self.entity_embeddings = nn.Embedding(num_entities, args.embedding_dim) if init_emb_e is None else init_emb_e
        self.relation_embeddings = nn.Embedding(num_relations, args.embedding_dim) if init_emb_rel is None else init_emb_rel

    def forward(self, h, r, t, v, adj, loss_function):
        x = self.entity_embeddings(torch.arange(self.num_entity, device=h.device))
        for i in range(self.layer_num):
            x = self.layers[i](self.get_merged_x(x, i), adj)
            x = {
                "ReLU": F.relu(x),
                "Tanh": F.tanh(x),
                "SoftMax": F.softmax(x, dim=1),
                "ELU": F.elu(x),
            }.get(self.ac_func[i])
            # x = self.dropout(x)
            self.outputs.append(x)
        return loss_function((self.entity_embeddings(h), self.relation_embeddings(r), self.entity_embeddings(t)), v)

    def build_structure(self):
        for i in range(self.layer_num):
            self.skip_to.append(self.random_select_skip_to_layers(i))
            self.ac_func.append(self.random_select_activate_function(i))

    def random_select_skip_to_layers(self, current_layer_num):
        """randomly select layers which current layer is skipping to
        """
        return random.sample(
            range(current_layer_num + 2, self.layer_num),
            random.randint(
                0,
                (self.layer_num - current_layer_num - 2)
                if current_layer_num < self.layer_num - 2
                else 0,
            ),
        )

    def random_select_activate_function(self, current_layer_num):
        """randomly select activate function for current layer
        """
        return random.choice(["ReLU", "Tanh", "SoftMax", "ELU"])

    def get_merged_x(self, x, current_layer_num):
        """get all input for current layer and merge them by `kernel()`
        """
        skip_from = []
        for i in range(current_layer_num):
            for layer in self.skip_to[i]:
                if layer == current_layer_num:
                    skip_from.append(i)
                    break
        x_list = [x]
        for layer in skip_from:
            x_list.append(self.outputs[layer])
        return self.kernel(x_list)

    def kernel(self, x_list):
        """kernel for merging inputs
        """
        try:
            sum(x_list)
        except:
            print(F.relu(x_list[0]))
            print(x_list)
            raise Exception
        return F.relu(sum(x_list))

    def get_structure(self):
        return self.skip_to, self.ac_func

    def evaluate(self, h, r, t):
        rank = args.hits
        if rank < 1.0:
            rank = int(num_entity * rank)
        pos_len = len(h)
        hits = 0
        total_rank = 0.0
        total_reciprocal_rank = 0.0
        all_entity_embeddings = self.entity_embeddings(torch.arange(self.num_entity, device=h.device))
        for head, relation, tail in zip(h, r, t):
            true_tail_embedding = self.entity_embeddings(head) + self.relation_embeddings(relation)
            distances = torch.norm(all_entity_embeddings - true_tail_embedding.repeat(self.num_entity, 1), dim=1)
            sorted_indices = torch.argsort(distances, descending=False)
            if tail in sorted_indices[:rank]:
                hits += 1
            current_rank = int((sorted_indices == tail).nonzero()) + 1
            total_rank += current_rank
            total_reciprocal_rank += 1.0 / current_rank
        accuracy_hits = hits / pos_len
        mean_rank = total_rank / pos_len
        mean_reciprocal_rank = total_reciprocal_rank / pos_len
        return accuracy_hits, mean_rank, mean_reciprocal_rank


class DoubleGCN(nn.Module):
    def __init__(self, num_entities, num_relations, init_emb_e=None, init_emb_rel=None):
        super().__init__()
        self.num_entity = num_entities
        self.num_relation = num_relations
        self.gcn1 = GCN(args.embedding_dim, args.embedding_dim)
        self.gcn2 = GCN(args.embedding_dim, args.embedding_dim)
        self.entity_embeddings = nn.Embedding(num_entities, args.embedding_dim) if init_emb_e is None else init_emb_e
        self.relation_embeddings = nn.Embedding(num_relations, args.embedding_dim) if init_emb_rel is None else init_emb_rel

    def forward(self, h, r, t, v, adj, loss_function):
        x = self.entity_embeddings(torch.arange(self.num_entity, device=h.device))
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = self.gcn2(x, adj)
        x = F.relu(x)
        return loss_function((self.entity_embeddings(h), self.relation_embeddings(r), self.entity_embeddings(t)), v)

    def evaluate(self, h, r, t):
        rank = args.hits
        if rank < 1.0:
            rank = int(num_entity * rank)
        pos_len = len(h)
        hits = 0
        total_rank = 0.0
        total_reciprocal_rank = 0.0
        all_entity_embeddings = self.entity_embeddings(torch.arange(self.num_entity, device=h.device))
        for head, relation, tail in zip(h, r, t):
            true_tail_embedding = self.entity_embeddings(head) + self.relation_embeddings(relation)
            distances = torch.norm(all_entity_embeddings - true_tail_embedding.repeat(self.num_entity, 1), dim=1)
            sorted_indices = torch.argsort(distances, descending=False)
            if tail in sorted_indices[:rank]:
                hits += 1
            current_rank = int((sorted_indices == tail).nonzero()) + 1
            total_rank += current_rank
            total_reciprocal_rank += 1.0 / current_rank
        accuracy_hits = hits / pos_len
        mean_rank = total_rank / pos_len
        mean_reciprocal_rank = total_reciprocal_rank / pos_len
        return accuracy_hits, mean_rank, mean_reciprocal_rank


class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, init_emb_e=None, init_emb_rel=None):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_e = nn.Embedding(num_entities, args.embedding_dim) if init_emb_e is None else init_emb_e
        self.emb_rel = nn.Embedding(num_relations, args.embedding_dim) if init_emb_rel is None else init_emb_rel
        nn.init.xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_normal_(self.emb_rel.weight.data)
        self.inp_drop = nn.Dropout(args.dropout)
        self.hidden_drop = nn.Dropout(args.dropout)
        self.feature_map_drop = nn.Dropout2d(args.dropout)
        self.loss = nn.BCELoss()
        self.emb_dim1 = int((args.embedding_dim * 0.5) ** 0.5)
        self.emb_dim2 = args.embedding_dim // self.emb_dim1
        self.conv1 = nn.Conv2d(1, args.out_channels, (3, 3), 1, 0, bias=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(args.out_channels)
        self.bn2 = nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        self.fc = nn.Linear(args.out_channels * (self.emb_dim1 * 2 - 2) * (self.emb_dim2 - 2), args.embedding_dim)

    def forward(self, e1, rel, e2):
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        true = F.one_hot(e2, self.num_entities).float()
        true = ((1.0-0.1)*true) + (1.0/true.size(1))
        return pred, self.loss(pred, true)

    def evaluate(self, h, r, t):
        rank = args.hits
        if rank < 1.0:
            rank = int(num_entity * rank)
        pos_len = len(h)
        hits = 0
        total_rank = 0.0
        total_reciprocal_rank = 0.0
        for head, relation, tail in zip(h, r, t):
            pred, loss = self(head[np.newaxis], relation[np.newaxis], tail[np.newaxis])
            sorted_indices = torch.argsort(pred[0], descending=True)
            if tail in sorted_indices[:rank]:
                hits += 1
            current_rank = int((sorted_indices == tail).nonzero()) + 1
            total_rank += current_rank
            total_reciprocal_rank += 1.0 / current_rank
        accuracy_hits = hits / pos_len
        mean_rank = total_rank / pos_len
        mean_reciprocal_rank = total_reciprocal_rank / pos_len
        return accuracy_hits, mean_rank, mean_reciprocal_rank
