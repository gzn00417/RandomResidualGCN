import os
from collections import defaultdict
import random
import torch
from torch.utils.data import Dataset
import torch.nn as nn


class WN18Dataset(Dataset):

    def __init__(self, data_path: str, dataset_name: str in ['train', 'valid', 'test'], entity2id: dict, relation2id: dict, has_negative_data: bool):
        super(WN18Dataset, self).__init__()
        self.data_path = os.path.join(data_path, dataset_name + '.txt')
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.entities = list(self.entity2id.values())
        self.relations = list(self.relation2id.values())
        self.edges = []# [{'h': h, 'r': r, 't': t, 'v': v}]
        self.edges_map = dict(zip(self.entities, [defaultdict(None) for _ in self.entities]))  # key1: h, key2: r, value: t
        self.load_pos_data()
        if has_negative_data:
            self.generate_neg_data()
        self.num_triple = len(self.edges)

    def __getitem__(self, index):
        return self.edges[index]['h'], self.edges[index]['r'], self.edges[index]['t'], self.edges[index]['v']

    def __len__(self):
        return self.num_triple

    @property
    def data(self):
        return self.edges

    def add_edge(self, h, r, t, v: int in [-1, 1] = 1):
        self.edges.append({'h': h, 'r': r, 't': t, 'v': v})

    def load_pos_data(self):
        with open(self.data_path) as f:
            lines = f.readlines()
        for line in lines:
            h, r, t = line.strip().split()
            h = self.entity2id[h]
            r = self.relation2id[r]
            t = self.entity2id[t]
            self.add_edge(h, r, t, 1)
            self.edges_map[h][r] = t

    def generate_neg_data(self, p: float = 0.2):
        for h in self.entities:
            for r in self.relations:
                if random.random() > p:  # p: generate probability
                    continue
                if r not in self.edges_map[h] or self.edges_map[h][r] is None:  # -1 if there is no edge, else >= 0
                    t = random.randint(0, len(self.entities) - 1)
                else:
                    t = random.randint(0, len(self.entities) - 1)
                    while t == self.edges_map[h][r]:
                        t = random.randint(0, len(self.entities) - 1)
                self.add_edge(h, r, t, -1)
