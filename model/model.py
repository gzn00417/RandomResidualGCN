import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

from model.layer import GCN


class GCNLayersRandomLeaps(nn.Module):
    def __init__(self, n_feat, dropout):
        super(GCNLayersRandomLeaps, self).__init__()

        self.gcn1 = GCN(n_feat, n_feat)
        self.gcn2 = GCN(n_feat, n_feat)
        self.gcn3 = GCN(n_feat, n_feat)
        self.gcn4 = GCN(n_feat, n_feat)
        self.gcn5 = GCN(n_feat, n_feat)
        self.layers = [self.gcn1, self.gcn2, self.gcn3, self.gcn4, self.gcn5]
        self.layer_num = len(self.layers)
        self.dropout = nn.Dropout(p=dropout)
        self.outputs = []
        self.skip_to = []
        self.ac_func = []

        self.build_structure()

    def forward(self, x, adj):
        for i in range(self.layer_num):
            x = self.layers[i](self.get_merged_x(x, i), adj)
            x = {
                "ReLU": F.relu(x),
                "Tanh": F.tanh(x),
                "SoftMax": F.softmax(x, dim=1),
                "ELU": F.elu(x),
            }.get(self.ac_func[i])
            x = self.dropout(x)
            self.outputs.append(x)
        return x

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


class ConvKB(nn.Module):
    def __init__(self, input_dim, in_channels, out_channels, drop_prob):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, (1, 3))
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((50 - 1 + 1) * out_channels, 1)
        self.criterion = nn.Softplus()

    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        out_conv = self.dropout(self.non_linearity(self.conv_layer(conv_input)))
        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output
