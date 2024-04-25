import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """ one layer of GCN """

    def __init__(self, input_dim, output_dim, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        if self.dropout is not None:
            h = self.dropout(h)
        x = h @ self.W
        x = adj @ x
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        return x
