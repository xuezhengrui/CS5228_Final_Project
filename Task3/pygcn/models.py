import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import numpy as np


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.mlp_q = nn.Linear(nfeat, nhid)
        self.mlp_k = nn.Linear(nfeat, nhid)

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        if adj is None:
            q = self.mlp_q(x)
            k = self.mlp_k(x)
            att = q.matmul(k.t()) / np.sqrt(k.shape[-1])
            adj = F.softmax(att, dim=-1)

            
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc11(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc12(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.gc2(x, adj)

        # return F.log_softmax(x, dim=1)
        return x
