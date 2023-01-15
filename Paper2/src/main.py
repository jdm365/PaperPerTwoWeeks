import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm
from handler import Handler



class RCGNLayer(nn.Module):
    def __init__(
            self, 
            input_dims: int, 
            output_dims: int, 
            adj_matrix: T.tensor = None,
            adj_tensor: T.sparse_coo_tensor = None
            ) -> None:
        super(RCGNLayer, self).__init__()
        '''
        input_dims  - input_dims
        output_dims - output_dims
        adj_matrix  - adjacency matrix with integer relationship ids
        adj_tensor  - sparse tensor with stacked adjacency matrices for
                      each relationship type.
        '''
        self.input_dims  = input_dims
        self.output_dims = output_dims

        if adj_matrix is not None:
            self.n_nodes     = adj_matrix.shape[-1]
            self.n_relations = int(T.max(adj_matrix).item()) + 1
        else:
            self.n_nodes     = adj_tensor.shape[-1]
            self.n_relations = adj_tensor.shape[0]

        self.adj_matrix  = adj_matrix
        self.adj_tensor  = adj_tensor

        ## Concat weight matrices of different relations
        self.W                 = nn.Parameter(T.empty(size=(self.n_relations, input_dims, output_dims)))
        self.W0                = nn.Parameter(T.empty((input_dims, output_dims)))
        self.inv_norm_constant = nn.Parameter(T.ones((self.n_relations, self.n_nodes)))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.W0)


    def forward(self, X: T.tensor) -> T.tensor:
        '''
        X - hidden states of all nodes incoming. dims -> (n_nodes, self.input_dims)
        '''

        vals = []
        
        if self.adj_matrix is None:
            for rel_idx in tqdm(range(self.n_relations)):
                rel_adj_matrix = self.adj_tensor[rel_idx]
                val = T.sparse.mm(rel_adj_matrix, X @ self.W[rel_idx])
                val *= self.inv_norm_constant[rel_idx].unsqueeze(-1)
                vals.append(val)

            incoming_messages = sum(vals)
            self_connection   = T.sparse.mm(X, self.W0)
        else:
            for rel_idx in range(self.n_relations):
                rel_adj_matrix = T.where(self.adj_matrix == rel_idx, 1, 0)
                val = rel_adj_matrix @ X @ self.W[rel_idx]
                val *= self.inv_norm_constant[rel_idx]

            incoming_messages = T.sum(vals)
            self_connection   = X @ self.W0

        return incoming_messages + self_connection




class RGCN(nn.Module):
    def __init__(
            self, 
            adj_matrix: T.tensor = None,
            adj_tensor: T.tensor = None,
            output_dims: int = 16,
            hidden_dims: int = 16,
            dtype: T.dtype = T.float32
            ) -> None:
        super(RGCN, self).__init__()
        ## input_dims first layer -> n_nodes; one-hot encoded.

        if adj_matrix is not None:
            n_nodes = adj_matrix.shape[-1]
        else:
            n_nodes = adj_tensor.shape[-1]

        self.dtype = dtype

        self.network = nn.Sequential(
                RCGNLayer(n_nodes, hidden_dims, adj_matrix, adj_tensor),
                nn.ReLU(),
                RCGNLayer(hidden_dims, output_dims, adj_matrix, adj_tensor)
                )


    def forward(self, X: T.tensor) -> T.tensor:
        if X.type != self.dtype:
            X = X.type(self.dtype)
            return self.network(X)
        return self.network(X)



class DistMult:
    def __init__(self):
        return



if __name__ == '__main__':
    ##################################
    ####         TESTING          ####
    ##################################

    handler = Handler(get_negative_examples=True)
    model   = RGCN(
            adj_matrix=None,
            adj_tensor=handler.train_adjacency_tensor
            )
    X       = T.diag(T.ones(handler.train_adjacency_matrix.shape[0])).type(T.long)
    print(model.forward(X))
