import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from handler import Handler



class RCGNLayer(nn.Module):
    def __init__(
            self, 
            input_dims: int, 
            output_dims: int, 
            adj_matrix: T.tensor,
            edges: T.tensor
            ) -> None:
        super(RCGNLayer, self).__init__()
        '''
        edges - edge triples of type (src, dst, rel_type). dims -> (3, n_edges)
        '''
        self.input_dims  = input_dims
        self.output_dims = output_dims
        self.n_nodes     = adj_matrix.shape[0] 
        self.n_relations = int(T.max(edges[2]).item()) + 1
        self.adj_matrix  = adj_matrix
        self.edges       = edges

        ## Concat weight matrices of different relations
        self.W                 = nn.Parameter(T.empty(size=(self.n_relations, input_dims, output_dims)))
        self.W0                = nn.Parameter(T.empty((input_dims, output_dims)))
        self.inv_norm_constant = nn.Parameter(T.ones((self.n_nodes, self.n_relations)))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.W0)



    def forward(self, X: T.tensor) -> T.tensor:
        '''
        X - hidden states of all nodes incoming. dims -> (batch_size, n_nodes, self.input_dims)
        '''
        ## TODO: Fix so no tensor creation in forward pass.
        output_hidden_states = T.zeros_like(X, requires_grad=True)
        for idx in range(self.n_nodes):
            output_hidden_states[:, self.edges[1, idx]] += self.pass_message(
                    X=X, 
                    src_idx=self.edges[0, idx], 
                    dst_idx=self.edges[1, idx], 
                    rel_idx=self.edges[2, idx]
                    )
        output_hidden_states += self.add_self_connections(X, output_hidden_states)
        return output_hidden_states



    def pass_message(
            self, 
            X: T.tensor, 
            src_idx: T.tensor, 
            dst_idx: T.tensor, 
            rel_idx: T.tensor
            ) -> T.tensor:
        '''
        X        - hidden states of all nodes incoming. dims -> (batch_size, n_nodes, self.input_dims)
        src_idx  - source node indices.
        dst_idx  - destination node indices.
        rel_idx  - relationship type indices.
        output_hidden_states - forward propogated hidden states of all nodes. dims -> (batch_size, n_nodes, self.input_dims)
        '''
        incoming_aggregated_state = X[:, src_idx] @ self.W[rel_idx]    ## dims -> (batch_size, output_dims)
        incoming_aggregated_state = self.inv_norm_constant[dst_idx, rel_idx]

        return incoming_aggregated_state 



    def add_self_connections(self, X: T.tensor, output_hidden_states: T.tensor) -> None:
        '''
        X - hidden states of all nodes. dims -> (batch_size, n_nodes, self.input_dims)
        output_hidden_states - forward propogated hidden states of all nodes. dims -> (n_nodes, self.input_dims)
        '''
        return X @ self.W0 



class RGCN(nn.Module):
    def __init__(
            self, 
            adj_matrix: T.tensor,
            edges: T.tensor,
            output_dims: int = 2,   ## 2 for link prediction. 
            hidden_dims: int = 16
            ) -> None:
        super(RGCN, self).__init__()
        ## input_dims -> n_nodes; one-hot encoded.
        n_nodes     = adj_matrix.shape[0]
        n_relations = edges.shape[-1]

        self.network = nn.Sequential(
                RCGNLayer(n_nodes, hidden_dims, adj_matrix, edges),
                nn.ReLU(),
                RCGNLayer(hidden_dims, output_dims, adj_matrix, edges),
                nn.Softmax(dim=-1)
                )


    def forward(self, X: T.tensor) -> T.tensor:
        if len(X.shape) == 2:
            ## Add batch dimension if needed.
            X = X.unsqueeze(dim=0)

        return self.network(X)








if __name__ == '__main__':
    ##################################
    ####         TESTING          ####
    ##################################

    handler = Handler()
    model   = RGCN(handler.train_adjacency_matrix, handler.train_graph)
    X       = T.diag(T.ones(handler.train_adjacency_matrix.shape[0]))
    print(model.forward(X))
