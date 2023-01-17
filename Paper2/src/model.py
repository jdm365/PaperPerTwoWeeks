import torch as T
import torch.nn as nn
import torch.nn.functional as F
from  torch.utils.data import Dataset, DataLoader
import torch.optim as optim
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
            for rel_idx in range(self.n_relations):
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
            dtype: T.dtype = T.float32,
            device: T.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            ) -> None:
        super(RGCN, self).__init__()
        ## input_dims first layer -> n_nodes; one-hot encoded.

        if adj_matrix is not None:
            n_nodes = adj_matrix.shape[-1]
        else:
            n_nodes = adj_tensor.shape[-1]

        self.dtype = dtype

        if adj_matrix is not None:
            adj_matrix = adj_matrix.to(device)

        if adj_tensor is not None:
            adj_tensor = adj_tensor.to(device)

        self.network = nn.Sequential(
                RCGNLayer(n_nodes, hidden_dims, adj_matrix, adj_tensor),
                nn.ReLU(),
                RCGNLayer(hidden_dims, output_dims, adj_matrix, adj_tensor)
                )

        self.device = device
        self.to(self.device)


    def forward(self, X: T.tensor) -> T.tensor:
        X = X.to(self.device)
        if X.type != self.dtype:
            X = X.type(self.dtype)
            return self.network(X)
        return self.network(X)



class DistMult(nn.Module):
    def __init__(
            self, 
            n_relations: int, 
            embed_dims: int = 16,
            device: T.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            ) -> None:
        super(DistMult, self).__init__()
        self.embed_dims = embed_dims

        ## Just store the diagonal
        self.M = nn.Parameter(T.empty(n_relations, embed_dims))

        nn.init.xavier_uniform_(self.M)

        self.device = device
        self.to(self.device)



    def forward(self, src_emb: T.tensor, dst_emb: T.tensor, rel_idx: int) -> T.tensor:
        ## Assume src_emb and dst_emb shapes are the same.
        batch_size, embed_dims = src_emb.shape
        identity = T.stack(batch_size * [T.eye(embed_dims, device=self.device)])

        M_diag = identity * self.M[rel_idx].unsqueeze(-1)
        src_emb = src_emb.unsqueeze(-1).transpose(-2, -1)
        dst_emb = dst_emb.unsqueeze(-1)

        return src_emb @ M_diag @ dst_emb





class LinkPredictionRCGN(nn.Module):
    def __init__(
            self, 
            adj_matrix: T.tensor = None,
            adj_tensor: T.tensor = None,
            output_dims: int = 16,
            hidden_dims: int = 16,
            dtype: T.dtype = T.float32,
            device: T.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu'),
            lr: float = 1e-1,
            triples_loader: DataLoader = None
            ) -> None:
        super(LinkPredictionRCGN, self).__init__()

        if adj_tensor is not None:
            n_relations = adj_tensor.shape[0]
        else:
            n_relations = T.max(adj_matrix) + 1

        self.rgcn = RGCN(
                adj_matrix=adj_matrix,
                adj_tensor=adj_tensor,
                output_dims=output_dims,
                hidden_dims=hidden_dims,
                dtype=dtype,
                device=device
                )
        self.dist_mult = DistMult(
                n_relations=n_relations, 
                embed_dims=hidden_dims,
                device=device
                )
        self.triples_loader = triples_loader

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device    = device
        self.to(self.device)


    def forward(self, X: T.tensor, triples: T.tensor) -> T.tensor:
        '''
        triples - (src, dst, rel_type); dims -> (3, n_nodes + m * n_nodes)
        n_nodes true triples. m * n_nodes corrupted triples.
        '''
        node_embeddings = self.rgcn(X)

        distances = []
        mask_vals = []
        for idx, (src_idx, dst_idx, rel_idx, mask_val) in enumerate(self.triples_loader):
            if idx == 1:
                break
            distances.append(
                    self.dist_mult(
                        src_emb=node_embeddings[src_idx],
                        dst_emb=node_embeddings[dst_idx],
                        rel_idx=rel_idx
                        )
                    )
            mask_vals.append(mask_val)
        return T.cat(distances).squeeze(), T.cat(mask_vals).squeeze()


if __name__ == '__main__':
    ##################################
    ####         TESTING          ####
    ##################################

    negative_sampling_ratio = 1.0

    handler = Handler(negative_sampling_ratio=negative_sampling_ratio)

    X = T.diag(T.ones(handler.train_adjacency_matrix.shape[0])).type(T.long)
    '''
    model   = RGCN(
            adj_matrix=None,
            adj_tensor=handler.train_adjacency_tensor
            )
    y = model.forward(X)

    dist_mult = DistMult(n_relations=handler.train_adjacency_tensor.shape[0])
    src_idx, dst_idx, rel_idx = handler.train_triples[:, -1]

    src_emb = y[src_idx] 
    dst_emb = y[dst_idx] 

    distance = dist_mult.forward(src_emb, dst_emb, rel_idx)
    print(distance)
    '''



    link_pred_model = LinkPredictionRCGN(
            hidden_dims=64,
            output_dims=64,
            adj_tensor=handler.train_adjacency_tensor,
            triples_loader=handler.train_triples_loader
            )
    loss_fn = nn.BCELoss()

    N_STEPS = 50

    progress_bar = tqdm(total=N_STEPS)
    for idx in range(N_STEPS):
        y, mask_vals = link_pred_model.forward(X, handler.train_triples)
        y    = T.sigmoid(y)
        loss = loss_fn(y, mask_vals)
        loss.backward()
        link_pred_model.optimizer.step()
        link_pred_model.optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_description(f'Loss: {loss.item()}')
