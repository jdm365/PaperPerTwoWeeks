import torch as T
from dgl.data import FB15k237Dataset



class Handler:
    def __init__(self, get_negative_examples=False) -> None:
        self.dataset = FB15k237Dataset()

        self.train_graph = self.get_graph_tensor(
                'train',
                get_negative_examples=get_negative_examples
                )
        self.train_adjacency_matrix = self.construct_adjacency_matrix('train')
        self.train_adjacency_tensor = self.construct_sparse_adjacency_tensor('train')

        #self.test_graph  = self.get_graph_tensor('test')
        #self.test_adjacency_matrix  = self.construct_adjacency_matrix('test')


    def get_graph_tensor(self, split='train', get_negative_examples=True) -> T.tensor:
        graph  = self.dataset[0]
        mask   = graph.edata[f'{split}_mask']
        if not get_negative_examples:
            idxs = T.nonzero(mask, as_tuple=False).squeeze()
            self.mask = None
        else:
            idxs = T.arange(len(mask)).squeeze()
            self.mask = mask

        src, dst = graph.find_edges(idxs)
        rel_ids  = graph.edata['etype'][idxs]
        rel_ids  = rel_ids % self.dataset.num_rels

        return T.stack([src, dst, rel_ids])


    def construct_adjacency_matrix(self, split='train') -> T.tensor:
        if split == 'train':
            graph = self.train_graph
        elif split == 'test':
            graph = self.test_graph

        n_nodes = T.max(graph[0:1]) + 1

        A = T.zeros((n_nodes, n_nodes))
        for idx in range(n_nodes):
            src_idx = graph[0, idx]
            dst_idx = graph[1, idx]
            rel_idx = graph[2, idx]

            A[src_idx, dst_idx] = rel_idx
        return A


    def construct_sparse_adjacency_tensor(self, split='train') -> T.tensor:
        if split == 'train':
            graph = self.train_graph
        elif split == 'test':
            graph = self.test_graph

        n_nodes     = T.max(graph[0:1]) + 1
        n_relations = T.max(graph[2]) + 1

        graph = T.stack([graph[2], graph[0], graph[1]])

        A = T.sparse_coo_tensor(
                indices=graph,
                values=T.ones(graph.shape[1]),
                size=(n_relations, n_nodes, n_nodes),
                dtype=T.float32
                )
        return A






if __name__ == '__main__':
    test = Handler()
    print(test.train_graph.shape)
    print(test.train_graph)
    print(test.train_adjacency_matrix)
    print(test.train_adjacency_tensor)
