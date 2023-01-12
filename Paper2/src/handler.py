import torch as T
from dgl.data import FB15k237Dataset



class Handler:
    def __init__(self) -> None:
        self.dataset = FB15k237Dataset()

        self.train_graph = self.get_graph_tensor('train')
        self.train_adjacency_matrix = self.construct_adjacency_matrix('train')

        #self.test_graph  = self.get_graph_tensor('test')
        #self.test_adjacency_matrix  = self.construct_adjacency_matrix('test')


    def get_graph_tensor(self, split='train') -> T.tensor:
        graph  = self.dataset[0]
        mask   = graph.edata[f'{split}_mask']
        idx    = T.nonzero(mask, as_tuple=False).squeeze()

        src, dst = graph.find_edges(idx)
        rel_ids  = graph.edata['etype'][idx]

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





if __name__ == '__main__':
    test = Handler()
    print(test.train_graph.shape)
    print(test.train_graph)
    print(test.train_adjacency_matrix)
