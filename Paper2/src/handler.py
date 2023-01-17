import torch as T
from torch.utils.data import Dataset, DataLoader
from dgl.data import FB15k237Dataset



class Handler:
    def __init__(self, negative_sampling_ratio=0.0) -> None:
        self.dataset = FB15k237Dataset()

        self.train_triples, self.train_mask = self.get_graph_tensor(
                'train',
                negative_sampling_ratio=negative_sampling_ratio
                )
        self.train_adjacency_matrix = self.construct_adjacency_matrix('train')
        self.train_adjacency_tensor = self.construct_sparse_adjacency_tensor('train')

        self.train_triples_loader = DataLoader(
                TriplesDataset(self.train_triples, self.train_mask),
                batch_size=1024,
                num_workers=0,
                #pin_memory=True,
                #prefetch_factor=1,
                #persistent_workers=True
                )

        #self.test_triples, self.test_mask = self.get_graph_tensor('test')
        #self.test_adjacency_matrix = self.construct_adjacency_matrix('test')


    def get_graph_tensor(self, split='train', negative_sampling_ratio=2.5) -> (T.tensor, T.tensor):
        graph  = self.dataset[0]
        mask   = graph.edata[f'{split}_mask']
        idxs = T.nonzero(mask, as_tuple=False).squeeze()

        src, dst = graph.find_edges(idxs)
        rel_ids  = graph.edata['etype'][idxs]
        rel_ids  = rel_ids % self.dataset.num_rels
        
        true_triples = T.stack([src, dst, rel_ids])

        n_false_triples = int(negative_sampling_ratio * true_triples.shape[-1])

        rand_srcs = T.randint(int(T.max(src) + 1), size=(n_false_triples // 2,))
        rand_dsts = T.randint(int(T.max(dst) + 1), size=(n_false_triples // 2,))
        rand_node_idxs = T.randint(T.max(dst) + 1, size=(n_false_triples,))

        false_triples = true_triples[:, rand_node_idxs]
        odd_idxs      = 1 + (T.arange(n_false_triples // 2) * 2)
        even_idxs     = T.arange(n_false_triples // 2) * 2
        false_triples[0, odd_idxs]  = rand_srcs
        false_triples[1, even_idxs] = rand_dsts

        mask    = T.cat((T.ones(true_triples.shape[-1]), T.zeros(false_triples.shape[-1])), dim=-1)
        triples = T.cat((true_triples, false_triples), dim=-1)

        permute_idxs = T.randperm(triples.shape[-1])
        return triples[:, permute_idxs], mask[permute_idxs]#.type(T.long)



    def construct_adjacency_matrix(self, split='train') -> T.tensor:
        if split == 'train':
            triples = self.train_triples
        elif split == 'test':
            triples = self.test_triples

        n_nodes = T.max(triples[0:1]) + 1

        A = T.zeros((n_nodes, n_nodes))
        for idx in range(n_nodes):
            src_idx = triples[0, idx]
            dst_idx = triples[1, idx]
            rel_idx = triples[2, idx]

            A[src_idx, dst_idx] = rel_idx
        return A


    def construct_sparse_adjacency_tensor(self, split='train') -> T.tensor:
        if split == 'train':
            triples = self.train_triples
        elif split == 'test':
            triples = self.test_triples

        n_nodes     = T.max(triples[0:1]) + 1
        n_relations = T.max(triples[2]) + 1

        triples = T.stack([triples[2], triples[0], triples[1]])

        A = T.sparse_coo_tensor(
                indices=triples,
                values=T.ones(triples.shape[1]),
                size=(n_relations, n_nodes, n_nodes),
                dtype=T.float32
                )
        return A


class TriplesDataset(Dataset):
    def __init__(
            self, 
            triples: T.tensor, 
            mask: T.tensor,
            device: T.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            ) -> None:
        self.triples = triples
        self.mask    = mask
        self.device  = device


    def __len__(self) -> int:
        return self.triples.shape[-1]


    def __getitem__(self, idx: int) -> (int, int, int, int):
        src_idx  = self.triples[0, idx].to(self.device)
        dst_idx  = self.triples[1, idx].to(self.device)
        rel_idx  = self.triples[2, idx].to(self.device)
        mask_val = self.mask[idx].to(self.device)
        return src_idx, dst_idx, rel_idx, mask_val






if __name__ == '__main__':
    test = Handler()
    print(test.train_triples.shape)
    print(test.train_triples)
    print(test.train_adjacency_matrix)
    print(test.train_adjacency_tensor)
