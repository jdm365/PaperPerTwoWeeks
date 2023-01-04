import torch as T
import torch.nn.functional as F
import numpy as np
import datasets
from tokenizers import Tokenizer
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader, RandomSampler
import multiprocessing as mp
import joblib
import sys


class TorchDatasetWrapper(Dataset):
    def __init__(
            self, 
            huggingface_dataset, 
            text_col='text', 
            subset_size=None,
            device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            ) -> None:
        super(Dataset, self).__init__()

        self.huggingface_dataset = huggingface_dataset[text_col]
        if subset_size is not None:
            self.huggingface_dataset = self.huggingface_dataset[:subset_size]

        self.device = device


    def __getitem__(self, idx) -> list:
        return self.huggingface_dataset[idx]


    def __len__(self) -> int:
        return len(self.huggingface_dataset)


class DataHandler:
    def __init__(
            self, 
            dataset_name='bookcorpus', 
            max_length=512, 
            subset_size=None,
            dtype=T.float16,
            device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            ) -> None:
        ## Read data function
        self.text_data   = datasets.load_dataset(dataset_name, 'plain_text', split='train')
        self.max_length  = max_length
        self.tokenizer   = BertTokenizer.from_pretrained('bert-base-cased')
        self.vocab_size  = len(self.tokenizer)
        self.subset_size = subset_size
        self.device      = device
        self.dtype       = dtype


    def get_dataloader(self, batch_size=1024, num_workers=4, pin_memory=True) -> DataLoader:
        collate_fn = CustomCollator(
                tokenizer=self.tokenizer,
                mlm_prob=0.15,
                max_length=self.max_length,
                dtype=self.dtype
                )

        dataloader = DataLoader(
                dataset=TorchDatasetWrapper(
                    self.text_data, 
                    subset_size=self.subset_size,
                    device=self.device
                    ),
                collate_fn=collate_fn,
                shuffle=True,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                prefetch_factor=2,
                persistent_workers=True
                )
        return dataloader



class CustomCollator:
    def __init__(
            self, 
            tokenizer, 
            mlm_prob=0.15, 
            dtype=T.float16, 
            max_length=512, 
            truncation=True
            ) -> None:
        self.tokenizer      = tokenizer
        self.mlm_prob       = mlm_prob
        self.dtype          = dtype
        self.max_length     = max_length
        self.truncation     = truncation

        self.pad_token_id   = self.tokenizer(['[PAD]'], add_special_tokens=False)['input_ids'][0][0]
        self.mask_token_id  = self.tokenizer(['[MASK]'], add_special_tokens=False)['input_ids'][0][0]


    def __call__(self, batch) -> (T.tensor, T.tensor):
        tokenized_output = self.tokenizer(
                batch,
                padding='max_length',
                truncation=self.truncation,
                max_length=self.max_length
                )
        token_ids      = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask']
        token_ids      = T.tensor(token_ids, dtype=T.long)
        attention_mask = T.tensor(attention_mask, dtype=T.long)

        token_ids, attention_mask, original_labels, mask_mask = self.mask_inputs(token_ids, attention_mask)

        return token_ids, attention_mask, original_labels, mask_mask 


    def encode_dense(self, batch: list, batch_size: int) -> list:
        batch = batch.join(' [SEP] ')

        token_ids      = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask']
        token_ids      = T.tensor(token_ids, dtype=T.long)
        attention_mask = T.tensor(attention_mask, dtype=T.long)

        num_tokens = len(token_ids)

        if num_tokens >= self.max_length * batch_size:
            token_ids      = token_ids[:self.max_length * batch_size].reshape(batch_size, self.max_length)
            attention_mask = attention_mask[:self.max_length * batch_size].reshape(batch_size, self.max_length)
        else:
            diff = num_tokens - self.max_length * batch_size

            padding_size          = diff // batch_size
            residual_padding_size = diff % batch_size

            token_ids_head = token_ids[:-1].reshape(batch_size - 1, self.max_length - padding_size)
            token_ids_tail = token_ids[-1].reshape(self.max_length - residual_padding_size)

            token_ids_head = token_ids_head.cat(
                    T.full(size=(batch_size - 1, padding_size), fill_value=self.pad_token_id),
                    dim=1
                    )
            token_ids_tail = token_ids_tail.cat(
                    T.full(size=residual_padding_size, fill_value=self.pad_token_id),
                    dim=0
                    )

            token_ids = token_ids_head.cat(token_ids_tail, dim=0)

            attention_mask_head = attention_mask[:-1].reshape(batch_size - 1, self.max_length - padding_size)
            attention_mask_tail = attention_mask[-1].reshape(self.max_length - residual_padding_size)

            attention_mask_head = attention_mask_head.cat(
                    T.full(size=(batch_size - 1, padding_size), fill_value=self.pad_token_id), 
                    dim=1
                    )
            attention_mask_tail = attention_mask_tail.cat(
                    T.full(size=residual_padding_size, fill_value=self.pad_token_id),
                    dim=0
                    )

            attention_mask = attention_mask_head.cat(attention_mask_tail, dim=0)
        return token_ids, attention_mask


    def mask_inputs(self, token_ids, attention_mask) -> (T.tensor, T.tensor, T.tensor, T.tensor):
        original_labels = token_ids.clone()

        ## mask (i.e. (1, 0, 0, 1, ...)) where mask tokens (i.e. '[MASK]') are applied.
        mask_mask = T.zeros_like(attention_mask)
        for idx, batch in enumerate(attention_mask):
            nonpad_idxs = T.argwhere(batch).squeeze()
            mask_idxs = np.random.choice(
                    nonpad_idxs, 
                    size=int(nonpad_idxs.shape[0] * self.mlm_prob), 
                    replace=False
                    )
            mask_mask[idx, mask_idxs] = 1
            token_ids[idx, mask_idxs] = self.mask_token_id
        return token_ids, attention_mask, original_labels, mask_mask





if __name__ == '__main__':
    handler = DataHandler(dataset_name='bookcorpus')
    dataloader = handler.get_dataloader(batch_size=32)

    for idx, (X, attention_mask, y) in enumerate(dataloader):
        print(X.shape, y.shape)
        if idx == 1:
            break










