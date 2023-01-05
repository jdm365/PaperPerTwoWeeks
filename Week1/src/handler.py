import torch as T
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import datasets
from tokenizers import Tokenizer
from transformers import BertTokenizer, DataCollatorForLanguageModeling, BatchEncoding
from torch.utils.data import Dataset, DataLoader, RandomSampler
import multiprocessing as mp
import joblib
import sys
import gc


class TorchDatasetWrapper(Dataset):
    def __init__(self, huggingface_dataset) -> None:
        super(Dataset, self).__init__()

        self.huggingface_dataset = huggingface_dataset


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
        self.text_data   = datasets.load_dataset(dataset_name, 'plain_text', split='train')['text']
        self.max_length  = max_length
        self.subset_size = subset_size

        ## Concatenate dataset to be of max_length.
        self.condense_dataset()

        self.tokenizer   = BertTokenizer.from_pretrained('bert-base-cased', cls_token='')
        self.vocab_size  = len(self.tokenizer)
        self.device      = device
        self.dtype       = dtype


    def condense_dataset(self):
        if self.subset_size is not None:
            self.text_data = self.text_data[:self.subset_size]

        self.text_data = ' [SEP] '.join(self.text_data)
        self.text_data = self.text_data.split(' ')

        n_batches = 1 + (len(self.text_data) // self.max_length)

        final_dataset = []
        for idx in tqdm(range(n_batches), desc='Preparing Dataset'):
            final_dataset.append(' '.join(self.text_data[idx * self.max_length:(idx + 1) * self.max_length]))
        final_dataset.append(' '.join(self.text_data[n_batches * self.max_length:]))

        self.text_data = final_dataset
        del final_dataset
        gc.collect()


    def get_dataloader(self, batch_size=1024, num_workers=4, pin_memory=True) -> DataLoader:
        collate_fn = CustomCollator(
                tokenizer=self.tokenizer,
                mlm_prob=0.15,
                max_length=self.max_length,
                dtype=self.dtype,
                batch_size=batch_size
                )

        dataloader = DataLoader(
                dataset=TorchDatasetWrapper(self.text_data),
                collate_fn=collate_fn,
                shuffle=False,
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
            truncation=True,
            batch_size=1024
            ) -> None:
        self.tokenizer      = tokenizer
        self.mlm_prob       = mlm_prob
        self.dtype          = dtype
        self.max_length     = max_length
        self.truncation     = truncation
        self.batch_size     = batch_size

        self.pad_token_id   = self.tokenizer(['[PAD]'], add_special_tokens=False)['input_ids'][0][0]
        self.mask_token_id  = self.tokenizer(['[MASK]'], add_special_tokens=False)['input_ids'][0][0]
        self.sep_token_id   = self.tokenizer(['[SEP]'], add_special_tokens=False)['input_ids'][0][0]


    def __call__(self, batch) -> (T.tensor, T.tensor):
        token_ids, attention_mask = self.encode(batch)

        token_ids      = T.tensor(token_ids, dtype=T.long)
        attention_mask = T.tensor(attention_mask, dtype=T.long)

        token_ids, attention_mask, original_labels, mask_idxs = self.mask_inputs(token_ids, attention_mask)

        return token_ids, attention_mask, original_labels, mask_idxs


    def encode(self, batch: list) -> (list, list):
        tokenized_output = self.tokenizer(
                batch,
                padding='max_length',
                truncation=self.truncation,
                max_length=self.max_length
                )
        token_ids      = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask']
        return token_ids, attention_mask



    def mask_inputs(self, token_ids, attention_mask) -> (T.tensor, T.tensor, T.tensor, T.tensor):
        original_ids    = token_ids.clone().flatten()
        token_idxs      = T.arange(0, original_ids.shape[0])

        '''
        original_labels = T.zeros((token_ids.shape[0] * token_ids.shape[1], self.tokenizer.vocab_size))
        original_labels[token_idxs, original_ids] = 1
        '''

        ## mask (i.e. (1, 0, 0, 1, ...)) where mask tokens (i.e. '[MASK]') are applied.
        mask_mask = T.zeros_like(attention_mask)
        ##for idx, batch in enumerate(attention_mask):
        for idx, batch in enumerate(token_ids):
            nonpad_idxs = T.argwhere(batch != self.sep_token_id).squeeze()
            mask_idxs = np.random.choice(
                    nonpad_idxs, 
                    size=int(nonpad_idxs.shape[0] * self.mlm_prob), 
                    replace=False
                    )
            mask_mask[idx, mask_idxs] = 1
            token_ids[idx, mask_idxs] = self.mask_token_id

        mask_idxs = T.argwhere(mask_mask.flatten() != 0).squeeze()

        #original_ids[mask_idxs] = -100
        return token_ids, attention_mask, original_ids[mask_idxs], mask_idxs 





if __name__ == '__main__':
    ###############################
    ####        TESTS          ####
    ###############################

    handler = DataHandler(dataset_name='bookcorpus', subset_size=10000)
    dataloader = handler.get_dataloader(batch_size=32)

    for idx, (X, attention_mask, y, mask_idxs) in enumerate(dataloader):
        if idx == 1:
            break










