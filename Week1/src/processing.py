import torch as T
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from torch.nn.data import Dataset, DataLoader




class TextDataset(Dataset):
    def __init__(self, encoded_text: T.tensor):
        self.encoded_text = encoded_text

    def __len__(self):
        return len(self.encoded_text)

    def __getitem__(self, idx):
        return self.encoded_text[idx]


class Preprocessing:
    def __init__(self, filename=None, word_piece=True) -> None:
        ## Read data function
        self.text_data = None

        if word_piece:
            self.tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    
    def encode(self, dtype=T.float16) -> None:
        self.encoded_data = self.tokenizer.encode(self.text_data)
        self.encoded_data = self.encoded_data.astype(dtype)


    def create_dataloader(self, batch_size=64, num_workers=4) -> None:
        try:
            self.encoded_text
        except:
            print('Must call `encode` method before creating a dataloader')

        self.dataloader = DataLoader(
                TextDataset(self.encoded_text),
                batch_size=batch_size,
                num_workers=num_workers
                )
