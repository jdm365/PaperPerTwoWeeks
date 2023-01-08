import torch as T
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm import tqdm
import pandas as pd
import gc
import sys
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from torch.utils.data import Dataset, DataLoader
from handler import DataHandler 
from model import *
from test_model import HuggingFaceBertWrapper
from config import train_configs


CONTINUE_FROM_CHECKPOINT     = True 
TEST_WITH_HUGGING_FACE_MODEL = False 
LOSS_RUNNING_MEAN_LENGTH     = 500
SHOW_PROGRESS                = True
DEBUG                        = False 
MICRO_BATCH_SIZE             = 32

def pretrain(
        n_epochs=1,
        lr=1e-4,
        batch_size=1536,
        dtype=T.float16,
        model_file='../trained_models/CrammingModel.pt',
        num_workers=4,
        pin_memory=True,
        seq_length=128,
        embed_dims=768, 
        num_heads=12, 
        has_bias=True,
        dropout_rate=0.2,
        n_encoder_blocks=12,
        mlp_expansion_factor=4,
        use_gpu=True,
        loss_fn=T.nn.CrossEntropyLoss(ignore_index=-100)
        ) -> None:
    ## Ensure empty cache. Should be done by operating system + cuda but can't hurt.
    T.cuda.empty_cache()

    device  = T.device('cuda:0' if use_gpu else 'cpu')
    handler = DataHandler(
            dataset_name='bookcorpus', 
            max_length=seq_length,
            dtype=dtype,
            device=device
            )

    ## Set `batch_size` to MICRO_BATCH_SIZE for memory reasons. Will make backward pass only 
    ## when actual `batch_size` inputs are processed.
    dataloader = handler.get_dataloader(batch_size=MICRO_BATCH_SIZE)

    if TEST_WITH_HUGGING_FACE_MODEL:
        ## Use Huggingface Bert model for MaskedLM to compare to custom model.
        cramming_model = HuggingFaceBertWrapper(
            vocab_size=handler.tokenizer.vocab_size, 
            hidden_size=embed_dims, 
            num_hidden_layers=n_encoder_blocks,
            hidden_dropout_prob=dropout_rate,
            lr=lr,
            device=device
            )
    else:
        cramming_model = CrammingTransformer(
                vocab_size=handler.tokenizer.vocab_size,
                seq_length=seq_length,
                embed_dims=embed_dims,
                num_heads=num_heads, 
                has_bias=has_bias,
                dropout_rate=dropout_rate,
                n_encoder_blocks=n_encoder_blocks,
                mlp_expansion_factor=mlp_expansion_factor,
                lr=lr,
                device=device
                )
        if CONTINUE_FROM_CHECKPOINT:
            cramming_model.load_model(model_file=model_file)


    progress_bar = tqdm(total=len(dataloader) * n_epochs)

    losses = []
    best_loss = 1e12
    for epoch in range(n_epochs):
        for idx, (X, attention_mask, y) in enumerate(dataloader):
            X              = X.to(cramming_model.device)

            ## Mask is None in fully packed token case.
            #attention_mask = attention_mask.to(cramming_model.device)
            y              = y.to(cramming_model.device)

            with T.cuda.amp.autocast():
                out = cramming_model.forward(X, None)
                out = out.view(-1, handler.tokenizer.vocab_size)
                
                loss = loss_fn(out, y)


            loss.backward()

            if idx % (batch_size // MICRO_BATCH_SIZE) == 0:
                T.nn.utils.clip_grad_value_(cramming_model.parameters(), clip_value=0.5)
                cramming_model.optimizer.step()
                cramming_model.scheduler.step()


                if DEBUG:
                    for param in cramming_model.parameters():
                        # Make sure not 0 gradients.
                        print(T.max(param.grad))

                ## Zeroing grad like this is faster. (https://h-huang.github.io/tutorials/recipes/recipes/tuning_guide.html)
                for param in cramming_model.parameters():
                    param.grad = None
                
            losses.append(loss.item())
            if len(losses) == LOSS_RUNNING_MEAN_LENGTH:
                losses.pop(0)

            if idx % 5000 == 0:

                if np.isnan(np.mean(losses)):
                    raise ValueError('Nan found in loss. Terminating.')

                if np.mean(losses) < best_loss:
                    best_loss = np.mean(best_loss)
                    print(f'Tokens ingested: {idx * 128 * MICRO_BATCH_SIZE // 1e6}M')
                    cramming_model.save_model(model_file=model_file)

                    ## Prediction sample
                    if SHOW_PROGRESS:
                        idxs = T.argwhere(y[:128] != -100).squeeze()
                        print(
                                #f'Original Text (Masked):   {handler.tokenizer.decode(X.flatten()[:128])}\n\n', 
                                #f'Predicted Text:           {handler.tokenizer.decode(T.argmax(out[:128], dim=-1))}\n\n',
                                f'Original Masked Tokens:    {handler.tokenizer.decode(y[idxs])}\n\n',
                                f'Predicted Masked Tokens:   {handler.tokenizer.decode(T.argmax(F.softmax(out[:128], dim=-1), dim=-1)[idxs])}\n\n'
                                )


            progress_bar.update(1)
            progress_bar.set_description(f'Running Loss: {np.mean(losses)}')

    
def run_inference(
    n_epochs=1,
    lr=1e-4,
    batch_size=1536,
    dtype=T.float16,
    model_file='../trained_models/CrammingModel.pt',
    num_workers=4,
    pin_memory=True,
    seq_length=128,
    embed_dims=768, 
    num_heads=12, 
    has_bias=True,
    dropout_rate=0.2,
    n_encoder_blocks=12,
    mlp_expansion_factor=4,
    use_gpu=True,
    loss_fn=T.nn.CrossEntropyLoss(ignore_index=-100)
    ) -> None:
    ## Ensure empty cache. Should be done by operating system + cuda but can't hurt.
    T.cuda.empty_cache()

    device  = T.device('cuda:0' if use_gpu else 'cpu')
    handler = DataHandler(
            dataset_name='bookcorpus',
            max_length=seq_length,
            dtype=dtype,
            device=device,
            eval=True
            )

    cramming_model = CrammingTransformer(
            vocab_size=handler.tokenizer.vocab_size,
            seq_length=seq_length,
            embed_dims=embed_dims,
            num_heads=num_heads, 
            has_bias=has_bias,
            dropout_rate=dropout_rate,
            n_encoder_blocks=n_encoder_blocks,
            mlp_expansion_factor=mlp_expansion_factor,
            lr=lr,
            device=device
            )
    cramming_model.load_model(model_file=model_file)


    while True:
        with T.no_grad():
            input_text = input('Enter a sentence with the token -> `[MASK]` ' \
                               'replacing various words and the model will try ' \
                               'to predict the masked value. (`quit` to quit)\n')
            if input_text == 'quit':
                sys.exit()


            tokenized_output = handler.tokenizer(
                    input_text,
                    padding='max_length',
                    truncation=False,
                    max_length=seq_length
                    )
            X, attention_mask = T.tensor(tokenized_output['input_ids']), T.tensor(tokenized_output['attention_mask'])

            X              = X.unsqueeze(dim=0).to(device)
            attention_mask = attention_mask.unsqueeze(dim=0).to(device)

            pred_probs  = F.softmax(cramming_model.forward(X, attention_mask), dim=-1)
            pred_tokens = T.argmax(pred_probs, dim=-1)

            print(handler.tokenizer.decode(pred_tokens.squeeze()))





if __name__ == '__main__':
    pretrain(**train_configs['bert_base'])
    #run_inference(**train_configs['bert_base'])
