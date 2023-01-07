import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

@T.jit.script
def fused_gelu(x):
    with T.autocast(device_type='cuda', dtype=T.float16):
        return x * 0.5 * (1.0 + T.erf(x / 1.41421))


class InputEmbedding(nn.Module):
    '''
    Learned word embedding + sinusodial positional embedding as in `Attention is all you Need` paper.
    '''
    def __init__(
            self, 
            vocab_size,
            embed_dims=768, 
            max_seq_length=512, 
            dtype=T.float16, 
            device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            ) -> None:
        super(InputEmbedding, self).__init__()

        self.input_emb  = nn.Embedding(vocab_size, embed_dims)
        self.embed_norm = nn.LayerNorm(embed_dims)

        self.embed_dims = embed_dims
        encoding = T.zeros(max_seq_length, embed_dims, device=device)
        position = T.arange(0, max_seq_length, dtype=dtype, device=device).unsqueeze(1)
        inv_denom = T.exp(T.arange(0, embed_dims, 2, dtype=dtype, device=device) * (-math.log(10000.0) / embed_dims))
        encoding[:, 0::2] = T.sin(position * inv_denom)
        encoding[:, 1::2] = T.cos(position * inv_denom)

        ## Add batch dimension.
        encoding = encoding.unsqueeze(0)
        ## Register as buffer (not parameter).
        self.register_buffer("encoding", encoding, persistent=False)
        self.to(device)

    def forward(self, X: T.tensor) -> T.tensor:
        X = self.input_emb(X)
        X = X + self.encoding 
        return self.embed_norm(X)


class SelfAttention(nn.Module):
    def __init__(self, embed_dims=768, num_heads=8, has_bias=False, dropout_rate=0.0) -> None:
        super(SelfAttention, self).__init__()
        
        self.embed_dims = embed_dims
        self.num_heads  = num_heads
        self.head_dims  = embed_dims // num_heads

        assert(self.head_dims * num_heads == embed_dims), "Embed dims needs to be divisible by heads"

        ## Concat qkv into one matrix
        self.qkv     = nn.Linear(embed_dims, 3 * embed_dims, bias=False)
        self.dropout = nn.Dropout1d(p=dropout_rate)
        self.fc_out  = nn.Linear(embed_dims, embed_dims, bias=has_bias)


    @staticmethod
    def scaled_dot_product(query, key, value, attention_mask=None) -> T.tensor:
        embed_dims = key.shape[-1]
        attention_logits = T.matmul(query, key.transpose(-2, -1)) / math.sqrt(embed_dims)

        if attention_mask is not None:
            mask_broadcast = attention_mask.unsqueeze(dim=1).unsqueeze(dim=1)

            ## Fill with fp16 min value
            attention_logits = attention_logits.masked_fill(mask_broadcast == 0, -1e+4)

        attention = F.softmax(attention_logits, dim=-1)
        values    = T.matmul(attention, value)
        return values


    def forward(self, X: T.tensor, attention_mask=None) -> T.tensor:
        batch_size, seq_length, _ = X.shape

        qkv = self.qkv(X)

        # Split embedding into self.heads pieces
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dims)

        ## (batch, heads, seq_length, dims)
        qkv = qkv.permute(0, 2, 1, 3)
        queries, keys, values = qkv.chunk(3, dim=-1)

        out = self.scaled_dot_product(queries, keys, values, attention_mask=attention_mask)

        ## Restore dims -> (batch, seq_length, heads, dims)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(batch_size, seq_length, self.embed_dims)
        X = self.dropout(out)

        X = self.fc_out(X)
        return X


class Encoder(nn.Module):
    def __init__(
            self,
            embed_dims=384,
            num_heads=8,
            has_bias=False,
            dropout_rate=0.0,
            mlp_expansion_factor=4
            ) -> None:
        super(Encoder, self).__init__()

        mlp_hidden_dims = mlp_expansion_factor * embed_dims

        self.attention_norm = nn.LayerNorm(embed_dims)
        self.attention_block = SelfAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                has_bias=has_bias,
                dropout_rate=dropout_rate
                )

        self.mlp_norm = nn.LayerNorm(embed_dims)

        self.fc1      = nn.Linear(embed_dims, mlp_hidden_dims, bias=has_bias)
        self.dropout1 = nn.Dropout1d(p=dropout_rate)
        self.fc2      = nn.Linear(mlp_hidden_dims, embed_dims, bias=has_bias)
        self.dropout2 = nn.Dropout1d(p=dropout_rate)

        ## Skeptical
        self.final_norm = nn.LayerNorm(embed_dims)
        

    def forward(self, X: T.tensor, attention_mask: T.tensor = None) -> (T.tensor, T.tensor):
        '''
        ## Post-norm
        _X = self.attention_block(X, attention_mask)
        X  = self.attention_norm(X + _X)
        
        _X = fused_gelu(self.dropout1(self.fc1(X)))
        _X = self.dropout2(self.fc2(_X))
        X  = self.mlp_norm(X + _X)
        '''

        ## Pre-norm
        _X = self.attention_norm(X)

        X  = X + self.attention_block(_X, attention_mask)
        
        _X = self.mlp_norm(X)

        _X = fused_gelu(self.fc1(_X))
        _X = self.dropout1(_X)
        _X = fused_gelu(self.fc2(_X))
        X  = X + self.dropout2(_X)

        X  = self.final_norm(X)
        return X, attention_mask


class CrammingTransformer(nn.Module):
    def __init__(
            self,
            vocab_size,
            seq_length,
            embed_dims=384, 
            num_heads=8, 
            has_bias=False,
            dropout_rate=0.0,
            n_encoder_blocks=8,
            mlp_expansion_factor=4,
            lr=5e-4,
            device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            ) -> None:
        super(CrammingTransformer, self).__init__()
        self.embed_dims = embed_dims

        self.input_embedding = InputEmbedding(
                vocab_size=vocab_size,
                embed_dims=embed_dims,
                max_seq_length=seq_length,
                device=device
                )
        self.model = nn.ModuleList([
            Encoder(
                embed_dims=embed_dims,
                num_heads=num_heads,
                has_bias=has_bias,
                dropout_rate=dropout_rate,
                mlp_expansion_factor=mlp_expansion_factor
                ) for _ in range(n_encoder_blocks)
            ])

        self.classifier_head = nn.Sequential(
                nn.Linear(embed_dims, mlp_expansion_factor * embed_dims, bias=has_bias),
                nn.LayerNorm(mlp_expansion_factor * embed_dims),
                nn.ReLU(),
                nn.Dropout1d(dropout_rate),
                nn.Linear(mlp_expansion_factor * embed_dims, vocab_size, bias=has_bias)
            )

        self.optimizer = optim.AdamW(
                self.parameters(), 
                lr=lr, 
                weight_decay=0.01, 
                eps=1e-6,
                betas=(0.9, 0.98)
                )
        self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=1000
                )

        self.device = device
        self.to(self.device)


    def forward(self, X: T.tensor, attention_mask: T.tensor = None) -> T.tensor:
        X = self.input_embedding(X)
        for encoder_block in self.model:
            X, attention_mask = encoder_block(X, attention_mask)
        return self.classifier_head(X)


    def save_model(self, model_file: str) -> None:
        print(f'...Saving Model to {model_file}...')
        T.save(self.state_dict(), model_file)
        T.save(self.optimizer.state_dict(), f'{model_file[:-3]}_optimizer.pt')


    def load_model(self, model_file: str, load_optimizer=False) -> None:
        print(f'...Loading Model from {model_file}...')
        self.load_state_dict(T.load(model_file))
        if load_optimizer:
            self.optimizer.load_state_dict(T.load(f'{model_file[:-3]}_optimizer.pt'))
