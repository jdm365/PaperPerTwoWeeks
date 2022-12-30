import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class InputEmbedding(nn.Module):
    def __init__(self, input_dims=512, embed_dim=384) -> None:
        super(InputEmbedding, self).__init__()
        self.input_dims = input_dims
        self.embed_dim  = embed_dim


    def forward(self, X: T.tensor) -> T.tensor:
        ## TODO
        return



class Encoder(nn.Module):
    def __init__(
            self, 
            embed_dim=384, 
            qkv_dim=64, 
            num_heads=8, 
            has_bias=False,
            activation=nn.GELU,
            dropout_rate=0.0,
            mlp_hidden_dims=None
            ) -> None:
        super(Encoder, self).__init__()
        self.embed_dim  = embed_dim
        self.qkv_dim    = qkv_dim
        self.num_heads  = num_heads
        if mlp_hidden_dims is None:
            mlp_hidden_dims = 4 * emb_dim


        self.attention_block = nn.Sequential(
                nn.MultiHeadedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads
                    dropout=droupout_rate,
                    bias=has_bias,
                    batch_first=True
                    ),
                nn.LayerNorm(emb_dim)
                )
        self.mlp_block = nn.Sequential(
                nn.Linear(emb_dim, mlp_hidden_dims, bias=has_bias),
                activation,
                nn.Linear(mlp_hidden_dims, mlp_hidden_dims, bias=has_bias),
                activation,
                nn.LayerNorm(emb_dim)
                )
        

    def forward(self, X: T.tensor) -> T.tensor:
        X = X + self.attention_block(X)
        X = X + self.mlp_block(X)
        return X


class CrammingTransformer(nn.Module):
    def __init__(
            self,
            input_dims=512,
            embed_dim=384, 
            qkv_dim=64, 
            num_heads=8, 
            has_bias=False,
            activation=nn.GELU,
            dropout_rate=0.0,
            mlp_hidden_dims=None,
            n_encoder_blocks=8,
            output_dims=None,
            lr=1e-4
            ) -> None:
        super(CrammingTransformer, self).__init__()

        self.model = [
                InputEmbedding(
                    input_dims=input_dims,
                    embed_dims=embed_dims
                    )]
        self.model += [
            Encoder(
                embed_dim=embed_dim,
                qkv_dim=qkv_dim,
                num_heads=num_heads,
                has_bias=has_bias,
                activation=activation,
                dropout_rate=dropout_rate,
                mlp_hidden_dims=mlp_hidden_dims
                ) for _ in n_encoder_blocks
            ]
        self.model += [
                nn.Linear(embed_dims, output_dims, bias=has_bias),
                nn.Softmax()
                ]

        self.model = nn.Sequential(nn.ModuleList(self.model))

        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
        self.to(self.device)


    def forward(self, X: T.tensor) -> T.tensor:
        return self.model(X)

        
