import torch as T
import torch.nn as nn
import transformers


class HuggingFaceBertWrapper(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            hidden_size, 
            num_hidden_layers,
            hidden_dropout_prob,
            lr,
            device
            ) -> None:
        self.model = BertForMaskedLM(
                vocab_size=vocab_size,
                hidden_size=hidden_size, 
                num_hidden_layers=num_hidden_layers,
                intermediate_size=4*num_hidden_layers,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=hidden_dropout_prob
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
        return self.model(X)['logits']


