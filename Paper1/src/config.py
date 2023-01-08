import torch as T


train_configs = {}

train_configs['bert_base'] = {
        'n_epochs': 4,
        'lr': 5e-4,
        'batch_size': 1536,
        'dtype': T.float16,
        'model_file': '../trained_models/CrammingModel.pt',
        'num_workers': 4,
        'pin_memory': True,
        'seq_length': 128,
        'embed_dims': 768,
        'num_heads': 12,
        'has_bias': False,
        'dropout_rate': 0.0,
        'n_encoder_blocks': 12,
        'mlp_expansion_factor': 4,
        'use_gpu': True,
        'loss_fn': T.nn.CrossEntropyLoss(ignore_index=-100)
        }

