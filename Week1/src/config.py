import torch as T


train_configs = {}

train_configs['bert_base_no_bias_no_dropout'] = {
        'n_epochs': 1,
        'lr': 5e-4,
        'batch_size': 1024,
        'dtype': T.float16,
        'model_file': '../trained_models/CrammingModel.pt',
        'num_workers': 6,
        'pin_memory': True,
        'input_dims': 128,
        'embed_dims': 768, 
        'num_heads': 12, 
        'has_bias': False,
        'dropout_rate': 0.0,
        'mlp_hidden_dims': None,
        'n_encoder_blocks': 12,
        'mlp_expansion_factor': 4,
        'output_dims': None,
        'use_gpu': True,
        'loss_fn': T.nn.CrossEntropyLoss()
        }

