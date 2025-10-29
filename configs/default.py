def get_config():
    return {
        'arch': 'wrn28_8',
        'train': {
            'base_learning_rate': 0.1,
            'num_epochs': 75,
            'batch_size': 128,
            'weight_decay': 0.0003,
            'weight_decay_vars': 'all',
            'momentum': 0.9,
            'adam_weight_decay': 0.0001,
            'num_classes': 10
        },
    }
