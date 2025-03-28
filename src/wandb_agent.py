sweep_config = {
    'method': 'random',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'val_loss',  # The metric to optimize
        'goal': 'minimize'  # The optimization goal of the metric
    },
    'parameters': {
        'lr': {
            'distribution': 'uniform',
            'min': 1e-6,
            'max': 1e-2
        },
        'num_epochs': {
            'values': [10, 20, 30]
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'adamw']
        },
        # You can add more hyperparameters as needed.
    }
}

import wandb
from utils import load_config
from train import trainer

# alternatively, you can load the sweep config from a file
# sweep_config = load_config("../config/sweep.yaml")

sweep_id = wandb.sweep(sweep_config, project="minialexnet")

print(f"Sweep ID: {sweep_id}")
wandb.agent(sweep_id, trainer)