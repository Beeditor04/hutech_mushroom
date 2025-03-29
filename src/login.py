import os
import wandb

# Read the WandB API key from the environment variable
api_key = os.getenv("WANDB_API_KEY")
if api_key:
    wandb.login(key=api_key)
else:
    print("WANDB_API_KEY not found in environment. Please set it to your WandB API key.")
wandb.require("core")
