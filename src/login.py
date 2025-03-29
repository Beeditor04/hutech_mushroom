import os
import wandb
from dotenv import load_dotenv

load_dotenv()
# Read the WandB API key from the environment variable
api_key = os.getenv("WANDB_API_KEY")
if api_key:
    wandb.login(key=api_key)
else:
    print("WANDB_API_KEY not found in environment. Please set it to your WandB API key.")
wandb.require("core")
