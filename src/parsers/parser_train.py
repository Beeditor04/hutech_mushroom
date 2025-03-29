import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Train with different models.")
    parser.add_argument("--model", default="alexnet", type=str, help="Model name to train")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--project", default="hutech_mushroom", type=str, help="WandB project name")
    parser.add_argument("--dataset", default="hutech-dataset:latest", type=str, help="Dataset name")
    return parser

def parse_args():
    parser = create_parser()
    args, unknown = parser.parse_known_args()
    return args
