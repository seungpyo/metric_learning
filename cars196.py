from torchvision.datasets import StanfordCars
import argparse
import os
from sys import exit
from typing import *


def get_dataset(root_dir: str) -> Dict[str, StanfordCars]:
    train = StanfordCars(root=root_dir, split="train", download=True)
    test = StanfordCars(root=root_dir, split="test", download=True)
    return {"train": train, "test": test}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/stanford_cars")
    args = parser.parse_args()
    if os.path.exists(args.data_dir) and len(os.listdir(args.data_dir)) > 0:
        opt = input(f"Data directory {args.data_dir} is not empty. Do you want to continue? [y/n]")
        if opt.lower() != "y":
            print("Exiting...")
            exit(0)
    os.makedirs(args.data_dir, exist_ok=True)
    train = StanfordCars(root=args.data_dir, split="train", download=True)
    test = StanfordCars(root=args.data_dir, split="test", download=True)