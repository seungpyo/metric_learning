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
    parser.add_argument("--data_dir", type=str, default="/data/")
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    get_dataset(args.data_dir)