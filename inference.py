import argparse
from pathlib import Path
from enum import Enum

class ModelSelection(str, Enum):
    BEST = "best"
    WORST = "worst"
    RANDOM = "random"
    CUSTOM = "custom"

def inference():
    pass

def main():
    args = parse_args()
    inference()

def parse_args():
    parser = argparse.ArgumentParser(description="RL inference")

    parser.add_argument(
        "--game",
        type=str,
        choices=("CartPole-v1", "Pendulum-v1", "CarRacing-v3", "BipedalWalker-v3"),
        default="CartPole-v1",
        help="Name of the game environment")

    parser.add_argument(
        "--model-selection",
        type=ModelSelection,
        choices=list(ModelSelection),
        default=ModelSelection.BEST,
        help="Model selection strategy"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the model when using custom selection"
    )

    parser.add_argument("--model-dir", type=Path, default="./models/")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


if __name__ == "__main__":
    main()