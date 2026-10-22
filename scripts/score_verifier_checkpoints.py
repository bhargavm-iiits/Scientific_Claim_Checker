"""Score saved checkpoints sequentially, sharing Python/GPU initialization."""
import argparse
import gc

import torch
from benchmark_verifier import main as benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", default="nli_scifact_train_v2")
    parser.add_argument("--epochs", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--include-title", action="store_true")
    parser.add_argument("--window-overlap", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    for epoch in args.epochs:
        benchmark(["--model", f"artifacts/models/{args.training_run}/epoch{epoch}",
                   "--size", "120", "--name", f"{args.training_run}_epoch{epoch}_dev120", "--device", "cuda",
                   "--window-overlap", str(args.window_overlap), "--top-k", str(args.top_k),
                   *(["--include-title"] if args.include_title else [])])
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
