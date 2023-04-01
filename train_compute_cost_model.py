import argparse
import os

from neuroshard.utils import load_compute_cost_data, dict2tensor
from neuroshard.compute_cost_model import ComputeCostModel

def main():
    parser = argparse.ArgumentParser("NeuroShard train compute cost model")
    parser.add_argument("--data_dir", type=str, default="data/cost_data/compute/")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--out_path", type=str, default="models/compute.pt")
    args = parser.parse_args()

    table_configs, X, y = load_compute_cost_data(args.data_dir)
    table_features = dict2tensor(table_configs)

    compute_cost_model = ComputeCostModel()
    compute_cost_model.train(
        X[:args.num_samples],
        y[:args.num_samples],
        table_features,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        epochs=args.epochs,
        lr=args.lr,
    )
    compute_cost_model.save(args.out_path)


if __name__ == '__main__':
    main()
    
