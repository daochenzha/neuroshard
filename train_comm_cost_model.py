import argparse
import os

from neuroshard.utils import load_comm_cost_data
from neuroshard.comm_cost_model import CommCostModel

def main():
    parser = argparse.ArgumentParser("NeuroShard train comm cost model")
    parser.add_argument("--data_dir", type=str, default="data/cost_data/comm_4/")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--fw_out_path", type=str, default="models/comm_4_fw.pt")
    parser.add_argument("--bw_out_path", type=str, default="models/comm_4_bw.pt")
    args = parser.parse_args()

    fw_X, fw_y, bw_X, bw_y = load_comm_cost_data(args.data_dir)

    fw_comm_cost_model = CommCostModel(ndevices=fw_y.shape[1])
    fw_comm_cost_model.train(
        fw_X[:args.num_samples],
        fw_y[:args.num_samples],
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        epochs=args.epochs,
        lr=args.lr,
    )
    fw_comm_cost_model.save(args.fw_out_path)

    bw_comm_cost_model = CommCostModel(ndevices=bw_y.shape[1])
    bw_comm_cost_model.train(
        bw_X[:args.num_samples],
        bw_y[:args.num_samples],
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        epochs=args.epochs,
        lr=args.lr,
    )
    bw_comm_cost_model.save(args.bw_out_path)

if __name__ == '__main__':
    main()
    
