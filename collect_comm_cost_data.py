import argparse
import os

from neuroshard.comm_bench import benchmark_comm
from neuroshard.utils import load_dlrm_dataset

def main():
    parser = argparse.ArgumentParser("NeuroShard comm cost data collection")
    parser.add_argument("--data_size", type=int, default=999999999)
    parser.add_argument("--gpu_devices", type=str, default="0,1,2,3")
    parser.add_argument('--T_range', type=str, default="10,60")
    parser.add_argument('--sleep_range', type=str, default="1,20")
    parser.add_argument("--max_dim", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="data/cost_data/comm_4")
    args = parser.parse_args()

    args.T_range = list(map(int, args.T_range.split(",")))
    args.sleep_range = list(map(int, args.sleep_range.split(",")))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    benchmark_comm(
        data_size=args.data_size,
        gpu_devices=args.gpu_devices,
        T_range=args.T_range,
        sleep_range=args.sleep_range,
        max_dim=args.max_dim,
        out_dir=args.out_dir,
    )

if __name__ == '__main__':
    main()

