import argparse
import os
import json

import numpy as np

from neuroshard.utils import table_size, load_dlrm_dataset


def main():
    parser = argparse.ArgumentParser("Generate DLRM tasks")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--data_size", type=int, default=100)
    parser.add_argument('--T_range', type=str, default="10,60")
    parser.add_argument("--max_mem", type=int, default=15) # 4 * 4 - 1
    parser.add_argument("--max_dim", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="data/dlrm_datasets")
    parser.add_argument('--out_dir', type=str, default="data/tasks/4_gpu")

    args = parser.parse_args()
    np.random.seed(args.seed)
    if "," in args.T_range:
        args.T_range = list(map(int, args.T_range.split(",")))
    else:
        args.T_range = [int(args.T_range), int(args.T_range)+1]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    table_configs = load_dlrm_dataset(args.data_dir, table_configs_only=True)

    dims = []
    cur_dim = args.max_dim
    while cur_dim % 4 == 0:
        dims.append(cur_dim)
        cur_dim //= 2

    gen_tasks(
        table_configs,
        args.data_size,
        args.T_range,
        args.max_mem,
        dims,
        args.out_dir,
    )
    print(f"{args.data_size} sharding tasks generated!")

def gen_tasks(
    table_configs,
    data_size,
    T_range,
    max_mem,
    dims,
    out_dir,
):
    out_path = os.path.join(out_dir, "data.txt")
    # Generate tasks
    with open(out_path, "w") as f:
        for _ in range(data_size):
            task_T = np.random.randint(T_range[0], T_range[1])
            table_ids, table_dims = gen_task(table_configs, task_T, max_mem, dims)
            table_ids = ",".join(list(map(str, table_ids)))
            table_dims = ",".join(list(map(str, table_dims)))
            f.write(f"{table_ids} {table_dims}\n")

def gen_task(
    table_configs,
    T,
    max_mem,
    dims,
):
    while True:
        table_ids = np.random.randint(len(table_configs), size=T)
        table_dims = np.random.choice(dims, size=T)
        size = sum([table_size(table_configs[task_id]["row"], table_dims[i]) for i, task_id in enumerate(table_ids)])
        if size <= max_mem:
            break
    return table_ids.tolist(), table_dims.tolist()




if __name__ == '__main__':
    main()
