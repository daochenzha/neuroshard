import os
import argparse
import json
import torch
import numpy as np
import copy
import time
import traceback

from neuroshard import sharders
from neuroshard.utils import (
    table_size,
    allocation2plan,
    read_tasks,
    load_dlrm_dataset,
    ShardConfig,
)
from neuroshard.evaluator import Evaluator
from neuroshard.neurosharder import NeurSharder
from neuroshard.compute_cost_model import ComputeCostModel
from neuroshard.comm_cost_model import CommCostModel


def main():
    parser = argparse.ArgumentParser("Benchmark sharding")
    parser.add_argument('--data_dir', type=str, default="data/dlrm_datasets")
    parser.add_argument('--task_path', type=str, default="data/tasks/4_gpu/data.txt")
    parser.add_argument('--compute_model', type=str, default="models/compute.pt")
    parser.add_argument('--fw_comm_model', type=str, default="models/comm_4_fw.pt")
    parser.add_argument('--bw_comm_model', type=str, default="models/comm_4_bw.pt")
    parser.add_argument('--alg', type=str, default="random")
    parser.add_argument('--max_mem', type=float, default=4)
    parser.add_argument('--gpu_devices', type=str, default="0,1,2,3")

    args = parser.parse_args()
    args.ndevices = len(args.gpu_devices.split(","))

    # Read data
    table_configs = load_dlrm_dataset(args.data_dir, table_configs_only=True)

    # Read tasks
    _, task_table_configs_list = read_tasks(args.task_path, table_configs)

    # Load models
    compute_cost_model = ComputeCostModel()
    compute_cost_model.load(args.compute_model)
    fw_comm_cost_model = CommCostModel(args.ndevices)
    fw_comm_cost_model.load(args.fw_comm_model)
    bw_comm_cost_model = CommCostModel(args.ndevices)
    bw_comm_cost_model.load(args.bw_comm_model)

    try:
        evaluator = Evaluator(
            args.data_dir,
            args.task_path,
            args.gpu_devices,
            args.max_mem,
        )
        latencies = [] 
        for task_id, task_table_configs in enumerate(task_table_configs_list):
            print("Task", str(task_id+1)+"/"+str(len(task_table_configs_list)))

            if args.alg == "neuroshard":
                sharder = NeurSharder(
                    compute_cost_model,
                    fw_comm_cost_model,
                    bw_comm_cost_model,
                    task_table_configs,
                    args.ndevices,
                    args.max_mem,
                )
                sharding_steps, shards = sharder.shard()
            else:
                sizes = [table_size(config["row"], config["dim"]) for config in task_table_configs]
                shard_config = ShardConfig(
                    args.ndevices,
                    args.max_mem,
                    len(task_table_configs),
                    task_table_configs,
                    sizes,
                )
                shards = sharders.shard(shard_config, args.alg)
                sharding_steps = []
            print(f"Sharding steps: {sharding_steps}, shards: {shards}")
            max_latency, latency = evaluator.evaluate(task_id, sharding_steps, shards)
            if max_latency is None:
                print("Out of memory")
            else:
                latencies.append(max_latency)
                print("Latency:", max_latency)
                print(latency)
        print("Average:", np.mean(latencies))
        print(f"Valid {len(latencies)} / {len(task_table_configs_list)}")
        
    except:
        traceback.print_exc()
    finally:
        evaluator.terminate()


if __name__ == '__main__':
    main()
