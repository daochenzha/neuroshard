import os
import subprocess
import json
import argparse
import time
import numpy as np
from typing import Dict, Any, Callable
from collections import OrderedDict
import gc
import copy
import traceback

import torch
import torch.distributed as dist
from fbgemm_gpu import split_table_batched_embeddings_ops
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType

import neuroshard
from neuroshard import extend_distributed as ext_dist
from neuroshard.utils import (
    Timer,
    allocation2plan,
    plan2allocation,
    read_tasks,
    get_data,
    load_dlrm_dataset,
    table_size,
)

evaluator_path = os.path.join(neuroshard.__path__[0], "evaluator.py")

class Evaluator:
    def __init__(
        self,
        data_dir,
        task_path,
        gpu_devices,
        max_mem,
    ):
        self.ndevices = len(gpu_devices.split(","))
        self.max_mem = max_mem

        # Read data
        table_configs = load_dlrm_dataset(data_dir, table_configs_only=True)

        # Read tasks
        _, self.task_table_configs_list = read_tasks(task_path, table_configs)

        # Construct command
        command = "OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={} python -m torch.distributed.run --nproc_per_node={} {} --data_dir {} --task_path {} --ndevices {}".format(
            gpu_devices,
            self.ndevices,
            evaluator_path,
            data_dir,
            task_path,
            self.ndevices,
        )

        # Run command
        self.process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding='utf8',
        )

        # Wait until it is ready
        while True:
            line = self.process.stdout.readline().strip()
            if line == "1":
                break
        print("Evaluator initiated!")

    def evaluate(self, task_id, sharding_steps, shards):
        aug_table_configs = [table_config.copy() for table_config in self.task_table_configs_list[task_id]]
        for index in sharding_steps:
            shard_dim = aug_table_configs[index]["dim"] // 2
            aug_table_configs[index]["dim"] = shard_dim
            aug_table_configs.append(aug_table_configs[index].copy())
        sizes = [table_size(config["row"], config["dim"]) for config in aug_table_configs]
        size_sums = [sum([sizes[i] for i in shard]) for shard in shards]
        if max(size_sums) > self.max_mem:
            return None, None
        
        sharding_steps = ",".join(map(str, sharding_steps))
        sharding = ",".join(map(str, plan2allocation(shards)))
        self.process.stdin.write(str(task_id) + "|" + sharding_steps + "|" + sharding+"\n")
        self.process.stdin.flush()

        while True:
            line = self.process.stdout.readline().strip()
            if line == "2":
                break

        # Read results
        latencies = []
        for device in range(self.ndevices):
            result_path = os.path.join("/tmp", f"neuroshard_{device}")
            with open(result_path, "r") as f:
                latency = list(map(float, f.readlines()[0].split(",")))
            latencies.append(latency)
        latencies = np.array(latencies)
        max_latency = np.max(np.sum(latencies, axis=1))

        return max_latency, latencies

    def terminate(self):
        #self.process.terminate()
        self.process.stdin.write("-1\n")
        self.process.stdin.flush()
        self.process.stdin.close()
        self.process.wait()
        print("Evaluator sub-process terminated!")


def main():
    parser = argparse.ArgumentParser("Multi GPU benchmark for embedding tables")
    parser.add_argument('--data_dir', type=str, default="data/dlrm_datasets")
    parser.add_argument('--task_path', type=str, default="data/tasks/4_gpu/data.txt")
    parser.add_argument('--ndevices', type=int, default=4)

    args = parser.parse_args()

    # Init distributed training
    ext_dist.init_distributed(use_gpu=True)
    device = "cuda:{}".format(ext_dist.my_local_rank)
    torch.set_num_threads(1)
    torch.cuda.set_device(device)

    # Read data
    table_configs, data = load_dlrm_dataset(args.data_dir)
    indices, offsets = data["indices"], data["offsets"]

    # Read tasks
    task_table_ids_list, task_table_configs_list = read_tasks(args.task_path, table_configs)

    # Create an evaluator for each task
    evaluators = []
    for task_table_ids, task_table_configs in zip(task_table_ids_list, task_table_configs_list):
        task_offsets = [offsets[table_id] for table_id in task_table_ids]
        task_indices = [indices[table_id] for table_id in task_table_ids]

        # Create evaluator
        evaluator = InternalEvaluator(
            table_configs=task_table_configs,
            indices=task_indices,
            offsets=task_offsets,
            ndevices=args.ndevices,
            device=device,
        )
        evaluators.append(evaluator)

    result_path = os.path.join("/tmp", f"neuroshard_{ext_dist.my_local_rank}")

    # Synchronize workers
    barrier(args.ndevices)
    print("1") # Show it is ready

    while True:
        if ext_dist.my_local_rank == 0:
            inputs = input()
            if inputs == "-1":
                task_id = [-1]
            else:
                task_id, sharding_steps, sharding = inputs.split("|")
                task_id = [int(task_id)]
                if len(sharding_steps) == 0:
                    sharding_steps = [[]]
                else:
                    sharding_steps = [list(map(int, sharding_steps.split(",")))]
                sharding = [list(map(int, sharding.split(",")))]
        else:
            task_id, sharding_steps, sharding = [None], [None], [None]
        dist.broadcast_object_list(task_id, src=0)
        task_id = task_id[0]
        if task_id == -1:
            break

        dist.broadcast_object_list(sharding_steps, src=0)
        sharding_steps = sharding_steps[0]
        dist.broadcast_object_list(sharding, src=0)
        sharding = sharding[0]

        # Get number of tables in each rank and local indices
        shards = allocation2plan(sharding, args.ndevices)

        # Benchmark
        latency = evaluators[task_id].eval(sharding_steps, shards)
        with open(result_path, "w") as f:
            f.write(",".join(list(map(str, latency))))
            f.flush()
        time.sleep(0.1) # Wait until file writing finished

        # Synchronize workers
        barrier(args.ndevices)
        print("2") # Succuss signal

class InternalEvaluator:
    def __init__(
        self,
        table_configs,
        indices,
        offsets,
        ndevices,
        warmup_iter=10,
        num_iter=100,
        device="cuda:0",
    ):

        self.table_configs = table_configs
        self.num_tables = len(self.table_configs)

        self.offsets = offsets
        self.indices = indices

        self.ndevices = ndevices
        self.warmup_iter = warmup_iter
        self.num_iter = num_iter
        self.device = device
        self.batch_size = self.offsets[0].shape[0] - 1

    def eval(self, sharding_steps, shards):
        aug_table_configs = [table_config.copy() for table_config in self.table_configs]
        index2raw = {i: i for i in range(len(self.table_configs))}
        for index in sharding_steps:
            shard_dim = aug_table_configs[index]["dim"] // 2
            aug_table_configs[index]["dim"] = shard_dim
            aug_table_configs.append(aug_table_configs[index].copy())
            index2raw[len(aug_table_configs)-1] = index2raw[index]
         
        table_indices = shards[ext_dist.my_local_rank]
        num_elements_per_rank = [sum([aug_table_configs[index]["dim"] for index in indices]) for indices in shards]

        gc.collect()
        torch.cuda.empty_cache()

        if len(table_indices) > 0:
            # Build the op
            shard_table_configs = [aug_table_configs[i] for i in table_indices]

            op = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
                [
                    (
                        table_config["row"],
                        table_config["dim"],
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.ComputeDevice.CUDA,
                    )
                    for table_config in shard_table_configs
                ],
                optimizer=OptimType.EXACT_SGD,
                cache_algorithm=split_table_batched_embeddings_ops.CacheAlgorithm.LFU,
                cache_reserved_memory=8.0,
                eps=0.01,
                device=self.device,
                weights_precision=SparseType.FP32,
            )

            # Get data
            shard_offsets = [self.offsets[index2raw[i]] for i in table_indices]
            shard_indices = [self.indices[index2raw[i]] for i in table_indices]
            args, kwargs, shard_grads_tensor = get_data(
                self.batch_size,
                shard_offsets,
                shard_indices,
                sum([aug_table_configs[index]["dim"] for index in table_indices]),
                self.device,
            )
        else:
            op, args, kwargs = None, [], {}
            shard_grads_tensor = torch.randn((self.batch_size, 0))

        grads_tensor = torch.randn(
            (
                self.batch_size // self.ndevices,
                sum(num_elements_per_rank),
            ),
        )

        time_records = benchmark_op(
            op,
            args,
            kwargs,
            grads_tensor,
            shard_grads_tensor,
            self.device,
            self.ndevices,
            num_elements_per_rank,
            num_iter=self.warmup_iter+self.num_iter,
        )[self.warmup_iter:]

        return np.median(time_records, axis=0)

def benchmark_op(
    op: Callable,
    args: Any,
    kwargs: Any,
    grads_tensor: Any,
    shard_grads_tensor: Any,
    device: str,
    ndevices: int,
    num_elements_per_rank: list,
    num_iter: int,
):
    batch_size = shard_grads_tensor.shape[0]

    # Benchmark compute only sequencially
    time_records = [[0] * 4 for _ in range(num_iter)]
    barrier(ndevices)
    for device_ in range(ndevices):
        if device_ == ext_dist.my_local_rank:
            for iter_id in range(num_iter):
                if op is None:
                    time_records[iter_id][1] = 0.0
                    time_records[iter_id][2] = 0.0
                else:
                    _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
                    torch.cuda.empty_cache()
                    with Timer(device_) as timer:
                        op(*args, **kwargs).backward(shard_grads_tensor)
                    time_records[iter_id][1] = timer.elapsed_time() * 1000

                    _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
                    torch.cuda.empty_cache()
                    with Timer(device_) as timer:
                        y = op(*args, **kwargs)
                    time_records[iter_id][2] = timer.elapsed_time() * 1000
                    time_records[iter_id][1] -= time_records[iter_id][2]
                    del y
        barrier(ndevices)

    for iter_time in time_records:
        tmp_grads_tensor = grads_tensor.to(device)
        torch.cuda.synchronize()

        if op is None:
            y = torch.randn((batch_size, 0, 1), device=device, requires_grad=True)
        else:
            y = op(*args, **kwargs)
            y = y.view(batch_size, -1, 1)
        a2a_req = ext_dist.alltoall([y], num_elements_per_rank, True)
        y = a2a_req.wait()
        y = torch.cat(y, dim=1)

        # Backward comm
        barrier(ndevices)
        with Timer(device) as timer:
            y.backward(tmp_grads_tensor)
        iter_time[0] = timer.elapsed_time() * 1000
        del y
        del tmp_grads_tensor

        torch.cuda.synchronize()
        barrier(ndevices)
        if op is not None:
            y = op(*args, **kwargs)

            with Timer(device) as timer:
                y = y.backward(shard_grads_tensor)
            iter_time[0] -= timer.elapsed_time() * 1000

        # Forward comm
        time_to_sleep = sum(iter_time[:3]) / 1000
        time_to_sleep = 0 # Tobe fixed
        if op is None:
            y = torch.randn((batch_size, 0, 1), device=device, requires_grad=True)
        else:
            y = op(*args, **kwargs)
            y = y.view(batch_size, -1, 1)
        torch.cuda.synchronize()

        barrier(ndevices)
        with Timer(device) as timer:
            time.sleep(time_to_sleep)
            a2a_req = ext_dist.alltoall([y], num_elements_per_rank, True)
            y = a2a_req.wait()
            barrier(ndevices)
        iter_time[3] = (timer.elapsed_time() - time_to_sleep) * 1000
        del y

    return time_records

def barrier(ndevices):
    a2a_req = ext_dist.alltoall([torch.zeros(ndevices, 4, 1).to(f"cuda:{ext_dist.my_local_rank}")], [4 for _ in range(ndevices)], True)
    a2a_req.wait()

if __name__ == "__main__":
    main()
