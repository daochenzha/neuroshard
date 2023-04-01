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
    read_tasks,
    get_data,
    load_dlrm_dataset,
)

comm_bench_path = os.path.join(neuroshard.__path__[0], "comm_bench.py")

def benchmark_comm(
    data_size,
    gpu_devices,
    T_range,
    sleep_range,
    max_dim,
    out_dir,
):
    ndevices = len(gpu_devices.split(","))
    try:
        comm_bench = CommBench(gpu_devices)
        dim_choices = []
        while max_dim % 4 == 0:
            dim_choices.append(max_dim)
            max_dim //= 2

        with open(os.path.join(out_dir, "data.txt"), "a") as f:
            for _ in range(data_size):
                task_T = np.random.randint(T_range[0], T_range[1])
                dims = np.random.choice(dim_choices, size=task_T)
                random_prob = np.random.random()
                dim_sums = random_dim_greedy(dims, random_prob, ndevices)

                sleep_time_diff = np.random.randint(sleep_range[0], sleep_range[1])
                sleep_times = np.random.randint(sleep_time_diff, size=ndevices)
                sleep_times -= np.min(sleep_times)
                fw_comm_costs, bw_comm_costs = comm_bench.evaluate(dim_sums, sleep_times)
                dim_sums = ",".join(list(map(str, dim_sums)))
                sleep_times = ",".join(list(map(str, sleep_times)))
                fw_comm_costs = ",".join(list(map(str, fw_comm_costs)))
                bw_comm_costs = ",".join(list(map(str, bw_comm_costs)))
                data = f"{dim_sums} {sleep_times} {fw_comm_costs} {bw_comm_costs}"
                print(data)
                f.write(data + "\n")
                f.flush()

                
            
    except:
        traceback.print_exc()
    finally:
        comm_bench.terminate()

def random_dim_greedy(dims, random_prob, ndevices):
    dims = sorted(dims, reverse=True)
    dim_sums = [0] * ndevices
    for dim in dims:
        if np.random.random() < random_prob:
            bin_id = np.random.randint(ndevices)
        else:
            bin_id = np.argmin(dim_sums)
        dim_sums[bin_id] += dim
    return dim_sums

class CommBench:
    def __init__(self, gpu_devices):
        self.ndevices = len(gpu_devices.split(","))

        # Construct command
        command = "OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={} python -m torch.distributed.run --nproc_per_node={} {} --ndevices {}".format(
            gpu_devices,
            self.ndevices,
            comm_bench_path,
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

    def evaluate(self, dims, sleep_times):
        dims = ",".join(map(str, dims))
        sleep_times = ",".join(map(str, sleep_times))

        self.process.stdin.write(f"{dims} {sleep_times}\n")
        self.process.stdin.flush()

        while True:
            line = self.process.stdout.readline().strip()
            if line == "2":
                break

        # Read results
        latencies = []
        for device in range(self.ndevices):
            result_path = os.path.join("/tmp", f"neuroshard_comm_{device}")
            with open(result_path, "r") as f:
                latency = list(map(float, f.readlines()[0].split(",")))
            latencies.append(latency)
        latencies = np.array(latencies)
        fw_comm_costs = latencies[:,0].tolist()
        bw_comm_costs = latencies[:,1].tolist()

        return fw_comm_costs, bw_comm_costs

    def terminate(self):
        #self.process.terminate()
        self.process.stdin.write("-1\n")
        self.process.stdin.flush()
        self.process.stdin.close()
        self.process.wait()
        print("Evaluator sub-process terminated!")


def main():
    parser = argparse.ArgumentParser("Multi GPU benchmark for embedding tables")
    parser.add_argument('--batch_size', type=int, default=65536)
    parser.add_argument('--ndevices', type=int, default=4)

    args = parser.parse_args()

    # Init distributed training
    ext_dist.init_distributed(use_gpu=True)
    device = "cuda:{}".format(ext_dist.my_local_rank)
    torch.set_num_threads(1)
    torch.cuda.set_device(device)

    internal_comm_bench = InternalCommBench(args.batch_size, args.ndevices, device=device)
    result_path = os.path.join("/tmp", f"neuroshard_comm_{ext_dist.my_local_rank}")

    # Synchronize workers
    barrier(args.ndevices)
    print("1") # Show it is ready

    while True:
        if ext_dist.my_local_rank == 0:
            inputs = input()
            if inputs == "-1":
                sleep_times = [-1]
            else:
                dims, sleep_times = inputs.split()
                dims = [list(map(int, dims.split(",")))]
                sleep_times = [list(map(int, sleep_times.split(",")))]
        else:
            dims, sleep_times = [None], [None]
        dist.broadcast_object_list(sleep_times, src=0)
        sleep_times = sleep_times[0]
        if sleep_times == -1:
            break
        dist.broadcast_object_list(dims, src=0)
        dims = dims[0]
        ext_dist.print_all(dims)

        latency = internal_comm_bench.eval(dims, sleep_times)

        with open(result_path, "w") as f:
            f.write(",".join(list(map(str, latency))))
            f.flush()
        time.sleep(0.1) # Wait until file writing finished

        # Synchronize workers
        barrier(args.ndevices)
        print("2") # Succuss signal

class InternalCommBench:
    def __init__(
        self,
        batch_size,
        ndevices,
        warmup_iter=5,
        num_iter=10,
        device="cuda:0",
    ):
        self.batch_size = batch_size
        self.ndevices = ndevices
        self.warmup_iter = warmup_iter
        self.num_iter = num_iter
        self.device = device

    def eval(self, dims, sleep_times):
        gc.collect()
        torch.cuda.empty_cache()

        grads_tensor = torch.randn((self.batch_size // self.ndevices, sum(dims)))

        time_records = benchmark(
            self.batch_size,
            dims,
            sleep_times[ext_dist.my_local_rank] / 1000,
            grads_tensor,
            self.device,
            self.ndevices,
            num_iter=self.warmup_iter+self.num_iter,
        )[self.warmup_iter:]

        return np.median(time_records, axis=0)

def benchmark(
    batch_size,
    dims,
    time_to_sleep,
    grads_tensor,
    device: str,
    ndevices: int,
    num_iter: int,
):
    time_records = []
    for _ in range(num_iter):
        tmp_grads_tensor = grads_tensor.to(device)
        torch.cuda.synchronize()

        iter_time = []

        y = torch.randn(
            batch_size,
            dims[ext_dist.my_local_rank],
            1,
            requires_grad=True,
            device=device,
        )

        # Forward communication
        barrier(ndevices)
        with Timer(device) as timer:
            time.sleep(time_to_sleep)
            a2a_req = ext_dist.alltoall([y], dims, True)
            y = a2a_req.wait()
            barrier(ndevices)
        iter_time.append((timer.elapsed_time() - time_to_sleep) * 1000)

        y = torch.cat(y, dim=1)

        # Backward communication
        barrier(ndevices)
        with Timer(device) as timer:
            y.backward(tmp_grads_tensor)
        iter_time.append(timer.elapsed_time() * 1000)
        del y
        del tmp_grads_tensor

        time_records.append(iter_time)

    return time_records

def barrier(ndevices):
    a2a_req = ext_dist.alltoall([torch.zeros(ndevices, 4, 1).to(f"cuda:{ext_dist.my_local_rank}")], [4 for _ in range(ndevices)], True)
    a2a_req.wait()

if __name__ == "__main__":
    main()

