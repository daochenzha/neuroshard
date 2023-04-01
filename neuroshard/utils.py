import os
import json
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch

@dataclass
class ShardConfig:
    ndevices: int
    max_mem: float
    num_tables: int
    table_configs: List[Dict]
    table_sizes: List[int]


def load_dlrm_dataset(dataset_dir, table_configs_only=False):
    # Load table configs
    table_config_path = os.path.join(dataset_dir, "table_configs.json")
    with open(table_config_path) as f:
        table_configs = json.load(f)
    if table_configs_only:
        return table_configs

    # Load indices and offsets
    data = torch.load(os.path.join(dataset_dir, "data.pt"))

    return table_configs, data

def load_compute_cost_data(data_dir):
    # Load table configs
    table_config_path = os.path.join(data_dir, "table_configs.json")
    with open(table_config_path) as f:
        table_configs = json.load(f)

    # Load cost data
    X, y, = [], []
    with open(os.path.join(data_dir, "data.txt"), "r") as f:
        lines = f.readlines()
    for line in lines:
        x, y_ = line.strip().split()
        x = list(map(int, x.split(",")))
        y_ = float(y_)
        X.append(x)
        y.append(y_)

    return table_configs, X, y

def load_comm_cost_data(data_dir):
    # Load cost data
    fw_X, fw_y, bw_X, bw_y = [], [], [], []
    with open(os.path.join(data_dir, "data.txt"), "r") as f:
        lines = f.readlines()
    for line in lines:
        dims, sleep_times, fw_costs, bw_costs = line.strip().split()
        dims = list(map(float, dims.split(",")))
        sleep_times = list(map(float, sleep_times.split(",")))
        fw_costs = list(map(float, fw_costs.split(",")))
        bw_costs = list(map(float, bw_costs.split(",")))

        fw_X.append(dims + sleep_times)
        fw_y.append(fw_costs)
        bw_X.append(dims)
        bw_y.append(bw_costs)

    return np.array(fw_X), np.array(fw_y), np.array(bw_X), np.array(bw_y)


def dict2tensor(feature_dicts):
    features = []
    for feature_dict in feature_dicts:
        feature = []
        # Sorted based on key
        for _, value in sorted(feature_dict.items(), key=lambda x: x[0]):
            if isinstance(value, bool):
                value = 1.0 if value is True else 0.0
            elif isinstance(value, int):
                value = float(value)
            feature.append(value)
        features.append(feature)

    return torch.tensor(features, dtype=torch.float32)

def table_size(hash_size, dim, fp16=False):
    gb_size = hash_size * dim / (1024 * 1024 * 1024)
    if fp16:
        gb_size = 2 * gb_size
    else:
        gb_size = 4 * gb_size
    return gb_size

def read_tasks(task_path, table_configs):
    with open(task_path, "r") as f:
        lines = f.readlines()

    task_table_ids_list = []
    task_table_configs_list = []
    for line in lines:
        table_ids, table_dims = line.strip().split()
        table_ids = list(map(int, table_ids.split(",")))
        table_dims = list(map(int, table_dims.split(",")))
        task_table_ids_list.append(table_ids)
        task_table_configs = [table_configs[table_id].copy() for table_id in table_ids]
        for i, table_dim in enumerate(table_dims):
            task_table_configs[i]["dim"] = table_dim
        task_table_configs_list.append(task_table_configs)

    return task_table_ids_list, task_table_configs_list

def allocation2plan(allocation, ndevices):
    plan = [[] for _ in range(ndevices)]
    for i, d in enumerate(allocation):
        if d != -1:
            plan[d].append(i)
    return plan

def plan2allocation(plan):
    num_tables = sum([len(shard) for shard in plan])
    table_device_indices = [-1] * num_tables
    for bin_id, partition in enumerate(plan):
        for index in partition:
            table_device_indices[index] = bin_id
    return table_device_indices
    
def get_data(batch_size, offsets, indices, dim, device):
    args_indices = torch.cat([x.view(-1) for x in indices], dim=0).int()
    E_offsets = [0] + np.cumsum([x.view(-1).shape[0] for x in indices]).tolist()
    args_offsets = torch.cat([torch.tensor([0])] + [x[1:] + y for x, y in zip(offsets, E_offsets[:-1])], dim=0).int()

    grads_tensor = (
        torch.randn(batch_size, dim)
    )

    return (
        [
            args_indices.to(device),
            args_offsets.to(device),
        ],
        {},
        grads_tensor.to(device),

    )

class IndexLoader:
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_len = len(self.data)

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            self._shuffle()
        return self

    def _shuffle(self):
        np.random.shuffle(self.data)

    def __next__(self):
        if self.index >= self.data_len:
            raise StopIteration
        end = min(self.index + self.batch_size, self.data_len)
        batch_index = self.data[self.index:end]
        self.index = end
        return batch_index

class Timer:
    def __init__(self, device: str):
        self.device: str = device
        self.start_time: float = 0
        self.end_time: float = 0
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if self.device == "cpu":
            self.start_time = time.perf_counter()
        else:
            torch.cuda.synchronize()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            self.start_time = 0
        return self

    def __exit__(self, type, value, traceback):
        if self.device == "cpu":
            self.end_time = time.perf_counter()
        else:
            self.end_event.record()
            torch.cuda.synchronize()
            self.end_time = self.start_event.elapsed_time(self.end_event) * 1.0e-3

    # returns time in seconds
    def elapsed_time(self):
        return self.end_time - self.start_time

