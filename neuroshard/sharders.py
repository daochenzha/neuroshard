from dataclasses import dataclass
import numpy as np

from neuroshard.utils import allocation2plan, plan2allocation, table_size

_sharders = {}

@dataclass
class TableInfo:
    index: int
    cost: float
    size: float

    def __lt__(self, o: "TableInfo") -> bool:
        return (self.cost, self.size, self.index) < (o.cost, o.size, o.index)

def register_sharder(sharder_name):
    def decorate(func):
        _sharders[sharder_name] = func
        return func
    return decorate

# get device indices for tables
# e.g 8 tables, No. [1,3,5,6] on device 0, No. [2,4,7,8] on device 1, then
# return [0, 1, 0, 1, 0, 0, 1, 1]
def shard(shard_config, alg="random"):
    if alg not in _sharders:
        import sys
        sys.exit("ERROR: sharder not found")
    return _sharders[alg](shard_config)

@register_sharder("dim_greedy")
def dim_greedy_shard(shard_config):
    # Get the embedding dims
    idx_weight_pairs = []
    for i, config in enumerate(shard_config.table_configs):
        dim = config["dim"]
        row = config["row"]
        size = shard_config.table_sizes[i]
        idx_weight_pairs.append(TableInfo(index=i, cost=dim, size=size))

    # Greedy algorithm
    num_bins = shard_config.ndevices
    sorted_idx_weight_pairs = sorted(idx_weight_pairs)
    partitions = [[] for p in range(num_bins)]
    partition_sums = [0.0] * num_bins
    partition_size_sums = [0.0] * num_bins

    mem_cap = [shard_config.max_mem] * num_bins

    while sorted_idx_weight_pairs:
        table_info = sorted_idx_weight_pairs.pop()
        min_sum = np.inf
        min_size_taken = np.inf
        min_r = -1
        for r in range(num_bins):
            if partition_size_sums[r] + table_info.size <= mem_cap[r]:
                if partition_sums[r] < min_sum or (
                    partition_sums[r] == min_sum
                    and partition_size_sums[r] < min_size_taken
                ):
                    min_sum = partition_sums[r]
                    min_r = r
                    min_size_taken = partition_size_sums[r]

        partitions[min_r].append(table_info)
        partition_sums[min_r] += table_info.cost
        partition_size_sums[min_r] += table_info.size

    partitions = [[table_info.index for table_info in partition] for partition in partitions]

    return partitions

@register_sharder("size_greedy")
def size_greedy_shard(shard_config):
    # Get the embedding dims
    idx_weight_pairs = []
    for i, config in enumerate(shard_config.table_configs):
        dim = config["dim"]
        row = config["row"]
        size = shard_config.table_sizes[i]
        idx_weight_pairs.append(TableInfo(index=i, cost=size, size=size))

    # Greedy algorithm
    num_bins = shard_config.ndevices
    sorted_idx_weight_pairs = sorted(idx_weight_pairs)
    partitions = [[] for p in range(num_bins)]
    partition_sums = [0.0] * num_bins
    partition_size_sums = [0.0] * num_bins

    mem_cap = [shard_config.max_mem] * num_bins

    while sorted_idx_weight_pairs:
        table_info = sorted_idx_weight_pairs.pop()
        min_sum = np.inf
        min_size_taken = np.inf
        min_r = -1
        for r in range(num_bins):
            if partition_size_sums[r] + table_info.size <= mem_cap[r]:
                if partition_sums[r] < min_sum or (
                    partition_sums[r] == min_sum
                    and partition_size_sums[r] < min_size_taken
                ):
                    min_sum = partition_sums[r]
                    min_r = r
                    min_size_taken = partition_size_sums[r]

        partitions[min_r].append(table_info)
        partition_sums[min_r] += table_info.cost
        partition_size_sums[min_r] += table_info.size

    partitions = [[table_info.index for table_info in partition] for partition in partitions]

    return partitions

@register_sharder("lookup_greedy")
def lookup_greedy_shard(shard_config):
    # Get the embedding dims
    idx_weight_pairs = []
    for i, config in enumerate(shard_config.table_configs):
        dim = config["dim"]
        row = config["row"]
        pooling_factor = config["pooling_factor"]
        size = shard_config.table_sizes[i]
        idx_weight_pairs.append(TableInfo(index=i, cost=dim*pooling_factor, size=size))

    # Greedy algorithm
    num_bins = shard_config.ndevices
    sorted_idx_weight_pairs = sorted(idx_weight_pairs)
    partitions = [[] for p in range(num_bins)]
    partition_sums = [0.0] * num_bins
    partition_size_sums = [0.0] * num_bins

    mem_cap = [shard_config.max_mem] * num_bins

    while sorted_idx_weight_pairs:
        table_info = sorted_idx_weight_pairs.pop()
        min_sum = np.inf
        min_size_taken = np.inf
        min_r = -1
        for r in range(num_bins):
            if partition_size_sums[r] + table_info.size <= mem_cap[r]:
                if partition_sums[r] < min_sum or (
                    partition_sums[r] == min_sum
                    and partition_size_sums[r] < min_size_taken
                ):
                    min_sum = partition_sums[r]
                    min_r = r
                    min_size_taken = partition_size_sums[r]

        partitions[min_r].append(table_info)
        partition_sums[min_r] += table_info.cost
        partition_size_sums[min_r] += table_info.size

    partitions = [[table_info.index for table_info in partition] for partition in partitions]

    return partitions

@register_sharder("size_lookup_greedy")
def size_lookup_greedy_shard(shard_config):
    # Get the embedding dims
    idx_weight_pairs = []
    for i, config in enumerate(shard_config.table_configs):
        dim = config["dim"]
        row = config["row"]
        pooling_factor = config["pooling_factor"]
        size = shard_config.table_sizes[i]
        idx_weight_pairs.append(TableInfo(index=i, cost=dim*pooling_factor*np.log10(row), size=size))

    # Greedy algorithm
    num_bins = shard_config.ndevices
    sorted_idx_weight_pairs = sorted(idx_weight_pairs)
    partitions = [[] for p in range(num_bins)]
    partition_sums = [0.0] * num_bins
    partition_size_sums = [0.0] * num_bins

    mem_cap = [shard_config.max_mem] * num_bins

    while sorted_idx_weight_pairs:
        table_info = sorted_idx_weight_pairs.pop()
        min_sum = np.inf
        min_size_taken = np.inf
        min_r = -1
        for r in range(num_bins):
            if partition_size_sums[r] + table_info.size <= mem_cap[r]:
                if partition_sums[r] < min_sum or (
                    partition_sums[r] == min_sum
                    and partition_size_sums[r] < min_size_taken
                ):
                    min_sum = partition_sums[r]
                    min_r = r
                    min_size_taken = partition_size_sums[r]

        partitions[min_r].append(table_info)
        partition_sums[min_r] += table_info.cost
        partition_size_sums[min_r] += table_info.size

    partitions = [[table_info.index for table_info in partition] for partition in partitions]

    return partitions

@register_sharder("random")
def random_shard(shard_config):
    table_device_indices = []
    for _ in range(shard_config.num_tables):
        table_device_indices.append(np.random.randint(shard_config.ndevices))

    return allocation2plan(table_device_indices, shard_config.ndevices)
