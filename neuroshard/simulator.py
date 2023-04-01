import numpy as np
import torch

from neuroshard.utils import (
    dict2tensor,
    table_size,
)


class ShardingSimulator:
    def __init__(
        self,
        compute_cost_model,
        fw_comm_cost_model,
        bw_comm_cost_model,
        table_configs,
        ndevices,
        max_mem,
    ):
        self._compute_cost_model = compute_cost_model
        self._fw_comm_cost_model = fw_comm_cost_model
        self._bw_comm_cost_model = bw_comm_cost_model
        self._backup_table_configs = [table_config.copy() for table_config in table_configs]
        self._ndevices = ndevices
        self._max_mem = max_mem
        self._snapshots = {}
        self._index2raw_snapshots = {}

        self._reset_tables()


    def apply_col_wise_shard(self, col_wise_sharding_steps):
        for index in col_wise_sharding_steps:
            shard_dim = self._table_configs[index]["dim"] // 2
            self._col_wise_shard(index, shard_dim)

    def get_final_cost(self, shards):
        if not self._check_size(shards):
            return None
        shard_dim_sums = self._get_dim_sums(shards)
        compute_costs = self._get_compute_cost(shards, self._table_features)
        bw_comm_costs = self._get_bw_comm_cost(shard_dim_sums)
        fw_comm_costs = self._get_fw_comm_cost(shard_dim_sums, compute_costs, bw_comm_costs)
        return max(
            [
                compute_costs[r] + bw_comm_costs[r] + fw_comm_costs[r]
                for r in range(self._ndevices)
            ]
        )

    def print_sharding_results(self, shards):
        if not self._check_size(shards):
            print("Out of memory")
            return
        results = {
            "sizes": [self._tables_size(shard) for shard in shards],
            "dims": self._get_dim_sums(shards),
            "compute": self._get_compute_cost(shards, self._table_features),
        }
        results["bw_comm"] = self._get_bw_comm_cost(results["dims"])
        results["fw_comm"] = self._get_fw_comm_cost(results["dims"], results["compute"], results["bw_comm"])
        results["total"] = [
            results["bw_comm"][r] + results["compute"][r] + results["fw_comm"][r]
            for r in range(self._ndevices)
        ]

        for key in results:
            print(f"[✅] {key}")
            print(results[key])
            print(
                f"Max: {np.max(results[key])}, mean: {np.mean(results[key])}, min: {np.min(results[key])}"
            )
            max_rank = np.argmax(results[key])
            min_rank = np.argmin(results[key])
            for rank_type, rank in [("Max", max_rank), ("Min", min_rank)]:
                print(
                        "{} rank: {}, shard: {}, size: {} dim: {}, bw_comm: {}, compute: {}, fw_comm: {}, total: {}, single computes: {}, single dims: {}, single sizes: {}".format(
                        rank_type,
                        rank,
                        shards[rank],
                        results["sizes"][rank],
                        results["dims"][rank],
                        results["bw_comm"][rank],
                        results["compute"][rank],
                        results["fw_comm"][rank],
                        results["total"][rank],
                        [
                            self._get_compute_cost([[i]], self._table_features)[0]
                            for i in shards[rank]
                        ],
                        [self._table_configs[i]["dim"] for i in shards[rank]],
                        [self._tables_size([i]) for i in shards[rank]],
                    )
                )

    def get_table_configs(self):
        return self._table_configs

    def _snapshot_tables(self, key):
        self._snapshots[key] = [table_config.copy() for table_config in self._table_configs]
        self._index2raw_snapshots[key] = self._index2raw.copy()

    def _snapshot_recover(self, key):
        self._table_configs = [table_config.copy() for table_config in self._snapshots[key]]
        self._index2raw = self._index2raw_snapshots[key]
        self._recover_from_table_configs()

    def _reset_tables(self):
        self._table_configs = [table_config.copy() for table_config in self._backup_table_configs]
        self._index2raw = {i: i for i in range(len(self._table_configs))}
        self._recover_from_table_configs()

    def _recover_from_table_configs(self):
        self._table_features = dict2tensor(self._table_configs)
        self._single_table_dims = [
            table_config["dim"] for table_config in self._table_configs
        ]
        self._mean_dim = float(sum(self._single_table_dims)) / self._ndevices
        self._single_table_sizes = [
            self._tables_size([i]) for i in range(len(self._table_configs))
        ]
        self._mean_size = sum(self._single_table_sizes) / self._ndevices

    def _check_size(self, shards):
        max_size_sum = max([self._tables_size(shard) for shard in shards])
        if max_size_sum > self._max_mem: 
            return False
        return True 

    def _get_compute_cost(self, X, table_features):
        return self._compute_cost_model.predict(X, table_features).tolist()

    def _get_bw_comm_cost(self, dims):
        return self._bw_comm_cost_model.predict(torch.tensor([dims])).flatten().tolist()

    def _get_fw_comm_cost(self, dims, compute, bw_comm):
        sleep_times = [compute[i] + bw_comm[i] for i in range(len(compute))]
        sleep_times = [x - min(sleep_times) for x in sleep_times]
        return self._fw_comm_cost_model.predict(torch.tensor([dims + sleep_times])).flatten().tolist()

    def _col_wise_shard(self, index, shard_dim):
        shard_size = self._single_table_sizes[index] * shard_dim / self._single_table_dims[index]
        self._table_configs[index]["dim"] = shard_dim
        self._table_configs.append(self._table_configs[index].copy())
        self._table_features = torch.cat(
            (
                self._table_features,
                dict2tensor([self._table_configs[index]]),
            )
        )
        self._table_features[index] = self._table_features[-1]

        self._single_table_dims[index] = shard_dim
        self._single_table_dims.append(shard_dim)
        self._single_table_sizes[index] = shard_size
        self._single_table_sizes.append(shard_size)
        self._index2raw[len(self._table_configs)-1] = self._index2raw[index]
        # print(f"Col-wise shard table {index}")

    def _tables_size(self, x):
        x_table_configs = [self._table_configs[i] for i in x]
        sizes = [
            table_size(table_config["row"], table_config["dim"])
            for table_config in x_table_configs
        ]
        return sum(sizes)

    def _get_dim_sums(self, shards):
        return [
            sum([self._table_configs[i]["dim"] for i in shard])
            for shard in shards
        ]

