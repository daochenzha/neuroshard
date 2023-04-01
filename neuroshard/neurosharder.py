import numpy as np
import torch

from neuroshard.simulator import ShardingSimulator

from neuroshard.utils import (
    dict2tensor,
    allocation2plan,
    table_size,
)

class NeurSharder(ShardingSimulator):
    def __init__(
        self,
        compute_cost_model,
        fw_comm_cost_model,
        bw_comm_cost_model,
        table_configs,
        ndevices,
        max_mem,
    ):
        super().__init__(
            compute_cost_model,
            fw_comm_cost_model,
            bw_comm_cost_model,
            table_configs,
            ndevices,
            max_mem,
        )
        self.sharded = False
        self._compute_snapshots = {}
        self._compute_cache = {}

    def shard(self, k=3, n=10, max_length=10, search_steps=11):
        assert self.sharded is False, "Already sharded"

        shards, cost = self._table_wise_shard(search_steps=search_steps)

        self._snapshot_tables(tuple([]))
        sharding_steps_list = [([], cost)]
        best = ([], shards, cost)
        for _ in range(max_length):
            new_sharding_steps_list = []
            for sharding_steps_, _ in sharding_steps_list:
                self._snapshot_recover(tuple(sharding_steps_))
                if n > len(self._single_table_sizes):
                    indices = list(range(len(self._single_table_sizes)))
                else:
                    top_size_indices = np.argpartition(self._single_table_sizes, -n)[-n:]
                    top_compute_indices = np.argpartition(self._single_table_compute_costs, -n)[-n:]
                    indices = np.union1d(top_size_indices, top_compute_indices)
                for index in indices:
                    shard_dim = self._single_table_dims[index] // 2
                    if shard_dim % 4 != 0:
                        continue
                    self._snapshot_recover(tuple(sharding_steps_))
                    self._col_wise_shard(index, shard_dim)
                    shards, cost = self._table_wise_shard(search_steps=search_steps)
                    new_sharding_steps = sharding_steps_.copy()
                    new_sharding_steps.append(index)
                    new_sharding_steps_list.append((new_sharding_steps, cost))
                    if cost < best[2]:
                        best = (new_sharding_steps, shards, cost)
                    self._snapshot_tables(tuple(new_sharding_steps))
            if not new_sharding_steps_list:
                break
            sharding_steps_list = sorted(new_sharding_steps_list, key=lambda x: x[1])[:k]

        self._snapshot_recover(tuple(best[0]))
        self.sharded = True

        return best[0], best[1]

    def _reset_tables(self):
        super()._reset_tables()
        self._single_table_compute_costs = self._get_compute_cost(
            [[i] for i in range(len(self._table_features))], self._table_features
        )

    def _snapshot_tables(self, key):
        super()._snapshot_tables(key)
        self._compute_snapshots[key] = self._single_table_compute_costs.copy()

    def _snapshot_recover(self, key):
        super()._snapshot_recover(key)
        self._single_table_compute_costs = self._compute_snapshots[key].copy()

    def _col_wise_shard(self, index, shard_dim):
        super()._col_wise_shard(index, shard_dim)
        self._single_table_compute_costs[index] = self._get_compute_cost(
            [[index]], self._table_features
        )[0]
        self._single_table_compute_costs.append(self._single_table_compute_costs[index])

    def _greedy(self, dim_constraint_scale=None):
        if dim_constraint_scale is None:
            dim_constraint_scale = float("inf")
        shards = [[] for p in range(self._ndevices)]
        shard_size_sums = [0.0] * self._ndevices
        shard_dim_sums = [0] * self._ndevices
        shard_compute_cost_sums = [0.0] * self._ndevices

        indices = list(range(self._table_features.shape[0]))
        indices = sorted(
            indices,
            key=lambda x: self._single_table_compute_costs[x],
        )
        while indices:
            index = indices.pop()
            min_sum = float("inf")
            min_size_taken = float("inf")
            min_r = -1

            for r in range(self._ndevices):
                if (
                    shard_size_sums[r] + self._single_table_sizes[index] <= self._max_mem
                    and shard_dim_sums[r] + self._single_table_dims[index] <= dim_constraint_scale * self._mean_dim
                ):
                    if shard_compute_cost_sums[r] < min_sum or (
                        shard_compute_cost_sums[r] == min_sum and shard_size_sums[r] < min_size_taken
                    ):
                        min_sum = shard_compute_cost_sums[r]
                        min_r = r
                        min_size_taken = shard_size_sums[r]

            if min_r == -1:
                return None, float("inf")

            shards[min_r].append(index)
            shard_size_sums[min_r] += self._single_table_sizes[index]
            shard_dim_sums[min_r] += self._single_table_dims[index]

            #shard_compute_cost_sums[min_r] = self._get_compute_cost([shards[min_r]], self._table_features)[0]
            key = tuple([tuple([self._index2raw[x] for x in shards[min_r]]), tuple([self._single_table_dims[x] for x in shards[min_r]])])
            if key in self._compute_cache:
                shard_compute_cost_sums[min_r] = self._compute_cache[key]
            else:
                shard_compute_cost_sums[min_r] = self._get_compute_cost([shards[min_r]], self._table_features)[0]
                self._compute_cache[key] = shard_compute_cost_sums[min_r]

        return shards, max(shard_compute_cost_sums)

    def _table_wise_shard(self, threshold_start=1.0, threshold_end=1.5, search_steps=11):

        step_size = (threshold_end - threshold_start) / (search_steps - 1)
        best_final_cost = float("inf")
        best_shards = None
        for i in range(search_steps):
            dim_constraint_scale = threshold_start + step_size * i
            shards, _ = self._greedy(dim_constraint_scale)
            if shards is not None:
                final_cost = self.get_final_cost(shards)
                if final_cost < best_final_cost:
                    best_final_cost = final_cost
                    best_shards = shards

        return best_shards, best_final_cost
