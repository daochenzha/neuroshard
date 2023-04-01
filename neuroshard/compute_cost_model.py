import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neuroshard.utils import IndexLoader

WEIGHT_TMP_PATH = "/tmp/compute_cost_net_weight.pt"
torch.set_num_threads(1)


class ComputeCostModel:
    def __init__(self):
        self._cost_net = None
        self._feature_means = None
        self._feature_stds = None
        self._num_table_features = None
        self._table_mlp_sizes = None
        self._pred_mlp_sizes = None

    def train(
        self,
        X,
        y,
        table_features,
        table_mlp_sizes=[128, 32],
        pred_mlp_sizes=[32,64],
        batch_size=512,
        eval_every=1,
        train_ratio=0.8,
        valid_ratio=0.1,
        epochs=10,
        lr=0.001,
    ):
        self._num_table_features = table_features.shape[1]
        self._table_mlp_sizes = table_mlp_sizes
        self._pred_mlp_sizes = pred_mlp_sizes

        y = torch.tensor(y, dtype=torch.float32)

        self._cost_net = ComputeCostNet(
            self._num_table_features,
            self._table_mlp_sizes,
            self._pred_mlp_sizes,
        )
        self._cost_net = self._cost_net

        # Normalize features
        self._feature_means = table_features.mean(dim=0)
        self._feature_stds = (
            table_features.std(dim=0) + 1e-6
        )  # Small value to avoid nan
        table_features = (table_features - self._feature_means) / self._feature_stds

        # Split train/test sets
        indices = np.array(list(range(len(X))))
        np.random.shuffle(indices)
        train_indices, val_indices, test_indices = (
            indices[: int(len(X) * train_ratio)],
            indices[
                int(len(X) * train_ratio) : int(len(X) * (train_ratio + valid_ratio))
            ],
            indices[int(len(X) * (train_ratio + valid_ratio)) :],
        )

        train_loader = IndexLoader(train_indices, batch_size, shuffle=True)
        val_loader = IndexLoader(val_indices, batch_size)
        test_loader = IndexLoader(test_indices, batch_size)

        optimizer = torch.optim.Adam(list(self._cost_net.parameters()), lr=lr)

        best_mses = {k: float("inf") for k in ["train", "val", "test"]}
        for epoch in range(1, epochs + 1):
            # Training
            self._cost_net.train()
            epoch_losses = []
            for batch_indices in train_loader:
                optimizer.zero_grad()
                batch_X, batch_y = get_batch(batch_indices, X, y, table_features)
                pred_y = self._cost_net.forward(batch_X)
                loss = ((batch_y - pred_y) ** 2).mean()
                epoch_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            # Validation, and Testing
            if epoch % eval_every == 0:
                mses = {"train": np.mean(epoch_losses)}
                for split, loader in [
                    ("val", val_loader),
                    ("test", test_loader),
                ]:
                    _, mses[split] = self.predict(
                        X,
                        table_features=table_features,
                        y=y,
                        loader=loader,
                        normalize=False,
                    )
                print(f"Epoch: {epoch}, train MSE: {mses['train']}, valid MSE {mses['val']}, test MSE: {mses['test']}")
                if mses["val"] < best_mses["val"]:
                    torch.save(self._cost_net.state_dict(), WEIGHT_TMP_PATH)
                    print(f"Found better model with val_mse={mses['val']}")
                    for k in mses:
                        best_mses[k] = mses[k]

        # Final results
        print(f"Final result, train MSE: {best_mses['train']}, valid MSE {best_mses['val']}, test MSE: {best_mses['test']}")

        self._cost_net.load_state_dict(torch.load(WEIGHT_TMP_PATH))
        os.remove(WEIGHT_TMP_PATH)

        return best_mses

    def predict(
        self,
        X,
        table_features,
        y=None,
        batch_size=512,
        loader=None,
        normalize=True,
    ):
        if normalize:
            table_features = (
                table_features - self._feature_means
            ) / self._feature_stds

        if isinstance(y, list):
            y = torch.tensor(y, dtype=torch.float32)

        self._cost_net.eval()

        collect_real_y = True
        if loader is None:
            loader = IndexLoader(np.array(list(range(len(X)))), batch_size)
            real_y = y
            collect_real_y = False

        pred_y = []
        if y is not None and collect_real_y:
            real_y = []
        with torch.no_grad():
            for batch_indices in loader:
                batch_X, batch_y = get_batch(
                    batch_indices,
                    X,
                    y,
                    table_features,
                )
                pred_y.append(self._cost_net.forward(batch_X))
                if y is not None and collect_real_y:
                    real_y.append(y[batch_indices])
        pred_y = torch.cat(pred_y)
        if y is None:
            return pred_y

        if collect_real_y:
            real_y = torch.cat(real_y)
        return pred_y, ((real_y - pred_y) ** 2).mean().item()

    def save(self, path):
        path_dir = os.path.dirname(path)
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        torch.save(
            {
                "state_dict": self._cost_net.state_dict(),
                "feature_means": self._feature_means,
                "feature_stds": self._feature_stds,
                "num_table_features": self._num_table_features,
                "table_mlp_sizes": self._table_mlp_sizes,
                "pred_mlp_sizes": self._pred_mlp_sizes,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self._feature_means = checkpoint["feature_means"]
        self._feature_stds = checkpoint["feature_stds"]
        self._num_table_features = checkpoint["num_table_features"]
        self._table_mlp_sizes = checkpoint["table_mlp_sizes"]
        self._pred_mlp_sizes = checkpoint["pred_mlp_sizes"]

        self._cost_net = ComputeCostNet(
            self._num_table_features,
            self._table_mlp_sizes,
            self._pred_mlp_sizes,
        )
        self._cost_net.load_state_dict(checkpoint["state_dict"])

class ComputeCostNet(nn.Module):
    def __init__(
        self,
        num_table_features,
        table_mlp_sizes,
        pred_mlp_sizes,
    ) -> None:
        super().__init__()

        table_mlp_sizes = [num_table_features] + table_mlp_sizes
        pred_mlp_sizes = pred_mlp_sizes + [1]

        # Table feature extraction
        self.t_hidden = nn.ModuleList()
        for k in range(len(table_mlp_sizes) - 1):
            self.t_hidden.append(nn.Linear(table_mlp_sizes[k], table_mlp_sizes[k + 1]))

        # Cost prediction
        self.c_hidden = nn.ModuleList()
        for k in range(len(pred_mlp_sizes) - 1):
            self.c_hidden.append(nn.Linear(pred_mlp_sizes[k], pred_mlp_sizes[k + 1]))

    def forward(self, X):
        X_len = torch.tensor([x.shape[0] for x in X], device=X[0].device)
        B = X_len.shape[0]

        out = torch.cat(X, dim=0)
        for layer in self.t_hidden:
            out = F.relu(layer(out))

        ind = torch.repeat_interleave(
            torch.arange(len(X_len), device=X[0].device), X_len
        )
        tmp = torch.zeros((X_len.shape[0], out.shape[1]), device=X[0].device)
        tmp.index_add_(0, ind, out)
        out = tmp

        for layer in self.c_hidden[:-1]:
            out = F.relu(layer(out))
        out = self.c_hidden[-1](out)
        out = out.view(B)

        return out


def get_batch(
    indices,
    X,
    y=None,
    table_features=None,
):
    batch_X = []
    for index in indices:
        _X = X[index]
        if table_features is not None:
            _X = table_features[_X]
        batch_X.append(_X)

    if y is None:
        return batch_X, None

    return batch_X, y[indices]

