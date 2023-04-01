import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neuroshard.utils import IndexLoader

WEIGHT_TMP_PATH = "/tmp/comm_cost_net_weight.pt"
torch.set_num_threads(1)


class CommCostModel:
    def __init__(self, ndevices):
        self._cost_net = None
        self._feature_means = None
        self._feature_stds = None
        self._num_features = None
        self._mlp_sizes = None
        self.ndevices = ndevices

    def train(
        self,
        X,
        y,
        mlp_sizes=[128, 64, 32, 16],
        batch_size=512,
        eval_every=1,
        train_ratio=0.8,
        valid_ratio=0.1,
        epochs=10,
        lr=0.001,
    ):
        self._num_features = X.shape[1]
        self._mlp_sizes = mlp_sizes

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        self._cost_net = CommCostNet(
            self._num_features,
            self._mlp_sizes,
            self.ndevices,
        )
        self._cost_net = self._cost_net

        # Normalize features
        self._feature_means = X.mean(dim=0)
        self._feature_stds = X.std(dim=0) + 1e-6  # Small value to avoid nan
        X = (X - self._feature_means) / self._feature_stds

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
                batch_X, batch_y = X[batch_indices], y[batch_indices]
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
        y=None,
        batch_size=512,
        loader=None,
        normalize=True,
    ):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if isinstance(y, list):
            y = torch.tensor(y, dtype=torch.float32)

        if normalize:
            X = (X - self._feature_means) / self._feature_stds

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
                batch_X = X[batch_indices]
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
                "num_features": self._num_features,
                "mlp_sizes": self._mlp_sizes,
                "ndevices": self.ndevices,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self._feature_means = checkpoint["feature_means"]
        self._feature_stds = checkpoint["feature_stds"]
        self._num_features = checkpoint["num_features"]
        self._mlp_sizes = checkpoint["mlp_sizes"]
        self.ndevices = checkpoint["ndevices"]

        self._cost_net = CommCostNet(
            self._num_features,
            self._mlp_sizes,
            self.ndevices,
        )
        self._cost_net.load_state_dict(checkpoint["state_dict"])

class CommCostNet(nn.Module):
    def __init__(
        self,
        num_features,
        mlp_sizes,
        num_outputs,
    ) -> None:
        super().__init__()

        self.num_outputs = num_outputs
        mlp_sizes = [num_features] + mlp_sizes + [num_outputs]

        # Cost prediction
        self.hidden = nn.ModuleList()
        for k in range(len(mlp_sizes) - 1):
            self.hidden.append(nn.Linear(mlp_sizes[k], mlp_sizes[k + 1]))

    def forward(self, X):
        out = X
        for layer in self.hidden[:-1]:
            out = F.relu(layer(out))
        out = self.hidden[-1](out)

        return out.view(-1, self.num_outputs)
