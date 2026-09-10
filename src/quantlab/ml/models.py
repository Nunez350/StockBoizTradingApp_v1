"""Deterministic model wrappers with a common fit/predict interface."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LGBMWrapper:
    def __init__(self, seed: int = 0, threads: int = 24, **params):
        self.params = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 200,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0} | params
        self.seed, self.threads = seed, threads
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(**self.params, random_state=self.seed, deterministic=True, force_col_wise=True,
                                       num_threads=self.threads, verbose=-1)
        self.model.fit(X.fillna(0.0).to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.fillna(0.0).to_numpy())

    def importances(self, names: list[str]) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=names).sort_values(ascending=False)


class TorchMLP:
    def __init__(self, seed: int = 0, hidden: int = 64, dropout: float = 0.1, epochs: int = 20, lr: float = 1e-3,
                 weight_decay: float = 1e-4, batch: int = 4096, device: str | None = None, **_):
        self.seed, self.hidden, self.dropout, self.epochs, self.lr, self.wd, self.batch = seed, hidden, dropout, epochs, lr, weight_decay, batch
        self.device = device
        self.net = None

    def _build(self, k: int):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(nn.Linear(k, self.hidden), nn.GELU(), nn.Dropout(self.dropout),
                                 nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Dropout(self.dropout),
                                 nn.Linear(self.hidden, 1)).to(self.device)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import torch

        Xa = torch.tensor(X.fillna(0.0).to_numpy(dtype=np.float32))
        ya = torch.tensor(y.to_numpy(dtype=np.float32))[:, None]
        self._build(Xa.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        g = torch.Generator().manual_seed(self.seed)
        n = len(Xa)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch):
                idx = perm[i : i + self.batch]
                xb, yb = Xa[idx].to(self.device), ya[idx].to(self.device)
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            xa = torch.tensor(X.fillna(0.0).to_numpy(dtype=np.float32)).to(self.device)
            return self.net(xa).cpu().numpy().ravel()


class RankEnsemble:
    def __init__(self, members: list):
        self.members = members

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        preds = [pd.Series(m.predict(X)).rank(pct=True).to_numpy() for m in self.members]
        return np.mean(preds, axis=0)


def make_model(kind: str, seed: int, params: dict):
    if kind == "lgbm":
        return LGBMWrapper(seed=seed, **params)
    if kind == "mlp":
        return TorchMLP(seed=seed, **params)
    if kind == "ensemble":
        return RankEnsemble([LGBMWrapper(seed=seed, **params.get("lgbm", {})), TorchMLP(seed=seed, **params.get("mlp", {}))])
    raise ValueError(kind)
