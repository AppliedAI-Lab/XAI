import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
import pandas as pd


class Regression(nn.Module):
      def __init__(self, input_dim, hidden_dim, output_dim):
            super(Regression, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

      def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


def get_fragment_elasticnet_pipeline(alphas=[0.01, 0.05, 0.1, 0.5, 1.0], l1_ratios=[0.5, 0.7, 0.9], max_iter=20000, cv=5, tol=1e-3, n_jobs=-1):
    """Build sklearn Pipeline: StandardScaler + ElasticNetCV for fragment-based pIC50 regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            alphas=list(alphas),
            l1_ratio=list(l1_ratios),
            cv=cv,
            n_jobs=n_jobs,
            max_iter=max_iter,
            tol=tol,
        )),
    ])


def get_fragment_importance(model, feature_names):
    """
    From fitted fragment ElasticNet pipeline, return Series of coefficient by fragment.
    coef > 0: fragment increases pIC50 (favorable for FAAH inhibitor).
    coef < 0: fragment decreases pIC50 (unfavorable).
    """
    coef = model.named_steps["enet"].coef_
    return pd.Series(coef, index=feature_names).sort_values()
    