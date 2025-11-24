import json
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from typing import Any, Union
import torch
import torch.nn as nn
import numpy as np
# 读取参数文件
model_params = {
    "adaboost": {
        "learning_rate": 0.6856988052045783,
        "n_estimators": 200,
        "random_state": 42
    },
    "decision_tree": {
        "criterion": "gini",
        "max_depth": 7,
        "min_samples_leaf": 20,
        "min_samples_split": 5,
        "random_state": 42
    },
    "gaussian_nb": {
        "var_smoothing": 1e-9
    },
    "gradient_boosting": {
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 100,
        "random_state": 42,
        "subsample": 1.0
    },
    "lightgbm": {
        "bagging_fraction": 0.7911667142271672,
        "bagging_freq": 10,
        "boosting_type": "gbdt",
        "feature_fraction": 0.5,
        "learning_rate": 0.01,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "min_sum_hessian_in_leaf": 10.0,
        "n_estimators": 748,
        "n_jobs": -1,
        "num_leaves": 30,
        "random_state": 42,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "subsample": 0.6
    },
    "linear_discriminant_analysis": {
        "solver": "svd",
        "tol": 0.0008145222883402799
    },
    "logistic_regression": {
        "C": 9.975836024182009,
        "n_jobs": -1,
        "random_state": 42,
        "solver": "liblinear"
    },
    "mlp_classifier": {
        "activation": "relu",
        "alpha": 0.0001,
        "early_stopping": True,
        "hidden_layer_sizes": [200, 50],
        "learning_rate": "adaptive",
        "max_iter": 1000,
        "random_state": 42,
        "solver": "adam",
        "tol": 0.0001
    },
    "random_forest": {
        "bootstrap": True,
        "max_depth": 9,
        "max_features": None,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "n_estimators": 300,
        "n_jobs": -1,
        "random_state": 42
    },
    "xgboost": {
        "colsample_bytree": 1.0,
        "device": "gpu",
        "eval_metric": "auc",
        "gamma": 0.0,
        "learning_rate": 0.022403069086742198,
        "max_depth": 5,
        "min_child_weight": 100,
        "n_estimators": 589,
        "n_jobs": -1,
        "random_state": 42,
        "reg_alpha": 0.14314863930500873,
        "reg_lambda": 100.0,
        "subsample": 0.7300248552604385
    }
}
# 定义模型类型别名
ModelType = Union[AdaBoostClassifier, DecisionTreeClassifier, GaussianNB, GradientBoostingClassifier, lgb.LGBMClassifier, LinearDiscriminantAnalysis, LogisticRegression,
                  MLPClassifier, RandomForestClassifier, xgb.XGBClassifier]


class StackingMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class PytorchModelWrapper:
    """包装 PyTorch 模型以符合 scikit-learn 接口"""

    def __init__(self, torch_model, device):
        self.model = torch_model
        self.device = device

    def predict_proba(self, X):
        """
        X 可以是 numpy 或 pandas
        返回 ndarray (n_samples, 2) [:,1] 为正类概率
        """
        self.model.eval()
        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        prob_pos = 1 / (1 + np.exp(-logits))  # sigmoid
        prob_neg = 1 - prob_pos
        return np.column_stack([prob_neg, prob_pos])


def get_model(model_name: str, use_optimized_params: bool = True) -> ModelType:
    """
    根据模型名称返回对应的模型实例。
    :param model_name: 模型名称字符串
    :param use_optimized_params: 是否使用优化参数,默认True使用优化参数,False使用默认参数
    :return: 对应的模型实例
    """
    # 模型映射字典
    model_mapping = {
        'AdaBoost': (AdaBoostClassifier, 'adaboost'),
        'DecisionTree': (DecisionTreeClassifier, 'decision_tree'),
        'GaussianNB': (GaussianNB, 'gaussian_nb'),
        'GradientBoosting': (GradientBoostingClassifier, 'gradient_boosting'),
        'LightGBM': (lgb.LGBMClassifier, 'lightgbm'),
        'LinearDiscriminantAnalysis': (LinearDiscriminantAnalysis, 'linear_discriminant_analysis'),
        'LogisticRegression': (LogisticRegression, 'logistic_regression'),
        'MLPClassifier': (MLPClassifier, 'mlp_classifier'),
        'RandomForest': (RandomForestClassifier, 'random_forest'),
        'XGBoost': (xgb.XGBClassifier, 'xgboost')
    }

    if model_name not in model_mapping:
        raise ValueError(f"未知模型名称:{model_name}")

    model_class, param_key = model_mapping[model_name]

    # 根据参数选择使用优化参数或默认参数
    if use_optimized_params:
        params = model_params[param_key]
    else:
        # 默认参数,只为需要并行的模型设置n_jobs
        params = {}
        if model_name in ['LightGBM', 'LogisticRegression', 'RandomForest', 'XGBoost']:
            params['n_jobs'] = -1

    return model_class(**params)
