import glob
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import os
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.utils import resample
from scipy import stats
import seaborn as sns
from typing import Any, List, Tuple
import torch
from tqdm import tqdm

def plot_roc_pr_curves(y_train_true: np.ndarray, y_train_proba: np.ndarray, y_test_true: np.ndarray, y_test_proba: np.ndarray, model_name: str = 'model') -> None:
    """
    同时绘制训练集和测试集的ROC曲线与Precision-Recall曲线。
    - y_train_true: 训练集真实标签
    - y_train_proba: 训练集预测概率
    - y_test_true: 测试集真实标签
    - y_test_proba: 测试集预测概率
    - model_name: 模型名称，用于保存图像文件
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # ROC
    fpr_train, tpr_train, _ = roc_curve(y_train_true, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test_true, y_test_proba)
    auroc_train = roc_auc_score(y_train_true, y_train_proba)
    auroc_test = roc_auc_score(y_test_true, y_test_proba)

    axs[0].plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC (AUC={auroc_train:.2f})')
    axs[0].plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC (AUC={auroc_test:.2f})')
    axs[0].plot([0, 1], [0, 1], color='#666666', lw=1, linestyle='--')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('ROC Curve')
    axs[0].legend(loc="lower right")
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])

    # PR
    precision_train, recall_train, _ = precision_recall_curve(y_train_true, y_train_proba)
    precision_test, recall_test, _ = precision_recall_curve(y_test_true, y_test_proba)
    auprc_train = average_precision_score(y_train_true, y_train_proba)
    auprc_test = average_precision_score(y_test_true, y_test_proba)

    axs[1].plot(recall_train, precision_train, color='blue', lw=2, label=f'Train PR (AUC={auprc_train:.2f})')
    axs[1].plot(recall_test, precision_test, color='darkorange', lw=2, label=f'Test PR (AUC={auprc_test:.2f})')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title('Precision-Recall Curve')
    axs[1].legend(loc="lower left")
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 1])

    os.makedirs('results_fig', exist_ok=True)
    plt.savefig(f'results_fig/roc_pr_curve_{model_name}.png')
    plt.show()

    print(f"Train AUROC: {auroc_train:.4f}, Test AUROC: {auroc_test:.4f}")
    print(f"Train AUPRC: {auprc_train:.4f}, Test AUPRC: {auprc_test:.4f}")
    print(f"过拟合程度：{auroc_train - auroc_test:.4f}")


def prob_distributions(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, model_name=None):
    """
    绘制模型预测概率的分布图，分别展示训练集和验证集中正负样本（0/1类）的概率分布。
    支持直方图和KDE曲线，便于分析模型区分能力和过拟合情况。

    参数：
        model: 已训练的分类模型，需支持 predict_proba 方法
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 验证集特征
        y_test: 验证集标签
        model_name: 模型名称（可选，用于图像保存和标题）
    """
    if model_name is None:
        model_name = model.__class__.__name__

    # 获取预测概率
    if hasattr(model, 'predict_proba'):
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    else:  # PyTorch
        model.eval()
        with torch.no_grad():
            y_pred_proba_train = torch.sigmoid(model(torch.tensor(X_train.values, dtype=torch.float32))).numpy().ravel()
            y_pred_proba_test = torch.sigmoid(model(torch.tensor(X_test.values, dtype=torch.float32))).numpy().ravel()

    y_train_numpy = y_train.to_numpy()
    y_test_numpy = y_test.to_numpy()

    # 分离训练集 0/1 类概率
    proba_train_class_0 = y_pred_proba_train[y_train_numpy == 0]
    proba_train_class_1 = y_pred_proba_train[y_train_numpy == 1]

    # 分离验证集 0/1 类概率
    proba_test_class_0 = y_pred_proba_test[y_test_numpy == 0]
    proba_test_class_1 = y_pred_proba_test[y_test_numpy == 1]

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = {'class_0': 'blue', 'class_1': 'red'}
    alpha_hist = 0.6
    alpha_kde = 0.8

    # ===== 训练集分布图 =====
    ax1.hist(proba_train_class_0, bins=40, alpha=alpha_hist, color=colors['class_0'], density=True, label=f'Class 0 (n={len(proba_train_class_0)})')
    ax1.hist(proba_train_class_1, bins=40, alpha=alpha_hist, color=colors['class_1'], density=True, label=f'Class 1 (n={len(proba_train_class_1)})')

    if len(proba_train_class_0) > 1:
        sns.kdeplot(proba_train_class_0, ax=ax1, color=colors['class_0'], linewidth=2, alpha=alpha_kde)
    if len(proba_train_class_1) > 1:
        sns.kdeplot(proba_train_class_1, ax=ax1, color=colors['class_1'], linewidth=2, alpha=alpha_kde)

    mean_train_0 = np.mean(proba_train_class_0)
    mean_train_1 = np.mean(proba_train_class_1)
    ax1.axvline(mean_train_0, color=colors['class_0'], linestyle='--', alpha=0.8, linewidth=2, label=f'Class 0 Mean: {mean_train_0:.3f}')
    ax1.axvline(mean_train_1, color=colors['class_1'], linestyle='--', alpha=0.8, linewidth=2, label=f'Class 1 Mean: {mean_train_1:.3f}')

    separation_train = abs(mean_train_1 - mean_train_0)
    auc_train = roc_auc_score(y_train_numpy, y_pred_proba_train)
    ax1.set_title(f'{model_name} - Training Set\nAUC: {auc_train:.3f} | Separation: {separation_train:.3f}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Density')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # ===== 验证集分布图 =====
    ax2.hist(proba_test_class_0, bins=40, alpha=alpha_hist, color=colors['class_0'], density=True, label=f'Class 0 (n={len(proba_test_class_0)})')
    ax2.hist(proba_test_class_1, bins=40, alpha=alpha_hist, color=colors['class_1'], density=True, label=f'Class 1 (n={len(proba_test_class_1)})')

    if len(proba_test_class_0) > 1:
        sns.kdeplot(proba_test_class_0, ax=ax2, color=colors['class_0'], linewidth=2, alpha=alpha_kde)
    if len(proba_test_class_1) > 1:
        sns.kdeplot(proba_test_class_1, ax=ax2, color=colors['class_1'], linewidth=2, alpha=alpha_kde)

    mean_test_0 = np.mean(proba_test_class_0)
    mean_test_1 = np.mean(proba_test_class_1)
    ax2.axvline(mean_test_0, color=colors['class_0'], linestyle='--', alpha=0.8, linewidth=2, label=f'Class 0 Mean: {mean_test_0:.3f}')
    ax2.axvline(mean_test_1, color=colors['class_1'], linestyle='--', alpha=0.8, linewidth=2, label=f'Class 1 Mean: {mean_test_1:.3f}')

    separation_test = abs(mean_test_1 - mean_test_0)
    auc_test = roc_auc_score(y_test_numpy, y_pred_proba_test)
    ax2.set_title(f'{model_name} - Validation Set\nAUC: {auc_test:.3f} | Separation: {separation_test:.3f}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    os.makedirs('results_fig', exist_ok=True)
    plt.savefig(f'results_fig/prob_distributions_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    print(f"=== {model_name} 概率分布统计 ===")
    print("训练集:")
    print(f"  Class 0: 均值={mean_train_0:.3f}, 标准差={np.std(proba_train_class_0):.3f}")
    print(f"  Class 1: 均值={mean_train_1:.3f}, 标准差={np.std(proba_train_class_1):.3f}")
    print(f"  分离度: {separation_train:.3f}, AUC: {auc_train:.3f}")
    print("验证集:")
    print(f"  Class 0: 均值={mean_test_0:.3f}, 标准差={np.std(proba_test_class_0):.3f}")
    print(f"  Class 1: 均值={mean_test_1:.3f}, 标准差={np.std(proba_test_class_1):.3f}")
    print(f"  分离度: {separation_test:.3f}, AUC: {auc_test:.3f}")
    print("过拟合指标:")
    print(f"  AUC差异: {auc_train - auc_test:.3f}")
    print(f"  分离度差异: {separation_train - separation_test:.3f}")
