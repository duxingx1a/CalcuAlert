import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import seaborn as sns
from typing import Any, Dict, List, Tuple
# import torch
from tqdm import tqdm
from matplotlib.figure import Figure


def report_sens_spec(
    y_train_true: np.ndarray | pd.Series,
    y_train_proba: np.ndarray,
    y_val_true: np.ndarray | pd.Series,
    y_val_proba: np.ndarray,
    output_dir: str = 'results_tmp',
    tag: str = 'model',
    show=True,
) -> Figure:
    """
    打印并可选绘图：训练集 vs 验证集的敏感度、特异度、C-index、Youden Index
    y_tr/y_va：训练/验证真实标签
    p_tr/p_va：训练/验证正类概率
    output_dir: 输出目录（未指定则为'results_tmp'）
    tag: 标签（可选，用于图像保存和标题）
    show：是否显示图像
    """

    def _calc(y, p):
        fpr, tpr, thr = roc_curve(y, p)
        sens = tpr
        spec = 1 - fpr
        youden = sens - fpr
        best_idx = np.argmax(youden)
        return {
            'sens': sens[best_idx],
            'spec': spec[best_idx],
            'youden': youden[best_idx],
            'cindex': roc_auc_score(y, p),
            'cut': thr[best_idx],
            'thr': thr,
            'sens_curve': sens,
            'spec_curve': spec
        }

    tr = _calc(y_train_true, y_train_proba)
    va = _calc(y_val_true, y_val_proba)

    print('>> 训练集 vs 验证集 最优阈值指标（Youden）')
    print(f'{"":<10} {"Train":<10} {"Test":<10}')
    print(f'{"Sensitivity":<10} {tr["sens"]:<10.3f} {va["sens"]:<10.3f}')
    print(f'{"Specificity":<10} {tr["spec"]:<10.3f} {va["spec"]:<10.3f}')
    print(f'{"C-index":<10} {tr["cindex"]:<10.3f} {va["cindex"]:<10.3f}')
    print(f'{"Youden":<10} {tr["youden"]:<10.3f} {va["youden"]:<10.3f}')
    print(f'{"Cut-point":<10} {tr["cut"]:<10.3f} {va["cut"]:<10.3f}')

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    fig.suptitle(f"Sensitivity-Specificity vs Threshold ({tag})", fontsize=16)
    plt.title('Scatter Plot Example')
    for i, (name, dic) in enumerate(zip(['Train', 'Test'], [tr, va])):
        ax[i].plot(dic['thr'], dic['sens_curve'], label='Sensitivity')
        ax[i].plot(dic['thr'], dic['spec_curve'], label='Specificity')
        ax[i].axvline(dic['cut'], ls='--', c='k', label=f'Best cut={dic["cut"]:.3f}')
        ax[i].set_xlabel('Threshold')
        ax[i].set_ylabel('Rate')
        ax[i].set_title(f'{name} Set')
        ax[i].legend()
        ax[i].grid()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/sens_spec_{tag}.png', dpi=300)
    if show:
        plt.show()
    plt.close()
    return fig


def plot_roc_pr_curves(y_train_true: np.ndarray | pd.Series,
                       y_train_proba: np.ndarray,
                       y_test_true: np.ndarray | pd.Series,
                       y_test_proba: np.ndarray,
                       output_dir: str = 'results_tmp',
                       tag: str = 'model',
                       show: bool = True) -> Figure:
    """
    同时绘制训练集和测试集的ROC曲线与Precision-Recall曲线。
    - y_train_true: 训练集真实标签
    - y_train_proba: 训练集预测概率
    - y_test_true: 测试集真实标签
    - y_test_proba: 测试集预测概率
    - output_dir: 输出目录（未指定则为'results_tmp'）
    - tag: 标签，用于保存图像文件
    - show: 是否显示图像
    """

    # ROC计算
    fpr_train, tpr_train, _ = roc_curve(y_train_true, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test_true, y_test_proba)
    auroc_train = roc_auc_score(y_train_true, y_train_proba)
    auroc_test = roc_auc_score(y_test_true, y_test_proba)

    # PR计算
    precision_train, recall_train, _ = precision_recall_curve(y_train_true, y_train_proba)
    precision_test, recall_test, _ = precision_recall_curve(y_test_true, y_test_proba)
    auprc_train = average_precision_score(y_train_true, y_train_proba)
    auprc_test = average_precision_score(y_test_true, y_test_proba)
    print(f"Train AUROC: {auroc_train:.4f}, Test AUROC: {auroc_test:.4f}")
    print(f"Train AUPRC: {auprc_train:.4f}, Test AUPRC: {auprc_test:.4f}")
    print(f"过拟合程度：{auroc_train - auroc_test:.4f}")

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    fig.suptitle(f"ROC & PR Curves ({tag})", fontsize=16)
    # ROC 曲线
    axs[0].plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC (AUC={auroc_train:.2f})')
    axs[0].plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC (AUC={auroc_test:.2f})')
    axs[0].plot([0, 1], [0, 1], color='#666666', lw=1, linestyle='--')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('ROC Curve')
    axs[0].legend(loc="lower right")
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])

    # PR 曲线
    axs[1].plot(recall_train, precision_train, color='blue', lw=2, label=f'Train PR (AUC={auprc_train:.2f})')
    axs[1].plot(recall_test, precision_test, color='darkorange', lw=2, label=f'Test PR (AUC={auprc_test:.2f})')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title('Precision-Recall Curve')
    axs[1].legend(loc="lower left")
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 1])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/roc_pr_curves_{tag}.png', dpi=300)
    if show:
        plt.show()
    plt.close()
    return fig


_ColorCycle = plt.cm.tab10(range(10))  # type: ignore
_LineStyle = ["-", "--", "-.", ":"]


def plot_multi_roc(
        proba_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],  # {model_name: (y_true, y_proba)}
        *,
        subset: str = "test",  # "train" | "test" | "both"
        title: str | None = None,
        figsize: Tuple[int, int] = (7, 6),
        output_dir: str = "results_tmp",
        tag: str = "multi_roc",
        show: bool = True,
) -> Figure:
    """
    一次性绘制多个模型的 ROC 曲线（单张图）

    参数
    ----
    proba_dict : dict[str, Tuple[np.ndarray, np.ndarray]]
        key   -> 模型名称（用于图例）
        value -> (y_true, y_proba)  二分类正类概率
    subset : str, 默认 "test"
        画哪部分数据： "train" / "test" / "both"
    title : str, 可选
        图标题，None 则自动生成
    figsize : tuple
        画布大小
    output_dir : str
        保存目录
    tag : str
        文件名后缀
    show : bool
        是否立即 show()

    返回
    ----
    fig : matplotlib.figure.Figure
        可继续二次编辑
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="gray", label="Luck")

    for idx, (name, (y_true, y_proba)) in enumerate(proba_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        color = _ColorCycle[idx % len(_ColorCycle)]
        linestyle = _LineStyle[idx % len(_LineStyle)]

        if subset == "both":
            label = f"{name} (AUC={roc_auc:.3f})"
        else:
            label = f"{name} {subset.title()} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, color=color, ls=linestyle, lw=2, label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or f"ROC Curves ({subset})", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/ulti_roc_{tag}.png', dpi=300)
    if show:
        plt.show()

    return fig


def prob_distributions(y_train_true: np.ndarray | pd.Series,
                       y_train_proba: np.ndarray,
                       y_test_true: np.ndarray | pd.Series,
                       y_test_proba: np.ndarray,
                       output_dir: str = 'results_tmp',
                       tag: str = 'model',
                       show: bool = True) -> None:
    """
    绘制模型预测概率的分布图，分别展示训练集和验证集中正负样本（0/1类）的概率分布。
    支持直方图和KDE曲线，便于分析模型区分能力和过拟合情况。

    参数：
        y_train_true: 训练集真实标签
        y_train_proba: 训练集预测概率
        y_test_true: 验证集真实标签
        y_test_proba: 验证集预测概率
        output_dir: 输出目录（未指定则为'results_tmp'）
        tag: 标签，用于保存图像文件和标题
        show: 是否显示图像
    """

    # 分离训练集 0/1 类概率
    proba_train_class_0 = y_train_proba[y_train_true == 0]
    proba_train_class_1 = y_train_proba[y_train_true == 1]

    # 分离验证集 0/1 类概率
    proba_test_class_0 = y_test_proba[y_test_true == 0]
    proba_test_class_1 = y_test_proba[y_test_true == 1]

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
    auc_train = roc_auc_score(y_train_true, y_train_proba)
    ax1.set_title(f'{tag} - Training Set\nAUC: {auc_train:.3f} | Separation: {separation_train:.3f}', fontsize=12, fontweight='bold')
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
    auc_test = roc_auc_score(y_test_true, y_test_proba)
    ax2.set_title(f'{tag} - Validation Set\nAUC: {auc_test:.3f} | Separation: {separation_test:.3f}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/prob_distribution_{tag}.png', dpi=300)
    if show:
        plt.show()

    # 打印统计信息
    print(f"=== {tag} 概率分布统计 ===")
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
