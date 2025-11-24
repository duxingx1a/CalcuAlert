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


def _apply_log1p_transform(X: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    对数值特征进行log1p变换的辅助函数
    参数：
        X: 特征数据框
    返回：
        tuple: (X_transformed, num_cols)
            - X_transformed: 经过log1p变换的特征数据框
            - num_cols: 需要变换的数值列名列表
    """
    num_cols = [col for col in X.select_dtypes(include=[np.number]).columns if X[col].nunique() > 3]
    X_transformed = X.copy()
    X_transformed[num_cols] = np.log1p(X[num_cols])
    return X_transformed, num_cols


def load_and_preprocess_ehr_data(csv_path: str, target_col: str = 'stone', split_data: bool = True, test_size: float = 0.2, random_state: int = 42, verbose: bool = True):
    """
    从CSV文件读取数据并进行预处理，返回log1p变换后的数据
    参数：
        csv_path (str): CSV文件路径
        target_col (str): 目标变量列名，默认为'stone'
        split_data (bool): 是否划分训练集和测试集，默认True
        test_size (float): 测试集比例，默认为0.2（仅在split_data=True时有效）
        random_state (int): 随机种子，默认为42（仅在split_data=True时有效）
        verbose (bool): 是否打印详细信息，默认True
    返回：
        如果 split_data=True:
            tuple: (X_train_log1p, X_val_log1p, y_train, y_val)
        如果 split_data=False:
            tuple: (X_log1p, y)
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 分离特征和目标变量
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if split_data:
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        # 对训练集和验证集分别进行log1p变换
        X_train_log1p, num_cols = _apply_log1p_transform(X_train)
        X_val_log1p, _ = _apply_log1p_transform(X_val)
        if verbose:
            print(f"总特征数：{len(X.columns)}，需要log1p变换的特征数：{len(num_cols)}")
            print(f"训练集样本数：{len(X_train)}，验证集样本数：{len(X_val)}")
        return X_train_log1p, X_val_log1p, y_train, y_val

    else:
        # 对整个数据集进行log1p变换
        X_log1p, num_cols = _apply_log1p_transform(X)
        if verbose:
            print(f"总特征数：{len(X.columns)}，需要log1p变换的特征数：{len(num_cols)}")
            print(f"缺失值统计：")
            print(X.isna().sum()[X.isna().sum() > 0])
            print(f"变换后缺失值统计：")
            print(X_log1p.isna().sum()[X_log1p.isna().sum() > 0])
        return X_log1p, y


# 保持向后兼容的包装函数
def preprocess_ehr_train_test_data(csv_path: str, target_col: str = 'stone', test_size: float = 0.2, random_state: int = 42) -> tuple:
    """向后兼容的训练集/测试集划分函数"""
    return load_and_preprocess_ehr_data(csv_path, target_col, split_data=True, test_size=test_size, random_state=random_state)


def load_external(csv_path: str, target_col: str = 'stone') -> tuple:
    """向后兼容的外部数据加载函数"""
    return load_and_preprocess_ehr_data(csv_path, target_col, split_data=False)


def handle_missing_values(df, method='simple', strategy='mean', n_neighbors=5, max_iter=10, random_state=42):
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.pipeline import FeatureUnion
    from sklearn.impute import MissingIndicator
    """
    处理缺失值的通用函数。

    参数:
        df (pd.DataFrame): 输入的 DataFrame。
        method (str): 处理缺失值的方法，可选 'simple'、'knn'、'iterative'、'drop' 或 'indicator'。
        strategy (str): 填充策略，用于 'simple' 和 'indicator' 方法，可选 'mean'、'median'、'most_frequent' 或 'constant'。
        n_neighbors (int): 用于 'knn' 方法的最近邻数量。
        max_iter (int): 用于 'iterative' 方法的最大迭代次数。
        random_state (int): 用于 'iterative' 方法的随机种子。

    返回:
        pd.DataFrame: 处理后的 DataFrame。
    """
    if method == 'simple':
        # 使用 SimpleImputer 填充缺失值
        imputer = SimpleImputer(strategy=strategy)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif method == 'knn':
        # 使用 KNNImputer 填充缺失值
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif method == 'iterative':
        # 使用 IterativeImputer 填充缺失值
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif method == 'drop':
        # 删除包含缺失值的行
        df_imputed = df.dropna()
    elif method == 'indicator':
        # 使用 MissingIndicator 标记缺失值，并用 SimpleImputer 填充
        indicator = MissingIndicator()
        imputer = SimpleImputer(strategy=strategy)
        transformer = FeatureUnion([("indicators", indicator), ("imputer", imputer)])
        df_imputed = pd.DataFrame(transformer.fit_transform(df), columns=df.columns)  # type: ignore
    else:
        raise ValueError("Invalid method. Choose from 'simple', 'knn', 'iterative', 'drop', or 'indicator'.")
    return df_imputed


def load_all_pkls(dir_path: str) -> List[Tuple[str, Any]]:
    """
    从指定目录加载所有 .pkl 模型文件。

    参数：
        dir_path (str): 模型文件所在目录。

    返回：
        List[Tuple[str, Any]]: 模型名称与模型实例的列表。
    """
    files = sorted(glob.glob(os.path.join(dir_path, '*.pkl')))
    models = []
    for fp in files:
        with open(fp, 'rb') as f:
            m = pickle.load(f)
        name = os.path.basename(fp).replace('.pkl', '')
        models.append((name, m))
    print(f'[INFO] 共加载 {len(models)} 个模型：{[n for n, _ in models]}')
    return models


def report_sens_spec(y_tr, p_tr, y_va, p_va, plot=True) -> None:
    """
    打印并可选绘图：训练集 vs 验证集的敏感度、特异度、C-index、Youden Index
    y_tr/y_va：训练/验证真实标签
    p_tr/p_va：训练/验证正类概率
    plot：是否画阈值-指标曲线
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

    tr = _calc(y_tr, p_tr)
    va = _calc(y_va, p_va)

    print('>> 训练集 vs 验证集 最优阈值指标（Youden）')
    print(f'{"":<10} {"Train":<10} {"Test":<10}')
    print(f'{"Sensitivity":<10} {tr["sens"]:<10.3f} {va["sens"]:<10.3f}')
    print(f'{"Specificity":<10} {tr["spec"]:<10.3f} {va["spec"]:<10.3f}')
    print(f'{"C-index":<10} {tr["cindex"]:<10.3f} {va["cindex"]:<10.3f}')
    print(f'{"Youden":<10} {tr["youden"]:<10.3f} {va["youden"]:<10.3f}')
    print(f'{"Cut-point":<10} {tr["cut"]:<10.3f} {va["cut"]:<10.3f}')
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
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
        plt.show()


def generate_meta_features(models, X, cache_name='meta_proba', cache_dir='cache', use_cache=True):
    """
    生成元特征(Meta Features)用于 Stacking 集成学习
    
    该函数将多个基础模型对输入数据的预测概率作为新的特征，用于训练二级模型。
    生成的元特征会被缓存到磁盘，避免重复计算。
    
    参数：
        models (List[Tuple[str, model]]): 基础模型列表，每个元素为 (模型名称, 模型对象) 的元组
        X (pd.DataFrame or np.ndarray): 输入特征数据，形状为 (N_samples, N_features)
        cache_name (str): 缓存文件名（不包含扩展名），默认为 'meta_proba'
        cache_dir (str): 缓存目录路径，默认为 'cache'，所有文件都保存在这个目录下
        use_cache (bool): 是否使用缓存，默认为 True

    返回：
        torch.Tensor: 元特征张量，形状为 (N_samples, n_models)
                     每一列对应一个基础模型对正类的预测概率
    
    注意：
        - predict_proba 返回形状为 (N_samples, 2)，[:, 1] 取正类概率
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{cache_name}.pt')
    if use_cache and os.path.exists(cache_file):
        print(f'[INFO] 直接读取缓存: {cache_file}')
        return torch.load(cache_file)
    probas = []
    for name, m in tqdm(models, desc=f'生成元特征 ({cache_name})'):
        probas.append(m.predict_proba(X)[:, 1])

    probas = np.column_stack(probas)
    tensor = torch.tensor(probas, dtype=torch.float32)
    torch.save(tensor, cache_file)
    print(f'[INFO] 元特征已保存: {cache_file} (shape: {tensor.shape})')

    return tensor
