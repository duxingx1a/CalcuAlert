# 读取数据
from typing import NamedTuple
import numpy as np
import pandas as pd
import ehr_utils, ehr_models, eval_utils

with open("timestamp.txt", "r", encoding="utf-8") as f:
    TIMESTAMP = f.read().strip()
MODEL_DIR = f"trained_models/{TIMESTAMP}"
OUTPUT_DIR = f"results/{TIMESTAMP}"
STACKING_DIR = f"{MODEL_DIR}/stacking"
CACHE_DIR = f"cache/{TIMESTAMP}"

LR = 1e-3
SEED = 42
N_EPOCHS = 100
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-4

RANDOM_STATE = 42
XGB_PARAM = dict(
    colsample_bytree=1.0,
    device="gpu",
    eval_metric="auc",
    gamma=0.0,
    learning_rate=0.0224,
    max_depth=5,
    min_child_weight=100,
    n_estimators=589,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    reg_alpha=0.143,
    reg_lambda=100.0,
    subsample=0.73,
    early_stopping_rounds=10,
)

logger = ehr_utils.get_logger("train_stacking_model")


def _train_eval_xgb(X_train: pd.DataFrame, Y_train: pd.Series, X_test: pd.DataFrame, Y_test: pd.Series, *, tag: str):
    """训练XGB并输出评估指标+图"""
    logger.info(f"\n=== 特征空间 [{tag}] 下训练 Stacking XGB 模型 ===")
    clf = ehr_models.get_model("XGBoost", use_optimized_params=False, **XGB_PARAM)
    clf.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=False)  # type: ignore
    train_proba = clf.predict_proba(X_train)[:, 1]  # type: ignore
    test_proba = clf.predict_proba(X_test)[:, 1]  # type: ignore
    eval_utils.report_sens_spec(Y_train, train_proba, Y_test, test_proba, output_dir=OUTPUT_DIR, tag=f'Stacking_XGB_{tag}', show=False)
    eval_utils.plot_roc_pr_curves(Y_train, train_proba, Y_test, test_proba, output_dir=OUTPUT_DIR, tag=f'Stacking_XGB_{tag}', show=False)
    eval_utils.prob_distributions(Y_train, train_proba, Y_test, test_proba, output_dir=OUTPUT_DIR, tag=f'Stacking_XGB_{tag}', show=False)
    score = f'{clf.best_score:.3f}'  # type: ignore
    ehr_utils.save_model_to_pkl(clf, STACKING_DIR, f'Stacking_XGB_{tag}_{score}')


class Dataset(NamedTuple):
    name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame


def main():
    logger.info("\n=== 1. 加载数据 ===")
    X_train, X_test, y_train, y_test = ehr_utils.preprocess_ehr_train_test_data("data_processed/benbu_baseline_cleaned_onehot_simulated80.csv")
    logger.info(f"原始训练集大小：{X_train.shape}")
    logger.info(f"原始测试集大小：{X_test.shape}")

    logger.info("=== 2. 生成元特征 ===")
    models = ehr_utils.load_all_trained_pkls(MODEL_DIR)
    logger.info(f"加载基模型数量: {len(models)}")

    meta_train = ehr_utils.generate_meta_features(models, X_train, cache_dir=CACHE_DIR, cache_name="meta_train")
    meta_test = ehr_utils.generate_meta_features(models, X_test, cache_dir=CACHE_DIR, cache_name="meta_test")

    # 三种特征空间
    datasets = [
        Dataset("only_proba", meta_train, meta_test),
        Dataset("base_and_probas", pd.concat([X_train.reset_index(drop=True), meta_train], axis=1), pd.concat([X_test.reset_index(drop=True), meta_test], axis=1)),
        Dataset("base_and_mean_proba",
                pd.concat([X_train.reset_index(drop=True), pd.DataFrame(np.mean(meta_train.values, axis=1), columns=["mean_prob"])], axis=1),
                pd.concat([X_test.reset_index(drop=True), pd.DataFrame(np.mean(meta_test.values, axis=1), columns=["mean_prob"])], axis=1)),
    ]

    for ds in datasets:
        logger.info(">>> 特征空间: %s  %s / %s", ds.name, ds.X_train.shape, ds.X_test.shape)
        _train_eval_xgb(ds.X_train, y_train, ds.X_test, y_test, tag=ds.name)


if __name__ == '__main__':
    main()
