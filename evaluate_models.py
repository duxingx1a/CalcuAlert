import os
import glob
import pickle
from re import X
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, confusion_matrix)
import ehr_utils, ehr_models, eval_utils
from typing import Tuple, Any, List

with open("timestamp.txt", "r", encoding="utf-8") as f:
    TIMESTAMP = f.read().strip()
MODEL_DIR = f"trained_models/{TIMESTAMP}"
OUTPUT_DIR = f"results/{TIMESTAMP}"
CACHE_DIR = f"cache/{TIMESTAMP}"


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, xl_path: str = "metrics.xlsx",
                   sheet_name: str = "Sheet1", threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    youden = sens + spec - 1
    res = {"Model" : model_name, "AUROC": auroc, "AUPRC": auprc, "Sensitivity": sens, "Specificity": spec,
           "Youden": youden}
    df_new = pd.DataFrame([res])

    # 如果文件不存在，先创建一个空文件
    if not os.path.isfile(xl_path):
        with pd.ExcelWriter(xl_path, engine='openpyxl') as writer:
            df_new.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # 追加/更新指定 sheet
        with pd.ExcelWriter(xl_path, mode='w', engine='openpyxl') as writer:
            book = writer.book
            if sheet_name in book.sheetnames:
                # 读取旧数据并追加
                df_old = pd.read_excel(xl_path, sheet_name=sheet_name)
                df_out = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_out = df_new
            df_out.to_excel(writer, sheet_name=sheet_name, index=False)
    return res


# ---------- 2. 批量评估 ----------
# def batch_evaluate(base_models: List[Tuple[str, Any]],
#                    meta_wrapper: ehr_models.PytorchModelWrapper,
#                    X_test,
#                    y_test,
#                    meta_test: torch.Tensor,
#                    output_path: str = "metrics.xlsx",
#                    sheet: str = "Sheet1"):
#     for name, m in base_models:
#         print(name)
#         if hasattr(m, "predict_proba"):
#             prob = m.predict_proba(X_test)[:, 1]
#         else:
#             prob = m.predict(X_test)
#         name = name.split('_')[0]
#         print(name)
#         evaluate_model(y_test, prob, name, output_path, sheet)

#     stack_prob = meta_wrapper.predict_proba(meta_test)[:, 1]
#     evaluate_model(y_test, stack_prob, "StackingMLP", output_path, sheet)


def eval_base_models():
    """评估基模型的表现
        
    """
    # 可以直接从cache中读取，如果存在的话。
    # cache中的meta_test就是由各个基模型预测得到的概率。
    X_train, X_test, y_train, y_test = ehr_utils.preprocess_ehr_train_test_data(
        'data_processed/benbu_baseline_cleaned_onehot_simulated80.csv')
    base_models = ehr_utils.load_all_trained_pkls(MODEL_DIR)
    meta_train = ehr_utils.generate_meta_features(base_models, X=None, cache_dir=CACHE_DIR, cache_name="meta_train")
    meta_test = ehr_utils.generate_meta_features(base_models, X=None, cache_dir=CACHE_DIR, cache_name="meta_test")
    train_dict = {base_models[i]: (y_train, meta_train[name]) for i, name in enumerate(meta_train.columns)}
    # test_dict = {name: (y_test, meta_test[name]) for name in meta_test.columns}
    eval_utils.plot_multi_roc(train_dict, subset="test", tag="stone_models")
    pass


def eval_stacking_models():
    pass


def main():
    # ---------- 1. 数据字典 ----------
    data_dict = {
        "Internal": "data_processed/benbu_baseline_cleaned_onehot.csv",
        "Shangjin": "data_processed/shangjin_baseline_cleaned_onehot.csv",
        "Tianfu"  : "data_processed/tianfu_baseline_cleaned_onehot.csv",
        "Wuhou"   : "data_processed/wuhou_baseline_cleaned_onehot.csv",
    }
    # ---------- 2. 加载基模型 ----------
    # ---------- 4. 一键全部评估 ----------
    # os.makedirs("result", exist_ok=True)
    # metrics_output_path = "result/all_metrics.xlsx"
    # cache_dir = "cache"
    # for sheet_name, csv_path in data_dict.items():
    #     print(f"\n>>> 正在评估 {sheet_name} ...")
    #     if sheet_name == "Internal":
    #         X_train, X_test, y_train, y_test = ehr_utils.preprocess_ehr_train_test_data(csv_path)
    #         X, y = X_test, y_test
    #     else:
    #         X, y = ehr_utils.load_external(csv_path)

    #     meta_prob = ehr_utils.generate_meta_features(base_models, X, cache_name=f"meta_{sheet_name.lower()}", cache_dir=cache_dir, use_cache=True)
    #     batch_evaluate(base_models=base_models, meta_wrapper=mlp_wrapper, X_test=X, y_test=y, meta_test=meta_prob.cpu().numpy(), output_path=metrics_output_path, sheet=sheet_name)

    # print("\n[INFO] 全部评估完成！文件 ->", metrics_output_path)
    pass


if __name__ == "__main__":
    eval_base_models()
    main()
