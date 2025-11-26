import ehr_utils, ehr_models
import os
import pickle
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import compute_sample_weight
from imblearn.over_sampling import SMOTE
import datetime

logger = ehr_utils.get_logger('train_10_basemodels')


def main(n_splits=2):
    """
    先划分20%的测试集，剩下的80%做交叉验证训练
    交叉验证训练的模型保存到trained_models_时间戳文件夹下
    """

    logger.info("\n=== 步骤1：获取训练集和测试集 ===")
    X_train, X_test, y_train, y_test = ehr_utils.preprocess_ehr_train_test_data('data_processed/benbu_baseline_cleaned_onehot_simulated80.csv')
    logger.info(f"原始训练集大小：{X_train.shape}")
    logger.info(f"原始测试集大小：{X_test.shape}")
    logger.info("\n=== 步骤2：准备所有模型 ===")
    # 定义所有要训练的模型
    all_models = [
        'AdaBoost', 'DecisionTree', 'GaussianNB', 'GradientBoosting', 'LightGBM', 'LinearDiscriminantAnalysis', 'LogisticRegression', 'MLPClassifier', 'RandomForest', 'XGBoost'
    ]
    logger.info(f"将训练以下 {len(all_models)} 个模型：")
    for i, model_name in enumerate(all_models, 1):
        logger.info(f"  {i}. {model_name}")

    # 存储所有模型的结果
    all_models_results = {}
    total_start_time = time.time()
    logger.info("=" * 80)
    logger.info("\n开始训练所有模型...")
    logger.info("-" * 80)

    # 创建保存模型的文件夹
    model_save_dir = f'trained_models_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}'
    os.makedirs(model_save_dir, exist_ok=True)

    for model_idx, model_name in enumerate(all_models, 1):
        if model_idx < 7:
            continue
        logger.info(f"{'='*60}")
        logger.info(f"正在训练模型 {model_idx}/{len(all_models)}：{model_name}")
        logger.info(f"{'='*60}")
        model_start_time = time.time()

        try:
            # 获取模型
            model = ehr_models.get_model(model_name)
            logger.info(f"模型参数：{model.get_params()}")
            logger.info(f"\n=== 开始 {model_name} {n_splits}折交叉验证 ===")
            # 交叉验证设置
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_results = {'train_aucs': [], 'val_aucs': [], 'models': [], 'fold_results': []}

            # K折交叉验证
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                logger.info(f'----- Fold {fold + 1}/{n_splits} -----')
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx].values
                y_fold_val = y_train.iloc[val_idx].values

                # 计算样本权重
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_fold_train)
                pos_mask = (y_fold_train == 1)
                sample_weights[pos_mask] *= 1.1

                # 每次都重新获取模型
                fold_model = ehr_models.get_model(model_name)

                if model_name == 'LinearDiscriminantAnalysis':
                    # LDA 不支持样本权重，使用 SMOTE
                    smote = SMOTE(random_state=42)
                    X_train_fold_res, y_train_fold_res = smote.fit_resample(X_fold_train, y_fold_train)  # type: ignore
                    fold_model.fit(X_train_fold_res, y_train_fold_res)  # type: ignore
                else:
                    fold_model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)  # type: ignore

                # 预测
                y_train_proba = np.asarray(fold_model.predict_proba(X_fold_train))[:, 1]
                y_val_proba = np.asarray(fold_model.predict_proba(X_fold_val))[:, 1]

                # 计算 AUC
                train_auc = roc_auc_score(y_fold_train, y_train_proba)
                val_auc = roc_auc_score(y_fold_val, y_val_proba)
                print(f"训练集 AUC：{train_auc:.4f}，验证集 AUC：{val_auc:.4f}，过拟合：{train_auc - val_auc:.4f}")
                # 保存结果
                cv_results['train_aucs'].append(train_auc)
                cv_results['val_aucs'].append(val_auc)
                cv_results['models'].append(fold_model)
                cv_results['fold_results'].append({'fold': fold, 'train_auc': train_auc, 'val_auc': val_auc, 'train_idx': train_idx, 'val_idx': val_idx})

            # 计算平均结果
            mean_train_auc = np.mean(cv_results['train_aucs'])
            mean_val_auc = np.mean(cv_results['val_aucs'])
            std_train_auc = np.std(cv_results['train_aucs'])
            std_val_auc = np.std(cv_results['val_aucs'])

            logger.info(f"\n{model_name} {n_splits}折交叉验证结果汇总：")
            logger.info(f"各折验证集 AUC: {[f'{auc:.4f}' for auc in cv_results['val_aucs']]}")
            logger.info(f"平均训练集 AUC: {mean_train_auc:.4f} ± {std_train_auc:.4f}")
            logger.info(f"平均验证集 AUC: {mean_val_auc:.4f} ± {std_val_auc:.4f}")
            logger.info(f"平均过拟合程度：{mean_train_auc - mean_val_auc:.4f}")
            # 选择最佳模型（验证集 AUC 最高的）
            best_fold_idx = np.argmax(cv_results['val_aucs'])
            best_model = cv_results['models'][best_fold_idx]
            best_val_auc = cv_results['val_aucs'][best_fold_idx]
            logger.info(f"最佳模型来自第 {best_fold_idx+1} 折，验证集 AUC：{best_val_auc:.4f}")

            # 保存最佳模型
            model_filename = f'{model_save_dir}/{model_name}_cv{n_splits}_{best_val_auc:.4f}.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump(best_model, f)
            logger.info(f"最佳模型已保存到：{model_filename}")

            # 计算训练时间
            model_end_time = time.time()
            model_duration = model_end_time - model_start_time

            # 暂存模型结果
            all_models_results[model_name] = {
                'mean_train_auc': mean_train_auc,
                'mean_val_auc': mean_val_auc,
                'std_train_auc': std_train_auc,
                'std_val_auc': std_val_auc,
                'best_val_auc': best_val_auc,
                'overfitting': mean_train_auc - mean_val_auc,
                'training_time': model_duration,
                'best_model': best_model,
                'cv_results': cv_results
            }
            logger.info(f"✓ {model_name} 训练完成！用时：{model_duration:.2f} 秒")

        except Exception as e:
            logger.error(f"✗ {model_name} 训练失败：{str(e)}")
            all_models_results[model_name] = {'error': str(e), 'training_time': time.time() - model_start_time}
    #保存模型结果到文件
    results_filename = f'{model_save_dir}/all_models_eval_results.csv'
    ehr_utils.append_metrics_to_csv(all_models_results, results_filename)

    # 计算总用时
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    logger.info("=" * 80)
    logger.info(f"\n所有模型训练完成！耗时{total_duration:.2f} 秒")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
