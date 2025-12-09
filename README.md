# CalcuAlert：结石预警模型项目

## 项目简介

CalcuAlert 是一个基于电子健康记录（EHR）的风险预警模型项目，提供**10 种基础模型训练、Stacking 集成学习及特征重要性分析**功能，支持模型训练日志记录、模型参数保存与评估指标可视化，适用于医疗数据风险预测场景（如疾病风险预警、诊疗效果预测等）。

## 核心功能

- 基础模型训练：支持 10 种经典机器学习模型的批量训练与调优
- 集成学习：提供 Stacking 集成策略，提升模型预测稳定性与精度
- 特征分析：内置特征重要性分析模块，可视化关键影响因子
- 全流程支持：自动记录训练日志、保存训练模型与评估指标，便于复现与迭代
- 数据适配：支持原始医疗数据预处理，含模拟数据生成示例（fake_data.ipynb）

## 环境配置

### 依赖环境

- Python 3.12
- 依赖库：详见 `requirements.txt`（含机器学习、数据处理、可视化相关库）

### 安装步骤

1. 创建并激活 conda 环境：
    ```bash
    conda create -n calculus python=3.12 -y
    conda activate calculus
    ```
    
2. 安装依赖包：
    ```bash
    pip install -r requirements.txt
    ```

## 目录结构说明

plaintext

```plaintext
CalcuAlert/
├── data_original/        # 原始医疗数据存储目录（需自行放入数据）
├── data_processed/       # 预处理后的数据存储目录（自动生成）
├── .gitignore            # Git忽略文件配置
├── README.md             # 项目说明文档（本文档）
├── ehr_models.py         # 医疗数据模型定义（基础模型+集成模型）
├── ehr_utils.py          # 数据预处理、日志记录等工具函数
├── eval_utils.py         # 模型评估工具（指标计算、结果可视化）
├── evaluate_models.py    # 模型评估主脚本
├── fake_data.ipynb       # 模拟医疗数据生成示例（Jupyter Notebook）
├── fea_importrance_analyze.ipynb  # 特征重要性分析脚本（可视化输出）
├── requirements.txt      # 项目依赖清单
├── timestamp.txt         # 训练时间戳记录（自动生成）
├── train_10_basemodels.py  # 10种基础模型批量训练脚本
├── train_stacking_model.py  # Stacking集成模型训练脚本
```

## 快速开始

### 1. 数据准备

- 将原始医疗数据放入 `data_original/` 目录（支持 CSV、Excel 等格式，需在 `ehr_utils.py` 中适配数据读取逻辑）
- 若无需真实数据测试，可运行 `fake_data.ipynb` 生成模拟医疗数据

### 2. 模型训练

#### 训练基础模型
```bash
python train_10_basemodels.py
```
- 训练结果（模型文件、日志、评估指标）自动保存至对应目录
- 支持通过 `ehr_utils.py` 调整训练参数（如迭代次数、学习率等）

#### 训练 Stacking 集成模型
```bash
python train_stacking_model.py
```
- 基于基础模型的预测结果进行集成训练，需先完成基础模型训练
### 3. 模型评估
```bash
python evaluate_models.py
```

- 输出准确率、召回率、AUC 等核心指标，支持混淆矩阵、ROC 曲线可视化

### 4. 特征重要性分析

打开 `fea_importrance_analyze.ipynb`，运行全部单元格：

- 生成特征重要性排序图、相关性热力图
- 支持导出分析报告（PNG/PDF 格式）

## 注意事项

1. 数据格式：确保 `data_original/` 中的数据字段与 `ehr_utils.py` 中的预处理逻辑一致，建议先运行 `fake_data.ipynb` 参考模拟数据格式
2. 环境兼容：仅支持 Python 3.12，其他版本可能存在依赖冲突
3. 模型保存：训练后的模型文件默认存储在 `trained_models/` 目录，可通过修改脚本配置自定义路径
4. 日志查看：训练日志实时写入 `logs`目录，可用于追溯训练过程与参数
