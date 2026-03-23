# CIFAR-10 Attack + REFINE Suite

统一的 CIFAR-10 攻防实验工程，包含：

1. 后门攻击训练（BadNets / Blended / Label-Consistent）
2. 防御训练与评估（REFINE / REFINE_CG / REFINE_SSL）
3. 结构化结果输出（JSON + Markdown）
4. 单次、单 case、批量 suite 三种运行模式

## 环境安装

建议 Python 3.10-3.13（推荐 3.11 或 3.12）。

```bash
pip install -r requirements.txt
```

若使用自定义 CUDA 轮子，可先安装匹配的 torch/torchvision，再安装其余依赖。

## 统一运行入口

所有运行模式统一通过 run.py。

### single: 单次直接运行

```bash
python run.py single \
  --only-attack badnets \
  --defense-variant refine_ssl \
  --refine-epochs 50 \
  --ssl-weight 0.02 \
  --ssl-temperature 0.07 \
  --output-root ./experiments/test/single_demo/runs \
  --adv-dataset-root ./experiments/test/single_demo/adv_dataset
```

### case: 运行矩阵中的单个实验

```bash
python run.py case --case badnets_refine_ssl_50
```

### suite: 批量运行多个 case

```bash
python run.py suite --cases all
```

### smoke: 1-epoch 快速健康检查

```bash
python run.py smoke --dry-run
python run.py smoke
```

## 1. 攻击与防御源码及原理概览

### 攻击方法

- BadNets
  - 源码: core/attacks/BadNets.py
  - 原理: 在部分训练样本叠加固定触发器，并将其标签篡改为目标类别，使模型学习触发器到目标类的后门映射。

- Blended
  - 源码: core/attacks/Blended.py
  - 原理: 将触发图案按一定权重混合到输入中（而不是纯贴片覆盖），再进行目标标签投毒，提升触发器隐蔽性。

- Label-Consistent
  - 源码: core/attacks/LabelConsistent.py
  - 原理: 通过对目标类样本做对抗扰动并叠加触发器，构造“标签一致”的投毒样本，使投毒样本在语义上更接近原标签，攻击更隐蔽。

### 防御方法

- REFINE
  - 源码: core/defenses/REFINE.py
  - 原理: 训练输入重编程网络（UNet）对输入进行变换，再送入已中毒模型；通过伪标签一致性（BCELoss）和对比约束抑制触发器效应。

- REFINE_CG
  - 源码: core/defenses/REFINE_CG.py
  - 原理: 在 REFINE 输出与原始模型输出之间加入置信度门控（Confidence Gate），按样本可疑度自适应融合，减少过修复。

- REFINE_SSL
  - 源码: core/defenses/REFINE_SSL.py
  - 原理: 在 REFINE 框架中引入自监督对比学习（SimCLR 风格）项，使用总损失 = 伪标签一致性损失 + ssl_weight * 自监督损失。

## 2. 参数说明（共有参数与方法特有参数）

参数定义集中在 runner/suite_config.py（RuntimeConfig + CLI），入口聚合在 run.py。

### 2.1 攻击与防御共有参数

- 数据与路径
  - --dataset-root
  - --output-root
  - --adv-dataset-root

- 复现实验与设备
  - --seed
  - --deterministic
  - --device-mode (GPU/CPU)
  - --cuda-selected-devices
  - --batch-size
  - --num-workers

- 训练轮次
  - --attack-epochs
  - --lc-epochs
  - --refine-epochs

- 通用任务控制
  - --only-attack (all/badnets/blended/label_consistent)
  - --force-rebuild
  - --attack-cache-root

### 2.2 攻击特有参数

- 目标标签
  - --y-target

- 投毒比例
  - --badnets-rate
  - --blended-rate
  - --lc-rate

- Label-Consistent 对抗样本参数
  - --lc-eps
  - --lc-alpha
  - --lc-steps

- 跳过 LC 分支
  - --skip-lc

### 2.3 防御特有参数

- 防御变体选择
  - --defense-variant (refine/refine_cg/refine_ssl)

- REFINE/UNet 容量参数
  - --refine-first-channels

- REFINE_CG 参数
  - --cg-threshold
  - --cg-temperature
  - --cg-strength

- REFINE_SSL 参数
  - --ssl-temperature
  - --ssl-weight

### 2.4 内部默认训练超参（来自流水线）

- 攻击训练默认优化设置（SGD）: lr=0.1, momentum=0.9, weight_decay=5e-4
- 防御训练默认优化设置（Adam）: lr=0.01, betas=(0.9, 0.999), eps=1e-8
- 学习率里程碑: 攻击 [150, 180]，防御 [100, 130]

## 3. 项目结构与模块功能

```text
.
├── run.py                     # 统一入口：single/case/suite/smoke
├── core/
│   ├── attacks/               # 攻击算法实现（BadNets/Blended/LabelConsistent）
│   ├── defenses/              # 防御算法实现（REFINE/REFINE_CG/REFINE_SSL）
│   ├── models/                # 模型定义（ResNet、UNet）
│   └── utils/                 # 日志、损失、PGD、辅助工具
├── runner/
│   ├── suite_pipeline.py      # 单次完整流水线编排与指标写出
│   ├── suite_config.py        # CLI 参数解析与 RuntimeConfig
│   └── pipeline_state.py      # 锁、阶段状态、日志与断点续跑状态管理
├── test/
│   ├── test_matrix.json       # 实验矩阵配置
│   ├── run_case.py            # 单 case 调度
│   ├── run_test_suite.py      # 多 case 批量调度
│   ├── matrix_utils.py        # 矩阵解析/参数映射/指标收集
│   └── wrapper_utils.py       # 公共封装工具
└── requirements.txt           # 依赖锁定文件
```

模块职责边界建议：

- core 仅放算法与模型，不放实验编排。
- runner 放单次流水线与运行时状态管理。
- test 放矩阵用例与批量实验调度逻辑。

## 4. 依赖清单

项目依赖定义在 requirements.txt。

### 核心依赖

- numpy==1.26.4
- Pillow==10.1.0
- opencv-python==4.8.1.78
- tqdm==4.66.1

### 深度学习依赖（按 Python 版本）

- Python 3.10/3.11
  - torch==2.2.0
  - torchvision==0.17.0

- Python 3.12/3.13
  - torch==2.5.1
  - torchvision==0.20.1

## 5. 统计指标与输出内容

指标生成逻辑位于 runner/suite_pipeline.py，结果写入 output_root。

### 5.1 攻击阶段指标

- clean.top1_acc: 干净测试集 Top-1 准确率（常称 BA）
- clean.top5_acc: 干净测试集 Top-5 准确率
- clean.mean_loss: 干净测试集平均损失
- poisoned.top1_acc: 投毒测试集 Top-1 准确率（常称 ASR，越高表示攻击越成功）
- poisoned.top5_acc: 投毒测试集 Top-5 准确率
- poisoned.mean_loss: 投毒测试集平均损失

### 5.2 防御阶段指标

- clean_after_refine.top1_acc: 防御后干净集 Top-1
- clean_after_refine.top5_acc: 防御后干净集 Top-5
- poisoned_after_refine.top1_acc: 防御后投毒集 Top-1（防御后期望下降）
- poisoned_after_refine.top5_acc: 防御后投毒集 Top-5

### 5.3 汇总报表指标

- Top1 Delta (Poisoned-Clean): 投毒 Top-1 与干净 Top-1 的差值
- Stage Timing: 每个阶段耗时（秒 + HH:MM:SS）
- Total elapsed: 全流水线总耗时

### 5.4 输出文件

- 单次流水线输出
  - <output_root>/metrics_summary.json
  - <output_root>/comparison_report.md

- 批量 suite 输出
  - ./experiments/test/summary/suite_summary_*.json
  - ./experiments/test/summary/suite_summary_*.md
