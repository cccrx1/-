# CIFAR-10 Attack + REFINE Suite

统一的 CIFAR-10 攻防实验工程，包含：

1. 干净模型训练（ResNet-18）
2. 三类后门攻击（BadNets / Blended / Label-Consistent）
3. 三类防御变体（REFINE / REFINE_CG / REFINE_SSL）
4. 结构化指标输出（json + markdown）

## 项目结构

- 根目录只保留：`run.py`、`README.md`、`requirements-lock.txt`
- `core/`: 核心算法与模型实现（攻击、防御、模型、工具函数）
- `runner/`: 运行时公共模块与流水线主实现
- `test/`: 实验编排脚本与矩阵配置（case/suite）

推荐按以下职责边界维护：

- 只在 `core/` 放算法与模型逻辑
- 只在 `test/` 放实验编排、case 选择、矩阵配置
- 顶层脚本只做入口聚合与兼容转发

其中：

- `runner/suite_config.py` 统一维护 suite 的 CLI 参数与 `RuntimeConfig`
- `runner/pipeline_state.py` 统一维护 `StageStatusManager`、`StageLogger`、`PipelineRunLock`
- `runner/suite_pipeline.py` 负责单次完整流水线编排（由 `run.py single` 调用）

## 环境安装

建议使用 Python 3.10-3.13（推荐 3.11 或 3.12）。

推荐使用锁定版本安装：

```bash
pip install -r requirements-lock.txt
```

若你使用自定义 CUDA/MUSA 轮子，可先安装对应 `torch/torchvision`，再补装其余依赖。

## 统一运行入口

所有运行模式都走 `run.py`：

### 1) 单次运行（single）

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

### 2) 运行单个矩阵 case（case）

```bash
python run.py case --case badnets_refine_ssl_50
```

可覆盖矩阵参数：

```bash
python run.py case --case badnets_refine_ssl_50 --ssl-weight 0.005 --force-rebuild
```

### 3) 批量运行矩阵（suite）

```bash
python run.py suite --cases all
```

只跑指定 case：

```bash
python run.py suite --cases badnets_refine,blended_refine_ssl_50
```

## 参数组织

`run.py` 统一暴露以下参数：

- 通用参数：数据路径、设备、batch、worker、epoch、攻击比例
- 防御特有参数：
  - REFINE_CG: `--cg-threshold`, `--cg-temperature`, `--cg-strength`
  - REFINE_SSL: `--ssl-temperature`, `--ssl-weight`
- 运行控制参数：`--skip-lc`, `--force-rebuild`, `--attack-cache-root`

## 可扩展性建议

新增一个实验 case 时，建议只改一个文件：

1. 在 `test/test_matrix.json` 增加新 case
2. 使用 `python run.py case --case <new_case>` 验证
3. 需要批量跑时，直接用 `python run.py suite --cases all`

这样可以避免为每个 case 新增重复脚本。

## 结果输出

- 单次流水线：`<output_root>/metrics_summary.json`
- 批量汇总：`./experiments/test/summary/suite_summary_*.json` 与 `.md`
