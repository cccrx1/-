# REFINE_ADAPTIVE 快速测试指南

## 最简单的运行方式

### 单条命令测试（推荐）

```bash
python run.py --defense-variant refine_adaptive --only-attack badnets --refine-epochs 10 --seed 666
```

这条命令会：
- 使用 REFINE_ADAPTIVE 方法
- 只运行 BadNets 攻击
- 训练 10 个 epoch（快速测试）
- 使用固定随机种子 666

### 如果上面的命令报错

#### 错误1: 找不到 run.py
```bash
# 确保在项目根目录
cd C:\Users\17672\Documents\Project\-
python run.py --help
```

#### 错误2: 导入错误
```bash
# 测试导入
python -c "import core; print(core.REFINE_ADAPTIVE)"
```

如果报错，检查：
- `core/defenses/REFINE_ADAPTIVE.py` 是否存在
- `core/defenses/__init__.py` 是否已更新
- `core/__init__.py` 是否已更新

#### 错误3: 参数错误
```bash
# 查看所有可用参数
python run.py --help | grep adaptive
```

应该能看到：
- `--defense-variant` 包含 `refine_adaptive`
- `--adaptive-mode`
- `--adaptive-initial-threshold`
- `--adaptive-final-threshold`
- `--adaptive-warmup-ratio`

## 完整测试流程

### 步骤1: 验证代码导入
```bash
python -c "from core.defenses import REFINE_ADAPTIVE; print('导入成功')"
```

### 步骤2: 运行最小实验（约5-10分钟）
```bash
python run.py \
    --output-root ./experiments/test_adaptive \
    --defense-variant refine_adaptive \
    --only-attack badnets \
    --refine-epochs 5 \
    --seed 666
```

### 步骤3: 查看结果
```bash
cat ./experiments/test_adaptive/*/metrics_summary.json
```

## 对比实验（完整版）

### 实验1: Baseline
```bash
python run.py \
    --output-root ./experiments/compare/baseline \
    --defense-variant refine \
    --only-attack badnets \
    --refine-epochs 50 \
    --seed 666
```

### 实验2: Adaptive
```bash
python run.py \
    --output-root ./experiments/compare/adaptive \
    --defense-variant refine_adaptive \
    --adaptive-mode progressive \
    --only-attack badnets \
    --refine-epochs 50 \
    --seed 666
```

## 常见问题

### Q: 脚本无法执行
A: 直接用上面的单条命令，不要用 .bat 或 .sh 脚本

### Q: 提示找不到预训练模型
A: 去掉 `--pretrained-attack-model-path` 参数，让它自动训练

### Q: 运行时间太长
A: 减少 `--refine-epochs` 到 10 或 5

### Q: 内存不足
A: 添加 `--batch-size 64 --num-workers 4`
