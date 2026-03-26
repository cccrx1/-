# BadNets + REFINE_PDB / REFINE_PDB_SSL 实验命令

## 1) 先设置 attacked checkpoint 路径

CKPT=./experiments/pdb_b_badnets_s666/runs/CIFAR10_ResNet18_BadNets_2026-03-25_15-19-31/ckpt_epoch_200.pth

## 2) 路径检查

test -f "$CKPT" && echo OK || echo NOT_FOUND

## 3) refine + pdb（无 ssl）

python run.py single --dataset-root ./datasets --output-root ./experiments/exp_refine_pdb_a_s666 --seed 666 --only-attack badnets --pretrained-attack-model-path $CKPT --defense-variant refine_pdb --refine-epochs 150 --pdb-weight 0.08 --pdb-batch-ratio 0.20 --no-pdb-inference-trigger

python run.py single --dataset-root ./datasets --output-root ./experiments/exp_refine_pdb_b_s666 --seed 666 --only-attack badnets --pretrained-attack-model-path $CKPT --defense-variant refine_pdb --refine-epochs 150 --pdb-weight 0.12 --pdb-batch-ratio 0.25 --no-pdb-inference-trigger

## 4) refine + pdb + ssl

python run.py single --dataset-root ./datasets --output-root ./experiments/exp_refine_pdb_ssl_a_s666 --seed 666 --only-attack badnets --pretrained-attack-model-path $CKPT --defense-variant refine_pdb_ssl --refine-epochs 150 --ssl-weight 0.001 --pdb-weight 0.05 --pdb-batch-ratio 0.10 --no-pdb-inference-trigger

python run.py single --dataset-root ./datasets --output-root ./experiments/exp_refine_pdb_ssl_b_s666 --seed 666 --only-attack badnets --pretrained-attack-model-path $CKPT --defense-variant refine_pdb_ssl --refine-epochs 150 --ssl-weight 0.002 --pdb-weight 0.08 --pdb-batch-ratio 0.20 --no-pdb-inference-trigger

## 5) 推荐执行顺序

1. exp_refine_pdb_ssl_a_s666
2. exp_refine_pdb_a_s666
3. exp_refine_pdb_ssl_b_s666
4. exp_refine_pdb_b_s666

## 6) 结果登记模板

请记录每次日志最后一行：
Stage[REFINE/BadNets]: done. BA(top1)=..., ASR(top1)=...

建议汇总格式：

- exp_refine_pdb_ssl_a_s666: BA=, ASR=
- exp_refine_pdb_a_s666: BA=, ASR=
- exp_refine_pdb_ssl_b_s666: BA=, ASR=
- exp_refine_pdb_b_s666: BA=, ASR=

## 7) 已有实验结果对比（自动汇总）

说明：

- 修改后（主动防御版本）：defense_variant 为 refine_pdb 或 refine_pdb_ssl。
- 修改前（旧逻辑/基线）：defense_variant 为 refine 或 refine_ssl，或无 defense_variant。
- 下表仅保留 badnets 主线的核心可比实验（已剔除 smoke 类测试）。

### 修改前（基线/旧逻辑）

| 实验 | defense_variant | refine_epochs | 关键参数 | Attack BA | Attack ASR | Refine BA | Refine ASR | 结论 |
|---|---|---:|---|---:|---:|---:|---:|---|
| experiments/badnets_refine/runs/metrics_summary.json | refine | 150 | 无 ssl/pdb | 0.9141 | 0.9729 | 0.9010 | 0.1049 | 稳定强基线 |
| experiments/final/seed777/runs/metrics_summary.json | refine_ssl | 30 | ssl=0.004 | 0.8048 | 0.9615 | 0.8203 | 0.1165 | seed777 下表现可接受 |
| experiments/quick_diag/ssl_w0004_e30_v2/runs/metrics_summary.json | refine_ssl | 30 | ssl=0.004 | 0.7960 | 0.9627 | 0.8156 | 0.1095 | 与 final/seed777 一致趋势 |
| experiments/badnets_refine_ssl_120_bs128/runs/metrics_summary.json | refine_ssl | 120 | 未显式记录 | 0.9141 | 0.9729 | 0.1438 | 0.2601 | 明显塌缩 |

### 修改后（refine+pdb / refine+pdb+ssl）

| 实验 | defense_variant | refine_epochs | ssl | pdb_weight | pdb_ratio | inference_trigger | Attack BA | Attack ASR | Refine BA | Refine ASR | 结论 |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| experiments/pdb_a_badnets_s666/runs/metrics_summary.json | refine_pdb | 30 | 0.02 | 0.40 | 0.60 | True | 0.9152 | 0.9731 | 0.0143 | 0.0923 | 参数过强导致塌缩 |
| experiments/pdb_b_badnets_s666/runs/metrics_summary.json | refine_pdb_ssl | 30 | 0.004 | 0.40 | 0.60 | True | 0.9182 | 0.9731 | 0.0180 | 0.0924 | 参数过强导致塌缩 |
| experiments/pdb_a_badnets_s666_refine_only_from_ckpt/metrics_summary.json | refine_pdb | 150 | 0.02 | 0.15 | 0.30 | False | 0.9152 | 0.9731 | 0.8783 | 0.1033 | 修复后最佳之一 |
| experiments/exp_refine_pdb_ssl_a_s666/metrics_summary.json | refine_pdb_ssl | 150 | 0.001 | 0.05 | 0.10 | False | 0.9182 | 0.9731 | 0.8403 | 0.1039 | 稳定且有效 |

### 当前结论（按可用性优先）

1. 最优综合：experiments/pdb_a_badnets_s666_refine_only_from_ckpt/metrics_summary.json（BA=0.8783, ASR=0.1033）。
2. 次优稳健：experiments/exp_refine_pdb_ssl_a_s666/metrics_summary.json（BA=0.8403, ASR=0.1039）。
3. 失败模式已定位：当 pdb_weight=0.4 且 pdb_ratio=0.6，且 inference_trigger=True 时，容易出现 BA 接近 0 的塌缩。

### 判定“修改前/后”的依据

- 修改后：metrics_summary 中 defense_variant 为 refine_pdb 或 refine_pdb_ssl，且存在 pdb_weight / pdb_batch_ratio 字段。
- 修改前：defense_variant 为 refine/refine_ssl（或为空），且无主动防御参数生效。

## 8) 新版平衡参数（降低 ssl/pdb 对 refine 的损害）

本仓库已加入三类稳定器：

- `--pdb-warmup-ratio`：PDB 损失 warmup 比例（前期弱约束，后期再加力）
- `--ssl-warmup-ratio`：SSL 损失 warmup 比例
- `--aux-loss-cap-ratio`：每个辅助损失相对 CE 的上限比例，防止辅助项“压过主任务”

推荐先跑以下两组（150 epoch 公平对比）：

### 8.1 refine + pdb（稳健优先）

python run.py single --dataset-root ./datasets --output-root ./experiments/exp_refine_pdb_v2_balanced_s666 --seed 666 --only-attack badnets --pretrained-attack-model-path $CKPT --defense-variant refine_pdb --refine-epochs 150 --pdb-weight 0.10 --pdb-batch-ratio 0.20 --pdb-warmup-ratio 0.35 --aux-loss-cap-ratio 1.2 --no-pdb-inference-trigger

### 8.2 refine + pdb + ssl（协同优先）

python run.py single --dataset-root ./datasets --output-root ./experiments/exp_refine_pdb_ssl_v2_balanced_s666 --seed 666 --only-attack badnets --pretrained-attack-model-path $CKPT --defense-variant refine_pdb_ssl --refine-epochs 150 --ssl-weight 0.0015 --pdb-weight 0.08 --pdb-batch-ratio 0.15 --ssl-warmup-ratio 0.40 --pdb-warmup-ratio 0.35 --aux-loss-cap-ratio 1.2 --no-pdb-inference-trigger

若目标是进一步降 ASR，可在保持 `aux-loss-cap-ratio=1.2` 不变时，逐步提升 `pdb-weight`（每次 +0.01），并观察 BA 是否仍 >= 0.85。

## 9) 自动汇总脚本（增量追加）

新增脚本：`tools/append_experiment_matrix.py`

用途：

- 自动扫描 `experiments/**/metrics_summary.json`
- 将每次实验的 BA/ASR 与关键参数追加到项目根目录 `experiment_matrix.csv`
- 同步生成便于阅读的表格（项目根目录）`experiment_matrix.md`
- 默认按 `run_path` 去重，只追加新实验
- 默认过滤“数据塌缩”实验：`refine_ba < 0.30` 不写入表格
- 方法缩写标签：`R=refine`、`RS=refine+ssl`、`RB=refine+pdb`、`RSB=refine+ssl+pdb`
- 表格展示规则：按 `Method` 排序；`Attack` 列展示攻击方式；`Run` 放在表格最后一列

执行命令：

python tools/append_experiment_matrix.py

首次或想全量重建时：

python tools/append_experiment_matrix.py --rebuild

如需调整“塌缩”阈值：

python tools/append_experiment_matrix.py --collapse-ba-threshold 0.35

## 10) 改进方法核心理论与公式（可直接用于报告）

### 10.1 目标与符号

- 目标：在保持较高干净精度（BA）的同时，显著降低后门攻击成功率（ASR）。
- 记攻击模型为 $f(\cdot)$，净化网络为 $g_\theta(\cdot)$，输入为 $x$。
- 攻击模型对原图的伪标签为：

$$
	ilde{y}=\arg\max f(x)
$$

### 10.2 核心改进思想（从“强耦合”改为“可控协同”）

1. 主任务优先：以 REFINE 的 clean-alignment 目标作为主目标，避免辅助项主导训练。
2. 渐进式注入：SSL/PDB 权重在前期 warmup，小步注入，后期再达到设定强度。
3. 辅助项上限保护：对每个辅助损失做相对 CE 的封顶，防止 BA 被拖垮。
4. 稳定选优：验证阶段用稳定主损失做 best checkpoint 选择，降低随机项扰动。

### 10.3 基础 REFINE 目标

令 $p_\theta(x)=\mathrm{softmax}(f(g_\theta(x)))$，$\hat{y}(x)$ 为 one-hot 伪标签。

$$
\mathcal{L}_{\mathrm{base}}=\mathcal{L}_{\mathrm{CE/BCE}}\big(p_\theta(x),\hat{y}(x)\big)
$$

原始实现中还包含监督对比项（SupCon），写作：

$$
\mathcal{L}_{\mathrm{refine}}=\mathcal{L}_{\mathrm{base}}+\lambda_{\mathrm{sup}}\,\mathcal{L}_{\mathrm{supcon}}
$$

### 10.4 PDB 主动防御项

- 设防御触发器变换为 $T(\cdot)$，目标类别平移映射为 $h(y)=y+s$（模类别数）。
- 在 batch 子集 $S$ 上定义主动约束：

$$
\mathcal{L}_{\mathrm{pdb}}=\mathrm{CE}\Big(f\big(T(g_\theta(x_S))\big),\,h\big(\tilde{y}_S\big)\Big)
$$

含义：净化后图像若再加“防御触发器”，模型应输出可控平移标签，从而削弱真实后门触发路径。

### 10.5 SSL 项

对同一样本两种增强视图 $v_1,v_2$ 经净化后表征做对比学习：

$$
\mathcal{L}_{\mathrm{ssl}}=\mathrm{NT\text{-}Xent}(z_1,z_2;\tau)
$$

其中 $\tau$ 为温度参数，$z_i$ 为归一化表示。

### 10.6 Warmup 与 Loss Cap（防塌缩关键）

对任一辅助分支 $u\in\{\mathrm{ssl},\mathrm{pdb}\}$：

$$
w_u(t)=w_u^{\star}\cdot\min\left(1,\frac{t+1}{\max(1,\lfloor r_uE\rfloor)}\right)
$$

- $w_u^{\star}$：目标权重；$r_u$：warmup 比例；$E$：总 epoch。

辅助项封顶（相对 CE）：

$$
\overline{\mathcal{L}}_u=\min\left(\mathcal{L}_u,\,\rho\,\mathcal{L}_{\mathrm{CE}}\right)
$$

- $\rho$ 即 `aux_loss_cap_ratio`。

### 10.7 最终训练目标

RB（refine+pdb）：

$$
\mathcal{L}_{\mathrm{RB}}=\mathcal{L}_{\mathrm{CE}}+\lambda_{\mathrm{sup}}\mathcal{L}_{\mathrm{supcon}}+w_{\mathrm{pdb}}(t)\,\overline{\mathcal{L}}_{\mathrm{pdb}}
$$

RSB（refine+ssl+pdb）：

$$
\mathcal{L}_{\mathrm{RSB}}=\mathcal{L}_{\mathrm{CE}}+w_{\mathrm{ssl}}(t)\,\overline{\mathcal{L}}_{\mathrm{ssl}}+w_{\mathrm{pdb}}(t)\,\overline{\mathcal{L}}_{\mathrm{pdb}}
$$

### 10.8 推理侧策略

- 默认建议 `--no-pdb-inference-trigger`（推理不加防御触发器），优先保 BA 稳定。
- 若启用推理触发器，需要做类别反平移补偿（$h^{-1}$）以避免系统性偏移。

### 10.9 参数与现象对应关系（经验）

- `pdb_weight`、`pdb_batch_ratio` 同时过大，且推理触发器开启时，最易 BA 塌缩。
- `ssl_weight` 建议小步搜索（如 0.001~0.005），避免与主任务冲突。
- `pdb_warmup_ratio`、`ssl_warmup_ratio` 增大通常更稳，但可能减慢 ASR 下降速度。
- `aux_loss_cap_ratio` 偏小更稳 BA，偏大更激进；建议先从 1.2~1.5 起步。
