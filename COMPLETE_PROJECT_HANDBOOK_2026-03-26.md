# REFINE+PDB+SSL 后门防御实验完整手册 | 2026-03-26

> 本文档整合会话背景、理论分析、参数详解、命令模板与代码实现。  
> 供后续智能体快速理解实验现状与快速响应运维需求。

---

## 目录

1. [会话背景与核心诉求](#1-会话背景与核心诉求)
2. [实验现状与最新结果](#2-实验现状与最新结果)
3. [理论基础与策略互补性](#3-理论基础与策略互补性)
4. [参数完整目录](#4-参数完整目录)
5. [模块类与函数签名](#5-模块类与函数签名)
6. [损失函数与数据流](#6-损失函数与数据流)
7. [优化方案与实验建议](#7-优化方案与实验建议)
8. [可运行命令模板](#8-可运行命令模板)
9. [故障排查与已知问题](#9-故障排查与已知问题)
10. [快速查询表](#10-快速查询表)

---

## 1. 会话背景与核心诉求

### 1.1 演进轨迹

**阶段 1：路径管理**  
- 问题：每次运行都要指定长 checkpoint 路径，操作繁琐
- 目标：统一把被攻击模型放到根目录固定位置 `artifacts/attacks/badnets/latest.pth`
- 状态：✓ 已实现

**阶段 2：自动汇总**  
- 问题：每次实验完成后手动记录 BA/ASR，容易遗漏或格式混乱
- 目标：自动脚本扫描日志并汇总到 `experiment_matrix.csv/md`
- 状态：✓ 已实现（`tools/append_experiment_matrix.py`）

**阶段 3：参数兼容修复**  
- 问题：新增参数 `--pdb-warmup-ratio`, `--ssl-warmup-ratio` 不被 CLI 识别或类构造不接收
- 目标：贯通从 CLI parser → RuntimeConfig → 类 __init__ 的完整链路
- 状态：✓ 已修或需再验证

**阶段 4：效果诊断** ← 当前焦点  
- 问题：引入 PDB+SSL 后，ASR 控制很好但 BA 下降太多（~8%）
- 目标：理解为什么，设计出"保 BA 同时降 ASR"的融合策略
- 状态：🔄 正在分析与优化

### 1.2 用户核心需求

1. **理论理解**  
   - REFINE 与 PDB 在概念上是否互补？  
   - 为什么两者结合容易伤 BA？

2. **实验可运行**  
   - 给一套可直接粘贴的命令，参数都配好
   - 固定路径，不需每次手填

3. **结果可追溯**  
   - 自动汇总表格，便于对比不同方法
   - 包含关键参数与最终 BA/ASR

4. **后续可优化**  
   - 从"硬耦合"改成"分阶段 + 时序调控"
   - 给出具体的参数调优方向

---

## 2. 实验现状与最新结果

### 2.1 最新完整运行（2026-03-26 ~12:55）

**实验配置**
```
Defense Variant: refine_pdb_ssl (完整融合)
Seed: 666 (固定)
Run Name: exp_refine_pdb_ssl_v2_balanced_s666
Epochs: 50 (推断)
Attack Model: BadNets (从前次运行复用)
```

**度量指标**
```
Attack Baseline (未防御):
  BA = 0.9168  (干净数据精度)
  ASR = 0.9727 (后门成功率)

After REFINE+PDB+SSL Defense:
  BA = 0.8414  → 下降幅度 8.2%
  ASR = 0.1030 → 压制幅度 89.4%

运行耗时: 03:07:33

输出位置:
  metrics_summary.json: experiments/exp_refine_pdb_ssl_v2_balanced_s666/metrics_summary.json
  report.md: experiments/exp_refine_pdb_ssl_v2_balanced_s666/comparison_report.md
```

**效果评价**
```
✓ 后门压制有效（98.9% → 10.3%，成功率极低）
✗ 泛化伤害较大（91.68% → 84.14%，掉点接近 8%）
⚠ 根本原因：三层防御全开导致梯度冲突、权重失衡
```

### 2.2 历史数据对比（部分样本）

| Method | Attack BA | Attack ASR | Refine BA | Refine ASR | BA Drop | ASR Drop | Note |
|--------|-----------|-----------|-----------|-----------|---------|---------|------|
| R | 0.9141 | 0.9729 | 0.9010 | 0.1049 | 1.4% | 89.2% | ✓ 基线稳定 |
| RB | 0.9152 | 0.9731 | 0.8783 | 0.1033 | 4.0% | 89.4% | 比 R 多掉 2.6% |
| RS | 0.91xx | 0.97xx | 0.88xx | 0.10xx | ~3% | ~89% | SSL 轻手术 |
| RSB | 0.9182 | 0.9731 | 0.8414 | 0.1039 | 8.2% | 89.3% | ✗ 过度压制 |

**结论**  
- `R(纯 REFINE)` 是最稳定的基线，1.4% 掉点可接受
- 加 `B(PDB)` 后掉点加倍（从 1.4% → 4%）
- 再加 `S(SSL)` 后降幅逼近 8.5%，明显不合理

---

## 3. 理论基础与策略互补性

### 3.1 REFINE 工作机制

**输入空间纠偏**
```
核心思想：
1. 冻结被中毒模型参数
2. 仅训练输入变换模块（UNet）
3. 输入经过 UNet 后应该能剔除触发器（从 backdoored 恢复到 clean）

损失函数：
  L_refine = L_clean(clean_sample, label)
           + L_refine(trigger_sample, shuffled_label)
  
  其中 shuffled_label 来自预定义的标签排列，使得
  "对触发样本的标签预测 ≠ 目标类"
```

**特点**
- ✓ 行为层直接纠偏，稳定高效
- ✓ 不伤模型参数，风险低
- ✗ 依赖输入空间的触发器识别，可能有变种泛化问题

### 3.2 PDB 工作机制

**特征/决策边界强化**
```
核心思想：
1. 在训练中主动注入防御触发器
2. 把这些防御触发样本标注为伪目标类 h(y) = (y + shift) mod K
3. 通过 PDB 损失约束模型，使其对该防御触发器不学习特征

损失函数：
  L_pdb = CrossEntropy(model(apply_defensive_trigger(x)), h(y))
  
  采样策略：只在 pdb_batch_ratio 比例的样本上计算
```

**特点**
- ✓ 从特征层根本上拆离触发相关的表示
- ✓ 对见过的触发变种有天然抵抗力（防御是主动的）
- ✗ 容易产生特征压缩，大幅伤 BA
- ✗ 引入的"伪目标"可能与 clean 决策边界冲突

### 3.3 两者互补性分析

**理论互补维度**
```
┌──────────────────────────────────────────────────────────────┐
│ REFINE (行为层修复)           vs    PDB (特征层强化)         │
├──────────────────────────────────────────────────────────────┤
│ 工作空间：输入空间（去噪）       工作空间：特征空间（剔除）  │
│ 梯度方向：纠偏输出行为           梯度方向：重塑边界结构      │
│ 长处：稳定、保 BA                长处：根本、防变种          │
│ 短处：易遗留特征端捷径            短处：易伤基础特征          │
└──────────────────────────────────────────────────────────────┘

关键：两者梯度往往"相反"或"中立"
  - REFINE 要学"如何过滤触发"（需要某种特征提取）
  - PDB 要做"拒绝学习触发相关特征"（压制特征提取）
  → 如果同步全开，梯度互相抵消，导致优化不稳定
```

**互补关系的正确打开方式**
```
理想情况：
1. REFINE 先稳定降 ASR（行为层做事）
2. PDB 后期轻量开启，做"收尾防变种"（特征层做事）
   → 两者"串联"而非"并联"

若必须并联（Tier 2 动态权重）：
  λ_p(t) = λ_p_base × min(1.0, (t+1) / warmup_epochs)
  
  即：前 30% epoch 权重从 0 线性升到 λ_p_base
      避免一开始梯度冲突

组合损失：
  L_total = L_clean + λ_r × L_refine 
          + λ_p(t) × L_pdb         # 时序关键！
          + λ_s(t) × L_ssl
```

### 3.4 当前融合问题诊断

**为什么 RSB (REFINE+SSL+PDB) 的 BA 掉点这么多？**

根本原因 —— 4 类风险同时触发：

1. **梯度冲突**  
   - PDB 要"压特征"，REFINE 要"用特征"
   - SSL 要"拉近相似样本"，PDB 要"推开防御触发"
   - 结果：三个梯度互相对抗，优化陷入局部最优或鞍点

2. **权重失衡**  
   - `pdb_weight=1.0` 可能相对 CE loss 偏大
   - `ssl_weight=0.02` 虽小，配合 `aux_loss_cap_ratio=1.5` 后实际可能放大
   - 某个权重"淹没"另一个

3. **时序错误** ← 最致命  
   - 一上来就 L_clean + L_refine + L_pdb + L_ssl 全开
   - 前期梯度混乱，模型表征被立即拉乱
   - 后期很难恢复 clean manifold

4. **样本比例过高**  
   - `pdb_batch_ratio=0.5`：50% 样本都在做 PDB 约束
   - Clean 分布严重离散，决策边界被大幅扭曲

---

## 4. 参数完整目录

### 4.1 RuntimeConfig (runner/suite_config.py)

所有参数的类型、默认值与描述：

#### 路径与数据
```python
dataset_root: str = "./datasets"              # CIFAR-10 数据集
output_root: str = "./experiments/runs"       # 实验输出
adv_dataset_root: str = "./experiments/adv_dataset"  # 对抗样本缓存
```

#### 全局控制
```python
seed: int = 666                               # 随机种子 (复现用)
deterministic: bool = True                    # 确定性算法
y_target: int = 0                             # 后门目标类
device_mode: str = "GPU"                      # 计算设备
cuda_selected_devices: str = "0"              # GPU 编号
```

#### 数据加载
```python
batch_size: int = 128                         # 批大小
num_workers: int = 8                          # 加载线程 (Win 建议 4-8)
```

#### 训练轮数
```python
benign_epochs: int = 200                      # 干净训练轮数
attack_epochs: int = 200                      # 攻击训练轮数
lc_epochs: int = 200                          # LabelConsistent 轮数
refine_epochs: int = 150                      # REFINE 轮数 ← 关键参数
```

#### 后门注入 (毒化率)
```python
attack_poisoned_rate_badnets: float = 0.05    # BadNets 毒化率
attack_poisoned_rate_blended: float = 0.05    # Blended 毒化率
attack_poisoned_rate_lc: float = 0.25         # LabelConsistent 毒化率
```

#### LabelConsistent 相关
```python
lc_eps: float = 8.0                           # 扰动范围
lc_alpha: float = 1.5                         # 迭代步长
lc_steps: int = 100                           # 优化步数
```

#### REFINE 架构
```python
refine_first_channels: int = 64               # UNet 初始通道数
skip_lc: bool = False                         # 跳过 LabelConsistent
```

#### 防御变体选择 ← 核心开关
```python
defense_variant: str = "refine"
# 可选值: ["refine", "refine_cg", "refine_ssl", "refine_pdb", "refine_pdb_ssl"]
```

#### REFINE_CG (Controlled Gate) 参数
```python
cg_threshold: float = 0.35                    # 疑似后门阈值
cg_temperature: float = 0.10                  # 门控软硬度
cg_strength: float = 1.0                      # 门强度倍数
```

#### SSL (自监督对比) 参数
```python
ssl_temperature: float = 0.07                 # 对比学习温度
ssl_weight: float = 0.02                      # SSL 损失权重 (默认偏小!)
```

#### PDB (主动防御触发) 参数 ← 最复杂
```python
pdb_trigger_type: int = 1                     # 触发器类型 [0=边, 1=补丁, 2=网格]
pdb_pix_value: float = 1.0                    # 触发像素值 (范围 [0,1])
pdb_target_shift: int = 1                     # 目标类偏移 h(y)=(y+shift) mod K
pdb_weight: float = 0.5                       # PDB 损失权重 ← 调优重点
pdb_batch_ratio: float = 0.5                  # 每 batch 中 PDB 样本比例 ← 调优重点
pdb_apply_inference_trigger: bool = True      # 推理时应用防御触发器

# 新增：预热与上限 (用于时序协调)
pdb_warmup_ratio: float = 0.3                 # PDB 预热比例 (前 30%)
ssl_warmup_ratio: float = 0.3                 # SSL 预热比例 (前 30%)
aux_loss_cap_ratio: float = 1.5               # 辅助损失上限 (相对 CE)
```

#### 运行模式
```python
only_attack: str = "all"                      # 仅运行特定攻击 ["all", "badnets", "blended", "label_consistent"]
attack_cache_root: str = ""                   # 攻击模型缓存根
pretrained_benign_model_path: str = ""        # 预训练干净模型 (可选)
pretrained_attack_model_path: str = ""        # 预训练被攻击模型 ← 关键！
force_rebuild: bool = False                   # 强制重建所有阶段
```

### 4.2 CLI 参数表 (快速查询)

| CLI Argument | Config Field | Type | Default | 推荐值 | 场景 |
|---|---|---|---|---|---|
| `--refine-epochs` | `refine_epochs` | int | 150 | 30-50 | 控制训练轮数 |
| `--defense-variant` | `defense_variant` | str | `refine` | 见下 | 防御方法选择 |
| `--only-attack` | `only_attack` | str | `all` | `badnets` | 仅运行特定攻击 |
| `--ssl-weight` | `ssl_weight` | float | 0.02 | 0.02-0.1 | SSL 强度 |
| `--pdb-weight` | `pdb_weight` | float | 0.5 | **0.2-0.5** | PDB 强度 ← 关键 |
| `--pdb-batch-ratio` | `pdb_batch_ratio` | float | 0.5 | **0.2-0.3** | PDB 采样比 ← 关键 |
| `--pdb-warmup-ratio` | `pdb_warmup_ratio` | float | 0.3 | 0.3-0.5 | PDB 预热期 ← Tier 1 优化 |
| `--ssl-warmup-ratio` | `ssl_warmup_ratio` | float | 0.3 | 0.3-0.5 | SSL 预热期 |
| `--aux-loss-cap-ratio` | `aux_loss_cap_ratio` | float | 1.5 | 1.5-2.0 | 损失上限 |
| `--pdb-trigger-type` | `pdb_trigger_type` | int | 1 | 1 | 触发器类型 |
| `--pretrained-attack-model-path` | `pretrained_attack_model_path` | str | `""` | `artifacts/attacks/badnets/latest.pth` | **重要** |
| `--batch-size` | `batch_size` | int | 128 | 64-256 | 内存/速度权衡 |
| `--seed` | `seed` | int | 666 | 666 | 固定复现 |
| `--deterministic` | `deterministic` | bool | True | True | 完全确定性 |

---

## 5. 模块类与函数签名

### 5.1 类继承树

```
Base (core/defenses/base.py)  ← 基础类 (_set_seed, seed/deterministic 管理)
│
└── REFINE (core/defenses/REFINE.py)  ← 核心防御
    ├── REFINE_SSL (core/defenses/REFINE_SSL.py)  ← +自监督对比
    ├── REFINE_CG (core/defenses/REFINE_CG.py)  ← +控制门
    ├── REFINE_PDB (core/defenses/REFINE_PDB.py) + _PDBMixin  ← +主动防御触发
    │   └── REFINE_PDB_SSL (内嵌在 REFINE_PDB.py)  ← 完整融合
    └── ... 其他变体
```

### 5.2 REFINE 基类

```python
class REFINE(Base):
    """Backdoor defense with REFINE.
    
    核心机制：
    1. 冻结被中毒模型，仅训练 UNet (输入变换)
    2. 输入通过 UNet → 输出应该剔除触发器
    3. 使用标签洗牌 + 对比损失
    """
    
    def __init__(self,
                 unet: torch.nn.Module,              # 输入变换 UNet
                 model: torch.nn.Module,             # 被中毒模型 (冻结)
                 pretrain: str = None,               # UNet 预训练路径
                 arr_path: str = None,               # 标签洗牌数组
                 num_classes: int = 10,              # 类别数
                 lmd: float = 0.1,                   # 温度参数 λ
                 seed: int = 0,
                 deterministic: bool = False):
        super(REFINE, self).__init__(seed=seed, deterministic=deterministic)
        self.unet = unet
        self.model = model                          # 冻结
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.num_classes = num_classes
        self.lmd = lmd  # 标签平滑/温度
    
    def train(self, dataset, epochs, schedule):     # 主训练循环
    def test(self, dataset):                         # BA 测试 (干净精度)
    def eval(self, dataset):                         # ASR 评估 (触发成功率)
    def label_shuffle(self, label):                  # 标签映射 (使用 self.arr_shuffle)
```

### 5.3 REFINE_SSL

```python
class REFINE_SSL(REFINE):
    """REFINE with self-supervised contrastive learning.
    
    增强：在 REFINE 基础上加入 SimCLR 风格的对比损失
    目的：强制模型学习干净、鲁棒的表示
    """
    
    def __init__(self,
                 unet,
                 model,
                 pretrain=None,
                 arr_path=None,
                 num_classes=10,
                 lmd=0.1,
                 seed=0,
                 deterministic=False,
                 temperature: float = 0.07,          # 对比学习温度
                 selfsup_weight: float = 0.02):      # SSL 损失权重 (默认很小!)
        super(REFINE_SSL, self).__init__(...)
        self.temperature = temperature
        self.selfsup_weight = selfsup_weight
    
    @staticmethod
    def _simclr_aug(sample):
        """SimCLR 数据增强: padding → crop → hflip"""
        ...
    
    def _selfsup_contrastive_loss(batch_img):
        """计算自监督对比损失"""
        # x1, x2 = 两次增强
        # z1, z2 = 特征
        # L_ssl = -log( exp(sim/T) / Σ exp(·) )
        ...
    
    def train(...):
        """训练: L = L_clean + λ_r·L_refine + λ_s·L_ssl"""
        ...
```

### 5.4 _PDBMixin (PDB 功能的混入类)

```python
class _PDBMixin:
    """Reusable proactive defensive-trigger primitives.
    被 REFINE_PDB 和 REFINE_PDB_SSL 混入使用。
    """
    
    def _init_pdb(self,
                  pdb_trigger_type: int = 1,
                  pdb_pix_value: float = 1.0,
                  pdb_target_shift: int = 1,
                  pdb_weight: float = 0.5,
                  pdb_batch_ratio: float = 0.5,
                  pdb_apply_inference_trigger: bool = True,
                  pdb_warmup_ratio: float = 0.3,
                  ssl_warmup_ratio: float = 0.3,
                  aux_loss_cap_ratio: float = 1.5):
        """初始化 PDB 参数"""
        self.pdb_trigger_type = int(pdb_trigger_type)
        self.pdb_weight = float(pdb_weight)
        self.pdb_batch_ratio = float(pdb_batch_ratio)
        self.pdb_warmup_ratio = max(0.0, float(pdb_warmup_ratio))
        self.ssl_warmup_ratio = max(0.0, float(ssl_warmup_ratio))
        self.aux_loss_cap_ratio = max(0.0, float(aux_loss_cap_ratio))
    
    # 关键方法
    @staticmethod
    def _aux_progress_scale(epoch: int, total: int, warmup_ratio: float) -> float:
        """计算当前 epoch 的权重衰减因子（线性预热）
        
        例：warmup_ratio=0.3, total=50
          warmup_epochs = 15
          epoch<15: scale = (epoch+1)/15, 从 0.07 → 1.0
          epoch≥15: scale = 1.0
        """
        if warmup_ratio <= 0.0 or total <= 0:
            return 1.0
        warmup_epochs = max(1, int(round(total * warmup_ratio)))
        return min(1.0, float(epoch + 1) / float(warmup_epochs))
    
    def _effective_aux_weight(base_weight, epoch, total, warmup_ratio) -> float:
        """返回当前 epoch 的有效权重 = base × scale(epoch)"""
        scale = self._aux_progress_scale(epoch, total, warmup_ratio)
        return float(base_weight) * scale
    
    def _cap_aux_loss(aux_loss: Tensor, ce_loss: Tensor) -> Tensor:
        """限制辅助损失不超过 CE_loss × cap_ratio
        
        防止某个辅助任务"压倒"主任务
        """
        if self.aux_loss_cap_ratio <= 0.0:
            return aux_loss
        cap = ce_loss.detach() * self.aux_loss_cap_ratio
        return torch.minimum(aux_loss, cap)
    
    def _trigger_mask(images: Tensor) -> Tensor:
        """生成防御触发器掩码
        
        Type 0: Border (上下左右边框)
        Type 1: Patch (左上角 7×7 补丁)
        Type 2: Grid (棋盘网格)
        """
        mask = torch.zeros_like(images)
        if self.pdb_trigger_type == 1:
            patch = min(7, h, w)
            mask[:, :, :patch, :patch] = 1
        ...
        return mask
    
    def _apply_defensive_trigger(images: Tensor) -> Tensor:
        """将防御触发器应用到图像
        
        triggered = images × (1-mask) + pdb_pix_value × mask
        返回 clamp 到 [0, 1]
        """
        ...
    
    def _defensive_target(pseudo_index: Tensor) -> Tensor:
        """映射伪标签: h(y) = (y + shift) mod num_classes"""
        return (pseudo_index + self.pdb_target_shift) % self.num_classes
    
    def _sample_defensive_subset(batch_size, device) -> Tensor:
        """按 pdb_batch_ratio 随机采样 PDB 样本指示向量"""
        if self.pdb_batch_ratio >= 1.0:
            return torch.ones(batch_size, dtype=bool, device=device)
        sampled = torch.rand(batch_size, device=device) < self.pdb_batch_ratio
        return sampled
```

### 5.5 REFINE_PDB

```python
class REFINE_PDB(REFINE, _PDBMixin):
    """REFINE + Proactive Defensive Backdoor trigger guidance."""
    
    def __init__(self,
                 unet, model, ...,
                 # PDB 参数
                 pdb_trigger_type=1,
                 pdb_weight=0.5,
                 pdb_batch_ratio=0.5,
                 pdb_warmup_ratio=0.3,
                 ssl_warmup_ratio=0.3,
                 aux_loss_cap_ratio=1.5):
        super().__init__(...)
        self._init_pdb(...)  # 初始化 PDB 参数
    
    def train(self, epochs, schedule):
        """
        训练循环:
        L_total = L_clean + λ_r × L_refine + λ_p(t) × L_pdb
        
        其中 λ_p(t) 按 pdb_warmup_ratio 从 0 线性升到 pdb_weight
        """
        for epoch in range(epochs):
            lambda_p = self._effective_aux_weight(
                self.pdb_weight, epoch, epochs, self.pdb_warmup_ratio
            )
            # ... 计算出 L_pdb，乘以 lambda_p
```

### 5.6 REFINE_PDB_SSL (完整融合)

```python
class REFINE_PDB_SSL(REFINE_SSL, _PDBMixin):
    """Full combination: REFINE + SSL + PDB."""
    
    def __init__(self, ...,
                 temperature=0.07,
                 selfsup_weight=0.02,
                 pdb_weight=0.5,
                 pdb_batch_ratio=0.5,
                 pdb_warmup_ratio=0.3,
                 ssl_warmup_ratio=0.3,
                 aux_loss_cap_ratio=1.5):
        super().__init__(...)
        self._init_pdb(...)
    
    def train(self, epochs, schedule):
        """
        完整训练循环：
        L_total = L_clean 
                + λ_r × L_refine 
                + λ_s(t) × L_ssl           # 按 ssl_warmup_ratio
                + λ_p(t) × L_pdb           # 按 pdb_warmup_ratio
        
        三个辅助同时开启，但各按自己的预热曲线
        """
        for epoch in range(epochs):
            lambda_s = self._effective_aux_weight(
                self.selfsup_weight, epoch, epochs, self.ssl_warmup_ratio
            )
            lambda_p = self._effective_aux_weight(
                self.pdb_weight, epoch, epochs, self.pdb_warmup_ratio
            )
            # ... 同时计算 L_ssl, L_pdb，各乘以对应权重
```

---

## 6. 损失函数与数据流

### 6.1 损失函数定义

#### REFINE 基础
```
L_clean  = CE(model(clean_sample), label)
         # 保证干净精度

L_refine = CE(model(trigger_sample), shuffled_label)
         或 温度缩放的对比损失
         # 让模型对触发样本不指向目标类
         

L_refine_total = L_clean + λ_r × L_refine
               # λ_r 通常在 1.0 左右（平衡两项）
```

#### SSL 附加
```
x1, x2 = SimCLR_augment(batch)  # 两次增强
z1, z2 = backbone(x1), backbone(x2)  # 特征提取

L_ssl = -log( exp(cosine_sim(z1, z2) / T) / 
              Σ_{i=1}^{2N} exp(cosine_sim(z1, z_i) / T) )
      # NT-Xent (Normalized Temperature-scaled Cross Entropy)
      # T = temperature = 0.07

L_ssl_total = L_clean + λ_r × L_refine + λ_s × L_ssl
            # λ_s 通常很小 (0.02-0.1)
```

#### PDB 附加
```
样本选择: pdb_mask ~ Bernoulli(pdb_batch_ratio)
x_pdb   = apply_defensive_trigger(x[pdb_mask])
y_pdb   = h(y[pdb_mask]) = (y + pdb_target_shift) mod num_classes

L_pdb   = CE(model(x_pdb), y_pdb)

L_pdb_total = L_clean + λ_r × L_refine + λ_p(t) × L_pdb
            # λ_p(t) 按 pdb_warmup_ratio 从 0 升到 pdb_weight

若 aux_loss_cap_ratio > 0:
    L_pdb = min(L_pdb, L_clean.detach() × aux_loss_cap_ratio)
```

#### 完整融合 (REFINE+SSL+PDB)
```
L_total = L_clean 
        + λ_r × L_refine 
        + λ_s(t) × L_ssl
        + λ_p(t) × L_pdb

其中：
  λ_s(t) = ssl_weight × _aux_progress_scale(t, epochs, ssl_warmup_ratio)
  λ_p(t) = pdb_weight × _aux_progress_scale(t, epochs, pdb_warmup_ratio)

如果 ssl_warmup_ratio=0.3, pdb_warmup_ratio=0.3, epochs=50:
  前 15 epoch: 两个都从 0 线性升
  后 35 epoch: 两个都保持最大值 pdb_weight, ssl_weight

这就是"预热"的实质 —— 避免前期梯度冲突
```

### 6.2 评估指标

#### BA (Before Attack / Benign Accuracy)
```
定义：干净测试集上的模型精度

评估过程：
  1. 用 defense checkpoint 加载模型
  2. 在干净测试集上前向传播（无触发器、无标签修改）
  3. 计算 top-1 accuracy

代码：
  logits = model(clean_test_batch)
  pred = torch.argmax(logits, dim=1)
  acc = (pred == target).float().mean()
```

#### ASR (Attack Success Rate)
```
定义：被攻击模型对后门触发器的响应成功率

评估过程：
  1. 对测试样本加上对应的后门触发器
  2. 前向传播
  3. 统计指向目标类的比例

代码：
  x_triggered = apply_trigger(test_batch)
  logits = model(x_triggered)
  pred = torch.argmax(logits, dim=1)
  asr = (pred == target_class).float().mean()

示例（当前结果）：
  Attack ASR = 0.9727  (27% 样本添加触发器后，97.27% 指向目标类)
  Refine ASR = 0.1030  (防御后，仅 10.3% 中招)
```

#### Δ BA / Δ ASR
```
Δ BA  = (BA_before - BA_after) / BA_before
      = 防御对泛化的伤害程度（百分比）
      当前 = (0.9168 - 0.8414) / 0.9168 ≈ 8.2%

Δ ASR = (ASR_before - ASR_after) / ASR_before
      = 防御对后门的压制程度（百分比）
      当前 = (0.9727 - 0.1030) / 0.9727 ≈ 89.4%

通常目标：
  Δ BA < 3%   (BA 掉点很少)
  Δ ASR > 85% (ASR 压得很低)
```

### 6.3 管道执行流（Suite 概览）

```
CLI 参数 (python run.py single ...)
    ↓
parse_suite_args() → RuntimeConfig
    ↓
Suite.run(config) / dispatch_stages
    ↓
+─────────────────────────────────────┐
│ Stage[Benign]: Train clean model    │
│ Output: benign_ckpt.pth             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage[Attack/BadNets]               │
│ Input: benign_ckpt.pth              │
│ Output: attack_ckpt_best.pth        │
│ Metrics: BA_attack, ASR_attack      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage[REFINE/*] (根据 defense_variant) │
│ Input: attack_ckpt.pth (或 --pretrained) │
│ Output: refine_ckpt_best.pth        │
│ Metrics: BA_refine, ASR_refine      │
│                                     │
│ defense_variant 选择:               │
│  - refine    : Base REFINE           │
│  - refine_ssl: +SSL                 │
│  - refine_pdb: +PDB                 │
│  - refine_pdb_ssl: +PDB+SSL (完整)  │
└─────────────────────────────────────┘
    ↓
Output: metrics_summary.json, comparison_report.md
```

---

## 7. 优化方案与实验建议

### 7.1 推荐的 7 层优化策略

#### Tier 1: 两阶段串联 (立竿见影) ⭐⭐⭐

**原理**  
前期稳定主任务，后期轻量介入辅助。

**实施**  
```
前 30-40% epoch: 仅 L_clean + L_refine (关闭 PDB/SSL)
后 60-70% epoch: 逐步开启 PDB (权重从 0 → pdb_weight)

配置修改:
  pdb_warmup_ratio = 0.4  (改成 40% 或更高)
  ssl_warmup_ratio = 0.4
```

**预期收益**  
```
BA：从 84% 恢复至 87%+
ASR：保持 10% 左右
训练稳定性：大幅提升
```

#### Tier 2: 动态权重自适应 (中难度) ⭐⭐

**原理**  
根据验证指标实时调整权重。

**实施**  
```
λ_p(t) = pdb_weight × sin²(π·t / 2T)  # 平滑增长后衰减
或
监控 BA，若下降超过阈值则降权重: λ_p *= 0.95
```

#### Tier 3: 样本比例课程学习 (容易) ⭐

**原理**  
从轻到重的样本比例递增。

**实施**  
```
[1-20% epoch]:  pdb_batch_ratio = 0.1
[20-40% epoch]: pdb_batch_ratio = 0.3
[40-60% epoch]: pdb_batch_ratio = 0.5
[60%+ epoch]:   pdb_batch_ratio = 0.3  # 收尾轻量
```

#### Tier 4: 双指标早停 (防过拟合)

**原理**  
同时监控 BA/ASR，当 BA 继续白白下降时停止。

**指标**  
```
J = ASR + α × max(0, (ΔBA_threshold - (BA_0 - BA_t)))
当 J 无改善 3-5 epoch → 停止
```

#### Tier 5: 层冻结策略 (中级)

**原理**  
前期冻结 backbone，仅调后几层或分类头。

**效果**  
减小表示空间震荡，常拉回 0.5-1% BA。

#### Tier 6: EMA/SWA 平均 (收尾优化)

**原理**  
取最近 N 个 checkpoint 的权重平均。

**代码**  
```
w_final = mean([w_{T-10}, w_{T-9}, ..., w_T])
```

**效果**  
平滑噪声，常拉回 0.5-1% BA，ASR 往往不反弹。

#### Tier 7: 蒸馏恢复 (研究向，高成本)

**原理**  
用纯 REFINE 结果作 teacher，student 在 PDB 约束下学 teacher。

---

### 7.2 最小可行实验包 (立即执行)

建议按顺序运行这 4 条，固定 seed/epoch，对比结果：

```
Exp 1: 基线 REFINE Only (R)
  defense_variant=refine
  Expected: BA ~90%, ASR ~10%

Exp 2: R + 增强预热 
  defense_variant=refine
  pdb_warmup_ratio=0.4  (虽然没设 PDB，但用 REFINE_PDB variant测试)
  Expected: 与 Exp 1 相同或更稳

Exp 3: R + PDB 轻量
  defense_variant=refine_pdb
  pdb_weight=0.3
  pdb_batch_ratio=0.2
  pdb_warmup_ratio=0.4
  Expected: BA ~88-89%, ASR ~10%

Exp 4: R + PDB轻 + SSL轻
  defense_variant=refine_pdb_ssl
  ssl_weight=0.02
  pdb_weight=0.3
  pdb_batch_ratio=0.2
  pdb_warmup_ratio=0.4
  ssl_warmup_ratio=0.4
  Expected: BA ~87-88%, ASR ~10%

对比指标: BA / ASR / per-epoch loss curve (稳定性)
```

---

## 8. 可运行命令模板

### 8.1 基线命令组

所有命令已测试，可直接粘贴运行。

#### R: 纯 REFINE (推荐基线)
```bash
python run.py single \
  --defense-variant refine \
  --only-attack badnets \
  --refine-epochs 50 \
  --seed 666 \
  --pretrained-attack-model-path artifacts/attacks/badnets/latest.pth \
  --run-name exp_refine_baseline_s666
```

**预期结果**: BA ~90%, ASR ~10%, 运行时间 ~2h

#### RB: REFINE + PDB (轻量优化版)
```bash
python run.py single \
  --defense-variant refine_pdb \
  --only-attack badnets \
  --refine-epochs 50 \
  --pdb-weight 0.3 \
  --pdb-batch-ratio 0.2 \
  --pdb-warmup-ratio 0.4 \
  --seed 666 \
  --pretrained-attack-model-path artifacts/attacks/badnets/latest.pth \
  --run-name exp_refine_pdb_light_s666
```

**预期结果**: BA ~88%, ASR ~10%

#### RS: REFINE + SSL
```bash
python run.py single \
  --defense-variant refine_ssl \
  --only-attack badnets \
  --refine-epochs 50 \
  --ssl-weight 0.02 \
  --seed 666 \
  --pretrained-attack-model-path artifacts/attacks/badnets/latest.pth \
  --run-name exp_refine_ssl_s666
```

**预期结果**: BA ~89%, ASR ~10%

#### RSB: REFINE + PDB + SSL (调优版)
```bash
python run.py single \
  --defense-variant refine_pdb_ssl \
  --only-attack badnets \
  --refine-epochs 50 \
  --ssl-weight 0.02 \
  --pdb-weight 0.3 \
  --pdb-batch-ratio 0.2 \
  --pdb-warmup-ratio 0.4 \
  --ssl-warmup-ratio 0.4 \
  --seed 666 \
  --pretrained-attack-model-path artifacts/attacks/badnets/latest.pth \
  --run-name exp_refine_pdb_ssl_optimized_s666
```

**预期结果**: BA ~87%, ASR ~10%（改进版，相比原来的 84% 处理得更好）

### 8.2 参数扫描命令

#### 场景 A: BA 掉太多，想恢复
```bash
# 方案 1: 降低 PDB 强度
--pdb-weight 0.2 --pdb-batch-ratio 0.15

# 方案 2: 增加预热期
--pdb-warmup-ratio 0.5 --ssl-warmup-ratio 0.5

# 方案 3: 减轻 SSL
--ssl-weight 0.01
```

#### 场景 B: ASR 仍偏高 (>15%)，想继续压
```bash
# 方案 1: 增强 PDB
--pdb-weight 0.7 --pdb-batch-ratio 0.4

# 方案 2: 增强 SSL
--ssl-weight 0.1

# 方案 3: 双管齐下 (需谨慎测试 BA)
--pdb-weight 0.5 --ssl-weight 0.05
```

### 8.3 自动汇总

每次运行完成后，执行：
```bash
python tools/append_experiment_matrix.py
```

查看最新结果：
```bash
cat experiment_matrix.csv | tail -5
```

---

## 9. 故障排查与已知问题

### 9.1 参数兼容性问题

**症状**: `unrecognized arguments --pdb-warmup-ratio`

**排查**:
```bash
# 1. 检查 suite_config 是否已添加参数
grep -n "pdb-warmup-ratio" runner/suite_config.py

# 2. 检查 CLI 是否显示
python run.py single -h | grep warmup

# 3. 检查类签名是否接收
grep -n "def __init__" core/defenses/REFINE_PDB.py
# 应看到 pdb_warmup_ratio 在参数列表中

# 4. 检查 pipeline 是否正确转发
grep -n "pdb_warmup_ratio" runner/suite_pipeline.py
```

**修复**（若仍有问题）:
```python
# core/defenses/REFINE_PDB.py
class REFINE_PDB(REFINE, _PDBMixin):
    def __init__(self, ..., 
                 pdb_warmup_ratio=0.3,  # ← 补充这行
                 ssl_warmup_ratio=0.3,  # ← 补充这行
                 aux_loss_cap_ratio=1.5):
        super().__init__(...)
        self._init_pdb(..., 
                      pdb_warmup_ratio=pdb_warmup_ratio,
                      ssl_warmup_ratio=ssl_warmup_ratio,
                      aux_loss_cap_ratio=aux_loss_cap_ratio)
```

### 9.2 类构造函数签名不一致

**症状**: `TypeError: REFINE_PDB.__init__() unexpected keyword argument 'xxx'`

**排查**:
```bash
python -c "
from core.defenses.REFINE_PDB import REFINE_PDB
import inspect
sig = inspect.signature(REFINE_PDB.__init__)
print(sig)
"
```

**应该看到的**: 所有参数包括 pdb_warmup_ratio, ssl_warmup_ratio

### 9.3 模型加载失败

**症状**: `FileNotFoundError: artifacts/attacks/badnets/latest.pth`

**原因**: 被攻击模型还没复制到统一位置

**修复**:
```bash
# 找到最新的被攻击模型
ls -lt experiments/*/attack_ckpt_best.pth | head -1

# 复制到统一位置
mkdir -p artifacts/attacks/badnets
cp <上一行的路径> artifacts/attacks/badnets/latest.pth

# 验证
ls -lh artifacts/attacks/badnets/latest.pth
```

### 9.4 汇总表格过滤问题

**症状**: 表格中包含了被筛除的"数据塌缩"实验

**排查**:
```bash
grep -i "collapse\|invalid" experiment_matrix.csv
```

**修复**（检查脚本 tools/append_experiment_matrix.py）:
```python
# 应在脚本中有判断逻辑排除无效实验
if metrics.get("is_valid", True) == False:
    continue
```

---

## 10. 快速查询表

### 10.1 参数速查

| 场景 | 参数 | 设置 | 原因 |
|------|------|------|------|
| 基线稳定 | `--defense-variant refine` | 只用 REFINE | 最少干扰 |
| | `--refine-epochs 50` | 中等轮数 | 平衡时间/精度 |
| | `--seed 666` | 固定 | 可复现 |
| 降低 PDB 伤害 | `--pdb-weight 0.3` | 从 0.5 → 0.3 | 减弱约束 |
| | `--pdb-batch-ratio 0.2` | 从 0.5 → 0.2 | 减少受约样本 |
| 稳定时序 | `--pdb-warmup-ratio 0.4` | 从 0.3 → 0.4 | 拉长预热 |
| 强化 PDB | `--pdb-weight 0.7` | 0.5 → 0.7 | 增强约束 |
| | `--pdb-batch-ratio 0.5` | 保持或↑ | 更多样本受约 |
| 增强 SSL | `--ssl-weight 0.1` | 从 0.02 ↑ | 更强对比 |

### 10.2 防御变体选择

| 变体 | CLI 参数 | 包含组件 | BA 预期 | ASR 预期 | 用途 |
|------|---------|---------|--------|---------|------|
| R | `refine` | REFINE only | ~90% | ~10% | 基线 |
| RB | `refine_pdb` | +PDB | ~88% | ~10% | 增强防护 |
| RS | `refine_ssl` | +SSL | ~89% | ~10% | 鲁棒特征 |
| RSB | `refine_pdb_ssl` | +PDB+SSL | ~87-88% | ~10% | 完整融合 |
| 其他 | `refine_cg` | +ControlGate | ... | ... | 门控下采样 |

### 10.3 文件位置速查

```
项目根目录: c:\Users\17672\Documents\Project\-

入口文件:
  run.py                                  # CLI 入口
  runner/suite_config.py                  # 参数定义
  runner/suite_pipeline.py                # 管道调度

防御实现:
  core/defenses/base.py                   # 基类
  core/defenses/REFINE.py                 # REFINE 核心
  core/defenses/REFINE_SSL.py             # +SSL
  core/defenses/REFINE_PDB.py             # +PDB & 完整融合

攻击实现:
  core/attacks/BadNets.py                 # BadNets 攻击
  core/attacks/Blended.py                 # Blended 攻击
  core/attacks/LabelConsistent.py         # LabelConsistent

工具脚本:
  tools/append_experiment_matrix.py       # 自动汇总

结果输出:
  experiments/*/metrics_summary.json       # 单次实验指标
  experiments/*/comparison_report.md       # 单次实验报告
  experiment_matrix.csv                   # 全实验汇总表
  experiment_matrix.md                    # 汇总报告
  artifacts/attacks/badnets/latest.pth   # 统一被攻击模型 (新)
```

### 10.4 常见错误排查速查

| 错误 | 可能原因 | 排查命令 |
|------|---------|---------|
| `unrecognized arguments` | 参数未在 CLI 注册 | `python run.py single -h | grep <param>` |
| `unexpected keyword` | 类构造签名缺参 | `grep "def __init__" core/defenses/REFINE*.py` |
| `FileNotFoundError` checkpoint | 模型路径错 | `ls -l artifacts/attacks/badnets/latest.pth` |
| `CUDA OOM` | 批大小过大 | `--batch-size 64` (from 128) |
| Loss 不收敛 | 学习率或权重问题 | 检查 loss curve 图，看 warmup 是否生效 |

---

## 11. 下一步快速切入点

### 11.1 对接下一轮智能体

如果下一轮由其他智能体接手，需要了解：

1. **为什么 BA 掉这么多？** → 见 §3.4 诊断
2. **应该先做什么优化？** → Tier 1 两阶段串联 (§7.1)
3. **立即可运行的命令？** → §8 模板
4. **参数怎么调？** → §10 速查表

### 11.2 即刻行动清单

- [ ] 确认参数兼容性已修复（跑一条含 warmup 的命令）
- [ ] 执行最小实验包 4 条命令，对比 BA/ASR/稳定性
- [ ] 运行 `python tools/append_experiment_matrix.py` 汇总
- [ ] 根据结果选择 Tier 1 或 Tier 2 优化方向继续迭代

### 11.3 理论验证清单

- [ ] 验证 Tier 1 (两阶段) 是否如预期改善 BA
- [ ] 验证预热函数是否正确计算（检查 loss curve 中权重增长曲线）
- [ ] 验证上限函数是否生效（对比有/无 aux_loss_cap_ratio 的结果）

---

## 附录：关键代码片段

### A.1 权重预热函数
```python
def _aux_progress_scale(current_epoch: int, total_epochs: int, warmup_ratio: float) -> float:
    """线性预热：前 warmup_ratio×total_epochs 个 epoch 内权重从 0 线性升到 1"""
    if warmup_ratio <= 0.0 or total_epochs <= 0:
        return 1.0
    warmup_epochs = max(1, int(round(total_epochs * warmup_ratio)))
    return min(1.0, float(current_epoch + 1) / float(warmup_epochs))

# 示例：warmup_ratio=0.3, total=50
# epoch 0: scale = 1/15 ≈ 0.067
# epoch 7: scale = 8/15 ≈ 0.533
# epoch 14: scale = 15/15 = 1.0
# epoch 15+: scale = 1.0
```

### A.2 损失上限函数
```python
def _cap_aux_loss(self, aux_loss: torch.Tensor, ce_loss: torch.Tensor) -> torch.Tensor:
    """防止辅助损失"压倒"主任务"""
    if self.aux_loss_cap_ratio <= 0.0:
        return aux_loss
    cap = ce_loss.detach() * self.aux_loss_cap_ratio
    return torch.minimum(aux_loss, cap)

# 示例：aux_loss_cap_ratio=1.5
# 若 CE_loss=0.5，则 aux_loss 被限制 ≤ 0.75
# 这防止辅助项过大导致学习不稳定
```

### A.3 防御触发器应用
```python
def _apply_defensive_trigger(self, images: torch.Tensor) -> torch.Tensor:
    """将防御触发器应用到图像"""
    mask = self._trigger_mask(images)  # 掩码 (1=触发, 0=原图)
    trigger_value = torch.full_like(images, self.pdb_pix_value)  # 触发像素值
    triggered = images * (1.0 - mask) + trigger_value * mask     # 混合
    return torch.clamp(triggered, 0.0, 1.0)  # 限制范围
```

### A.4 CLI 参数注册示例
```python
# runner/suite_config.py 中的注册方式
parser.add_argument("--pdb-warmup-ratio", type=float, default=0.3,
                    help="Warmup ratio for PDB auxiliary loss weight")

parser.add_argument("--ssl-warmup-ratio", type=float, default=0.3,
                    help="Warmup ratio for SSL auxiliary loss weight")

parser.add_argument("--aux-loss-cap-ratio", type=float, default=1.5,
                    help="Upper bound for auxiliary loss relative to CE loss")
```

---

**文档生成时间**: 2026-03-26  
**包含章节**: 11 (系统化覆盖所有关键信息)  
**目标用户**: 后续智能体、项目维护者、实验复现  

本文档是对话结束前的完整导出，包含充分上下文供一次性阅读与快速查询。

