# 三个 Observation 的实验设计规范
## ——可直接用于编写实验代码

---

# Experiment 1：验证良性相似度矩阵的低秩性
**对应 Observation 1**

## 1.1 实验目标
证明：全良性场景下，客户端Jaccard相似度矩阵的奇异值能量
高度集中在前几个奇异值，且该性质在不同non-IID程度下稳定存在。

## 1.2 数据与模型配置

```
数据集:       CIFAR-10
客户端数:     K = 20（全部良性）
数据划分:     Dirichlet(α)，α ∈ {0.1, 0.3, 0.5, 1.0}
公共数据集:   从CIFAR-10测试集随机抽取 N_pub = 1000 张无标签图片
              （不参与任何客户端训练，仅用于生成logits）
本地模型:     各客户端独立训练，允许使用不同小型CNN架构
              （如ResNet-8 / VGG-small / MobileNetV1的混合）
本地训练:     每个客户端训练 E_local = 20 个epoch（模拟收敛后状态）
重复次数:     5次独立随机种子，取均值±标准差
```

## 1.3 计算步骤（逐步）

### Step 1：生成 soft-label 预测
```
对每个客户端 k = 1..K：
  z_k ∈ R^{N_pub × C}  ← 在公共数据集上的logit输出（C=10类）
  p_k = softmax(z_k)    ← 转为概率分布
  
  对每个公共样本 n：
    topk_k(n) = argsort(p_k[n])[-κ:]  ← top-κ 预测类别的下标集合
    κ = 5（超参数）
```

### Step 2：构建 Jaccard 相似度矩阵
```
S ∈ R^{K×K}，其中：
  S[i, j] = (1/N_pub) * Σ_n  |topk_i(n) ∩ topk_j(n)| / |topk_i(n) ∪ topk_j(n)|

注意：S 是对称矩阵，S[i,i] = 1
```

### Step 3：SVD 分解与能量分析
```
U, σ, V^T = SVD(S)          ← σ 为奇异值向量，降序排列

# 累积能量曲线
energy_ratio[r] = Σ_{i=1}^{r} σ_i² / Σ_{i=1}^{K} σ_i²   for r=1..K

# 有效秩（Effective Rank）
σ̄_i = σ_i / Σ_j σ_j         ← 归一化
H = -Σ_i σ̄_i * log(σ̄_i)    ← 香农熵
effective_rank = exp(H)
```

## 1.4 可视化规格

### 图1A：奇异值累积能量曲线（主图）
```
类型:   折线图
X轴:    奇异值序号 r（1 → K=20），标签"Singular Value Index"
Y轴:    累积能量占比 energy_ratio[r]，范围[0, 1]，标签"Cumulative Energy Ratio"
曲线:   4条，对应α ∈ {0.1, 0.3, 0.5, 1.0}，不同颜色+线型
标注:   在 energy_ratio = 0.90 处画水平虚线，标注"90% energy"
        标注每条曲线达到90%时对应的奇异值数量
图例:   "α=0.1 (Highly non-IID)" ... "α=1.0 (Near-IID)"
误差带: 5次重复的均值±标准差（shaded area）
```

### 图1B：相似度矩阵热图（辅助图，选α=0.5展示）
```
类型:   热图（imshow / seaborn heatmap）
数据:   S矩阵（按层次聚类重排行列顺序，使块状结构更明显）
色彩:   viridis 或 RdYlBu，值域[0,1]
标注:   每个轴标"Client Index"，colorbar标"Jaccard Similarity"
尺寸:   正方形，20×20
```

### 图1C：有效秩 vs. α（可做成小插图或表格）
```
X轴:    α 值（0.1, 0.3, 0.5, 1.0）
Y轴:    effective_rank（均值±std）
参考线: r=K=20（无结构的上界）
```

## 1.5 预期结果
- α=0.5时，前3个奇异值能量占比 > 85%
- 即使α=0.1（极度non-IID），effective_rank << K
- 说明：低秩性是任务对齐的内在性质，对数据异构度鲁棒

---

# Experiment 2：验证 Byzantine 扰动的行稀疏结构
**对应 Observation 2**

## 2.1 实验目标
证明：Byzantine客户端的存在，在相似度矩阵中引入的扰动
高度集中在对应Byzantine客户端的行列上（行稀疏性），
且该性质在不同攻击类型下一致成立。

## 2.2 数据与模型配置

```
数据集:       CIFAR-10
客户端数:     K = 20，其中 m = 4 为 Byzantine（20%）
Byzantine ID: 固定为客户端 {0, 1, 2, 3}（实验中已知，用于验证）
数据划分:     Dirichlet α = 0.5
公共数据集:   同 Exp 1，N_pub = 1000
攻击类型:     4种（每种独立运行）
  - Gaussian:  z_k^adv = z_k + ε,  ε ~ N(0, σ²I),  σ=1.0
  - Label Flip: z_k^adv[n] = one_hot(1 - argmax(p_k[n])) * 10  (翻转最高类)
  - Targeted:  z_k^adv[n] = one_hot(target_class) * 10  (固定目标类=0)
  - ALIE:      z_k^adv = μ_benign + z_scale * σ_benign（需先收集良性统计量）
重复次数:     5次随机种子
```

## 2.3 计算步骤（逐步）

### Step 1：构建两个相似度矩阵
```
# 场景A：全良性（K=20，无攻击）→ 作为 L* 的估计
S_benign = compute_jaccard_matrix(all_benign_predictions)  # K×K

# 场景B：含Byzantine（相同良性客户端 + Byzantine客户端）
S_observed = compute_jaccard_matrix(mixed_predictions)     # K×K
# 注意：S_observed 的行列顺序与 S_benign 一致（客户端编号对应）
```

### Step 2：计算扰动矩阵
```
E = S_observed - S_benign    # K×K
# E[i,j] 表示引入Byzantine客户端后，客户端i和j相似度的变化量
```

### Step 3：行能量分析
```
row_energy[k] = ||E[k, :]||_2    for k = 0..K-1   # 每行的L2范数

# 归一化（方便跨攻击类型比较）
row_energy_norm[k] = row_energy[k] / Σ_k row_energy[k]

# 统计指标：
byzantine_energy_ratio = Σ_{k∈Byzantine} row_energy[k] / Σ_k row_energy[k]
# 预期：这个比例 > 0.85，说明能量高度集中在Byzantine行
```

### Step 4：计算稀疏性统计
```
对每种攻击：
  concentration = Σ_{k∈Byzantine} row_energy[k] / Σ_k row_energy[k]
  sparsity      = ||E||_0 / (K*K)    # 非零元素比例（用阈值1e-3判断非零）
```

## 2.4 可视化规格

### 图2A：行能量条形图（每种攻击一个子图，2×2布局）
```
类型:   分组条形图（grouped bar chart），4个子图
X轴:    客户端编号 0..19，标签"Client Index"
Y轴:    归一化行能量 row_energy_norm[k]，标签"Normalized Row Energy ||E_k||₂"
颜色:   Byzantine客户端（k∈{0,1,2,3}）→ 红色
        良性客户端 → 蓝色
标注:   在Byzantine列上方标注 "Byzantine"（小箭头或星号）
        在图内右上角标注 "Concentration: XX.X%"（Byzantine能量占比）
子图标题: "Gaussian Attack" / "Label Flipping" / "Targeted" / "ALIE"
```

### 图2B：稀疏性统计表（在论文中以表格呈现）
```
| Attack Type   | Byzantine Energy Concentration | Sparsity (||E||_0/K²) |
|---------------|-------------------------------|----------------------|
| Gaussian      |             XX.X%             |         XX.X%        |
| Label Flip    |             XX.X%             |         XX.X%        |
| Targeted      |             XX.X%             |         XX.X%        |
| ALIE          |             XX.X%             |         XX.X%        |
均值±标准差，来自5次重复
```

### 图2C（可选）：E 矩阵热图
```
类型:   热图，显示 |E| 的绝对值
色彩:   Reds（值越大越深）
标注:   在Byzantine对应的行列上画红色边框
直观展示：热图中高亮区域应集中在Byzantine行列的交叉处
```

## 2.5 预期结果
- Byzantine Energy Concentration > 85% 对所有攻击类型成立
- 即使是ALIE（最隐蔽的攻击），扰动仍主要集中在Byzantine行
- 直觉验证：Byzantine客户端有限（m=4），只能影响自身对应行列

---

# Experiment 3：验证净化后图的导电率差异与PPR信任对齐
**对应 Observation 3**

## 3.1 实验目标
证明：经过低秩图净化后，良性客户端在图中的连通性显著高于
Byzantine客户端；且PPR信任分与这种连通性结构高度对齐，
能够区分良性与Byzantine客户端。

## 3.2 数据与模型配置

```
数据集:       CIFAR-10
客户端数:     K = 20，m = 6 Byzantine（30%，比Exp2更严苛）
Byzantine ID: {0,1,2,3,4,5}
攻击类型:     ALIE（最难防御，作为worst-case展示）
数据划分:     Dirichlet α = 0.5
RPCA求解:     用 ADMM 求解 min ||L||_* + λ||E||_1，λ = 1/√K
              最大迭代次数 500，收敛阈值 1e-4
PPR参数:      阻尼因子 β = 0.85，种子分布 = 均匀分布
              收敛阈值 1e-6，最大迭代 200
重复次数:     5次随机种子
```

## 3.3 计算步骤（逐步）

### Step 1：构建观测相似度矩阵
```
S_observed = compute_jaccard_matrix(all_predictions)   # K×K，含Byzantine
```

### Step 2：低秩图净化（RPCA）
```
# 求解：min_{L,E} ||L||_* + λ||E||_1  s.t. L+E = S_observed
L_hat, E_hat = solve_rpca(S_observed, lambda_=1/sqrt(K))

# 后处理：将L_hat的负值截断为0，归一化到[0,1]
L_hat = clip(L_hat, min=0)
L_hat = L_hat / L_hat.max()
```

### Step 3：计算图连通性指标
```
# 对净化前后两个图分别计算：
for Graph in [S_observed, L_hat]:
  # 加权节点度
  degree[k] = Σ_j Graph[k, j]   for k=0..K-1
  
  # 良性子图内部平均边权（intra-group connectivity）
  W_benign = mean(Graph[i,j]) for i,j both in Benign
  
  # Byzantine节点到良性核心的平均边权（cross-group connectivity）  
  W_cross = mean(Graph[i,j]) for i in Byzantine, j in Benign
  
  # 连通性差异（越大越好）
  connectivity_gap = W_benign - W_cross
```

### Step 4：计算图谱间隙（Spectral Gap）
```
# 构建归一化图Laplacian
D = diag(degree)
L_graph = I - D^{-1/2} @ Graph @ D^{-1/2}   # 归一化Laplacian

# 特征值分解
eigenvalues = sorted(eigvalsh(L_graph))        # 升序

# 谱间隙 = 第2小特征值（Fiedler值）
spectral_gap = eigenvalues[1]                  # 越大→图社区结构越清晰
```

### Step 5：计算 PPR 信任分
```
# 行归一化权重矩阵
W_norm[i,j] = L_hat[i,j] / Σ_j L_hat[i,j]

# PPR迭代（幂迭代法）
v = ones(K) / K                                # 初始均匀分布
seed = ones(K) / K                             # 均匀种子
for iter in range(max_iter):
  v_new = (1 - β) * seed + β * W_norm.T @ v
  if ||v_new - v||_2 < tol: break
  v = v_new

# 归一化信任分
trust_score = v / v.sum()
```

## 3.4 可视化规格

### 图3A：净化前后谱间隙对比（条形图）
```
类型:   分组条形图（2组：净化前 vs 净化后）
X轴:    两组（"Before Purification" / "After Purification"）
Y轴:    Fiedler值（谱间隙），标签"Spectral Gap (Fiedler Value)"
颜色:   净化前=灰色，净化后=蓝色
误差条: 5次重复的std
标注:   箭头标出谱间隙增大的幅度
解读文字: 谱间隙越大→图社区结构越清晰→良性/Byzantine分离度越高
```

### 图3B：PPR信任分条形图（核心图）
```
类型:   条形图
X轴:    客户端编号 0..19，按 trust_score 降序排列
Y轴:    PPR信任分 trust_score[k]，标签"PPR Trust Score"
颜色:   良性客户端（真实标签）→ 蓝色
        Byzantine客户端 → 红色
标注:   在图内画水平虚线 = 均值信任分（1/K）
        标注"Benign avg: XX" / "Byzantine avg: XX"
期望效果: 蓝色条明显高于红色条，形成视觉上的明显分隔
```

### 图3C：连通性差异对比（净化前 vs 净化后）
```
类型:   双组条形图或折线对比图
指标:   W_benign（良性内部连通性）/ W_cross（跨组连通性）/ connectivity_gap
X轴:    两组（净化前 / 净化后）
Y轴:    平均边权值
期望效果: 净化后 connectivity_gap 显著增大
```

### 图3D（可选）：PPR收敛动态图
```
类型:   多曲线折线图
X轴:    PPR迭代次数（0, 5, 10, 20, 50, 收敛）
Y轴:    各客户端的信任分 v_k
线条:   良性客户端=蓝色细线，Byzantine=红色细线
期望效果: 随迭代，蓝线上升，红线下降，最终收敛到分离状态
```

## 3.5 预期结果
- 净化后 spectral_gap 比净化前提升 > 50%
- PPR信任分：良性客户端均值 >> Byzantine客户端均值
- connectivity_gap 在净化后显著增大
- 即使ALIE攻击（最隐蔽），PPR仍能有效区分身份

---

# 代码框架建议（三个实验共用）

## 文件结构
```
experiments/
├── setup.py              # 数据加载、Dirichlet划分、模型定义
├── federated_sim.py      # 模拟FD轮次、生成logits、Byzantine攻击
├── metrics.py            # Jaccard相似度、行能量、谱间隙、PPR
├── rpca_solver.py        # ADMM求解低秩分解
├── exp1_lowrank.py       # Observation 1 实验
├── exp2_rowsparse.py     # Observation 2 实验
├── exp3_conductance.py   # Observation 3 实验
└── plot_utils.py         # 统一绘图风格
```

## 关键函数签名

```python
# setup.py
def get_dirichlet_partition(dataset, K, alpha, seed):
    """返回 K 个客户端的数据索引列表"""

def get_public_dataset(dataset, N_pub, seed):
    """返回公共数据集（无标签）"""

# federated_sim.py
def train_local_models(partitions, arch_list, epochs, device):
    """返回 K 个训练好的本地模型"""

def generate_logits(models, public_dataset, device):
    """返回 logits: List[Tensor(N_pub, C)]"""

def apply_attack(logits_k, attack_type, benign_stats=None):
    """对单个客户端logit施加攻击，返回攻击后logit"""

# metrics.py
def jaccard_similarity_matrix(logits_list, kappa=5):
    """输入: List[Tensor(N_pub, C)]，输出: ndarray(K, K)"""

def row_energy(E):
    """输入: ndarray(K, K)，输出: ndarray(K,) 每行L2范数"""

def effective_rank(S):
    """输入: 相似度矩阵，输出: 标量有效秩"""

def spectral_gap(W):
    """输入: 加权邻接矩阵，输出: Fiedler值"""

def personalized_pagerank(W, beta=0.85, tol=1e-6, max_iter=200):
    """输入: 邻接矩阵，输出: ndarray(K,) 信任分"""

# rpca_solver.py
def solve_rpca_admm(S, lam, rho=1.0, max_iter=500, tol=1e-4):
    """输入: 观测矩阵S，输出: (L_hat, E_hat)"""
```

## 攻击实现参考

```python
def apply_attack(logits, attack_type, **kwargs):
    if attack_type == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return logits + torch.randn_like(logits) * sigma

    elif attack_type == 'label_flip':
        # 将最高概率类的logit置为最低，最低置为最高
        adv = logits.clone()
        top_cls = logits.argmax(dim=1)
        bot_cls = logits.argmin(dim=1)
        for n in range(len(logits)):
            adv[n, top_cls[n]], adv[n, bot_cls[n]] = \
                logits[n, bot_cls[n]], logits[n, top_cls[n]]
        return adv

    elif attack_type == 'targeted':
        target = kwargs.get('target_class', 0)
        adv = torch.zeros_like(logits)
        adv[:, target] = 10.0
        return adv

    elif attack_type == 'alie':
        # 需要先收集良性统计量
        mu = kwargs['benign_mean']    # (N_pub, C)
        sigma = kwargs['benign_std']  # (N_pub, C)
        z_scale = kwargs.get('z_scale', 1.5)
        return mu + z_scale * sigma
```

---

# 三个实验的输出汇总

| 实验 | 核心指标 | 主图 | 辅助图 |
|------|---------|------|-------|
| Exp1 | 有效秩、累积能量占比 | 奇异值衰减曲线（4条α） | 相似度矩阵热图 |
| Exp2 | Byzantine能量集中度 | 行能量条形图（2×2攻击） | E矩阵热图、统计表 |
| Exp3 | 谱间隙、PPR信任分 | PPR条形图（蓝红对比） | 连通性对比、收敛动态 |

*版本: 2026-04-28*
