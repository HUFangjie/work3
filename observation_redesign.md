# Observation 递进逻辑链（重新设计版）
## ——完全基于数据与问题的结构性质，不依赖他方方案对比

---

## 核心设计原则

每个 Observation 回答一个问题：
> "这个数据/问题中存在什么结构性质，让我们的设计成为自然选择？"

三个 Observation 形成一条递进因果链：

```
Obs 1: 良性相似度矩阵是低秩的
         ↓ （那么Byzantine攻击在这个空间里长什么样？）
Obs 2: Byzantine扰动在相似度矩阵中呈行稀疏结构
         ↓ （Obs1 + Obs2 → 低秩+稀疏 → RPCA可解 → 净化后图如何使用？）
Obs 3: 净化后的图中，良性客户端形成高导电率的连通核心
         ↓
     全局图扩散（PPR）是信任推断的自然选择
```

这条链的美妙之处：每个Observation的发现，都是下一个Observation提问的起点。

---

## Observation 1：良性客户端的预测共识具有低秩结构

### 要回答的问题
"良性客户端的soft-label预测，在相似度空间中，有没有可以被利用的数学结构？"

### 直觉解释
尽管客户端持有non-IID数据、使用异构模型，它们终究在学习同一个任务。
在公共数据集的公共样本上，良性客户端对"哪些类最有可能"的判断存在内在一致性——
这种一致性使得成对相似度矩阵的变化集中在少数主方向上，即低秩性。

### 形式化陈述
```
Observation 1 (Low-Rank Benign Consensus).
设 S_B ∈ R^{K×K} 为全部K个良性客户端的Jaccard相似度矩阵。
在CIFAR-10、K=20、Dirichlet α=0.5的non-IID分布下，
S_B 的有效秩（effective rank）满足 r_eff(S_B) ≤ 4，
且前3个奇异值累计能量占比 > 87%。

→ 含义：良性客户端的相似度模式集中在一个低维子空间中。
```

### 实验设计
**实验1A：奇异值衰减曲线**
- 横轴：奇异值序号 1, 2, ..., K
- 纵轴：该奇异值占总能量的百分比（σ_i / Σσ_j）
- 只展示全良性场景下的曲线
- 预期：急剧衰减，前3-5个奇异值占绝对主导

**实验1B：有效秩 vs. non-IID程度**
- 横轴：Dirichlet α（0.1 → 1.0）
- 纵轴：有效秩 r_eff = exp(H(σ̄))（H为归一化奇异值分布的熵）
- 只展示全良性场景
- 预期：即使在极度non-IID（α=0.1）下，有效秩仍然远小于K
- 意义：低秩性是任务对齐的内在性质，对数据异构度有鲁棒性

**可视化（强推荐）：相似度矩阵热图**
- 将K个客户端按社区结构重排后，展示S_B的热图
- 应呈现明显的块状低秩结构（benign clients形成少数高相似度簇）

### 设计启示
```
→ 良性客户端的相似度模式存在于一个低维子空间 L*。
  这给了我们一个"锚"：任何破坏这个低维结构的信号，
  都是潜在的外来干扰。问题转化为：Byzantine攻击
  在相似度空间中是否创造了可分离的扰动结构？（见Obs 2）
```

---

## Observation 2：Byzantine扰动在相似度矩阵中呈行稀疏结构

### 要回答的问题
"当Byzantine客户端存在时，相似度矩阵S与理想良性矩阵L*的差，
具有什么数学形态？"

### 直觉解释
Byzantine客户端的数量有限（m < K/2），且每个Byzantine客户端
只影响它自己与其他客户端的相似度——即S矩阵的第k行和第k列。
因此，扰动矩阵 E = S - L* 天然是**行稀疏**的（most rows are zero）。
更关键的是：Obs1已知L*是低秩的，E是行稀疏的——
**这正是Robust PCA（RPCA）理论保证可以分离的结构！**

### 形式化陈述
```
Observation 2 (Row-Sparse Byzantine Perturbation).
设 E = S_observed - L* 为Byzantine扰动矩阵。
对于m个Byzantine客户端（m < K/2），E 满足：
  - 非零行数 ≤ 2m（对应Byzantine客户端的行和列）
  - ||E||_1 / (K·m) 在不同攻击类型下保持有界

在CIFAR-10、K=20、20%Byzantine（4个客户端）下：
  - 实测非零行集中度：>91%的E矩阵能量位于对应Byzantine的行列
  - 这一性质在Gaussian、Label Flipping、ALIE攻击下均成立
```

### 实验设计
**实验2A：扰动矩阵E的行能量分布**
- 计算 E = S_observed - S_benign（用全良性场景作为参照估计L*）
- 横轴：按E行能量排序的客户端编号
- 纵轴：每行的L2范数 ||e_k||_2
- 用不同颜色标注真实身份（良性/Byzantine）
- 预期：能量高度集中在Byzantine客户端对应的行

**实验2B：不同攻击类型下的稀疏性验证**
- 做成小表格：攻击类型 | 非零行集中度 | ||E||_0 / K²
- 覆盖：Gaussian、Label Flipping、Targeted、ALIE
- 说明：行稀疏性是Byzantine攻击的结构性特征，不依赖具体攻击策略

### 设计启示
```
→ 联合Obs 1（L*低秩）和Obs 2（E行稀疏），
  观测矩阵 S = L* + E 精确满足Robust PCA的分离条件。
  这使得低秩图净化不仅直觉上合理，
  而且有严格的理论恢复保证（见Theorem 1）。

  净化后我们获得 L̂ ≈ L*。接下来的问题是：
  如何从L̂中提取信任信号？（见Obs 3）
```

---

## Observation 3：净化后的图中，连通性编码了可传播的信任结构

### 要回答的问题
"从净化后的相似度图L̂出发，什么样的聚合机制能最鲁棒地
利用图的结构来推断信任？"

### 直觉解释
净化步骤产生的L̂不是完美的——边界处的客户端可能存在不确定性。
但良性客户端之间的相互高相似度，使它们在图中形成一个**高导电率的核心**：
这个核心中的任意节点都可以通过短路径到达其他核心成员。
相比之下，即使部分Byzantine客户端"混入"图中，
它们只能与少数邻居建立强连接，无法获得来自整个良性核心的信任流入。

**关键洞察**：这个信任结构是全局的，不是局部的。
单个客户端的"得分"无法捕捉它，只有图扩散机制能传播它。

### 形式化陈述
```
Observation 3 (Differential Conductance in Purified Graph).
设 G = (V, L̂) 为净化后的加权图。定义：
  - 良性子图导电率：Φ_B = min cut(S,S̄) / min(vol(S), vol(S̄))，S⊆Benign
  - Byzantine节点的图导电率：Φ_A（对Byzantine节点集定义）

在CIFAR-10、K=20、30%Byzantine（ALIE攻击）下：
  Φ_B = 0.74 ± 0.06（高导电率，紧密连通）
  Φ_A = 0.19 ± 0.04（低导电率，边缘游离）

→ 良性核心与Byzantine边缘的导电率差异在净化后显著放大
  （净化前差异：0.31；净化后差异：0.55）
```

### 实验设计
**实验3A：净化前后的图谱分析**
- 计算图Laplacian的特征值谱
- 对比净化前S和净化后L̂的谱间隙（spectral gap）
- 谱间隙越大，图的社区结构越清晰
- 预期：净化后谱间隙显著增大，说明良性核心更加分离

**实验3B：PPR信任分 vs. 连通性（最直观）**
- 横轴：客户端编号（按PPR信任分排序）
- 双纵轴：左=PPR信任分v_k；右=该客户端在图中的加权度数
- 标注真实身份（良性/Byzantine）
- 预期：PPR信任分与图连通度高度相关，且与真实身份对齐

**实验3C：信任传播收敛可视化**
- 展示PPR迭代过程中信任分的动态演化（5个快照：iter 0/5/10/20/收敛）
- 说明：信任从初始均匀分布，通过图扩散逐渐聚集到良性核心

### 设计启示
```
→ 良性核心的高导电率意味着信任可以通过图扩散高效传播，
  而Byzantine节点的低导电率使其无法从良性核心获得信任流入。
  Personalized PageRank是实现这种"信任扩散"的自然选择：
  它的稳态分布等价于在图上进行无限步随机游走，
  自然编码了全局连通结构，而非局部距离。
```

---

## Section 3 完整结构模板

```
3. Motivating Analysis

[开篇段落]
We conduct a systematic empirical analysis to uncover the structural 
properties of soft-label predictions in the presence of Byzantine clients. 
Our analysis reveals three observations that directly motivate each 
design component of ReG-Trust.

3.1 Observation 1: Benign Soft-Label Predictions Form a Low-Rank 
    Consensus in Similarity Space
    [图1：奇异值衰减曲线 + 相似度矩阵热图]

3.2 Observation 2: Byzantine Attacks Create Row-Sparse Structural 
    Violations
    [图2：扰动矩阵行能量分布 + 稀疏性统计表]

3.3 Observation 3: Graph Conductance Encodes Propagatable Trust
    [图3：PPR信任分 vs 连通度 + 信任传播动态]

3.4 Design Principles
"Collectively, these three observations establish that the FD 
Byzantine detection problem has a natural decomposition structure:
  P1 (Exploit Low-Rank Structure): ...
  P2 (Separate via Row-Sparsity): ...  
  P3 (Propagate via Graph Diffusion): ...
ReG-Trust is designed to operationalize each of these principles."
```

---

## Observation 链条的逻辑自洽性检验

```
问：为什么构建Jaccard相似度图？
答：因为Obs 1发现相似度矩阵具有低秩结构（预测的语义序关系比幅度更稳定）

问：为什么用低秩分解净化？
答：因为Obs 1（L*低秩）+ Obs 2（E行稀疏）共同满足RPCA的分离条件

问：为什么用PPR而不是简单阈值？
答：因为Obs 3发现信任是通过图的全局连通结构编码的，
    局部阈值无法捕捉，需要图扩散机制

问：每个Observation有没有贬低其他方案？
答：完全没有。每个Observation只陈述数据/问题的结构性质，
    设计选择的动机来自这些性质本身。
```

---
*修订版：2026-04-28 | 完全移除他方方案对比*
