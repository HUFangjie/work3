# FedGraphGuard / ReG-Trust Observation Scripts

> 说明：FedGraphGuard 与 ReG-Trust 指同一套防御思路。本目录中的 observation 代码是**独立实现**，不耦合 `defense_fedgraphguard.py`。

## 文件说明

- `observation1_lowrank.py`：Observation 1（良性相似度矩阵低秩性）
- `observation2_rowsparse.py`：Observation 2（Byzantine 扰动的行稀疏性）
- `observation3_conductance_ppr.py`：Observation 3（净化后导电率/连通性与 PPR 信任对齐）
- `plotting.py`：绘图代码（与实验逻辑独立）
- `common_metrics.py`：可复用度量与算法（Jaccard/RPCA/PPR/谱间隙等）

## 运行示例

```bash
python -m observations.fedgraphguard_regtrust.observation1_lowrank \
  --logits-npz path/to/obs1_logits.npz \
  --out-dir observations/outputs/obs1

python -m observations.fedgraphguard_regtrust.observation2_rowsparse \
  --benign-logits-npy path/to/benign_logits.npy \
  --byzantine-ids 0 1 2 3 \
  --out-dir observations/outputs/obs2

python -m observations.fedgraphguard_regtrust.observation3_conductance_ppr \
  --benign-logits-npy path/to/benign_logits.npy \
  --byzantine-ids 0 1 2 3 4 5 \
  --out-dir observations/outputs/obs3
```

## 输入数据格式

- `obs1_logits.npz`: key 形如 `alpha_0.5_seed_1`，值 shape `(K, N_pub, C)`
- `benign_logits.npy`: shape `(K, N_pub, C)`
