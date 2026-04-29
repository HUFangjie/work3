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

## 直接脚本运行（不通过 -m）

如果你在该目录下直接执行脚本（例如 `python observation1_lowrank.py ...`），
现在也支持；脚本会自动处理导入路径。

```bash
cd observations/fedgraphguard_regtrust
python observation1_lowrank.py --logits-npz path/to/obs1_logits.npz --out-dir ../../observations/outputs/obs1
```


## 如何获取 `obs1_logits.npz` 和 `benign_logits.npy`

你有两种方式：

1. **真实实验数据（推荐）**
   - 从你自己的联邦训练流程里导出每个客户端在公共数据集上的 logits。
   - `obs1_logits.npz`：包含多个 key（例如 `alpha_0.5_seed_1`），每个 value 的 shape 都是 `(K, N_pub, C)`。
   - `benign_logits.npy`：shape 为 `(K, N_pub, C)`。

2. **先用 mock 数据跑通流程（快速验证脚本）**
   - 运行：

```bash
python -m observations.fedgraphguard_regtrust.generate_mock_inputs   --out-dir observations/mock_inputs
```

   - 输出：
     - `observations/mock_inputs/obs1_logits.npz`
     - `observations/mock_inputs/benign_logits.npy`

   - 然后可直接运行三个 observation：

```bash
python -m observations.fedgraphguard_regtrust.observation1_lowrank   --logits-npz observations/mock_inputs/obs1_logits.npz   --out-dir observations/outputs/obs1

python -m observations.fedgraphguard_regtrust.observation2_rowsparse   --benign-logits-npy observations/mock_inputs/benign_logits.npy   --out-dir observations/outputs/obs2

python -m observations.fedgraphguard_regtrust.observation3_conductance_ppr   --benign-logits-npy observations/mock_inputs/benign_logits.npy   --out-dir observations/outputs/obs3
```

> 注意：mock 数据仅用于检查代码链路是否打通，不代表真实实验结论。
