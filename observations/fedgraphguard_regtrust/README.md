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


## 如何从真实联邦训练导出 logits

你需要在**真实训练流程**中导出每个客户端在公共数据集上的 logits（不是 mock）。

### 目标文件格式

- `benign_logits.npy`：shape `(K, N_pub, C)`，表示 K 个客户端在公共集上的 logits。
- `obs1_logits.npz`：多个 key（如 `alpha_0.5_seed_1`），每个 value shape `(K, N_pub, C)`。

### 推荐导出流程

1. 按你的实验配置完成一轮真实联邦训练（或收敛后的模型）。
2. 固定公共数据集（与 observation 一致）。
3. 逐客户端前向公共数据集，拼接得到每个客户端的 `N_pub x C` logits。
4. `np.save('benign_logits.npy', logits_all_clients)`。
5. 对 Observation-1，不同 `alpha` 与 `seed` 重复上述过程，然后将每次结果写入 `obs1_logits.npz` 对应 key。



### 一键真实导出脚本（推荐）

已提供 `export_real_logits.py`，可直接复用当前项目 federated 流程（DataManager / Client / Server / FD 主循环）训练后导出：

- `benign_logits.npy`
- `obs1_logits.npz`

运行示例：

```bash
python -m observations.fedgraphguard_regtrust.export_real_logits   --dataset cifar10   --num-clients 20   --clients-per-round 20   --num-rounds 20   --local-epochs 3   --data-root /autodl-tmp/t3code/t3_code/analysis/data   --benign-alpha 0.5   --benign-seed 42   --obs1-alphas 0.1 0.3 0.5 1.0   --obs1-seeds 0 1 2 3 4   --out-benign-npy observations/real_inputs/benign_logits.npy   --out-obs1-npz observations/real_inputs/obs1_logits.npz   --run-tag expA   --auto-suffix
```

如果你本地已有 CIFAR-10（例如 `analysis/data/cifar-10-batches-py`），请将 `--data-root` 设为其上级目录（即 `analysis/data`）。

默认**不会自动下载**（export 脚本默认不加 `--allow-download`），若目录中文件缺失会直接报错；仅当你显式加 `--allow-download` 时才会下载。

为避免同名文件覆盖，建议使用 `--run-tag`（例如 `expA`）或打开 `--auto-suffix` 自动追加 `_v1/_v2`。

导出完成后，直接运行 observation 脚本即可。

### 运行 observation

```bash
python -m observations.fedgraphguard_regtrust.observation1_lowrank   --logits-npz path/to/obs1_logits.npz   --out-dir observations/outputs/obs1

python -m observations.fedgraphguard_regtrust.observation2_rowsparse   --benign-logits-npy path/to/benign_logits.npy   --out-dir observations/outputs/obs2

python -m observations.fedgraphguard_regtrust.observation3_conductance_ppr   --benign-logits-npy path/to/benign_logits.npy   --out-dir observations/outputs/obs3
```


### 常见报错排查

- `ModuleNotFoundError: No module named config`
  - 已在 `export_real_logits.py` 内自动将项目根目录加入 `sys.path`，支持你在非仓库根目录启动命令。

- `libgomp: Invalid value for environment variable OMP_NUM_THREADS`
  - `export_real_logits.py` 启动时会自动检查 `OMP_NUM_THREADS`，若值非法则回退到 `1`。
