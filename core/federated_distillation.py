# core/federated_distillation.py
"""
Federated distillation main training loop (with optional attacks/defenses).

Server-side global student:
  - clients upload logits on public data
  - server aggregates logits (defense)
  - server distills a REAL student model on public data
  - evaluation always on REAL student

Extra support:
  - impersonation attack requires benign logits: collect benign first, then malicious.
  - reliability raw dump: save per-sample confidence/pred/target/correct + per-bin stats
    to .npz per evaluation round for later multi-panel reliability plots.

Added:
  (1) Stealth W1 (Wasserstein-1 / Earth Mover Distance):
      W1( entropy(malicious uploaded logits in current round),
          entropy(benign reference history) )
      - computed only when malicious uploads exist and benign history has enough samples
      - logged to <log_dir>/<exp_name>_stealth_rounds.csv
      - TensorBoard: stealth/stealth_w1_teacher

  (2) Client efficiency & communication overhead (per-client per-round):
      - total time spent in compute_public_logits (includes forward + attack + cpu transfer)
      - uplink communication size of uploaded logits (bytes/MB)
      - malicious breakdown: diagnosis / TTA / PGD time (from attack.last_overhead)
      - logged to <log_dir>/<exp_name>_client_overhead_rounds.csv
      - TensorBoard: client_overhead/* (means over selected clients)
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple
from collections import deque
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.client import Client
from core.server import Server
from core.metrics import evaluate_model, evaluate_with_calibration_and_raw
from core.utils import Timer
from attacks.utils import compute_entropy

_IMP_CTX_AVAILABLE = True
try:
    from attacks.impersonation_context import set_benign_pool, clear_benign_pool
except Exception:
    _IMP_CTX_AVAILABLE = False


# =========================================================
# Paths / CSV helpers
# =========================================================
def _get_metrics_csv_path(config: Dict) -> str:
    log_cfg = config.get("logging_config", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    exp_name = log_cfg.get("exp_name", "debug_run")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{exp_name}_metrics_rounds.csv")


def _get_stealth_csv_path(config: Dict) -> str:
    log_cfg = config.get("logging_config", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    exp_name = log_cfg.get("exp_name", "debug_run")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{exp_name}_stealth_rounds.csv")


def _get_client_overhead_csv_path(config: Dict) -> str:
    log_cfg = config.get("logging_config", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    exp_name = log_cfg.get("exp_name", "debug_run")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{exp_name}_client_overhead_rounds.csv")


def _get_aurc_client_scores_csv_path(config: Dict) -> str:
    log_cfg = config.get("logging_config", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "aurc_client_scores.csv")


def _get_aurc_rc_points_csv_path(config: Dict) -> str:
    log_cfg = config.get("logging_config", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "aurc_risk_coverage_points.csv")


def _get_aurc_per_round_csv_path(config: Dict) -> str:
    log_cfg = config.get("logging_config", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "aurc_per_round.csv")


def _get_aurc_tail20_curve_csv_path(config: Dict) -> str:
    log_cfg = config.get("logging_config", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "aurc_tail20_curve.csv")


def _get_reliability_dir(config: Dict) -> str:
    """
    Store reliability raw files under:
      <log_dir>/<exp_name>/reliability/
    """
    log_cfg = config.get("logging_config", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    exp_name = log_cfg.get("exp_name", "debug_run")
    out_dir = os.path.join(log_dir, exp_name, "reliability")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _save_reliability_npz(
    out_dir: str,
    round_idx: int,
    split: str,
    raw: Dict,
    extra: Dict,
) -> str:
    """
    Save raw reliability data to a compact npz file.
    """
    fname = f"{split}_round{round_idx:04d}.npz"
    path = os.path.join(out_dir, fname)

    payload = dict(raw)
    # attach extra scalar meta (loss/acc/ece/ks, etc.)
    for k, v in extra.items():
        payload[f"meta_{k}"] = np.array(v)

    np.savez_compressed(path, **payload)
    return path


def _init_metrics_csv_if_needed(csv_path: str) -> None:
    """
    Initialize (or upgrade) the per-round metrics CSV.

    Backward compatibility:
      - If the file already exists but does not contain `test_coe` column,
        we will upgrade it in-place by inserting the column (filled with NaN
        for historical rows).
    """
    header = [
        "round",
        "test_loss",
        "test_accuracy",
        "test_avg_conf",
        "test_coe",
        "test_ece",
        "test_ks",
        "test_aurc",
    ]

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                old_header = next(reader, None)
                old_rows = list(reader)
            need_upgrade = (old_header is not None) and (
                ("test_coe" not in old_header) or ("test_aurc" not in old_header)
            )
            if need_upgrade:
                idx = {k: i for i, k in enumerate(old_header)}
                new_rows = []
                for r in old_rows:
                    def getv(name: str, default: str = "nan") -> str:
                        i = idx.get(name, None)
                        if i is None or i >= len(r):
                            return default
                        return r[i]
                    new_rows.append([
                        getv("round", "0"),
                        getv("test_loss", "nan"),
                        getv("test_accuracy", "nan"),
                        getv("test_avg_conf", "nan"),
                        getv("test_coe", "nan"),
                        getv("test_ece", "nan"),
                        getv("test_ks", "nan"),
                        getv("test_aurc", "nan"),
                    ])
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(new_rows)
            return
        except Exception:
            # fall through to re-init
            pass

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)



def _append_round_metrics(
    csv_path: str,
    round_idx: int,
    test_metrics: Dict[str, float],
) -> None:
    # KEEP original metrics schema unchanged
    row = [
        round_idx,
        test_metrics.get("loss", 0.0),
        test_metrics.get("accuracy", 0.0),
        test_metrics.get("avg_confidence", 0.0),
        float(test_metrics.get("confidence_on_error", float("nan"))),
        test_metrics.get("ece", 0.0),
        test_metrics.get("ks_confidence", 0.0),
        test_metrics.get("aurc", float("nan")),
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _init_stealth_csv_if_needed(csv_path: str) -> None:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "teacher_entropy_mean", "stealth_w1_teacher", "history_size", "malicious_clients"])


def _append_round_stealth(
    csv_path: str,
    round_idx: int,
    teacher_entropy_mean: float,
    stealth_w1_teacher: float,
    history_size: int,
    malicious_clients: int,
) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round_idx, teacher_entropy_mean, stealth_w1_teacher, history_size, malicious_clients])


def _save_tail20_aurc_summary(config: Dict, csv_path: str) -> None:
    """Compute and save avg AURC over last 20% rounds from metrics CSV."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rnd = int(float(r.get("round", "0")))
                aurc = float(r.get("test_aurc", "nan"))
            except Exception:
                continue
            if np.isfinite(aurc):
                rows.append((rnd, aurc))
    if len(rows) == 0:
        return
    rows = sorted(rows, key=lambda x: x[0])
    n = len(rows)
    tail_n = max(1, int(np.ceil(0.2 * n)))
    tail = rows[-tail_n:]
    tail_avg = float(np.mean([x[1] for x in tail]))

    log_cfg = config.get("logging_config", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    exp_name = log_cfg.get("exp_name", "debug_run")
    out_path = os.path.join(log_dir, f"{exp_name}_aurc_tail20.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp_name", "num_eval_points", "tail20_points", "tail20_start_round", "tail20_end_round", "tail20_avg_aurc"])
        w.writerow([exp_name, n, tail_n, tail[0][0], tail[-1][0], f"{tail_avg:.10f}"])


def compute_risk_coverage_curve(trust_scores, is_byzantine):
    trust_scores = np.asarray(trust_scores, dtype=np.float64)
    is_byzantine = np.asarray(is_byzantine).astype(int)
    K = len(trust_scores)
    ranking = np.argsort(-trust_scores)
    coverage_list, risk_list, retained_byz_list = [], [], []
    byz_count = 0
    for r, idx in enumerate(ranking, start=1):
        byz_count += int(is_byzantine[idx])
        coverage_list.append(float(r / K))
        risk_list.append(float(byz_count / r))
        retained_byz_list.append(int(byz_count))
    coverage_auc = np.array([0.0] + coverage_list, dtype=np.float64)
    risk_auc = np.array([0.0] + risk_list, dtype=np.float64)
    aurc = float(np.trapz(risk_auc, coverage_auc))
    return coverage_list, risk_list, aurc, ranking, retained_byz_list


def _init_client_overhead_csv_if_needed(csv_path: str) -> None:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    header = [
        "round",
        "client_id",
        "is_malicious",
        "num_public_batches",
        "uplink_logits_MB",
        "t_total_s",
        "t_attack_total_s",
        "t_diag_s",
        "t_tta_s",
        "t_pgd_s",
        "t_forward_est_s",
        "hard_cnt_sum",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def _append_client_overhead_row(csv_path: str, row: List[Any]) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _init_aurc_tracking_csvs_if_needed(config: Dict) -> None:
    paths_and_headers = [
        (_get_aurc_client_scores_csv_path(config), ["dataset", "attack", "method", "seed", "round", "client_id", "trust_score", "is_byzantine"]),
        (_get_aurc_rc_points_csv_path(config), ["dataset", "attack", "method", "seed", "round", "coverage", "risk", "rank", "retained_clients", "retained_byzantine"]),
        (_get_aurc_per_round_csv_path(config), ["dataset", "attack", "method", "seed", "round", "aurc"]),
        (_get_aurc_tail20_curve_csv_path(config), ["dataset", "attack", "method", "seed", "coverage", "risk_mean", "risk_std", "aurc_mean", "aurc_std", "tail20_start_round", "tail20_end_round"]),
    ]
    for p, h in paths_and_headers:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        with open(p, "w", newline="") as f:
            csv.writer(f).writerow(h)


def _tensor_nbytes(x: Any) -> int:
    try:
        if isinstance(x, torch.Tensor):
            return int(x.numel() * x.element_size())
    except Exception:
        pass
    return 0


# =========================================================
# Stealth W1 helper (1D)
# =========================================================
def wasserstein_1d(a: torch.Tensor, b: torch.Tensor, quantiles: int = 256) -> float:
    """
    Approximate 1D Wasserstein-1 distance via quantiles:
        W1 ≈ mean_p |Q_a(p) - Q_b(p)|
    """
    with torch.no_grad():
        a = a.detach().flatten()
        b = b.detach().flatten()
        a = a[torch.isfinite(a)]
        b = b[torch.isfinite(b)]
        if a.numel() < 2 or b.numel() < 2:
            return float("nan")
        a = a.float().cpu()
        b = b.float().cpu()
        q = torch.linspace(0.0, 1.0, int(max(16, quantiles)), dtype=torch.float32)
        qa = torch.quantile(a, q)
        qb = torch.quantile(b, q)
        return float(torch.mean(torch.abs(qa - qb)).item())


# =========================================================
# Attack / role helpers
# =========================================================


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _scheduled_lr(
    fd_cfg: Dict[str, Any],
    base_lr: float,
    round_idx: int,
    num_rounds: int,
    min_lr_key: str = "lr_min",
) -> float:
    """Round-level LR schedule controlled by fd_config.

    Supported:
      - "none": keep base LR.
      - "cosine": warmup + cosine decay to lr_min.
      - "step": multiply by lr_gamma at each milestone in lr_milestones.
    """
    schedule = str(fd_cfg.get("lr_schedule", "none")).lower()
    if schedule in ("", "none", "constant"):
        return float(base_lr)

    warmup_rounds = max(0, int(fd_cfg.get("lr_warmup_rounds", 0)))
    warmup_factor = float(fd_cfg.get("lr_warmup_factor", 0.2))
    if warmup_rounds > 0 and round_idx <= warmup_rounds:
        alpha = float(round_idx) / float(max(1, warmup_rounds))
        return float(base_lr) * (warmup_factor + (1.0 - warmup_factor) * alpha)

    if schedule == "cosine":
        min_lr = float(fd_cfg.get(min_lr_key, fd_cfg.get("lr_min", 1e-4)))
        denom = max(1, num_rounds - warmup_rounds)
        progress = min(1.0, max(0.0, (round_idx - warmup_rounds) / float(denom)))
        return min_lr + 0.5 * (float(base_lr) - min_lr) * (1.0 + math.cos(math.pi * progress))

    if schedule == "step":
        gamma = float(fd_cfg.get("lr_gamma", 0.1))
        milestones = [int(m) for m in fd_cfg.get("lr_milestones", [])]
        drops = sum(1 for m in milestones if round_idx >= m)
        return float(base_lr) * (gamma ** drops)

    raise ValueError(f"Unsupported lr_schedule: {schedule}")


def _needs_impersonation_attack(config: Dict) -> bool:
    attack_cfg = config.get("attack_config", {})
    if not bool(attack_cfg.get("enabled", False)):
        return False
    name = str(attack_cfg.get("name", "none")).lower()
    return name in ("impersonation", "impersonate")


def _is_client_malicious(client: Client) -> bool:
    return bool(getattr(getattr(client, "attack", None), "is_malicious", False))


def _split_benign_malicious(
    clients: Dict[int, Client],
    selected_clients: List[int],
) -> Tuple[List[int], List[int]]:
    benign_ids: List[int] = []
    malicious_ids: List[int] = []
    for cid in selected_clients:
        c = clients[int(cid)]
        if _is_client_malicious(c):
            malicious_ids.append(int(cid))
        else:
            benign_ids.append(int(cid))
    return benign_ids, malicious_ids


# =========================================================
# Main FD loop
# =========================================================
def run_federated_distillation(
    config: Dict,
    server: Server,
    clients: Dict[int, Client],
    public_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    logger,
    writer,
) -> None:
    fd_cfg = config["fd_config"]
    eval_cfg = config["evaluation_config"]

    num_rounds = int(fd_cfg["num_rounds"])
    clients_per_round = int(fd_cfg["clients_per_round"])

    # Speed: limit number of public batches per round (0 => use full public_loader)
    public_batches_per_round = int(fd_cfg.get("public_batches_per_round", 0))
    public_batches_per_round = max(0, public_batches_per_round)
    num_clients_total = len(clients)
    client_ids = list(clients.keys())

    device = server.device
    ce_loss = nn.CrossEntropyLoss()

    # --- output files ---
    metrics_csv_path = _get_metrics_csv_path(config)
    _init_metrics_csv_if_needed(metrics_csv_path)

    stealth_csv_path = _get_stealth_csv_path(config)
    _init_stealth_csv_if_needed(stealth_csv_path)

    client_overhead_csv_path = _get_client_overhead_csv_path(config)
    _init_client_overhead_csv_if_needed(client_overhead_csv_path)

    reliability_dir = _get_reliability_dir(config)
    _init_aurc_tracking_csvs_if_needed(config)
    aurc_scores_csv = _get_aurc_client_scores_csv_path(config)
    aurc_points_csv = _get_aurc_rc_points_csv_path(config)
    aurc_round_csv = _get_aurc_per_round_csv_path(config)
    aurc_tail20_curve_csv = _get_aurc_tail20_curve_csv_path(config)

    # --- stealth config (benign history) ---
    attack_cfg = config.get("attack_config", {}) if isinstance(config.get("attack_config", {}), dict) else {}
    t3_cfg = attack_cfg.get("t3", {}) if isinstance(attack_cfg.get("t3", {}), dict) else {}

    benign_history_window = int(t3_cfg.get("history_window", 2048))
    benign_history_window = max(64, benign_history_window)

    stealth_min_ref = int(t3_cfg.get("stealth_w1_min_ref", min(128, benign_history_window)))
    stealth_min_ref = max(2, min(stealth_min_ref, benign_history_window))

    stealth_quantiles = int(t3_cfg.get("stealth_w1_quantiles", 256))
    stealth_quantiles = max(16, stealth_quantiles)

    benign_entropy_hist: "deque[torch.Tensor]" = deque()
    benign_entropy_hist_n: int = 0

    logger.info(
        f"Starting federated distillation (server-side global student): "
        f"rounds={num_rounds}, clients_per_round={clients_per_round}, "
        f"num_clients_total={num_clients_total}"
    )

    rng = np.random.RandomState(config.get("seed", 42))
    seed = int(config.get("seed", 42))
    dataset_name = str(config.get("data_config", {}).get("dataset", "unknown"))
    attack_name = str(config.get("attack_config", {}).get("name", "none"))
    defense_name = str(config.get("defense_config", {}).get("name", "none"))
    method_name = "UniformTrust" if defense_name.lower() == "none" else defense_name
    rc_round_records: List[Dict[str, Any]] = []
    aurc_round_records: List[Tuple[int, float]] = []
    eval_every = int(eval_cfg.get("eval_every", 5))
    calib_num_bins = int(eval_cfg.get("calibration_num_bins", 15))
    base_client_lr = float(fd_cfg.get("lr", 0.01))
    base_server_lr = float(fd_cfg.get("server_lr", base_client_lr))

    need_impersonation = _needs_impersonation_attack(config)
    need_manipulating_kd = (
        bool(config.get("attack_config", {}).get("enabled", False))
        and str(config.get("attack_config", {}).get("name", "none")).lower() == "manipulating_kd"
    )
    if need_impersonation and (not _IMP_CTX_AVAILABLE):
        raise RuntimeError(
            "Impersonation attack is enabled, but attacks/impersonation_context.py "
            "cannot be imported. Please add it (set_benign_pool / clear_benign_pool)."
        )

    for round_idx in range(1, num_rounds + 1):
        logger.info(f"=== Round {round_idx}/{num_rounds} ===")

        client_lr = _scheduled_lr(fd_cfg, base_client_lr, round_idx, num_rounds, min_lr_key="lr_min")
        server_lr = _scheduled_lr(fd_cfg, base_server_lr, round_idx, num_rounds, min_lr_key="server_lr_min")
        for c in clients.values():
            _set_optimizer_lr(c.optimizer, client_lr)
        _set_optimizer_lr(server.optimizer, server_lr)
        if round_idx == 1 or round_idx % max(1, eval_every) == 0:
            logger.info(f"Round {round_idx}: lr(client)={client_lr:.6g}, lr(server)={server_lr:.6g}")

        # 1) Sample clients (teachers) for this round
        selected_clients: List[int] = list(
            rng.choice(
                client_ids,
                size=min(clients_per_round, num_clients_total),
                replace=False,
            )
        )
        logger.info(f"Selected clients: {selected_clients}")

        benign_ids, malicious_ids = _split_benign_malicious(clients, selected_clients)

        # Per-client accumulators (per round)
        per_client_batches: Dict[int, int] = {int(cid): 0 for cid in selected_clients}
        per_client_uplink_bytes: Dict[int, int] = {int(cid): 0 for cid in selected_clients}
        per_client_t_total: Dict[int, float] = {int(cid): 0.0 for cid in selected_clients}

        # Attack breakdown (malicious only; summed over batches)
        per_client_t_attack_total: Dict[int, float] = {int(cid): 0.0 for cid in selected_clients}
        per_client_t_diag: Dict[int, float] = {int(cid): 0.0 for cid in selected_clients}
        per_client_t_tta: Dict[int, float] = {int(cid): 0.0 for cid in selected_clients}
        per_client_t_pgd: Dict[int, float] = {int(cid): 0.0 for cid in selected_clients}
        per_client_hard_cnt_sum: Dict[int, int] = {int(cid): 0 for cid in selected_clients}

        # 2) One epoch over public_loader: teachers -> aggregate logits -> train server student
        round_student_kd_losses: list[float] = []
        teacher_ent_means: list[float] = []
        stealth_w1_batches: list[float] = []

        for batch_idx, (x_pub, y_pub) in enumerate(public_loader):
            if public_batches_per_round > 0 and batch_idx >= public_batches_per_round:
                break
            client_logits: Dict[int, torch.Tensor] = {}

            if need_manipulating_kd and len(malicious_ids) > 0:
                raw_logits_by_cid: Dict[int, torch.Tensor] = {}
                for cid in selected_clients:
                    client = clients[int(cid)]
                    t0 = time.perf_counter()
                    logits = client.compute_public_logits(
                        x_public=x_pub,
                        y_public=y_pub,
                        round_idx=round_idx,
                        apply_attack=False,
                    )
                    dt = time.perf_counter() - t0

                    per_client_t_total[int(cid)] += float(dt)
                    per_client_batches[int(cid)] += 1
                    raw_logits_by_cid[int(cid)] = logits

                selected_order = [int(cid) for cid in selected_clients]
                all_client_logits = torch.stack([raw_logits_by_cid[cid].float() for cid in selected_order], dim=0)
                selected_pos = {cid: pos for pos, cid in enumerate(selected_order)}
                malicious_positions = [selected_pos[int(cid)] for cid in malicious_ids]

                attack_owner = clients[int(malicious_ids[0])].attack
                t0 = time.perf_counter()
                with torch.enable_grad():
                    shared_malicious_logits = attack_owner.build_shared_malicious_logits(
                        all_client_logits=all_client_logits,
                        malicious_ids=malicious_positions,
                        round_idx=round_idx,
                    ).detach().float().cpu()
                attack_dt = time.perf_counter() - t0

                for cid in selected_order:
                    if cid in malicious_ids:
                        uploaded = shared_malicious_logits.clone()
                        client_logits[cid] = uploaded
                        per_client_uplink_bytes[cid] += _tensor_nbytes(uploaded)
                        per_client_t_total[cid] += float(attack_dt)

                        atk = getattr(clients[cid], "attack", None)
                        if atk is not None:
                            atk.last_overhead = dict(getattr(attack_owner, "last_overhead", {}) or {})
                            atk.last_overhead["round"] = int(round_idx)
                            atk.last_overhead["t_total_s"] = float(attack_dt)
                            atk.last_overhead["t_pgd_s"] = float(attack_dt)

                        per_client_t_attack_total[cid] += float(attack_dt)
                        per_client_t_pgd[cid] += float(attack_dt)
                        oh = getattr(atk, "last_overhead", None) if atk is not None else None
                        if isinstance(oh, dict):
                            per_client_hard_cnt_sum[cid] += int(oh.get("hard_cnt", 0))
                    else:
                        uploaded = raw_logits_by_cid[cid]
                        client_logits[cid] = uploaded
                        per_client_uplink_bytes[cid] += _tensor_nbytes(uploaded)

            elif need_impersonation:
                # benign first
                for cid in benign_ids:
                    client = clients[int(cid)]
                    t0 = time.perf_counter()
                    logits = client.compute_public_logits(x_public=x_pub, y_public=y_pub, round_idx=round_idx)
                    dt = time.perf_counter() - t0

                    per_client_t_total[int(cid)] += float(dt)
                    per_client_uplink_bytes[int(cid)] += _tensor_nbytes(logits)
                    per_client_batches[int(cid)] += 1

                    client_logits[int(cid)] = logits

                set_benign_pool({cid: client_logits[cid] for cid in benign_ids})

                # malicious second
                for cid in malicious_ids:
                    client = clients[int(cid)]
                    t0 = time.perf_counter()
                    logits = client.compute_public_logits(x_public=x_pub, y_public=y_pub, round_idx=round_idx)
                    dt = time.perf_counter() - t0

                    per_client_t_total[int(cid)] += float(dt)
                    per_client_uplink_bytes[int(cid)] += _tensor_nbytes(logits)
                    per_client_batches[int(cid)] += 1

                    # pull breakdown from client.attack.last_overhead if available
                    atk = getattr(client, "attack", None)
                    oh = getattr(atk, "last_overhead", None) if atk is not None else None
                    if isinstance(oh, dict) and int(oh.get("round", -999)) == int(round_idx):
                        per_client_t_attack_total[int(cid)] += float(oh.get("t_total_s", 0.0))
                        per_client_t_diag[int(cid)] += float(oh.get("t_diag_s", 0.0))
                        per_client_t_tta[int(cid)] += float(oh.get("t_tta_s", 0.0))
                        per_client_t_pgd[int(cid)] += float(oh.get("t_pgd_s", 0.0))
                        per_client_hard_cnt_sum[int(cid)] += int(oh.get("hard_cnt", 0))

                    client_logits[int(cid)] = logits

                clear_benign_pool()
            else:
                for cid in selected_clients:
                    client = clients[int(cid)]
                    t0 = time.perf_counter()
                    logits = client.compute_public_logits(x_public=x_pub, y_public=y_pub, round_idx=round_idx)
                    dt = time.perf_counter() - t0

                    per_client_t_total[int(cid)] += float(dt)
                    per_client_uplink_bytes[int(cid)] += _tensor_nbytes(logits)
                    per_client_batches[int(cid)] += 1

                    if _is_client_malicious(client):
                        atk = getattr(client, "attack", None)
                        oh = getattr(atk, "last_overhead", None) if atk is not None else None
                        if isinstance(oh, dict) and int(oh.get("round", -999)) == int(round_idx):
                            per_client_t_attack_total[int(cid)] += float(oh.get("t_total_s", 0.0))
                            per_client_t_diag[int(cid)] += float(oh.get("t_diag_s", 0.0))
                            per_client_t_tta[int(cid)] += float(oh.get("t_tta_s", 0.0))
                            per_client_t_pgd[int(cid)] += float(oh.get("t_pgd_s", 0.0))
                            per_client_hard_cnt_sum[int(cid)] += int(oh.get("hard_cnt", 0))

                    client_logits[int(cid)] = logits

            # -------------------------------------------------
            # Stealth W1 (malicious entropy vs benign history)
            # -------------------------------------------------
            with torch.no_grad():
                # update benign history from benign uploads in this batch
                if len(benign_ids) > 0:
                    ent_list: List[torch.Tensor] = []
                    for cid in benign_ids:
                        if cid in client_logits:
                            ent_list.append(compute_entropy(client_logits[cid]).detach().float().cpu())
                    if len(ent_list) > 0:
                        ent_benign = torch.cat(ent_list, dim=0)
                        benign_entropy_hist.append(ent_benign)
                        benign_entropy_hist_n += int(ent_benign.numel())

                        # maintain FIFO by sample count
                        while benign_entropy_hist_n > benign_history_window and len(benign_entropy_hist) > 0:
                            left = benign_entropy_hist[0]
                            left_n = int(left.numel())
                            overflow = benign_entropy_hist_n - benign_history_window
                            if overflow >= left_n:
                                benign_entropy_hist.popleft()
                                benign_entropy_hist_n -= left_n
                            else:
                                benign_entropy_hist[0] = left[overflow:]
                                benign_entropy_hist_n -= overflow
                                break

                # compute W1 if malicious exists and reference is sufficient
                if len(malicious_ids) > 0 and benign_entropy_hist_n >= stealth_min_ref and len(benign_entropy_hist) > 0:
                    mal_list: List[torch.Tensor] = []
                    for cid in malicious_ids:
                        if cid in client_logits:
                            mal_list.append(compute_entropy(client_logits[cid]).detach().float().cpu())
                    if len(mal_list) > 0:
                        ent_mal = torch.cat(mal_list, dim=0)
                        ref = torch.cat(list(benign_entropy_hist), dim=0)
                        w1 = wasserstein_1d(ent_mal, ref, quantiles=stealth_quantiles)
                        if np.isfinite(w1):
                            stealth_w1_batches.append(float(w1))
            # Aggregate logits for student distillation
            # If clients upload FP16 logits to save communication, cast to float32 here for stable aggregation.
            for _cid, _logits in list(client_logits.items()):
                if isinstance(_logits, torch.Tensor) and _logits.dtype != torch.float32:
                    client_logits[_cid] = _logits.float()
            teacher_logits = server.aggregate_logits(client_logits, y_public=y_pub)

            with torch.no_grad():
                ent_teacher = compute_entropy(teacher_logits.to(device))
                teacher_ent_means.append(float(ent_teacher.mean().item()))

            loss_val = server.distill_student_on_public(x_pub, teacher_logits)
            round_student_kd_losses.append(loss_val)

        avg_student_kd_loss = float(np.mean(round_student_kd_losses)) if round_student_kd_losses else 0.0
        logger.info(f"Round {round_idx} avg STUDENT KD loss on public data: {avg_student_kd_loss:.4f}")
        writer.add_scalar("train/student_kd_loss", avg_student_kd_loss, global_step=round_idx)

        avg_teacher_ent = float(np.mean(teacher_ent_means)) if teacher_ent_means else float("nan")
        if teacher_ent_means:
            logger.info(f"Round {round_idx} mean teacher entropy on public data: {avg_teacher_ent:.4f}")
            writer.add_scalar("train/teacher_entropy", avg_teacher_ent, global_step=round_idx)

        # round-level stealth W1: mean over batches
        if len(malicious_ids) > 0:
            avg_stealth_w1 = float(np.mean(stealth_w1_batches)) if len(stealth_w1_batches) > 0 else float("nan")
            logger.info(
                f"Round {round_idx} Stealth W1 (malicious entropy vs benign history): "
                f"{avg_stealth_w1:.6f} (benign_hist={benign_entropy_hist_n}, malicious_clients={len(malicious_ids)})"
            )
            writer.add_scalar("stealth/stealth_w1_teacher", avg_stealth_w1, global_step=round_idx)
            _append_round_stealth(
                stealth_csv_path,
                round_idx,
                avg_teacher_ent,
                avg_stealth_w1,
                int(benign_entropy_hist_n),
                int(len(malicious_ids)),
            )
        else:
            # still write a row for alignment (stealth_w1=nan)
            _append_round_stealth(
                stealth_csv_path,
                round_idx,
                avg_teacher_ent,
                float("nan"),
                int(benign_entropy_hist_n),
                0,
            )

        # per-client overhead rows
        benign_total_list: List[float] = []
        malicious_total_list: List[float] = []
        malicious_diag_list: List[float] = []
        malicious_tta_list: List[float] = []
        malicious_pgd_list: List[float] = []

        for cid in selected_clients:
            cid = int(cid)
            client = clients[cid]
            is_mal = 1 if _is_client_malicious(client) else 0

            nb = int(per_client_batches.get(cid, 0))
            uplink_MB = float(per_client_uplink_bytes.get(cid, 0)) / (1024.0 * 1024.0)

            t_total = float(per_client_t_total.get(cid, 0.0))
            t_atk = float(per_client_t_attack_total.get(cid, 0.0))
            t_diag = float(per_client_t_diag.get(cid, 0.0))
            t_tta = float(per_client_t_tta.get(cid, 0.0))
            t_pgd = float(per_client_t_pgd.get(cid, 0.0))
            hard_sum = int(per_client_hard_cnt_sum.get(cid, 0))

            t_forward = max(t_total - t_atk, 0.0) if is_mal else t_total

            _append_client_overhead_row(
                client_overhead_csv_path,
                [
                    round_idx,
                    cid,
                    is_mal,
                    nb,
                    uplink_MB,
                    t_total,
                    t_atk,
                    t_diag,
                    t_tta,
                    t_pgd,
                    t_forward,
                    hard_sum,
                ],
            )

            if is_mal:
                malicious_total_list.append(t_total)
                malicious_diag_list.append(t_diag)
                malicious_tta_list.append(t_tta)
                malicious_pgd_list.append(t_pgd)
            else:
                benign_total_list.append(t_total)

        # TensorBoard: quick Attack vs Benign mean curves
        if len(benign_total_list) > 0:
            writer.add_scalar("client_overhead/benign_t_total_mean", float(np.mean(benign_total_list)), round_idx)
        if len(malicious_total_list) > 0:
            writer.add_scalar("client_overhead/malicious_t_total_mean", float(np.mean(malicious_total_list)), round_idx)
        if len(malicious_diag_list) > 0:
            writer.add_scalar("client_overhead/malicious_t_diag_mean", float(np.mean(malicious_diag_list)), round_idx)
        if len(malicious_tta_list) > 0:
            writer.add_scalar("client_overhead/malicious_t_tta_mean", float(np.mean(malicious_tta_list)), round_idx)
        if len(malicious_pgd_list) > 0:
            writer.add_scalar("client_overhead/malicious_t_pgd_mean", float(np.mean(malicious_pgd_list)), round_idx)

        # 2.5) Save trust score / RC points / per-round AURC for this round
        round_cids = [int(cid) for cid in selected_clients]
        is_byz = np.array([1 if _is_client_malicious(clients[cid]) else 0 for cid in round_cids], dtype=int)
        K = len(round_cids)
        with open(aurc_scores_csv, "a", newline="") as f_sc, open(aurc_points_csv, "a", newline="") as f_pt, open(aurc_round_csv, "a", newline="") as f_ar:
            w_sc, w_pt, w_ar = csv.writer(f_sc), csv.writer(f_pt), csv.writer(f_ar)

            if method_name == "UniformTrust":
                num_trials = 100
                risk_acc = np.zeros(K, dtype=np.float64)
                for _ in range(num_trials):
                    perm = rng.permutation(K)
                    byz_count = 0
                    for r, j in enumerate(perm, start=1):
                        byz_count += int(is_byz[j])
                        risk_acc[r - 1] += byz_count / r
                risk_list = (risk_acc / float(num_trials)).tolist()
                coverage_list = [float((r + 1) / K) for r in range(K)]
                aurc = float(np.trapz(np.array([0.0] + risk_list), np.array([0.0] + coverage_list)))
                trust_scores = np.ones(K, dtype=np.float64)
                retained_byz_list = [int(round(risk_list[i] * (i + 1))) for i in range(K)]
            else:
                # Fallback trust proxy when explicit trust scores are unavailable in current defense:
                # lower compute time -> slightly higher trust.
                trust_scores = np.array([1.0 / (1.0 + float(per_client_t_total.get(cid, 0.0))) for cid in round_cids], dtype=np.float64)
                coverage_list, risk_list, aurc, ranking, retained_byz_list = compute_risk_coverage_curve(trust_scores, is_byz)
                # reorder lists to ranking order for consistent rank writing
                round_cids = [round_cids[int(i)] for i in ranking]
                is_byz = is_byz[ranking]
                trust_scores = trust_scores[ranking]

            for i, cid in enumerate(round_cids):
                w_sc.writerow([dataset_name, attack_name, method_name, seed, round_idx, cid, f"{float(trust_scores[i]):.10f}", int(is_byz[i])])
            for i, (cov, risk) in enumerate(zip(coverage_list, risk_list), start=1):
                rb = int(retained_byz_list[i - 1])
                w_pt.writerow([dataset_name, attack_name, method_name, seed, round_idx, f"{cov:.10f}", f"{risk:.10f}", i, i, rb])
                rc_round_records.append({"round": int(round_idx), "coverage": float(cov), "risk": float(risk)})
            w_ar.writerow([dataset_name, attack_name, method_name, seed, round_idx, f"{aurc:.10f}"])
            aurc_round_records.append((int(round_idx), float(aurc)))

        # 3) Optional: supervised training on private data (refresh teachers)
        local_epochs = int(fd_cfg.get("local_epochs", 0))
        if local_epochs > 0:
            logger.info(
                f"Round {round_idx}: running {local_epochs} local epochs "
                f"on private data for selected clients (teacher refresh)."
            )
            for cid in selected_clients:
                client = clients[int(cid)]
                avg_local_loss = client.train_on_private(local_epochs=local_epochs)
                logger.info(f"  Client {cid}: avg private loss = {avg_local_loss:.4f}")

        # 4) Periodic evaluation on validation/test sets using REAL global student
        do_eval = (round_idx % eval_every == 0) or (round_idx == num_rounds)
        if do_eval:
            logger.info(f"Evaluating GLOBAL STUDENT at round {round_idx}...")

            student_model = server.get_student()

            with Timer() as t_eval:
                val_metrics = evaluate_model(student_model, val_loader, device, ce_loss)

                # One-pass: metrics + raw reliability values on TEST
                test_metrics, test_raw = evaluate_with_calibration_and_raw(
                    student_model,
                    test_loader,
                    device,
                    num_bins=calib_num_bins,
                    criterion=ce_loss,
                )

            logger.info(
                f"[Round {round_idx}] "
                f"Val loss: {val_metrics['loss']:.4f}, "
                f"Val acc: {val_metrics['accuracy']*100:.2f}%, "
                f"Test loss: {test_metrics['loss']:.4f}, "
                f"Test acc: {test_metrics['accuracy']*100:.2f}%, "
                f"Test AvgConf: {test_metrics['avg_confidence']:.4f}, "
                f"Test CoE: {test_metrics.get('confidence_on_error', float('nan')):.4f}, "
                f"Test ECE: {test_metrics['ece']:.4f}, "
                f"Test KS: {test_metrics['ks_confidence']:.4f}, "
                f"(eval_time={t_eval.elapsed:.2f}s)"
            )

            # TensorBoard scalars
            writer.add_scalar("val/loss", val_metrics["loss"], round_idx)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], round_idx)

            writer.add_scalar("test/loss", test_metrics["loss"], round_idx)
            writer.add_scalar("test/accuracy", test_metrics["accuracy"], round_idx)
            writer.add_scalar("test/ece", test_metrics["ece"], round_idx)
            writer.add_scalar("test/avg_confidence", test_metrics["avg_confidence"], round_idx)
            writer.add_scalar("test/confidence_on_error", test_metrics.get("confidence_on_error", float("nan")), round_idx)
            writer.add_scalar("test/ks_confidence", test_metrics["ks_confidence"], round_idx)
            writer.add_scalar("test/aurc", test_metrics.get("aurc", float("nan")), round_idx)

            _append_round_metrics(metrics_csv_path, round_idx, test_metrics)

            # dump reliability raw
            saved_path = _save_reliability_npz(
                out_dir=reliability_dir,
                round_idx=round_idx,
                split="test",
                raw=test_raw,
                extra=test_metrics,
            )
            logger.info(f"[ReliabilityDump] saved: {saved_path}")

    _save_tail20_aurc_summary(config=config, csv_path=metrics_csv_path)
    if len(aurc_round_records) > 0 and len(rc_round_records) > 0:
        rounds_sorted = sorted(aurc_round_records, key=lambda x: x[0])
        tail_n = max(1, int(np.ceil(0.2 * len(rounds_sorted))))
        tail_rounds = [r for r, _ in rounds_sorted[-tail_n:]]
        tail_set = set(tail_rounds)
        tail_aurc_vals = [v for r, v in rounds_sorted if r in tail_set]
        tail_start, tail_end = min(tail_rounds), max(tail_rounds)
        pts = [p for p in rc_round_records if p["round"] in tail_set]
        cov_keys = sorted(set(float(p["coverage"]) for p in pts))
        with open(aurc_tail20_curve_csv, "a", newline="") as f:
            w = csv.writer(f)
            for cov in cov_keys:
                rv = [float(p["risk"]) for p in pts if float(p["coverage"]) == cov]
                if not rv:
                    continue
                w.writerow([
                    dataset_name, attack_name, method_name, seed,
                    f"{cov:.10f}", f"{float(np.mean(rv)):.10f}", f"{float(np.std(rv)):.10f}",
                    f"{float(np.mean(tail_aurc_vals)):.10f}", f"{float(np.std(tail_aurc_vals)):.10f}",
                    tail_start, tail_end
                ])
    logger.info("Federated distillation completed.")
