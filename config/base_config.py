# config/base_config.py
"""
Base configuration for the Fed-T3-FD project.

This file defines a Python dictionary `BASE_CONFIG` that serves as the
default configuration. It is meant to be overridden/updated by:
  1) Experiment-specific YAML/JSON/Python files
  2) Command-line arguments parsed in main.py / utils/config_parser.py
"""

from typing import Dict, Any


BASE_CONFIG: Dict[str, Any] = {
    # Random seed & device
    "seed": 42,
    "device": "cuda",  # or "cpu"

    # Dataset & partitioning
    "data_config": {
        "dataset": "tiny_imagenet",         # ["fmnist", "cifar10", "tiny_imagenet", ...]
        # "data_root": "./data/MedMNIST",
        "data_root": "./data",
        "num_clients": 10,
        "public_ratio": 0.3,          # Tiny-ImageNet FD needs broad public coverage
        "partition_type": "dirichlet",  # ["dirichlet", "shard", "label_separation"]
        "dirichlet_alpha": 2.0,
        "num_shards": 40,             # for shard partition
        "label_separation_classes_per_client": 2,  # for extreme Non-IID
        "batch_size_private": 128,
        "batch_size_public": 128,
        "num_workers": 8,
    },

    # Model configuration
    "model_config": {
        "name": "wrn28_4_tiny",   # ["fmnist_cnn", "cifar10_cnn", "wrn28_4_tiny", "wrn28_8_tiny", "resnet18_tiny", "resnet34_tiny"...]
        "num_classes": 200,      # e.g., CIFAR10: 10; Tiny-ImageNet: 200; pathmnist: 9
        "input_channels": 3,    # FEMNIST: 1, CIFAR10: 3
        "width_mult": 1.0,
        "dropout": 0.1,
    },

    # Federated distillation protocol configuration
    "fd_config": {
        "offload_clients_to_cpu": True,
        "public_logits_amp": True,
        # Speed knobs (especially useful on Tiny-ImageNet)
        # - public_logits_micro_bs: micro-batch for teacher inference on public data
        # - uplink_logits_dtype: cast uploaded logits on CPU to save memory/"communication"
        # - public_batches_per_round: limit public batches per round (0 => full epoch)
        "public_logits_micro_bs": 64,
        "uplink_logits_dtype": "float32",  # {"float32","float16"}
        "public_batches_per_round": 0,
        "num_rounds": 400,
        "clients_per_round": 10,
        "local_epochs": 2,
        "optimizer": "sgd",
        "lr": 3e-2,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "server_lr": 2e-2,
        "lr_schedule": "cosine",
        "lr_warmup_rounds": 10,
        "lr_warmup_factor": 0.2,
        "lr_min": 1e-4,
        "server_lr_min": 1e-4,
        # Knowledge distillation temperature
        "kd_temperature": 4.0,
        # Coefficient for distillation loss vs. supervised loss (if any)
        "kd_alpha": 1.0,
    },

    # Attack configuration
    "attack_config": {
        "enabled": False,
        "name": "none",  # ["none","t3","gaussian","label_flip","topk","impersonation","naive_sharpening","manipulating_kd","fed_ace","fed_oca"]
        "malicious_client_fraction": 0.2,
        "fixed_malicious_clients": [0,1],  # list of client ids, or None

        # --- T3-specific hyperparameters ---
        "t3": {
            "rho": 0.4,          # budget quantile for hard sample selection
            "lambda_epistemic": 1.0,
            "lambda_stealth": 1.0,
            "lambda_align": 1.0,
            "epsilon": 2.0,      # l_inf bound for logit perturbation
            "pgd_steps": 40,
            "pgd_step_size": 0.2,
            "history_window": 1,  # for global alignment
            "tta_type": "strong",  # ["weak", "strong", "custom"]
            "debug": True,
        },

        # Gaussian logit attack (baseline)
        "gaussian": {
            "sigma": 0.1,
            "mode": "targeted_mean_shift",  # ["targeted_mean_shift","max_prediction_masking"]
            "scale_with_span": True,
            "span_scale": 0.35,
            "span_bias": 0.10,
            "min_scale": 0.5,
            "max_scale": None,
            "target_offset": 1,
            "mask_strength": 1.0,
            "mix_with_original": 0.15,
        },

        # Label flip / targeted attack (baseline)
        "label_flip": {
            "flip_probability": 0.5,
            "use_hard_target": True,
            "target_logit": 10.0,
            "non_target_logit": -10.0,
            "fixed_target_offset": 1,  # target = (source + offset) % K
            "amplitude_scale": 1.5,    # dynamic scaling from original logit range
            "amplitude_bias": 0.5,     # additive boost on amplitude
            "min_amplitude": 2.0,      # lower bound for malicious confidence
            "mix_with_original": 0.2,  # keep soft structure for KD gradients
        },
        "topk": {
        "k": 3,
        "delta": -10.0,       # paper describes -10 after normalization
        "normalize": True,
        "norm_low": -10.0,
        "norm_high": 10.0,
        "rank_weighted": True,      # stronger suppression on top-1 than top-2/3
        "promote_non_top1": True,   # actively boost a non-top1 target class
        "promote_strength": 0.8,    # boost magnitude coefficient
        },
        "impersonation": {
        # no hyperparams needed in the paper's definition
        },

        # --- New attacks (calibration/confidence baselines) ---
        "naive_sharpening": {
            "temperature": 0.5,   # < 1 => sharper (preserves argmax)
            "max_abs_logit": None,
        },
        "manipulating_kd": {
            "tau": 5.0,               # KL(reference || poisoned) temperature
            "num_ascent_steps": 80,   # projected Lagrangian ascent iterations
            "attack_lr": 0.35,        # normalized ascent step as a fraction of stealth radius
            "dual_lr": 0.1,           # dual ascent step size for stealth constraints
            "init_ratio": 0.10,       # small feasible random jitter to avoid KL=0 stationarity
            "boundary_ratio": 1.0,    # initialize directional candidates on stealth boundary
            "num_restarts": 4,        # random boundary candidates before ascent
            "grad_eps": 1e-8,         # gradient normalization floor
            "eps": 1e-12,
        },
        "fed_ace": {
            "tau": 0.7,           # confidence threshold
            "gamma": 0.5,         # how strongly to push up/down confidence
            "min_margin": 1e-3,
            "max_margin": 50.0,
            "l2_budget": None,
        },
        "fed_oca": {
            "gamma": 0.8,         # increase margin by (1+gamma)
            "min_margin": 1e-3,
            "max_margin": 50.0,
            "l2_budget": None,
        },
    },

    # Defense configuration
    "defense_config": {
        "enabled": False,
        "name": "none",  # ["none","cronus","entropy_clip","mkrum","trimean","fedmdr","fedtgd"]
        "none": {},

        "entropy_clip": {
            "max_entropy": 2.5,  # example threshold
        },
        "cronus": {
            # Placeholder hyperparams; to be refined for defense
            "temperature": 1.0,
            "gamma": 2.0,
            "trimming_fraction": 0.2,  # 每个样本丢掉最远 20% 的客户端
            "min_clients_kept": 2,
        },

        # --- New defenses (robust aggregation for distillation logits) ---
        "mkrum": {
            "f": 1,               # assumed Byzantine clients
            "m": 1,               # multi-krum selection count (1 => krum)
            "use_squared": True,
        },
        "trimean": {
            "q": 0.25,            # quartile for Q1/Q3
        },
        "fedmdr": {
            "trim_ratio": 0.2,    # trim bottom clients by batch accuracy
            "softmax_temp": 1.0,  # softmax temperature over accuracies
            "max_iter": 50,       # Weiszfeld iterations
            "eps": 1e-6,
            "min_clients_kept": 2,
        },
        "fedtgd": {
            "topk": 5,            # k for top-k truncation/features
            "dbscan_eps": 0.5,
            "dbscan_min_samples": 2,
            "cosine_keep_ratio": 1.0,
            "fallback": "mean",   # ["mean","median"]
        },
    },

    # Evaluation / calibration configuration (仅 ID 测试集相关)
    "evaluation_config": {
        "eval_every": 5,              # evaluate every N rounds on val/test
        "calibration_num_bins": 15,   # num bins for ECE / KS
    },

    # Logging / checkpointing
    "logging_config": {
        "log_dir": "./logs",
        "exp_name": "tiny_imagenet_wrn28_4_tuned",
        "save_checkpoint_every": 50,
        "print_every": 1,
        "use_tensorboard": True,
    },
}


def get_base_config() -> Dict[str, Any]:
    """
    Return a deep copy of the base configuration dictionary.

    This function exists to ensure that callers get an independent
    config object that they can modify without affecting the global constant.
    """
    import copy
    return copy.deepcopy(BASE_CONFIG)
