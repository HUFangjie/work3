import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare metrics between benign and attack runs."
    )
    parser.add_argument(
        "--benign_csv",
        type=str,
        required=True,
        help="Path to metrics_rounds.csv of benign run",
    )
    parser.add_argument(
        "--attack_csv",
        type=str,
        required=True,
        help="Path to metrics_rounds.csv of attack run",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./figs_compare",
        help="Directory to save comparison figures and summary",
    )
    return parser.parse_args()


def load_metrics(csv_path, label):
    df = pd.read_csv(csv_path)
    if "round" not in df.columns:
        raise ValueError(f"'round' column not found in {csv_path}")
    df = df.copy()
    df["setting"] = label
    return df


def plot_metric(df_all, metric, out_path):
    """
    df_all: concat of benign/attack, with columns: round, setting, metric
    """
    plt.figure(figsize=(6, 4))

    for setting, group in df_all.groupby("setting"):
        group = group.sort_values("round")

        # 转成 numpy，避免 pandas 在 x[:, None] 时抛错
        x = group["round"].to_numpy()
        y = group[metric].to_numpy()

        plt.plot(x, y, marker="o", label=setting)

    plt.xlabel("Round")
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def summarize_metrics(df_benign, df_attack, metrics, out_path):
    """
    生成一个 summary.txt，对比：
      - 每个指标在“最后一轮”的数值（benign vs attack）
      - 每个指标在“所有轮次的平均值”（benign vs attack）
    """
    lines = []
    lines.append("Summary of benign vs attack runs\n")
    lines.append("-" * 60 + "\n")

    # 最后一轮：各自取各自的最大 round
    last_round_benign = int(df_benign["round"].max())
    last_round_attack = int(df_attack["round"].max())

    lines.append(f"Benign last round = {last_round_benign}\n")
    lines.append(f"Attack last round = {last_round_attack}\n\n")

    df_b_last = df_benign[df_benign["round"] == last_round_benign]
    df_a_last = df_attack[df_attack["round"] == last_round_attack]

    lines.append("=== Final round metrics ===\n")
    for m in metrics:
        if m not in df_benign.columns or m not in df_attack.columns:
            continue
        v_b = float(df_b_last[m].mean())
        v_a = float(df_a_last[m].mean())
        diff = v_a - v_b
        lines.append(
            f"{m}: benign={v_b:.6f}, attack={v_a:.6f}, "
            f"attack-benign={diff:.6f}\n"
        )

    lines.append("\n=== Mean over all rounds ===\n")
    for m in metrics:
        if m not in df_benign.columns or m not in df_attack.columns:
            continue
        mean_b = float(df_benign[m].mean())
        mean_a = float(df_attack[m].mean())
        diff = mean_a - mean_b
        lines.append(
            f"{m}: benign_mean={mean_b:.6f}, attack_mean={mean_a:.6f}, "
            f"attack-benign={diff:.6f}\n"
        )

    with open(out_path, "w") as f:
        f.writelines(lines)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 读两次实验的 csv
    df_benign = load_metrics(args.benign_csv, label="benign")
    df_attack = load_metrics(args.attack_csv, label="attack")
    df_all = pd.concat([df_benign, df_attack], ignore_index=True)

    # 2) 你关心的指标列表（新的 ID-only 指标）
    metrics_to_plot = [
        "test_accuracy",
        "test_avg_conf",
        "test_ece",
        "test_ks",
    ]

    # 过滤掉当前 csv 里不存在的列（避免 KeyError）
    metrics_available = [m for m in metrics_to_plot if m in df_all.columns]
    if not metrics_available:
        raise ValueError(
            "No matching metrics found in csv. "
            "Check metrics_rounds.csv column names."
        )

    print("Will plot metrics:", metrics_available)

    # 3) 每个指标画一张“benign vs attack”的折线图
    for metric in metrics_available:
        out_path = os.path.join(args.out_dir, f"{metric}.png")
        plot_metric(df_all, metric, out_path)

    # 4) 生成一个 summary.txt，方便快速比较攻击前后效果
    summary_path = os.path.join(args.out_dir, "summary.txt")
    summarize_metrics(df_benign, df_attack, metrics_available, summary_path)
    print(f"Summary saved to: {summary_path}")

    print(f"All figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
