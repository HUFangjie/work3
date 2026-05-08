from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from obs_utils import resolve_out_dir, save_json
else:
    from .obs_utils import resolve_out_dir, save_json


def load_logits(path):
    z = np.load(path)
    if z.ndim != 3:
        raise ValueError(f"Expected logits shape (K,N_pub,C), got {z.shape}")
    return z


def jaccard_similarity_matrix(logits, kappa=5):
    K, N, C = logits.shape
    kk = min(kappa, C)
    topk = np.argpartition(logits, kth=C - kk, axis=2)[:, :, -kk:]
    S = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            vals = []
            for a, b in zip(topk[i], topk[j]):
                sa, sb = set(a.tolist()), set(b.tolist())
                vals.append(len(sa & sb) / len(sa | sb))
            S[i, j] = S[j, i] = float(np.mean(vals))
    return S


def apply_gaussian_attack(logits, sigma, rng):
    return logits + rng.normal(0.0, sigma, size=logits.shape)


def apply_label_flip_swap_attack(logits):
    # label_flip_swap: swap highest and lowest logit per sample
    adv = logits.copy()
    top = np.argmax(logits, axis=1)
    bot = np.argmin(logits, axis=1)
    rows = np.arange(logits.shape[0])
    adv[rows, top], adv[rows, bot] = logits[rows, bot], logits[rows, top]
    return adv


def apply_targeted_attack(logits, target_class):
    adv = np.zeros_like(logits)
    adv[:, target_class] = 10.0
    return adv


def apply_alie_attack(benign_logits_all, benign_ids, z_scale):
    mu = np.mean(benign_logits_all[benign_ids], axis=0)
    std = np.std(benign_logits_all[benign_ids], axis=0)
    return mu + z_scale * std


def build_observed_logits(benign_logits, byz_ids, attack, args):
    obs = benign_logits.copy()
    K = benign_logits.shape[0]
    benign_ids = [i for i in range(K) if i not in byz_ids]
    rng = np.random.default_rng(args.seed)

    for b in byz_ids:
        if attack == "gaussian":
            obs[b] = apply_gaussian_attack(obs[b], args.gaussian_sigma, rng)
        elif attack == "label_flip":
            obs[b] = apply_label_flip_swap_attack(obs[b])
        elif attack == "targeted":
            obs[b] = apply_targeted_attack(obs[b], args.target_class)
        elif attack == "alie":
            obs[b] = apply_alie_attack(benign_logits, benign_ids, args.alie_z_scale)
        else:
            raise ValueError(attack)
    return obs


def compute_row_column_energy(E):
    K = E.shape[0]
    en = np.zeros(K)
    for i in range(K):
        en[i] = np.sqrt(np.sum(E[i, :] ** 2) + np.sum(E[:, i] ** 2))
    return en


def compute_mask_energy_concentration(E, byz_ids):
    K = E.shape[0]
    M = np.zeros_like(E)
    for i in range(K):
        for j in range(K):
            M[i, j] = 1.0 if (i in byz_ids or j in byz_ids) else 0.0
    num = np.sum((M * E) ** 2)
    den = np.sum(E ** 2) + 1e-12
    return float(num / den), M


def _color_ticks(ax, byz_ids):
    for tick, idx in zip(ax.get_xticklabels(), range(len(ax.get_xticklabels()))):
        if idx in byz_ids: tick.set_color('red')
    for tick, idx in zip(ax.get_yticklabels(), range(len(ax.get_yticklabels()))):
        if idx in byz_ids: tick.set_color('red')


def _save(fig, base):
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(base + '.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logits-npy', required=True)
    p.add_argument('--out-dir', default='observations/outputs/obs2')
    p.add_argument('--kappa', type=int, default=5)
    p.add_argument('--byzantine-ids', nargs='+', type=int, default=[0, 1])
    p.add_argument('--attack', default='alie', choices=['gaussian', 'label_flip', 'targeted', 'alie', 'all'])
    p.add_argument('--target-class', type=int, default=0)
    p.add_argument('--gaussian-sigma', type=float, default=1.0)
    p.add_argument('--alie-z-scale', type=float, default=1.5)
    p.add_argument('--eps', type=float, default=1e-6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sort-energy', action='store_true')
    args = p.parse_args()

    benign_logits = load_logits(args.logits_npy)
    K, N, C = benign_logits.shape
    byz = sorted(set(args.byzantine_ids))
    if len(byz) == 0: raise ValueError('byzantine_ids cannot be empty')
    if max(byz) >= K or min(byz) < 0: raise ValueError('byzantine_ids out of range')
    benign_ids = [i for i in range(K) if i not in byz]

    out_dir = resolve_out_dir(args.out_dir)

    attacks = ['gaussian', 'label_flip', 'targeted', 'alie'] if args.attack == 'all' else [args.attack]

    L_star = jaccard_similarity_matrix(benign_logits, kappa=args.kappa)

    for attack in attacks:
        S_obs = jaccard_similarity_matrix(build_observed_logits(benign_logits, byz, attack, args), kappa=args.kappa)
        E = S_obs - L_star
        abs_E = np.abs(E)

        energy = compute_row_column_energy(E)
        energy_n = energy / (np.max(energy) + 1e-12)
        c_mask, M = compute_mask_energy_concentration(E, byz)
        bb = np.sum(E[np.ix_(benign_ids, benign_ids)] ** 2) / (np.sum(E ** 2) + 1e-12)
        support = np.sum(np.abs(E) > args.eps) / (K * K)
        mask_support = np.sum((np.abs(E) > args.eps) & (M == 1)) / (np.sum(np.abs(E) > args.eps) + 1e-12)

        # (a)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(L_star, cmap='Blues', vmin=0, vmax=1)
        ax.set_title('(a) Ideal benign consistency matrix L*')
        ax.set_xlabel('Client j'); ax.set_ylabel('Client i')
        fig.colorbar(im, ax=ax, label='Jaccard similarity')
        ax.text(0.02, -0.12, 'Benign clients form structured consistency patterns.', transform=ax.transAxes, fontsize=9)
        _save(fig, str(out_dir / f'obs2a_ideal_benign_Lstar_{attack}'))

        # (b)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(S_obs, cmap='Blues', vmin=0, vmax=1)
        ax.set_title('(b) Observed matrix S with Byzantine clients')
        ax.set_xlabel('Client j'); ax.set_ylabel('Client i')
        fig.colorbar(im, ax=ax, label='Jaccard similarity')
        for b in byz:
            ax.axhline(b, color='red', alpha=0.4); ax.axvline(b, color='red', alpha=0.4)
        _color_ticks(ax, byz)
        ax.text(0.02, -0.12, 'Byzantine uploads alter similarities involving themselves.', transform=ax.transAxes, fontsize=9)
        _save(fig, str(out_dir / f'obs2b_observed_S_{attack}'))

        # (c)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(abs_E, cmap='Reds')
        ax.set_title('(c) Perturbation |E| = |S - L*|')
        ax.set_xlabel('Client j'); ax.set_ylabel('Client i')
        fig.colorbar(im, ax=ax, label='|E|')
        for b in byz:
            ax.axhline(b, color='red', ls='--', alpha=0.7); ax.axvline(b, color='red', ls='--', alpha=0.7)
        b0, b1 = min(benign_ids), max(benign_ids)
        rect = plt.Rectangle((b0-0.5,b0-0.5), b1-b0+1, b1-b0+1, fill=False, ls='--', ec='black', lw=1)
        ax.add_patch(rect)
        ax.text(0.02, -0.12, 'Perturbation is row/column-localized, not globally spread.', transform=ax.transAxes, fontsize=9)
        _save(fig, str(out_dir / f'obs2c_perturbation_absE_{attack}'))

        # (d)
        fig, ax = plt.subplots(figsize=(6, 4))
        order = np.arange(K)
        if args.sort_energy:
            order = np.argsort(-energy)
        cols = ['red' if i in byz else 'tab:blue' for i in order]
        ax.bar(np.arange(K), energy[order], color=cols)
        ax.set_title('(d) Perturbation energy by client')
        ax.set_xlabel('Client ID'); ax.set_ylabel('Row/column perturbation energy')
        ax.text(0.02, 0.95, 'Most energy concentrates on Byzantine-related rows/columns.', transform=ax.transAxes, va='top', fontsize=9)
        _save(fig, str(out_dir / f'obs2d_energy_by_client_{attack}'))

        save_json(out_dir / f'obs2_metrics_{attack}.json', {
            'logits_npy': args.logits_npy, 'K': K, 'N_pub': N, 'C': C, 'kappa': args.kappa,
            'attack': 'label_flip_swap' if attack == 'label_flip' else attack,
            'byzantine_ids': byz, 'benign_ids': benign_ids,
            'mask_energy_concentration': c_mask,
            'benign_block_energy_ratio': float(bb),
            'support_ratio': float(support),
            'mask_support_ratio': float(mask_support),
            'row_column_energy': energy.tolist(),
            'row_column_energy_normalized': energy_n.tolist(),
        })


if __name__ == '__main__':
    main()
