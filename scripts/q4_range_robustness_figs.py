#!/usr/bin/env python3
"""Figures for the Q4 range-robustness analysis (reads the sweep's JSON dump).

Usage: python scripts/q4_range_robustness_figs.py --dump <q4_range_robustness.json> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"      # curve / series-1
VIOLET = "#4a3aa7"    # reference taus
RED = "#e34948"       # per-metric optimum
GRAY = "#c3c2b7"
INK = "#1a1a19"
BAND = "#2a78d6"      # band shading (low alpha)
SEQ_BLUES = ["#a8c8ee", "#6ba0de", "#2a78d6", "#1b4e8f"]  # tau-ordered bar series

OFFPAIR = ("OFF-PAIR / INDICATIVE: stage9-gemini-gpt-medium (legacy Gemini+GPT), "
           "not the current Claude+GPT pair")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.loads(Path(args.dump).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    grid = np.array(d["grid_taus"])
    curves = {k: np.array(v) for k, v in d["curves"].items()}
    u = d["uncalibrated"]
    band = d["verdict"]["band_tolerant"]
    refs = {k: v for k, v in d["meta"]["reference_taus"].items() if k != "identity"}
    ref_short = {"tau_oc_qwen": "qwen 1.60", "tau_oc_llama31": "llama 1.86",
                 "tau_oc_gemma31": "gemma 4.88", "tau_oc_glm_soft": "glm 8.84"}

    # ---- Fig 1: delta curves vs tau, band shaded --------------------------
    panels = [
        ("reliability", "Δ Murphy reliability (− is better)", u["reliability"], "min"),
        ("resolution", "Δ Murphy resolution (+ is better)", u["resolution"], "max"),
        ("mean_rps", "Δ mean RPS (− is better)", u["mean_rps"], "min"),
        ("mean_w1", "Δ mean W1 (− is better)", u["mean_w1"], "min"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (key, title, base, sense) in zip(axes.flat, panels):
        delta = curves[key] - base
        ax.axhspan(0, 0, color=GRAY)  # no-op keeps zorder simple
        ax.axvspan(band["lower"], band["upper"], color=BAND, alpha=0.08, zorder=0)
        ax.axhline(0, color=GRAY, lw=1, zorder=1)
        for name, tau in refs.items():
            ax.axvline(tau, color=VIOLET, lw=1, ls=":", alpha=0.7, zorder=1)
        ax.plot(grid, delta, color=BLUE, lw=2, zorder=3)
        ax.margins(y=0.12)
        i_opt = int(np.argmin(delta)) if sense == "min" else int(np.argmax(delta))
        ax.plot(grid[i_opt], delta[i_opt], "o", ms=7, color=RED, zorder=4)
        ax.annotate(f"τ*={grid[i_opt]:.2f}", (grid[i_opt], delta[i_opt]),
                    textcoords="offset points",
                    xytext=(6, -12) if sense == "max" else (6, 8),
                    fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10, color=INK)
        ax.grid(axis="y", color="#eeeeea", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("temperature τ (log scale)", fontsize=9)
    # direct labels for the reference taus, once, on the top-left panel
    ax0 = axes[0][0]
    ymin, ymax = ax0.get_ylim()
    for i, (name, tau) in enumerate(sorted(refs.items(), key=lambda kv: kv[1])):
        ax0.annotate(ref_short[name], (tau, ymax), rotation=90, fontsize=7,
                     color=VIOLET, va="top", ha="right", alpha=0.9,
                     textcoords="offset points", xytext=(-2, -2))
    fig.suptitle("Q4 range robustness — closed-pair metric deltas vs temperature τ\n"
                 f"shaded: tolerant benefit band [{band['lower']:.2f}, {band['upper']:.1f}] "
                 "(reliability improves, RPS not worse, resolution ≥ −10%)",
                 fontsize=11, color=INK)
    fig.text(0.5, 0.005, OFFPAIR, ha="center", fontsize=8, color="#8a8a85")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out / "q4_range_robustness_curves.png", dpi=160)
    plt.close(fig)

    # ---- Fig 2: heterogeneity — mean dRPS by Article and GT level ---------
    het = d["heterogeneity"]
    tau_keys = ["tau_oc_qwen", "tau_oc_llama31", "tau_oc_gemma31", "tau_oc_glm_soft"]
    tau_labels = [f"τ={het[k]['tau']:.2f}" for k in tau_keys]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (field, xlab) in zip(axes, [("by_article", "Article"), ("by_gt_level", "GT argmax level")]):
        groups = list(het[tau_keys[0]][field].keys())
        x = np.arange(len(groups))
        wbar = 0.19
        for i, tk in enumerate(tau_keys):
            vals = [het[tk][field][g]["mean_d_rps"] for g in groups]
            ax.bar(x + (i - 1.5) * wbar, vals, wbar * 0.92, color=SEQ_BLUES[i],
                   label=tau_labels[i])
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([g if field == "by_gt_level" else f"Art {g}" for g in groups],
                           fontsize=8)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel("mean Δ RPS (− is better)", fontsize=9)
        ax.grid(axis="y", color="#eeeeea", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Per-cell heterogeneity of the correction — where the RPS gain lives",
                 fontsize=11, color=INK)
    fig.text(0.5, 0.005, OFFPAIR, ha="center", fontsize=8, color="#8a8a85")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out / "q4_range_robustness_heterogeneity.png", dpi=160)
    plt.close(fig)
    print(f"[figs] wrote {out}/q4_range_robustness_{{curves,heterogeneity}}.png")


if __name__ == "__main__":
    main()
