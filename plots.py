#!/usr/bin/env python3
"""
plots.py - turns the CSV / NPZ files in ./results into the figures in ./figures

Run after experiments.py:   python plots.py
Colours: validated categorical palette, assigned per ENTITY (scenario) and kept fixed in every figure.
Every line also has its own marker and the legends are always present, so colour is never the only code.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)

SLA = {"mean_delay_ms": 10.0, "p99_delay_ms": 50.0, "blocking_prob": 0.005}

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#ffffff"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "magenta": "#e87ba4",
     "violet": "#6250d6", "green": "#008300", "red": "#e34948", "yellow": "#eda100"}

# fixed colour / marker per scenario (entity), same in every figure
SC = {
    "Baseline (5 Mbps)": (C["blue"], "o", "-"),
    "Constrained (2.8 Mbps)": (C["orange"], "s", "-"),
    "Improved A: +bandwidth (4 Mbps)": (C["aqua"], "^", "-"),
    "Improved B: compression (2.8 Mbps)": (C["magenta"], "D", "-"),
    "Improved C: compression + 3.2 Mbps": (C["violet"], "v", "-"),
    "Severe constraint (2.4 Mbps)": (C["red"], "X", "--"),
    "Severe + compression (2.4 Mbps)": (C["green"], "P", "--"),
}
SHORT = {"Baseline (5 Mbps)": "baseline", "Constrained (2.8 Mbps)": "constrained",
         "Improved A: +bandwidth (4 Mbps)": "impA", "Improved B: compression (2.8 Mbps)": "impB",
         "Improved C: compression + 3.2 Mbps": "impC", "Severe constraint (2.4 Mbps)": "severe",
         "Severe + compression (2.4 Mbps)": "severeB"}
CLASS_COL = {"Student": C["blue"], "Faculty": C["orange"], "Admin": C["aqua"], "IoT": C["magenta"]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#9c9b96", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 6,
    "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 220,
})


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def sla_line(ax, y, label=None, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(y, color=INK2, ls=":", lw=1.3)
    if label:
        ax.text(ax.get_xlim()[1], y, " " + label, va="bottom", ha="right", fontsize=8, color=INK2)


def errband(ax, x, y, ci, color, label, marker="o", ls="-"):
    ax.plot(x, y, color=color, marker=marker, ls=ls, label=label)
    ax.fill_between(x, np.array(y) - np.array(ci), np.array(y) + np.array(ci), color=color, alpha=0.18, lw=0)


# ---------------------------------------------------------------------------------------
def fig_validation():
    d = pd.read_csv(RES / "validation.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    # (a) delay vs rho: M/M/1 and M/D/1
    a = ax[0]
    for model, col, mk in (("M/M/1", C["blue"], "o"), ("M/D/1", C["orange"], "s")):
        s = d[(d.model == model) & (d.metric == "mean_delay_ms")]
        r = np.linspace(0.25, 0.9, 100)
        mu, ES = 125.0, 8.0
        if model == "M/M/1":
            th = ES / (1 - r)
        else:
            th = ES + r * ES / (2 * (1 - r))
        a.plot(r, th, color=col, lw=1.6, label=f"{model} theory")
        a.errorbar(s.rho, s["sim"], yerr=s["ci"], fmt=mk, color=col, ms=6, capsize=3, label=f"{model} simulation")
    a.set_xlabel("Traffic intensity ρ"); a.set_ylabel("Mean delay (ms)")
    a.set_title("(a) M/M/1 and M/D/1 mean delay"); a.legend()
    # (b) M/M/1/K blocking
    b = ax[1]
    s = d[(d.model == "M/M/1/K (K=10)") & (d.metric == "blocking_prob")]
    r = np.linspace(0.5, 1.6, 200); K = 10
    pk = np.where(np.abs(r - 1) < 1e-9, 1 / (K + 1), (1 - r) * r ** K / (1 - r ** (K + 1)))
    b.plot(r, pk, color=C["blue"], lw=1.6, label="M/M/1/K theory (K = 10)")
    b.errorbar(s.rho, s["sim"], yerr=s["ci"], fmt="o", color=C["blue"], capsize=3, label="simulation")
    b.set_xlabel("Offered load ρ"); b.set_ylabel("Blocking probability")
    b.set_title("(b) Finite buffer: blocking probability"); b.legend()
    # (c) campus mix vs P-K
    c = ax[2]
    s = d[(d.model == "Campus mix vs P-K") & (d.metric == "mean_delay_ms")].sort_values("rho")
    c.plot(s.rho, s.theory, color=C["blue"], marker=None, lw=1.6, label="Pollaczek–Khinchine theory")
    c.errorbar(s.rho, s["sim"], yerr=s["ci"], fmt="o", color=C["blue"], capsize=3, label="simulation")
    c.set_xlabel("Traffic intensity ρ (campus traffic mix)"); c.set_ylabel("Mean delay (ms)")
    c.set_title("(c) Four-class campus mix vs M/G/1"); c.legend()
    fig.tight_layout()
    save(fig, "fig_validation.png")


def fig_baseline():
    raw = np.load(RES / "raw_baseline.npz")
    conv = pd.read_csv(RES / "run_length_convergence.csv")
    byc = pd.read_csv(RES / "baseline_by_class.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    a = ax[0]
    for name in ("Student", "Faculty", "Admin", "IoT"):
        x = np.sort(raw[f"delay_{name}"])
        y = np.arange(1, x.size + 1) / x.size
        a.plot(x, y, color=CLASS_COL[name], label=name, lw=1.8)
    a.set_xlim(0, 12); a.set_xlabel("Packet delay (ms)"); a.set_ylabel("Cumulative probability")
    a.set_title("(a) Delay distribution by class"); a.legend(loc="lower right")
    b = ax[1]
    tr = raw["trace"]
    b.step(tr[:, 0] - tr[0, 0], tr[:, 1], where="post", color=C["blue"], lw=1.4)
    b.set_xlim(0, 0.5)
    b.set_xlabel("Time (s)"); b.set_ylabel("Packets in system")
    b.set_title("(b) Gateway occupancy, 0.5 s excerpt")
    c = ax[2]
    c.errorbar(conv.sim_time_s, conv.rel_ci_pct, fmt="o-", color=C["blue"], capsize=0)
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlabel("Simulated time per replication (s)"); c.set_ylabel("95 % CI half-width / mean (%)")
    c.set_title("(c) Run-length study (16 replications)")
    c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); c.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    c.axhline(0.5, color=INK2, ls=":", lw=1.2); c.text(20, 0.55, "0.5 % target", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "fig_baseline.png")

    # per-class bar
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    d = byc[byc["class"] != "All"]
    x = np.arange(len(d)); w = 0.38
    ax.bar(x - w / 2, d.mean_delay_ms, w, color=[CLASS_COL[c] for c in d["class"]], label="mean delay")
    ax.bar(x + w / 2, d.p99_delay_ms, w, color=[CLASS_COL[c] for c in d["class"]], alpha=0.45, label="99th percentile")
    for i, (m, p) in enumerate(zip(d.mean_delay_ms, d.p99_delay_ms)):
        ax.text(i - w / 2, m + 0.1, f"{m:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, p + 0.1, f"{p:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(d["class"]); ax.set_ylabel("Delay (ms)")
    ax.set_title("Baseline delay per traffic class (solid: mean, light: p99)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_baseline_classes.png")


def fig_bandwidth():
    d = pd.read_csv(RES / "bandwidth_sweep.csv").sort_values("rate_mbps")
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))
    a = ax[0, 0]
    errband(a, d.rate_mbps, d.mean_delay_ms, d.mean_delay_ms__ci, C["blue"], "mean delay (simulation)")
    errband(a, d.rate_mbps, d.p99_delay_ms, d.p99_delay_ms__ci, C["orange"], "99th percentile (simulation)", "s")
    ok = np.isfinite(d.pk_delay_ms)
    a.plot(d.rate_mbps[ok], d.pk_delay_ms[ok], color=C["blue"], ls="--", lw=1.4, marker=None,
           label="M/G/1 theory, infinite buffer")
    a.set_yscale("log"); a.set_xlabel("Channel capacity R (Mbit/s)"); a.set_ylabel("Delay (ms)")
    a.axhline(10, color=C["blue"], ls=":", lw=1.1); a.axhline(50, color=C["orange"], ls=":", lw=1.1)
    a.text(10.0, 10.8, "mean SLA 10 ms", ha="right", fontsize=8, color=INK2)
    a.text(10.0, 54, "p99 SLA 50 ms", ha="right", fontsize=8, color=INK2)
    a.set_title("(a) Delay versus bandwidth"); a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93))
    b = ax[0, 1]
    b.plot(d.rate_mbps, d.rho_offered, color=C["blue"], marker="o", label="offered load ρ (demand / capacity)")
    b.plot(d.rate_mbps, d.utilization, color=C["orange"], marker="s", ls="--", label="measured channel utilization")
    b.axhline(1.0, color=INK2, ls=":", lw=1.2)
    b.set_xlabel("Channel capacity R (Mbit/s)"); b.set_ylabel("Fraction of time")
    b.set_title("(b) Offered load and utilization"); b.legend()
    c = ax[1, 0]
    c.plot(d.rate_mbps, d.offered_mbps, color=C["blue"], marker="o", label="offered traffic (channel bits)")
    c.plot(d.rate_mbps, d.goodput_mbps, color=C["aqua"], marker="^", label="goodput (delivered payload)")
    c.plot(d.rate_mbps, d.rate_mbps, color=INK2, ls=":", lw=1.3, marker=None, label="capacity R")
    c.set_ylim(1.8, 2.8); c.set_xlim(2.1, 4.0)
    c.set_xlabel("Channel capacity R (Mbit/s)"); c.set_ylabel("Rate (Mbit/s)")
    c.set_title("(c) Throughput saturates when R < demand (zoom)"); c.legend(loc="lower right")
    e = ax[1, 1]
    e.plot(d.rate_mbps, 100 * d.blocking_prob, color=C["orange"], marker="s", label="blocking (buffer full)")
    e.plot(d.rate_mbps, 100 * d.loss_prob, color=C["blue"], marker="o", label="total loss (blocking + bit errors)")
    e.axhline(0.5, color=INK2, ls=":", lw=1.2); e.text(9.9, 1.0, "blocking SLA 0.5 %", ha="right", fontsize=8, color=INK2)
    e.set_xlabel("Channel capacity R (Mbit/s)"); e.set_ylabel("Probability (%)")
    e.set_title("(d) Packet loss versus bandwidth"); e.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_bandwidth.png")


def fig_load():
    d = pd.read_csv(RES / "load_sweep.csv")
    cols = {3.0: (C["orange"], "s"), 5.0: (C["blue"], "o"), 10.0: (C["aqua"], "^")}
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for R, (col, mk) in cols.items():
        s = d[d.rate_mbps == R].sort_values("load_scale")
        ax[0].plot(s.rho_offered, s.mean_delay_ms, color=col, marker=mk, label=f"R = {R:g} Mbit/s")
        ax[1].plot(s.offered_mbps, s.goodput_mbps, color=col, marker=mk, label=f"R = {R:g} Mbit/s")
        ax[2].plot(s.rho_offered, 100 * s.blocking_prob, color=col, marker=mk, label=f"R = {R:g} Mbit/s")
    ax[0].set_yscale("log"); ax[0].axhline(10, color=INK2, ls=":", lw=1.2)
    ax[0].text(0.02, 10.8, "mean-delay SLA 10 ms", fontsize=8, color=INK2)
    ax[0].set_xlabel("Offered traffic intensity ρ"); ax[0].set_ylabel("Mean delay (ms)")
    ax[0].set_title("(a) Delay knee"); ax[0].legend(loc="lower right")
    ax[1].plot([0, 5], [0, 5], color=INK2, ls=":", lw=1.2, marker=None)
    ax[1].set_xlabel("Offered traffic (Mbit/s)"); ax[1].set_ylabel("Goodput (Mbit/s)")
    ax[1].set_title("(b) Throughput versus offered load"); ax[1].legend(loc="upper left")
    ax[2].set_xlabel("Offered traffic intensity ρ"); ax[2].set_ylabel("Blocking probability (%)")
    ax[2].set_title("(c) Blocking onset beyond ρ ≈ 1"); ax[2].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_load.png")

    s = pd.read_csv(RES / "sustainable_load.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax[0].plot(s.rate_mbps, s.rho_max, color=C["blue"], marker="o")
    for i, (x, y) in enumerate(zip(s.rate_mbps, s.rho_max)):
        ax[0].text(x + (0.25 if i == 0 else 0.0), y + (-0.03 if i == 0 else 0.012), f"{y:.2f}", ha="center", fontsize=8)
    ax[0].set_ylim(0.7, 1.0); ax[0].set_xlabel("Channel capacity R (Mbit/s)")
    ax[0].set_ylabel("Maximum sustainable ρ"); ax[0].set_title("(a) Sustainable utilization grows with capacity")
    ax[1].plot(s.rate_mbps, s.packets_per_s_max, color=C["orange"], marker="s")
    ax[1].plot(s.rate_mbps, 380 * np.ones(len(s)), color=INK2, ls=":", marker=None, lw=1.3)
    ax[1].set_ylim(0, 1500)
    ax[1].text(5.0, 300, "expected load 380 pkt/s", fontsize=8, color=INK2)
    ax[1].set_xlabel("Channel capacity R (Mbit/s)"); ax[1].set_ylabel("Packets per second")
    ax[1].set_title("(b) Maximum load meeting the SLA")
    fig.tight_layout()
    save(fig, "fig_sustainable.png")


def fig_scenarios():
    s = pd.read_csv(RES / "scenarios.csv")
    raw = np.load(RES / "raw_scenarios.npz")
    main = [n for n in SC if n in set(s.scenario) and not n.startswith("Severe")]
    # (1) CDFs + stacked occupancy traces
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12.5, 5.2))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.15, 1], hspace=0.35, wspace=0.22)
    a0 = fig.add_subplot(gs[:, 0])
    for n in main:
        col, mk, ls = SC[n]
        x = np.sort(raw[f"delay_{SHORT[n]}"])
        y = np.arange(1, x.size + 1) / x.size
        a0.plot(x, y, color=col, ls=ls, label=n)
    a0.set_xscale("log"); a0.set_xlabel("Packet delay (ms, log scale)"); a0.set_ylabel("Cumulative probability")
    a0.axvline(50, color=INK2, ls=":", lw=1.2); a0.text(54, 0.12, "p99 SLA\n50 ms", fontsize=8, color=INK2)
    a0.set_title("(a) Delay distribution")
    a0.legend(loc="center right", bbox_to_anchor=(1.0, 0.33), fontsize=8, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.95)
    show = ("Baseline (5 Mbps)", "Constrained (2.8 Mbps)", "Improved B: compression (2.8 Mbps)")
    for i, n in enumerate(show):
        ax_ = fig.add_subplot(gs[i, 1])
        col, mk, ls = SC[n]
        tr = raw[f"trace_{SHORT[n]}"]
        ax_.step(tr[:, 0] - tr[0, 0], tr[:, 1], where="post", color=col, lw=1.1)
        ax_.set_ylim(0, 13); ax_.set_xlim(0, 2)
        ax_.set_ylabel("Packets", fontsize=8)
        ax_.text(0.99, 0.92, n, transform=ax_.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
        if i == 0:
            ax_.set_title("(b) Gateway occupancy, 2 s window")
        if i < 2:
            ax_.set_xticklabels([])
        else:
            ax_.set_xlabel("Time (s)")
    save(fig, "fig_scenarios_cdf.png")

    # (2) bars of key metrics for all scenarios
    names = [n for n in SC if n in set(s.scenario)]
    lab = {"Baseline (5 Mbps)": "Baseline\n5 Mbps", "Constrained (2.8 Mbps)": "Constr.\n2.8 Mbps",
           "Improved A: +bandwidth (4 Mbps)": "A: +BW\n4 Mbps",
           "Improved B: compression (2.8 Mbps)": "B: compr.\n2.8 Mbps",
           "Improved C: compression + 3.2 Mbps": "C: compr.\n+3.2 Mbps",
           "Severe constraint (2.4 Mbps)": "Severe\n2.4 Mbps",
           "Severe + compression (2.4 Mbps)": "Severe\n+ compr."}
    d = s.set_index("scenario").loc[names]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    x = np.arange(len(names)); cols = [SC[n][0] for n in names]
    for a, key, ttl, unit, thr in ((ax[0], "mean_delay_ms", "(a) Mean delay", "ms", 10),
                                    (ax[1], "p99_delay_ms", "(b) 99th-percentile delay", "ms", 50),
                                    (ax[2], "loss_prob", "(c) Total packet loss", "%", None)):
        v = d[key] * (100 if unit == "%" else 1)
        a.bar(x, v, color=cols, width=0.68)
        a.set_yscale("log") if unit == "ms" else None
        for xi, vi in zip(x, v):
            a.text(xi, vi * (1.08 if unit == "ms" else 1) + (0 if unit == "ms" else 0.03),
                   f"{vi:.2f}" if vi < 10 else f"{vi:.0f}", ha="center", fontsize=8)
        if thr:
            a.axhline(thr, color=INK2, ls=":", lw=1.3)
            a.text(len(names) - 0.5, thr * 1.08, f"SLA {thr} ms", ha="right", fontsize=8, color=INK2)
        a.set_xticks(x); a.set_xticklabels([lab[n] for n in names], fontsize=7.5)
        a.set_ylabel(f"{ttl[4:]} ({unit})"); a.set_title(ttl); a.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_scenarios_bars.png")

    # (3) per-class mean delay: baseline / constrained / improved B
    c = pd.read_csv(RES / "scenarios_by_class.csv")
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    show = ["Baseline (5 Mbps)", "Constrained (2.8 Mbps)", "Improved A: +bandwidth (4 Mbps)",
            "Improved B: compression (2.8 Mbps)"]
    w = 0.2
    for i, n in enumerate(show):
        v = [c[(c.scenario == n) & (c["class"] == k)].mean_delay_ms.iloc[0] for k in CLASS_COL]
        ax.bar(np.arange(4) + (i - 1.5) * w, v, w, color=SC[n][0], label=n)
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_COL.keys()); ax.set_ylabel("Mean delay (ms)")
    ax.set_title("Mean delay per class: the delay penalty is shared by all classes"); ax.legend(fontsize=7.5)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_scenarios_classes.png")


def fig_buffer_comp():
    b = pd.read_csv(RES / "buffer_sweep.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    cols = {"rho~0.9 (2.8 Mbps)": (C["blue"], "o"), "rho~1.1 (2.3 Mbps)": (C["orange"], "s")}
    lab = {"rho~0.9 (2.8 Mbps)": "R = 2.8 Mbit/s (ρ ≈ 0.89)", "rho~1.1 (2.3 Mbps)": "R = 2.3 Mbit/s (ρ ≈ 1.07)"}
    for k, (col, mk) in cols.items():
        s = b[b.case == k].sort_values("buffer_pkts")
        ax[0].plot(s.buffer_pkts, 100 * s.blocking_prob, color=col, marker=mk, label=lab[k])
        ax[1].plot(s.buffer_pkts, s.mean_delay_ms, color=col, marker=mk, label=lab[k])
        ax[2].plot(s.buffer_pkts, s.goodput_mbps, color=col, marker=mk, label=lab[k])
    ax[0].axhline(100 * (1 - 1 / 1.0726), color=INK2, ls=":", lw=1.2)
    ax[0].text(30, 4.4, "dotted: long-run floor 1 − 1/ρ = 6.8 %\n(B = 3000: 300 s run too short to fill the buffer)", fontsize=7.5, color=INK2)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Buffer size B (packets)"); ax[0].set_ylabel("Blocking probability (%)")
    ax[0].set_title("(a) Blocking"); ax[0].legend(loc="upper right")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("Buffer size B (packets)")
    ax[1].set_ylabel("Mean delay (ms)"); ax[1].set_title("(b) Delay"); ax[1].legend(loc="upper left")
    ax[2].set_xscale("log"); ax[2].set_xlabel("Buffer size B (packets)"); ax[2].set_ylabel("Goodput (Mbit/s)")
    ax[2].set_title("(c) Goodput"); ax[2].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_buffer.png")

    d = pd.read_csv(RES / "compression_sweep.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    x = 100 * d.compression
    ax[0].plot(x, d.mean_delay_ms, color=C["blue"], marker="o", label="mean delay")
    ax[0].plot(x, d.p99_delay_ms, color=C["orange"], marker="s", label="99th percentile")
    ax[0].axhline(10, color=C["blue"], ls=":", lw=1.1); ax[0].axhline(50, color=C["orange"], ls=":", lw=1.1)
    ax[0].set_yscale("log"); ax[0].set_xlabel("Payload reduction of compressible classes (%)")
    ax[0].set_ylabel("Delay (ms)"); ax[0].set_title("(a) Delay at R = 2.8 Mbit/s"); ax[0].legend()
    ax[1].plot(x, d.rho_offered, color=C["blue"], marker="o")
    ax[1].set_xlabel("Payload reduction of compressible classes (%)"); ax[1].set_ylabel("Traffic intensity ρ")
    ax[1].set_title("(b) Offered load falls with compression")
    ax[2].plot(x, 100 * d.loss_prob, color=C["blue"], marker="o")
    ax[2].set_xlabel("Payload reduction of compressible classes (%)"); ax[2].set_ylabel("Total packet loss (%)")
    ax[2].set_title("(c) Fewer bits on air, fewer bit-error losses")
    fig.tight_layout()
    save(fig, "fig_compression.png")


def fig_capacity():
    c = pd.read_csv(RES / "capacity_planning.csv")
    r = pd.read_csv(RES / "robust_capacity.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.1))
    a = ax[0]
    a.plot(c.offered_pps, c.R_required_sim_mbps, color=C["blue"], marker="o", label="required R, simulation (SLA met)")
    a.plot(c.offered_pps, c.R_required_pk_meandelay_mbps, color=C["orange"], marker="s", ls="--",
           label="required R, M/G/1 theory (mean delay ≤ 10 ms)")
    a.plot(c.offered_pps, c.R_stability_mbps, color=INK2, ls=":", marker=None, lw=1.4, label="stability limit (ρ = 1)")
    a.set_xlabel("Offered packet rate (packets/s)"); a.set_ylabel("Channel capacity R (Mbit/s)")
    a.set_title("(a) Capacity needed for nominal demand"); a.legend(loc="upper left")
    b = ax[1]
    b.plot(r.rate_mbps, 100 * r.p_sla_met, color=C["blue"], marker="o")
    b.fill_between(r.rate_mbps, 100 * r.p_ci_low, 100 * r.p_ci_high, color=C["blue"], alpha=0.18, lw=0)
    b.axhline(95, color=INK2, ls=":", lw=1.3); b.text(r.rate_mbps.max(), 90, "95 % target", ha="right", fontsize=8, color=INK2)
    b.axvline(2.94, color=C["orange"], ls="--", lw=1.3); b.text(3.02, 42, "nominal requirement\n2.94 Mbit/s", fontsize=8, color=INK2)
    b.set_xlabel("Channel capacity R (Mbit/s)"); b.set_ylabel("P(SLA met) under uncertainty (%)")
    b.set_title("(b) Robust capacity (Monte Carlo, 300 trials)")
    fig.tight_layout()
    save(fig, "fig_capacity.png")


def fig_montecarlo():
    m = pd.read_csv(RES / "montecarlo.csv")
    names = [n for n in SC if n in set(m.scenario)]
    short = {"Baseline (5 Mbps)": "Baseline\n5 Mbps", "Constrained (2.8 Mbps)": "Constrained\n2.8 Mbps",
             "Improved A: +bandwidth (4 Mbps)": "A: +bandwidth\n4 Mbps", "Improved B: compression (2.8 Mbps)":
                 "B: compression\n2.8 Mbps", "Improved C: compression + 3.2 Mbps": "C: compr. +\n3.2 Mbps",
             "Severe constraint (2.4 Mbps)": "Severe\n2.4 Mbps", "Severe + compression (2.4 Mbps)": "Severe +\ncompr."}
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]
    data = [m[m.scenario == n].mean_delay_ms.values for n in names]
    bp = a.boxplot(data, positions=range(len(names)), widths=0.55, patch_artist=True, showfliers=True,
                   medianprops=dict(color=INK, lw=1.6), flierprops=dict(marker="o", ms=3, mfc="none", mec=INK2))
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(SC[n][0]); patch.set_alpha(0.55)
    a.set_yscale("log"); a.axhline(10, color=INK2, ls=":", lw=1.3); a.text(-0.45, 10.8, "SLA 10 ms", ha="left", fontsize=8, color=INK2)
    a.set_xticks(range(len(names))); a.set_xticklabels([short[n] for n in names], fontsize=7.5)
    a.set_ylabel("Mean delay per trial (ms)"); a.set_title("(a) Distribution of mean delay over 300 uncertain-input trials")
    a.grid(axis="x", visible=False)
    b = ax[1]
    p = [m[m.scenario == n].sla_ok.mean() * 100 for n in names]
    b.barh(range(len(names)), p, color=[SC[n][0] for n in names], height=0.62)
    for i, v in enumerate(p):
        b.text(v + 1.5, i, f"{v:.0f} %", va="center", fontsize=9)
    b.set_yticks(range(len(names))); b.set_yticklabels([short[n].replace("\n", " ") for n in names], fontsize=8)
    b.invert_yaxis(); b.set_xlim(0, 115); b.set_xlabel("Probability that all three SLA limits are met (%)")
    b.set_title("(b) SLA compliance"); b.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_montecarlo.png")

    s = pd.read_csv(RES / "montecarlo_sensitivity.csv")
    inp = ["demand_factor", "mult_Student", "mult_Faculty", "mult_Admin", "mult_IoT", "ber"]
    lab = ["Overall demand", "Student activity", "Faculty activity", "Admin activity", "IoT activity", "Bit error rate"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for a, out, ttl in ((ax[0], "mean_delay_ms", "(a) Mean delay"), (ax[1], "loss_prob", "(b) Total packet loss")):
        pv = s[s.output == out].pivot(index="input", columns="scenario", values="spearman").loc[inp]
        pv = pv[[n for n in ("Baseline (5 Mbps)", "Constrained (2.8 Mbps)", "Improved A: +bandwidth (4 Mbps)")]]
        w = 0.26
        for i, n in enumerate(pv.columns):
            a.barh(np.arange(len(inp)) + (i - 1) * w, pv[n], w, color=SC[n][0], label=n)
        a.set_yticks(range(len(inp))); a.set_yticklabels(lab); a.invert_yaxis()
        a.set_xlabel("Spearman rank correlation with output"); a.set_title(ttl)
        a.axvline(0, color=INK2, lw=0.8); a.set_xlim(-0.2, 1.05); a.grid(axis="y", visible=False)
    ax[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_sensitivity.png")


if __name__ == "__main__":
    fig_validation()
    fig_baseline()
    fig_bandwidth()
    fig_load()
    fig_scenarios()
    fig_buffer_comp()
    fig_capacity()
    fig_montecarlo()
