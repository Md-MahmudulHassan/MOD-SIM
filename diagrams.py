#!/usr/bin/env python3
"""diagrams.py - system block diagram and simulation flowchart (matplotlib, no external tools)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
LIGHT = {BLUE: "#e4eefb", ORANGE: "#fdeae2", AQUA: "#dff5ec", MAGENTA: "#fbe6ee", "#ffffff": "#ffffff",
         "#9c9b96": "#f1f0ed"}


def box(ax, x, y, w, h, text, color=BLUE, fs=9, bold=False, round_=0.02, fc=None, ec=None, lw=1.6, ha="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}", fc=fc or LIGHT.get(color, "#fff"),
                       ec=ec or color, lw=lw)
    ax.add_patch(p)
    ax.text(x + (w / 2 if ha == "center" else 0.012), y + h / 2, text, ha=ha, va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.25)
    return p


def arrow(ax, p0, p1, color=INK2, lw=1.6, style="-|>", ls="-", rad=0.0, label=None, lpos=None, fs=8, lcolor=None):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13, color=color, lw=lw, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label:
        lx, ly = lpos or ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.012)
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=fs, color=lcolor or INK2)


def system_diagram():
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.975, "Group 7 – Wireless gateway link with limited bandwidth: system block diagram",
            ha="center", va="top", fontsize=13, fontweight="bold")

    # sources
    srcs = [("Student devices", "400 × 0.5 pkt/s\n1000 B payload", 0.745),
            ("Faculty devices", "50 × 1 pkt/s\n1200 B payload", 0.585),
            ("Administrative PCs", "30 × 1 pkt/s\n600 B payload", 0.425),
            ("IoT sensors", "500 × 0.2 pkt/s\n100 B payload", 0.265)]
    for t, d, y in srcs:
        box(ax, 0.015, y, 0.15, 0.115, f"{t}\n{d}", BLUE, fs=8.6)
    ax.text(0.09, 0.885, "TRAFFIC SOURCES\n(independent Poisson processes)", ha="center", fontsize=8.5, color=INK2)

    # merge
    box(ax, 0.215, 0.43, 0.095, 0.30, "Superposition\n\nλ = Σ λᵢ\n= 380 pkt/s\n\nPoisson", AQUA, fs=8.6)
    for _, _, y in srcs:
        arrow(ax, (0.165, y + 0.0575), (0.215, 0.58), color=INK2)

    # system boundary
    ax.add_patch(Rectangle((0.345, 0.17), 0.43, 0.63, fc="none", ec=INK2, lw=1.4, ls=(0, (6, 4))))
    ax.text(0.36, 0.785, "SYSTEM BOUNDARY (gateway + shared wireless link)", fontsize=8.5, color=INK2, va="center")

    box(ax, 0.37, 0.43, 0.12, 0.30, "FIFO buffer\n(queue)\n\nB = 100 waiting\nslots\n\nfull → packet\nblocked", ORANGE, fs=8.6)
    box(ax, 0.525, 0.43, 0.115, 0.30, "Gateway +\ntransmitter\n(single server)\n\nS = 8(L+H)/R\n+ t_proc\nR = 5 Mbit/s", BLUE, fs=8.4)
    box(ax, 0.675, 0.43, 0.085, 0.30, "Wireless\nchannel\n\nBER = 10⁻⁶\n(bit errors)", MAGENTA, fs=8.6)
    arrow(ax, (0.31, 0.58), (0.37, 0.58))
    arrow(ax, (0.49, 0.58), (0.525, 0.58))
    arrow(ax, (0.64, 0.58), (0.675, 0.58))

    # blocked
    box(ax, 0.37, 0.205, 0.12, 0.105, "Blocked packets\n(buffer overflow)\n→ congestion loss", ORANGE, fs=8.2)
    arrow(ax, (0.43, 0.43), (0.43, 0.31), color=ORANGE, label="arrival finds\nbuffer full", lpos=(0.395, 0.33), fs=7.5)

    # destination
    box(ax, 0.82, 0.60, 0.16, 0.13, "Destination\n(error-free packets)\n→ goodput", AQUA, fs=8.6)
    box(ax, 0.82, 0.31, 0.16, 0.13, "Corrupted packets\ndiscarded\n→ error loss", MAGENTA, fs=8.6)
    arrow(ax, (0.76, 0.62), (0.82, 0.655), color=AQUA, label="OK", lpos=(0.79, 0.665), fs=8)
    arrow(ax, (0.76, 0.54), (0.82, 0.375), color=MAGENTA, label="packet error prob.\n1−(1−BER)^bits", lpos=(0.905, 0.47), fs=7.5)

    # inputs / outputs / entities (bottom row, outside the boundary)
    ax.text(0.02, 0.135, "INPUTS (parameters)", fontsize=9, fontweight="bold", va="top")
    ax.text(0.02, 0.105, "• arrival rates λ_c, payload sizes L_c, header H\n• channel capacity R (varied 2.2 – 10 Mbit/s)\n"
                         "• buffer size B, processing time t_proc, BER\n• compression ratio (improvement option)",
            fontsize=8.2, va="top", color=INK, linespacing=1.4)
    ax.text(0.37, 0.135, "OUTPUTS (metrics)", fontsize=9, fontweight="bold", va="top")
    ax.text(0.37, 0.105, "• mean / p95 / p99 delay, waiting time\n• queue length (mean, max), utilization\n"
                         "• blocking probability, packet loss, PDR\n• goodput / throughput", fontsize=8.2, va="top",
            color=INK, linespacing=1.4)
    ax.text(0.73, 0.135, "ENTITIES · RESOURCES · EVENTS", fontsize=9, fontweight="bold", va="top")
    ax.text(0.73, 0.105, "Entity: packet\nResources: channel (R), B buffer slots\nEvents: ARRIVAL, DEPARTURE\n"
                         "Queue: FIFO, one server", fontsize=8.2, va="top", color=INK, linespacing=1.4)
    fig.savefig(FIG / "fig_system_diagram.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig_system_diagram.png")


def diamond(ax, cx, cy, w, h, text, color=ORANGE, fs=8.2):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True,
                         fc=LIGHT[color], ec=color, lw=1.6))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.2)


def flowchart():
    fig, ax = plt.subplots(figsize=(11.6, 9.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.99, "Discrete-event simulation algorithm (next-event time advance)", ha="center", va="top",
            fontsize=13, fontweight="bold")
    cx = 0.30
    box(ax, 0.12, 0.895, 0.36, 0.05, "START: read configuration (R, B, BER, λ_c, L_c, T)", BLUE, fs=8.6)
    box(ax, 0.12, 0.815, 0.36, 0.055, "Generate arrivals: N_c ~ Poisson(λ_c T), times ~ U(0,T)\nsort & merge classes; draw packet sizes", AQUA, fs=8.4)
    box(ax, 0.12, 0.735, 0.36, 0.055, "Per packet: bits, service time S = bits/R + t_proc,\nerror flag ~ Bernoulli(1−(1−BER)^bits)", AQUA, fs=8.4)
    box(ax, 0.12, 0.665, 0.36, 0.045, "Initialise: queue = ∅, server idle, t_dep = ∞", BLUE, fs=8.6)
    for y0, y1 in ((0.895, 0.870), (0.815, 0.790), (0.735, 0.710), (0.665, 0.6475)):
        arrow(ax, (cx, y0), (cx, y1))
    diamond(ax, cx, 0.60, 0.34, 0.095, "More arrivals or\npacket in service?", BLUE)
    box(ax, 0.12, 0.485, 0.36, 0.05, "t = min(next arrival time, t_dep);  update time-weighted\nareas  ∫N dt, ∫Nq dt, busy time", BLUE, fs=8.2)
    arrow(ax, (cx, 0.5525), (cx, 0.535), label="yes", lpos=(cx + 0.03, 0.538), fs=8)
    diamond(ax, cx, 0.405, 0.30, 0.085, "Event = ARRIVAL?", ORANGE)
    arrow(ax, (cx, 0.485), (cx, 0.4475))

    # arrival branch
    diamond(ax, 0.17, 0.30, 0.25, 0.085, "Server idle?", ORANGE)
    arrow(ax, (cx - 0.15, 0.405), (0.17, 0.3425), label="yes", lpos=(0.118, 0.372), fs=8)
    box(ax, 0.02, 0.155, 0.19, 0.06, "Start service now\nt_dep = t + S", AQUA, fs=8.2)
    arrow(ax, (0.17, 0.2575), (0.12, 0.215), label="yes", lpos=(0.115, 0.245), fs=8)
    diamond(ax, 0.39, 0.205, 0.22, 0.08, "Queue < B?", ORANGE)
    arrow(ax, (0.295, 0.30), (0.39, 0.245), label="no", lpos=(0.325, 0.285), fs=8)
    box(ax, 0.29, 0.085, 0.14, 0.05, "Enqueue packet", AQUA, fs=8.2)
    arrow(ax, (0.39, 0.165), (0.36, 0.135), label="yes", lpos=(0.395, 0.148), fs=8)
    box(ax, 0.455, 0.085, 0.14, 0.05, "Block packet\n(count loss)", MAGENTA, fs=8.2)
    arrow(ax, (0.50, 0.205), (0.525, 0.135), label="no", lpos=(0.525, 0.18), fs=8)

    # departure branch
    box(ax, 0.60, 0.33, 0.37, 0.095, "DEPARTURE: record delay = t − arrival time;\nmark corrupted / delivered", BLUE, fs=8.2)
    arrow(ax, (cx + 0.15, 0.405), (0.60, 0.38), label="no", lpos=(0.52, 0.405), fs=8)
    diamond(ax, 0.785, 0.235, 0.30, 0.09, "Queue empty?", ORANGE)
    arrow(ax, (0.785, 0.33), (0.785, 0.28))
    box(ax, 0.64, 0.075, 0.13, 0.07, "Server idle\nt_dep = ∞", AQUA, fs=8.2)
    box(ax, 0.80, 0.075, 0.17, 0.07, "Pop next packet,\nwait = t − arrival;\nt_dep = t + S", AQUA, fs=7.8)
    arrow(ax, (0.635, 0.235), (0.69, 0.145), label="yes", lpos=(0.600, 0.185), fs=8)
    arrow(ax, (0.935, 0.235), (0.90, 0.145), label="no", lpos=(0.94, 0.19), fs=8)

    # loop-back bus
    bus = 0.035
    for x0, y0 in ((0.115, 0.155), (0.36, 0.085), (0.525, 0.085), (0.705, 0.075), (0.885, 0.075)):
        ax.plot([x0, x0], [y0, bus], color=INK2, lw=1.6)
    ax.plot([0.0125, 0.885], [bus, bus], color=INK2, lw=1.6)
    ax.plot([0.0125, 0.0125], [bus, 0.60], color=INK2, lw=1.6)
    arrow(ax, (0.0125, 0.60), (cx - 0.17, 0.60), color=INK2)
    ax.text(0.45, bus - 0.006, "next event", fontsize=8, ha="center", va="top", color=INK2)

    # exit
    arrow(ax, (cx + 0.17, 0.60), (0.60, 0.60), label="no", lpos=(0.50, 0.612), fs=8)
    box(ax, 0.60, 0.55, 0.37, 0.10, "END: compute metrics for packets arriving in [warm-up, T]:\ndelay, wait, p99, queue length, utilization,\nblocking, loss, PDR, goodput", BLUE, fs=8.2)
    fig.savefig(FIG / "fig_flowchart.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig_flowchart.png")


if __name__ == "__main__":
    system_diagram()
    flowchart()
