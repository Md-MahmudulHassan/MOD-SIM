#!/usr/bin/env python3
"""
experiments.py  -  All numerical experiments for the Group 7 study.

Usage:   python experiments.py [stage ...]
Stages:  validation baseline bandwidth load improve buffer capacity montecarlo robust   (default: all)

Every stage writes CSV files to ./results/ ; plots.py turns them into figures.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from wnet_sim import (SLA, NetworkConfig, TrafficClass, baseline_config, default_classes, mm1k_metrics,
                      pk_metrics, required_rate_pk, run_and_summarize, run_replications, simulate,
                      stability_rate, summarize)

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

SEED = 2026
WORKERS = 2

# --- scenario definitions ---------------------------------------------------------------
R_BASE = 5.0e6            # baseline channel capacity (bit/s)
R_CONSTRAINED = 2.8e6     # bandwidth-limited channel
COMPRESSION = 0.30        # payload reduction for text/web/administrative traffic
COMPRESSION_PROC = 100e-6 # extra gateway time per compressed packet (s)


def base() -> NetworkConfig:
    return baseline_config(rate_bps=R_BASE)


def constrained() -> NetworkConfig:
    return baseline_config(rate_bps=R_CONSTRAINED)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def flat_row(label: str, s: dict, **extra) -> dict:
    row = {"scenario": label}
    row.update(extra)
    row.update(s)
    return row


# =========================================================================================
# 0. Validation against queueing theory
# =========================================================================================
def stage_validation():
    log("validation: M/M/1, M/D/1, M/M/1/K, P-K for the multi-class mix, Little's law")
    rows = []
    R = 1.0e6
    mean_bytes = 1000.0                       # E[S] = 8000/1e6 = 8 ms  -> mu = 125 pkt/s
    mu = R / (8 * mean_bytes)
    ideal = dict(rate_bps=R, header_bytes=0.0, proc_time_s=0.0, ber=0.0, sim_time_s=3000.0, warmup_s=200.0,
                 buffer_pkts=10 ** 9)
    # (a) M/M/1 and (b) M/D/1 : delay vs rho
    for dist, name in (("exp", "M/M/1"), ("fixed", "M/D/1")):
        for rho in (0.3, 0.5, 0.7, 0.85):
            lam = rho * mu
            tc = TrafficClass("X", n_devices=1, rate_per_device=lam, payload_bytes=mean_bytes, size_dist=dist)
            cfg = NetworkConfig(classes=(tc,), **ideal)
            s = run_and_summarize(cfg, n_reps=8, seed=SEED, workers=WORKERS)
            th = pk_metrics(cfg)
            rows.append(dict(model=name, rho=rho, metric="mean_delay_ms", sim=s["mean_delay_ms"],
                             ci=s["mean_delay_ms__ci"], theory=th["W_ms"]))
            rows.append(dict(model=name, rho=rho, metric="mean_wait_ms", sim=s["mean_wait_ms"],
                             ci=s["mean_wait_ms__ci"], theory=th["Wq_ms"]))
            rows.append(dict(model=name, rho=rho, metric="mean_queue_len", sim=s["mean_queue_len"],
                             ci=s["mean_queue_len__ci"], theory=th["Lq"]))
            rows.append(dict(model=name, rho=rho, metric="utilization", sim=s["utilization"],
                             ci=s["utilization__ci"], theory=rho))
            rows.append(dict(model=name, rho=rho, metric="little_ratio", sim=s["little_ratio"],
                             ci=s["little_ratio__ci"], theory=1.0))
            log(f"   {name} rho={rho}: W sim={s['mean_delay_ms']:.3f} th={th['W_ms']:.3f} ms")
    # (c) M/M/1/K blocking and mean number in system (K = B + 1 = 10)
    B = 9
    for rho in (0.6, 0.8, 1.0, 1.2, 1.5):
        lam = rho * mu
        tc = TrafficClass("X", n_devices=1, rate_per_device=lam, payload_bytes=mean_bytes, size_dist="exp")
        cfg = NetworkConfig(classes=(tc,), **{**ideal, "buffer_pkts": B})
        s = run_and_summarize(cfg, n_reps=8, seed=SEED, workers=WORKERS)
        th = mm1k_metrics(lam, mu, B + 1)
        rows.append(dict(model="M/M/1/K (K=10)", rho=rho, metric="blocking_prob", sim=s["blocking_prob"],
                         ci=s["blocking_prob__ci"], theory=th["blocking"]))
        rows.append(dict(model="M/M/1/K (K=10)", rho=rho, metric="mean_in_system", sim=s["mean_in_system"],
                         ci=s["mean_in_system__ci"], theory=th["L"]))
        rows.append(dict(model="M/M/1/K (K=10)", rho=rho, metric="mean_delay_ms", sim=s["mean_delay_ms"],
                         ci=s["mean_delay_ms__ci"], theory=th["W_s"] * 1e3))
        rows.append(dict(model="M/M/1/K (K=10)", rho=rho, metric="utilization", sim=s["utilization"],
                         ci=s["utilization__ci"], theory=th["util"]))
        log(f"   M/M/1/K rho={rho}: Pb sim={s['blocking_prob']:.4f} th={th['blocking']:.4f}")
    # (d) The real multi-class network: simulation vs P-K over several capacities (large buffer)
    for R_ in (3.0e6, 3.5e6, 4.0e6, 5.0e6, 7.0e6):
        cfg = baseline_config(rate_bps=R_, ber=0.0, buffer_pkts=10 ** 9, sim_time_s=1500.0, warmup_s=100.0)
        s = run_and_summarize(cfg, n_reps=8, seed=SEED, workers=WORKERS)
        th = pk_metrics(cfg)
        rows.append(dict(model="Campus mix vs P-K", rho=th["rho"], metric="mean_delay_ms", sim=s["mean_delay_ms"],
                         ci=s["mean_delay_ms__ci"], theory=th["W_ms"]))
        rows.append(dict(model="Campus mix vs P-K", rho=th["rho"], metric="utilization", sim=s["utilization"],
                         ci=s["utilization__ci"], theory=th["rho"]))
        log(f"   campus mix R={R_/1e6} Mbps rho={th['rho']:.3f}: W sim={s['mean_delay_ms']:.3f} th={th['W_ms']:.3f}")
    df = pd.DataFrame(rows)
    df["rel_err_pct"] = 100 * (df["sim"] - df["theory"]) / df["theory"]
    df["within_ci"] = (df["sim"] - df["theory"]).abs() <= df["ci"] * 1.0
    df.to_csv(OUT / "validation.csv", index=False)
    log(f"validation saved ({len(df)} rows); max |rel err| = {df['rel_err_pct'].abs().max():.2f}%")


# =========================================================================================
# 1. Baseline performance
# =========================================================================================
def stage_baseline():
    log("baseline: 30 replications x 600 s")
    cfg = base().with_(sim_time_s=600.0, warmup_s=60.0)
    results = run_replications(cfg, 30, SEED, WORKERS)
    s = summarize(results)
    pd.DataFrame([s]).to_csv(OUT / "baseline_summary.csv", index=False)
    rows = []
    for tc in cfg.classes:
        p = tc.name + "."
        rows.append({"class": tc.name, "rate_pps": tc.rate, "payload_B": tc.payload_bytes,
                     **{k: s[p + k] for k in ("mean_delay_ms", "mean_wait_ms", "p99_delay_ms", "blocking_prob",
                                              "corruption_prob", "loss_prob", "pdr", "goodput_mbps")}})
    rows.append({"class": "All", "rate_pps": s["arrival_rate_pps"], "payload_B": np.nan,
                 **{k: s[k] for k in ("mean_delay_ms", "mean_wait_ms", "p99_delay_ms", "blocking_prob",
                                      "corruption_prob", "loss_prob", "pdr", "goodput_mbps")}})
    pd.DataFrame(rows).to_csv(OUT / "baseline_by_class.csv", index=False)

    # run-length / stability study: how the estimate of mean delay converges with simulated time
    conv = []
    for T in (20, 50, 100, 200, 400, 800, 1600):
        c = base().with_(sim_time_s=float(T), warmup_s=min(30.0, 0.1 * T))
        rr = run_replications(c, 16, SEED, WORKERS)
        v = np.array([r["mean_delay_ms"] for r in rr])
        conv.append({"sim_time_s": T, "mean_delay_ms": v.mean(),
                     "ci95": stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)),
                     "rel_ci_pct": 100 * stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) / v.mean()})
    pd.DataFrame(conv).to_csv(OUT / "run_length_convergence.csv", index=False)

    # one long run with raw delays and a queue trace (for CDF / time-series plots)
    for label, c in (("baseline", base()), ("constrained", constrained())):
        c = c.with_(sim_time_s=300.0, warmup_s=30.0)
        r = simulate(c, np.random.default_rng(SEED), trace_window=(100.0, 102.0), keep_delays=True)
        np.savez_compressed(OUT / f"raw_{label}.npz", trace=np.array(r["_trace"]),
                            **{f"delay_{k}": v for k, v in r["_delays_ms"].items()})
    log("baseline done")


# =========================================================================================
# 2. Bandwidth sweep (traffic fixed, capacity R varied)
# =========================================================================================
def stage_bandwidth():
    log("bandwidth sweep")
    rows = []
    for R in (2.2e6, 2.4e6, 2.5e6, 2.6e6, 2.8e6, 3.0e6, 3.25e6, 3.5e6, 4.0e6, 5.0e6, 6.0e6, 8.0e6, 10.0e6):
        cfg = baseline_config(rate_bps=R, sim_time_s=300.0, warmup_s=30.0)
        s = run_and_summarize(cfg, n_reps=12, seed=SEED, workers=WORKERS)
        th = pk_metrics(cfg)
        rows.append(flat_row("bandwidth_sweep", s, rate_mbps=R / 1e6, rho_theory=th["rho"],
                             pk_delay_ms=th["W_ms"], pk_wait_ms=th["Wq_ms"],
                             mean_service_ms=th["ES"] * 1e3))
        log(f"   R={R/1e6:5.2f} Mbps rho={th['rho']:.3f} W={s['mean_delay_ms']:.2f} ms "
            f"p99={s['p99_delay_ms']:.1f} Pb={s['blocking_prob']:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "bandwidth_sweep.csv", index=False)


# =========================================================================================
# 3. Offered-load sweep at several capacities
# =========================================================================================
def stage_load():
    log("load sweep")
    rows = []
    for R in (3.0e6, 5.0e6, 10.0e6):
        for g in (0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0):
            cfg = baseline_config(rate_bps=R, load_scale=g, sim_time_s=300.0, warmup_s=30.0)
            s = run_and_summarize(cfg, n_reps=10, seed=SEED, workers=WORKERS)
            th = pk_metrics(cfg)
            rows.append(flat_row("load_sweep", s, rate_mbps=R / 1e6, load_scale=g, rho_theory=th["rho"],
                                 pk_delay_ms=th["W_ms"]))
        log(f"   R={R/1e6} Mbps done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "load_sweep.csv", index=False)

    # maximum sustainable traffic intensity: bisection on the load factor g (common random numbers)
    out = []
    for R in (2.8e6, 3.0e6, 4.0e6, 5.0e6, 7.0e6, 10.0e6):
        lo, hi = 0.1, 8.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            ok, _s = _sla_ok(baseline_config(rate_bps=R, load_scale=mid, sim_time_s=250.0, warmup_s=25.0), 6)
            if ok:
                lo = mid
            else:
                hi = mid
        cfg_ok = baseline_config(rate_bps=R, load_scale=lo, sim_time_s=250.0, warmup_s=25.0)
        s_ok = run_and_summarize(cfg_ok, n_reps=10, seed=SEED, workers=WORKERS)
        s_hi = run_and_summarize(baseline_config(rate_bps=R, load_scale=hi, sim_time_s=250.0, warmup_s=25.0),
                                 n_reps=10, seed=SEED, workers=WORKERS)
        binding = ", ".join(k for k, v in SLA.items() if s_hi[k] > v) or "none"
        out.append({"rate_mbps": R / 1e6, "load_scale_max": lo, "packets_per_s_max": 380.0 * lo,
                    "rho_max": s_ok["rho_offered"], "goodput_mbps_max": s_ok["goodput_mbps"],
                    "mean_delay_ms": s_ok["mean_delay_ms"], "p99_delay_ms": s_ok["p99_delay_ms"],
                    "blocking_prob": s_ok["blocking_prob"], "binding_criterion_just_above": binding})
        log(f"   max sustainable load at R={R/1e6:4.1f} Mbps: g={lo:.3f} ({380*lo:.0f} pkt/s)  rho={s_ok['rho_offered']:.3f}"
            f"  binding={binding}")
    pd.DataFrame(out).to_csv(OUT / "sustainable_load.csv", index=False)
    log("load sweep done")


# =========================================================================================
# 4. Constrained vs improved designs
# =========================================================================================
COMP = dict(compression_ratio=COMPRESSION, compression_proc_s=COMPRESSION_PROC)
# (short id, label, configuration)
SCENARIO_LIST = [
    ("baseline", "Baseline (5 Mbps)", baseline_config(rate_bps=R_BASE)),
    ("constrained", "Constrained (2.8 Mbps)", baseline_config(rate_bps=R_CONSTRAINED)),
    ("impA", "Improved A: +bandwidth (4 Mbps)", baseline_config(rate_bps=4.0e6)),
    ("impB", "Improved B: compression (2.8 Mbps)", baseline_config(rate_bps=R_CONSTRAINED, **COMP)),
    ("impC", "Improved C: compression + 3.2 Mbps", baseline_config(rate_bps=3.2e6, **COMP)),
    ("severe", "Severe constraint (2.4 Mbps)", baseline_config(rate_bps=2.4e6)),
    ("severeB", "Severe + compression (2.4 Mbps)", baseline_config(rate_bps=2.4e6, **COMP)),
]


def scenarios() -> dict:
    return {label: cfg for _, label, cfg in SCENARIO_LIST}


def stage_improve():
    log("scenario comparison: 30 replications x 600 s")
    rows, cls_rows = [], []
    for name, cfg in scenarios().items():
        cfg = cfg.with_(sim_time_s=600.0, warmup_s=60.0)
        s = run_and_summarize(cfg, n_reps=30, seed=SEED, workers=WORKERS)
        th = pk_metrics(cfg)
        rows.append(flat_row(name, s, rate_mbps=cfg.rate_bps / 1e6, buffer_pkts=cfg.buffer_pkts,
                             compression=cfg.compression_ratio, rho_theory=th["rho"], pk_delay_ms=th["W_ms"]))
        for tc in cfg.classes:
            p = tc.name + "."
            cls_rows.append({"scenario": name, "class": tc.name,
                             **{k: s[p + k] for k in ("mean_delay_ms", "p99_delay_ms", "blocking_prob",
                                                      "loss_prob", "pdr", "goodput_mbps")},
                             **{k + "__ci": s[p + k + "__ci"] for k in ("mean_delay_ms", "p99_delay_ms")}})
        log(f"   {name:40s} rho={s['rho_offered']:.3f} W={s['mean_delay_ms']:.2f} ms p99={s['p99_delay_ms']:.1f} "
            f"Pb={s['blocking_prob']:.4f} PDR={s['pdr']:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "scenarios.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(OUT / "scenarios_by_class.csv", index=False)

    # raw delay samples for CDFs and queue traces
    arrays = {}
    for key, name, cfg in SCENARIO_LIST:
        r = simulate(cfg.with_(sim_time_s=300.0, warmup_s=30.0), np.random.default_rng(SEED),
                     trace_window=(100.0, 102.0), keep_delays=True)
        arrays[f"delay_{key}"] = r["_delays_ms"]["All"]
        arrays[f"trace_{key}"] = np.array(r["_trace"])
    np.savez_compressed(OUT / "raw_scenarios.npz", **arrays)

    # compression-ratio sensitivity at the constrained capacity
    comp = []
    for cr in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        cfg = baseline_config(rate_bps=R_CONSTRAINED, compression_ratio=cr,
                              compression_proc_s=COMPRESSION_PROC if cr > 0 else 0.0,
                              sim_time_s=300.0, warmup_s=30.0)
        s = run_and_summarize(cfg, n_reps=12, seed=SEED, workers=WORKERS)
        comp.append(flat_row("compression_sweep", s, compression=cr, rho_theory=pk_metrics(cfg)["rho"]))
    pd.DataFrame(comp).to_csv(OUT / "compression_sweep.csv", index=False)
    log("scenario comparison done")


# =========================================================================================
# 5. Buffer sensitivity (does a bigger buffer cure a bandwidth shortage?)
# =========================================================================================
def stage_buffer():
    log("buffer sensitivity")
    rows = []
    for label, R in (("rho~0.9 (2.8 Mbps)", 2.8e6), ("rho~1.1 (2.3 Mbps)", 2.3e6)):
        for B in (5, 10, 25, 50, 100, 200, 500, 1000, 3000):
            cfg = baseline_config(rate_bps=R, buffer_pkts=B, sim_time_s=300.0, warmup_s=30.0)
            s = run_and_summarize(cfg, n_reps=10, seed=SEED, workers=WORKERS)
            rows.append(flat_row("buffer_sweep", s, case=label, rate_mbps=R / 1e6, buffer_pkts=B,
                                 rho_theory=pk_metrics(cfg)["rho"]))
        log(f"   {label} done")
    pd.DataFrame(rows).to_csv(OUT / "buffer_sweep.csv", index=False)


# =========================================================================================
# 6. Capacity planning: minimum R that meets the SLA for a given demand
# =========================================================================================
def _sla_ok(cfg, n_reps=8):
    s = run_and_summarize(cfg, n_reps=n_reps, seed=SEED, workers=WORKERS)
    return all(s[k] <= v for k, v in SLA.items()), s


def stage_capacity():
    log("capacity planning (bisection on R with common random numbers)")
    rows = []
    for g in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
        lo, hi = 1.0e6, 30.0e6
        for _ in range(11):
            mid = 0.5 * (lo + hi)
            ok, _ = _sla_ok(baseline_config(rate_bps=mid, load_scale=g, sim_time_s=250.0, warmup_s=25.0), 6)
            if ok:
                hi = mid
            else:
                lo = mid
        cfg = baseline_config(rate_bps=hi, load_scale=g, sim_time_s=250.0, warmup_s=25.0)
        _, s = _sla_ok(cfg, 8)
        base_cfg = baseline_config(load_scale=g)
        Rpk = required_rate_pk(base_cfg, SLA["mean_delay_ms"] * 1e-3)
        th = pk_metrics(cfg)
        rows.append({"load_scale": g, "offered_pps": 380 * g, "offered_channel_mbps": s["offered_mbps"],
                     "R_stability_mbps": stability_rate(base_cfg) / 1e6,
                     "R_required_sim_mbps": hi / 1e6, "R_required_pk_meandelay_mbps": Rpk / 1e6,
                     "rho_at_required": th["rho"], "mean_delay_ms": s["mean_delay_ms"],
                     "p99_delay_ms": s["p99_delay_ms"], "blocking_prob": s["blocking_prob"]})
        log(f"   g={g}: R_req(sim)={hi/1e6:.3f} Mbps   R_req(P-K mean only)={Rpk/1e6:.3f}   rho={th['rho']:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "capacity_planning.csv", index=False)


# =========================================================================================
# 7. Monte Carlo over uncertain inputs
# =========================================================================================
SIGMA_GLOBAL = 0.15   # log-std of overall demand uncertainty (e.g. +-15 %)
SIGMA_CLASS = 0.10    # log-std of independent per-class activity uncertainty
SIGMA_BER = 0.50      # log-std of BER (interference / weather)
N_MC = 300
MC_SIM_TIME = 300.0
MC_WARMUP = 30.0


def _mc_draw(cfg: NetworkConfig, rng):
    g = float(np.exp(rng.normal(0, SIGMA_GLOBAL)))
    mult = np.exp(rng.normal(0, SIGMA_CLASS, len(cfg.classes)))
    ber = float(cfg.ber * np.exp(rng.normal(0, SIGMA_BER)))
    classes = tuple(TrafficClass(tc.name, tc.n_devices, tc.rate_per_device * g * m, tc.payload_bytes, tc.size_dist,
                                 tc.compressible) for tc, m in zip(cfg.classes, mult))
    return cfg.with_(classes=classes, ber=ber), {"demand_factor": g, **{f"mult_{tc.name}": m for tc, m in
                                                                      zip(cfg.classes, mult)}, "ber": ber}


def _mc_job(args):
    cfg, params, seed_seq = args
    r = simulate(cfg, np.random.default_rng(seed_seq))
    keep = {k: r[k] for k in ("mean_delay_ms", "p99_delay_ms", "blocking_prob", "loss_prob", "pdr",
                              "utilization", "goodput_mbps", "rho_offered", "mean_queue_len")}
    return {**params, **keep}


def _mc_draw_set():
    """The SAME uncertain-input draws are reused for every design (paired comparison)."""
    child = np.random.SeedSequence(SEED + 7).spawn(N_MC)
    draws = [_mc_draw(baseline_config(), np.random.default_rng(c))[1] for c in child]
    return draws, np.random.SeedSequence(SEED + 11).spawn(N_MC)


def _mc_run(cfg0: NetworkConfig, draws, sim_seeds) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for p, sd in zip(draws, sim_seeds):
        classes = tuple(TrafficClass(tc.name, tc.n_devices,
                                     tc.rate_per_device * p["demand_factor"] * p[f"mult_{tc.name}"],
                                     tc.payload_bytes, tc.size_dist, tc.compressible) for tc in cfg0.classes)
        jobs.append((cfg0.with_(classes=classes, ber=p["ber"]), p, sd))
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(_mc_job, jobs, chunksize=4))
    df = pd.DataFrame(res)
    df["sla_ok"] = (df["mean_delay_ms"] <= SLA["mean_delay_ms"]) & (df["p99_delay_ms"] <= SLA["p99_delay_ms"]) \
        & (df["blocking_prob"] <= SLA["blocking_prob"])
    return df


def stage_montecarlo():
    log(f"Monte Carlo: {N_MC} trials per scenario")
    draws, sim_seeds = _mc_draw_set()
    rows = []
    for key, name, cfg in SCENARIO_LIST:
        df = _mc_run(cfg.with_(sim_time_s=MC_SIM_TIME, warmup_s=MC_WARMUP), draws, sim_seeds)
        df.insert(0, "scenario", name)
        rows.append(df)
        log(f"   {name:40s} P(SLA met)={df['sla_ok'].mean():.3f}  median mean-delay={df['mean_delay_ms'].median():.2f} ms"
            f"  p95={df['mean_delay_ms'].quantile(0.95):.2f} ms")
    alldf = pd.concat(rows, ignore_index=True)
    alldf.to_csv(OUT / "montecarlo.csv", index=False)

    # sensitivity: Spearman rank correlation between each uncertain input and each output
    sens = []
    for name, d in alldf.groupby("scenario"):
        for inp in ("demand_factor", "mult_Student", "mult_Faculty", "mult_Admin", "mult_IoT", "ber"):
            for out in ("mean_delay_ms", "p99_delay_ms", "loss_prob"):
                rho, p = stats.spearmanr(d[inp], d[out])
                sens.append({"scenario": name, "input": inp, "output": out, "spearman": rho, "p_value": p})
    pd.DataFrame(sens).to_csv(OUT / "montecarlo_sensitivity.csv", index=False)
    log("Monte Carlo done")


def stage_robust():
    """Capacity needed so that the SLA is met with high PROBABILITY under uncertain demand and BER."""
    log("robust capacity: P(SLA met) vs channel capacity")
    draws, sim_seeds = _mc_draw_set()
    rows = []
    for R in (2.8e6, 3.0e6, 3.25e6, 3.5e6, 3.75e6, 4.0e6, 4.5e6, 5.0e6, 5.5e6, 6.0e6, 7.0e6):
        df = _mc_run(baseline_config(rate_bps=R, sim_time_s=MC_SIM_TIME, warmup_s=MC_WARMUP), draws, sim_seeds)
        n = len(df)
        p = df["sla_ok"].mean()
        lo, hi = stats.beta.ppf(0.025, df["sla_ok"].sum() + 0.5, n - df["sla_ok"].sum() + 0.5), \
            stats.beta.ppf(0.975, df["sla_ok"].sum() + 0.5, n - df["sla_ok"].sum() + 0.5)
        rows.append({"rate_mbps": R / 1e6, "p_sla_met": p, "p_ci_low": lo, "p_ci_high": hi,
                     "median_mean_delay_ms": df["mean_delay_ms"].median(),
                     "p95_mean_delay_ms": df["mean_delay_ms"].quantile(0.95),
                     "mean_blocking": df["blocking_prob"].mean(), "frac_rho_ge_1": (df["rho_offered"] >= 1).mean()})
        log(f"   R={R/1e6:4.2f} Mbps  P(SLA met)={p:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "robust_capacity.csv", index=False)


STAGES = {
    "validation": stage_validation,
    "baseline": stage_baseline,
    "bandwidth": stage_bandwidth,
    "load": stage_load,
    "improve": stage_improve,
    "buffer": stage_buffer,
    "capacity": stage_capacity,
    "montecarlo": stage_montecarlo,
    "robust": stage_robust,
}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(STAGES)
    t0 = time.time()
    for st in todo:
        STAGES[st]()
    log(f"finished in {time.time() - t0:.0f} s")
