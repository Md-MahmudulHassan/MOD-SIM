#!/usr/bin/env python3
"""
wnet_sim.py  -  Discrete-event simulation of a bandwidth-limited campus wireless link
=====================================================================================
Modeling and Simulation | ECE 3rd year | Group 7: Limited wireless bandwidth

System modelled
---------------
Four traffic classes (student, faculty, administrative, IoT) generate packets
according to independent Poisson processes.  All packets are forwarded through
ONE gateway that owns ONE shared wireless channel of capacity R (bit/s).

    sources --> [ finite FIFO buffer (B waiting slots) ] --> [ gateway + channel (rate R) ] --> destination
                       blocked when full                        service time = bits / R + t_proc

* Arrivals            : Poisson process per class (exponential inter-arrivals).
* Packet size         : class-dependent payload + fixed header.
* Service (transmit)  : S = 8 (payload + header) / R  +  t_proc   (deterministic given size).
* Channel errors      : independent bit errors with BER p_b  ->  packet error
                        probability  1 - (1 - p_b)^(bits)  (a corrupted packet still
                        occupies the channel and is discarded at the destination;
                        no retransmission - that is the subject of another group).
* Buffer              : B waiting slots + 1 packet in service.  An arrival that finds
                        the buffer full is blocked (lost).
* Discipline          : FIFO, single server, non-preemptive.

The engine is a next-event-time-advance discrete-event simulator written for speed in
pure Python/NumPy (SimPy is not required).  Only two event types exist (ARRIVAL,
DEPARTURE) and the future-event list has at most two entries, so no heap is needed.

The module also contains closed-form queueing results (Pollaczek-Khinchine M/G/1,
M/M/1/K) that are used to validate the simulator.

Units: time in seconds, size in bytes, rate in bit/s, delays reported in milliseconds.
"""
from __future__ import annotations

import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------------------
# 1.  Model description
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrafficClass:
    """One population of identical devices.

    n_devices * rate_per_device is the aggregate Poisson arrival rate (packets/s).
    """
    name: str
    n_devices: int
    rate_per_device: float          # packets / s / device
    payload_bytes: float            # mean payload size (bytes)
    size_dist: str = "fixed"        # "fixed" or "exp" (exp is used for validation only)
    compressible: bool = False      # can the gateway compress this payload?

    @property
    def rate(self) -> float:
        return self.n_devices * self.rate_per_device


@dataclass(frozen=True)
class NetworkConfig:
    classes: Tuple[TrafficClass, ...]
    rate_bps: float = 5.0e6         # channel capacity R (bit/s)
    header_bytes: float = 40.0      # per-packet protocol overhead (bytes)
    buffer_pkts: int = 100          # B: waiting slots (excludes the packet in service)
    proc_time_s: float = 50e-6      # fixed gateway processing time per packet (s)
    ber: float = 1e-6               # channel bit error rate
    load_scale: float = 1.0         # multiplies every class arrival rate
    compression_ratio: float = 0.0  # fraction of payload removed for compressible classes
    compression_proc_s: float = 0.0 # extra gateway time for compressing one packet (s)
    sim_time_s: float = 300.0       # simulated time per replication (s)
    warmup_s: float = 30.0          # initial transient that is discarded (s)

    def with_(self, **kw) -> "NetworkConfig":
        return replace(self, **kw)


def default_classes() -> Tuple[TrafficClass, ...]:
    """Baseline traffic mix: 380 packets/s in total."""
    return (
        TrafficClass("Student", n_devices=400, rate_per_device=0.5, payload_bytes=1000, compressible=True),
        TrafficClass("Faculty", n_devices=50, rate_per_device=1.0, payload_bytes=1200, compressible=True),
        TrafficClass("Admin", n_devices=30, rate_per_device=1.0, payload_bytes=600, compressible=True),
        TrafficClass("IoT", n_devices=500, rate_per_device=0.2, payload_bytes=100, compressible=False),
    )


def baseline_config(**kw) -> NetworkConfig:
    """Normal operating conditions: R = 5 Mbit/s, B = 100 packets, BER = 1e-6."""
    return NetworkConfig(classes=default_classes(), **kw)


# SLA used throughout the study (acceptable-quality thresholds)
SLA = {"mean_delay_ms": 10.0, "p99_delay_ms": 50.0, "blocking_prob": 0.005}


# --------------------------------------------------------------------------------------
# 2.  Per-packet quantities
# --------------------------------------------------------------------------------------


def _class_arrays(cfg: NetworkConfig):
    comp = np.array([tc.compressible for tc in cfg.classes], dtype=bool)
    return comp


def _packet_bits_and_service(cfg: NetworkConfig, cls: np.ndarray, payload: np.ndarray):
    comp_flag = _class_arrays(cfg)[cls]
    pay_eff = np.where(comp_flag, payload * (1.0 - cfg.compression_ratio), payload)
    bits = 8.0 * (pay_eff + cfg.header_bytes)
    svc = bits / cfg.rate_bps + cfg.proc_time_s + comp_flag * cfg.compression_proc_s
    return bits, svc, pay_eff


def _generate_traffic(cfg: NetworkConfig, rng: np.random.Generator):
    """Superposition of independent Poisson processes, simulated exactly:
    N ~ Poisson(lambda T) and, given N, arrival instants are i.i.d. U(0, T)."""
    times, cls, pay = [], [], []
    for k, tc in enumerate(cfg.classes):
        lam = tc.rate * cfg.load_scale
        n = rng.poisson(lam * cfg.sim_time_s)
        times.append(rng.random(n) * cfg.sim_time_s)
        cls.append(np.full(n, k, dtype=np.int16))
        if tc.size_dist == "fixed":
            pay.append(np.full(n, float(tc.payload_bytes)))
        elif tc.size_dist == "exp":
            pay.append(rng.exponential(tc.payload_bytes, n))
        else:
            raise ValueError(f"unknown size_dist {tc.size_dist!r}")
    t = np.concatenate(times)
    c = np.concatenate(cls)
    p = np.concatenate(pay)
    order = np.argsort(t, kind="stable")
    return t[order], c[order], p[order]


# --------------------------------------------------------------------------------------
# 3.  Discrete-event engine
# --------------------------------------------------------------------------------------


def _run_queue(arr: np.ndarray, svc: np.ndarray, buffer_pkts: int, t_warm: float, t_end: float,
               trace_window: Optional[Tuple[float, float]] = None):
    """Single-server FIFO queue with a finite waiting room.

    State: q (waiting packets), busy (0/1), t_dep (time of the scheduled departure).
    Events: ARRIVAL  -> start service / enqueue / block;
            DEPARTURE -> record departure, start next service or go idle.
    Time-weighted statistics are integrated over the measurement window [t_warm, t_end].
    """
    arr_l = arr.tolist()
    svc_l = svc.tolist()
    n = len(arr_l)
    INF = float("inf")

    admitted = bytearray(n)
    depart = [0.0] * n
    wait = [0.0] * n
    q: deque = deque()
    busy = 0
    cur = -1
    t_dep = INF
    i = 0
    t_prev = 0.0
    a_sys = a_q = a_busy = 0.0
    max_q = 0
    tr = [] if trace_window is not None else None
    tr_lo, tr_hi = trace_window if trace_window is not None else (0.0, 0.0)

    while i < n or busy:
        t_arr = arr_l[i] if i < n else INF
        if t_arr < t_dep:
            t = t_arr
            is_arrival = True
        else:
            t = t_dep
            is_arrival = False

        # integrate the state over [t_prev, t] intersected with the measurement window
        lo = t_prev if t_prev > t_warm else t_warm
        hi = t if t < t_end else t_end
        if hi > lo:
            dt = hi - lo
            lq = len(q)
            a_q += lq * dt
            a_sys += (lq + busy) * dt
            a_busy += busy * dt
        t_prev = t

        if is_arrival:
            if not busy:                       # idle server: transmit immediately
                busy = 1
                cur = i
                t_dep = t + svc_l[i]
                admitted[i] = 1
            elif len(q) < buffer_pkts:         # room in the buffer: wait
                q.append(i)
                admitted[i] = 1
                if t >= t_warm and len(q) > max_q:
                    max_q = len(q)
            # else: buffer full -> packet blocked (admitted[i] stays 0)
            i += 1
        else:                                  # departure of the packet in service
            depart[cur] = t
            if q:
                j = q.popleft()
                wait[j] = t - arr_l[j]
                cur = j
                t_dep = t + svc_l[j]
            else:
                busy = 0
                cur = -1
                t_dep = INF

        if tr is not None and tr_lo <= t <= tr_hi:
            tr.append((t, len(q) + busy))

    window = t_end - t_warm
    return (np.frombuffer(admitted, dtype=np.uint8).astype(bool), np.array(depart), np.array(wait),
            {"L": a_sys / window, "Lq": a_q / window, "util": a_busy / window, "max_q": max_q}, tr)


# --------------------------------------------------------------------------------------
# 4.  One simulation replication -> flat dictionary of metrics
# --------------------------------------------------------------------------------------


def _subset_metrics(prefix: str, mask: np.ndarray, admitted, corrupt, delay, wait, svc, pay0, window):
    adm = mask & admitted
    offered = int(mask.sum())
    n_adm = int(adm.sum())
    blocked = offered - n_adm
    corr = int((adm & corrupt).sum())
    deliv = n_adm - corr
    d = delay[adm] * 1e3
    w = wait[adm] * 1e3
    nan = float("nan")
    out = {
        "offered_pkts": offered,
        "blocked_pkts": blocked,
        "corrupted_pkts": corr,
        "delivered_pkts": deliv,
        "blocking_prob": blocked / offered if offered else nan,
        "corruption_prob": corr / n_adm if n_adm else nan,
        "loss_prob": (blocked + corr) / offered if offered else nan,
        "pdr": deliv / offered if offered else nan,
        "mean_delay_ms": float(d.mean()) if n_adm else nan,
        "mean_wait_ms": float(w.mean()) if n_adm else nan,
        "p50_delay_ms": float(np.percentile(d, 50)) if n_adm else nan,
        "p95_delay_ms": float(np.percentile(d, 95)) if n_adm else nan,
        "p99_delay_ms": float(np.percentile(d, 99)) if n_adm else nan,
        "max_delay_ms": float(d.max()) if n_adm else nan,
        "goodput_mbps": float((pay0[adm & ~corrupt] * 8.0).sum() / window / 1e6),
        "offered_work": float(svc[mask].sum() / window),     # = offered traffic intensity rho
    }
    return {(prefix + k): v for k, v in out.items()}


def simulate(cfg: NetworkConfig, rng: np.random.Generator, trace_window: Optional[Tuple[float, float]] = None,
             keep_delays: bool = False) -> Dict[str, float]:
    """Run ONE replication and return overall and per-class metrics."""
    arr, cls, pay0 = _generate_traffic(cfg, rng)
    bits, svc, pay_eff = _packet_bits_and_service(cfg, cls, pay0)
    if cfg.ber > 0:
        p_err = -np.expm1(bits * np.log1p(-cfg.ber))
    else:
        p_err = np.zeros_like(bits)
    corrupt = rng.random(arr.size) < p_err

    admitted, depart, wait, qs, tr = _run_queue(arr, svc, cfg.buffer_pkts, cfg.warmup_s, cfg.sim_time_s,
                                                trace_window)
    delay = depart - arr
    window = cfg.sim_time_s - cfg.warmup_s
    in_win = arr >= cfg.warmup_s

    res: Dict[str, float] = {}
    res.update(_subset_metrics("", in_win, admitted, corrupt, delay, wait, svc, pay0, window))
    for k, tc in enumerate(cfg.classes):
        res.update(_subset_metrics(tc.name + ".", in_win & (cls == k), admitted, corrupt, delay, wait, svc,
                                   pay0, window))

    adm_w = in_win & admitted
    res["offered_mbps"] = float(bits[in_win].sum() / window / 1e6)
    res["carried_mbps"] = float(bits[adm_w].sum() / window / 1e6)
    res["utilization"] = qs["util"]
    res["mean_queue_len"] = qs["Lq"]
    res["mean_in_system"] = qs["L"]
    res["max_queue_len"] = float(qs["max_q"])
    lam_adm = adm_w.sum() / window
    res["little_ratio"] = qs["L"] / (lam_adm * res["mean_delay_ms"] * 1e-3) if lam_adm > 0 else float("nan")
    res["rho_offered"] = res["offered_work"]
    res["arrival_rate_pps"] = float(in_win.sum() / window)

    if keep_delays:
        res["_delays_ms"] = {"All": delay[adm_w] * 1e3}
        for k, tc in enumerate(cfg.classes):
            res["_delays_ms"][tc.name] = delay[adm_w & (cls == k)] * 1e3
    if tr is not None:
        res["_trace"] = tr
    return res


# --------------------------------------------------------------------------------------
# 5.  Replications, parallel execution and confidence intervals
# --------------------------------------------------------------------------------------


def _worker(args):
    cfg, seed_seq = args
    return simulate(cfg, np.random.default_rng(seed_seq))


def run_replications(cfg: NetworkConfig, n_reps: int, seed: int = 2026, workers: int = 2) -> List[Dict[str, float]]:
    """Independent replications.  Using the same `seed` for different configurations
    gives common random numbers (variance reduction when comparing designs)."""
    seeds = np.random.SeedSequence(seed).spawn(n_reps)
    jobs = [(cfg, s) for s in seeds]
    if workers <= 1:
        return [_worker(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_worker, jobs))


def summarize(results: Sequence[Dict[str, float]], conf: float = 0.95) -> Dict[str, float]:
    """Mean and half-width of the t confidence interval for each scalar metric.
    Keys: <metric> (mean) and <metric>__ci (half width)."""
    out: Dict[str, float] = {}
    n = len(results)
    tcrit = stats.t.ppf(0.5 + conf / 2, n - 1) if n > 1 else float("nan")
    for k in results[0]:
        if k.startswith("_"):
            continue
        v = np.array([r[k] for r in results], dtype=float)
        out[k] = float(np.nanmean(v))
        out[k + "__ci"] = float(tcrit * np.nanstd(v, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    out["n_reps"] = n
    return out


def run_and_summarize(cfg: NetworkConfig, n_reps: int = 10, seed: int = 2026, workers: int = 2):
    return summarize(run_replications(cfg, n_reps, seed, workers))


# --------------------------------------------------------------------------------------
# 6.  Analytical results used for validation and capacity planning
# --------------------------------------------------------------------------------------


def service_moments(cfg: NetworkConfig):
    """Mean and second moment of the service time of the aggregate (mixed) traffic,
    plus the total arrival rate.  Works for fixed and exponential payload sizes."""
    lam_tot = sum(tc.rate * cfg.load_scale for tc in cfg.classes)
    ES = ES2 = 0.0
    for tc in cfg.classes:
        lam = tc.rate * cfg.load_scale
        pay = tc.payload_bytes * ((1.0 - cfg.compression_ratio) if tc.compressible else 1.0)
        extra = cfg.proc_time_s + (cfg.compression_proc_s if tc.compressible else 0.0)
        fixed_part = 8.0 * cfg.header_bytes / cfg.rate_bps + extra
        var_part = 8.0 * pay / cfg.rate_bps
        if tc.size_dist == "fixed":
            m, var = fixed_part + var_part, 0.0
        else:                                    # payload ~ Exp(mean pay): variance = mean^2
            m, var = fixed_part + var_part, var_part ** 2
        w = lam / lam_tot
        ES += w * m
        ES2 += w * (var + m * m)
    return lam_tot, ES, ES2


def pk_metrics(cfg: NetworkConfig) -> Dict[str, float]:
    """Pollaczek-Khinchine formulas for an M/G/1 queue (infinite buffer, no blocking):
        rho = lambda E[S],  Wq = lambda E[S^2] / (2 (1 - rho)),  W = Wq + E[S]."""
    lam, ES, ES2 = service_moments(cfg)
    rho = lam * ES
    if rho >= 1:
        return {"rho": rho, "lambda": lam, "ES": ES, "Wq_ms": math.inf, "W_ms": math.inf,
                "Lq": math.inf, "L": math.inf}
    Wq = lam * ES2 / (2 * (1 - rho))
    W = Wq + ES
    return {"rho": rho, "lambda": lam, "ES": ES, "Wq_ms": Wq * 1e3, "W_ms": W * 1e3,
            "Lq": lam * Wq, "L": lam * W}


def mm1k_metrics(lam: float, mu: float, K: int) -> Dict[str, float]:
    """M/M/1/K (K = total capacity incl. the packet in service)."""
    rho = lam / mu
    if abs(rho - 1) < 1e-12:
        pK = 1.0 / (K + 1)
        L = K / 2.0
    else:
        pK = (1 - rho) * rho ** K / (1 - rho ** (K + 1))
        L = rho / (1 - rho) - (K + 1) * rho ** (K + 1) / (1 - rho ** (K + 1))
    lam_eff = lam * (1 - pK)
    W = L / lam_eff
    return {"rho": rho, "blocking": pK, "L": L, "W_s": W, "Wq_s": W - 1.0 / mu, "util": lam_eff / mu}


def required_rate_pk(cfg: NetworkConfig, mean_delay_target_s: float, hi: float = 1e9) -> float:
    """Smallest channel rate R (bit/s) for which the P-K mean delay is <= target
    (infinite-buffer approximation).  Solved by bisection."""
    a, b = stability_rate(cfg), hi          # W(R) is decreasing on (R_stab, inf)
    for _ in range(200):
        mid = 0.5 * (a + b)
        w = pk_metrics(cfg.with_(rate_bps=mid))["W_ms"] * 1e-3
        if w > mean_delay_target_s:
            a = mid
        else:
            b = mid
    return b


def stability_rate(cfg: NetworkConfig) -> float:
    """Smallest capacity R (bit/s) with rho < 1, i.e. the absolute minimum for a finite queue.
    rho(R) is monotonically decreasing, so it is found by bisection on rho(R) = 1."""
    a, b = 1.0, 1e12
    for _ in range(200):
        mid = 0.5 * (a + b)
        if pk_metrics(cfg.with_(rate_bps=mid))["rho"] >= 1:
            a = mid
        else:
            b = mid
    return b


def meets_sla(summary: Dict[str, float], sla: Dict[str, float] = SLA) -> bool:
    return all(summary[k] <= v for k, v in sla.items())


# --------------------------------------------------------------------------------------
# 7.  Quick demonstration
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = baseline_config()
    r = run_and_summarize(cfg, n_reps=4, workers=2)
    th = pk_metrics(cfg)
    print(f"Baseline: rho(theory)={th['rho']:.3f}  rho(sim)={r['rho_offered']:.3f}")
    print(f"  mean delay  sim={r['mean_delay_ms']:.3f} ms   P-K theory={th['W_ms']:.3f} ms")
    print(f"  p99 delay   sim={r['p99_delay_ms']:.2f} ms")
    print(f"  blocking    sim={r['blocking_prob']:.2e}   loss={r['loss_prob']:.3%}")
    print(f"  utilization sim={r['utilization']:.3f}   goodput={r['goodput_mbps']:.3f} Mbit/s")
