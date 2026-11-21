# Group 7 – Limited Wireless Bandwidth

**Modeling and Simulation · ECE 3rd year · Design and Performance Optimization of a Stochastic Wireless Communication Network**

Discrete-event simulation of a campus gateway whose shared wireless channel has limited capacity.
It measures how channel capacity affects transmission time, delay, congestion, loss and throughput,
estimates the capacity the expected traffic requires, and tests improvements (add bandwidth,
payload compression, buffer enlargement) under demand and bit-error-rate uncertainty.

## Contents

| File / folder | Purpose |
|---|---|
| `Group7_Limited_Wireless_Bandwidth_Report.docx` | Full report (system, assumptions, math model, method, validation, results, interpretation, limitations) |
| `wnet_sim.py` | Simulation model, event engine, replications + confidence intervals, analytical formulas (M/G/1, M/M/1/K, required capacity) |
| `experiments.py` | All experiments → CSV files in `results/` |
| `plots.py` | Result figures → `figures/` |
| `diagrams.py` | System block diagram and simulation flowchart → `figures/` |
| `build_report.py` | Builds the Word report from the CSVs and figures (all numbers in the report come from the data) |
| `results/` | CSV outputs, raw samples (`.npz`), run log |
| `figures/` | All figures used in the report |

## Run

```bash
pip install -r requirements.txt
python experiments.py          # all stages, ~7 min on 2 cores (or e.g. `python experiments.py validation baseline`)
python plots.py
python diagrams.py
python build_report.py
python wnet_sim.py             # 5-second demo: baseline vs Pollaczek–Khinchine theory
```

The master random seed (2026) is in `experiments.py`. Parameters (traffic mix, capacity, buffer, BER,
service targets) are defined at the top of `wnet_sim.py` and `experiments.py`.
Set the number of worker processes with `WORKERS` in `experiments.py`.

## Model in one paragraph

Four Poisson traffic classes (student, faculty, administrative, IoT; 380 packets/s in total) share one
FIFO gateway with a 100-packet buffer and one channel of capacity R. Service time is
`8(payload+header)/R + 50 µs`; packets arriving to a full buffer are blocked; bit errors (BER 1e-6)
corrupt packets with probability `1-(1-BER)^bits`. Service targets used to define "acceptable":
mean delay ≤ 10 ms, p99 ≤ 50 ms, blocking ≤ 0.5 %.

## Headline results (nominal conditions)

| Network | Mean delay | p99 delay | Targets met |
|---|---|---|---|
| Baseline, 5 Mbit/s | 2.17 ms | 7.2 ms | yes |
| Constrained, 2.8 Mbit/s | 13.7 ms | 58 ms | no |
| A: 4 Mbit/s | 3.4 ms | 12 ms | yes |
| B: compression at 2.8 Mbit/s | 4.0 ms | 15 ms | yes |
| C: compression + 3.2 Mbit/s | 3.0 ms | 10 ms | yes |

Required capacity for the expected traffic: **2.94 Mbit/s** nominal; **≈ 4 Mbit/s** for ≥ 95 % assurance under
uncertain demand and BER (Monte Carlo, 300 trials).

## Notes

* The event loop is hand-written (SimPy was not available in the build environment); it is validated against
  M/M/1, M/D/1, M/M/1/K and Pollaczek–Khinchine results (69 of 70 comparisons inside the 95 % CI, max error 1.8 %).
* Traffic mix, service targets, uncertainty distributions and the 30 % compression ratio are assumptions
  (stated and discussed in the report), not measurements.
