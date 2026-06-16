# Production Network Topology and Monetary Policy Transmission

*Does a country's position in the euro area input–output network shape how strongly ECB rate changes pass through into bank credit standards?*

---

## Abstract

This project constructs the intra–euro area production network from OECD Trade in Value Added (TiVA 2025) data for 1995–2022, measures how concentrated each country's production structure is, and tests whether that concentration moderates the transmission of ECB rate changes into bank credit standards (ECB Bank Lending Survey, 2003–2026).

The central result is conditional but robust: **during ECB tightening cycles, countries with more concentrated production networks tighten credit standards significantly less in response to rate hikes.** The interaction between rate changes and network concentration is −234 (p = 0.017) in the tightening-only sample and −178 (p = 0.045) after removing two structural outliers. The effect is absent in the full sample — which is expected, since the mechanism can only operate when rates are actually moving.

A temporal exponential random graph model (TERGM) shows the production network itself does **not** respond to monetary policy: bilateral supply-chain links persist almost deterministically year to year (memory coefficient 3.04), and ECB rate changes do not predict link formation or dissolution. This closes the causal loop — the network shapes credit transmission, not the reverse.

---

## The Mechanism

Standard monetary transmission models treat sectors as isolated borrowers whose response to higher rates depends on their own leverage and cash flow. Network position is absent.

This project adds one idea: a sector that supplies critical intermediate inputs to many others cannot easily cut its own activity when credit tightens, because its customers depend on its output. It is partially insulated. A country whose production is concentrated in such central sectors should therefore transmit monetary policy more weakly into credit conditions.

The prediction is a cross-country one: higher production network concentration → weaker pass-through of ECB rate changes into credit standards.

---

## Data

All data is free and publicly available; no proprietary sources are used.

| Dataset | Source | Role |
|---|---|---|
| TiVA 2025 — EXGR_INT (intermediate exports) | OECD Data Explorer | Production network edges |
| TiVA 2025 — PROD (gross output) | OECD Data Explorer | Node normalisation |
| Bank Lending Survey — credit standards, enterprises | ECB Data Portal | Transmission outcome |
| ECB main refinancing rate | ECB | Policy/treatment variable |

**Coverage:** 19 EA countries × 43 ISIC sectors × 28 years for the network; 18 countries × 94 quarters (2003 Q1–2026 Q2) for the BLS. The regression sample is 15 countries after excluding Cyprus and Malta (BLS samples of 3–5 banks, where one bank moves the national reading by 20–30 points) and Luxembourg (a financial centre whose BLS reflects international wholesale conditions rather than domestic transmission).

---

## Method

### Network construction

An annual directed weighted graph of intra-EA intermediate input flows. Each node is a country–sector pair; each edge is the USD-million value of intermediate inputs shipped from one country-sector to another within the euro area. Aggregate ISIC codes (e.g. "C" for all manufacturing) are filtered out so individual sectors are not double-counted against their own totals.

**Centrality — forward linkage (out-strength):** the total value a sector ships as intermediate inputs to the rest of the EA network. This is preferred to eigenvector centrality (degenerate on this disconnected directed graph) and PageRank (which rewards universal *recipients* such as postal and transport services). Forward linkage directly measures the research concept — how much the rest of the network depends on a sector's output — and corresponds to the influence vector of Acemoglu et al. (2012).

**Network Centralisation Index (NCI):** the Gini coefficient of sector-level forward linkage within a country. High NCI = one or two sectors dominate (Ireland, Luxembourg). Low NCI = many sectors contribute evenly (the Baltics).

### Transmission regression

```
BLS_tightening(i,t) = α_i + γ_t + β₁·(ΔRATE_t × NCI_i,t−1) + β₂·NCI_i,t−1 + ε(i,t)
```

Country fixed effects (α_i) absorb every time-invariant national characteristic. Time fixed effects (γ_t) absorb every common quarterly shock — including the ECB rate *level*, which is identical across countries. `ΔRATE_t` alone is therefore absorbed; the interaction `ΔRATE × NCI` survives because NCI varies across countries. β₁ is the coefficient of interest. Standard errors are clustered by country.

### Reverse-causality check (TERGM)

A temporal ERGM on the country-level network (19 nodes, 28 years) tests whether ECB rate changes predict the formation or dissolution of bilateral supply-chain links, controlling for endogenous structure (reciprocity, transitivity, memory) and node NCI.

---

## Results

### Main regression

| Specification | β (ΔRATE × NCI) | p-value | N |
|---|---:|---:|---:|
| Full sample | −17.5 | 0.483 | 1,056 |
| **Tightening episodes only** | **−233.7** | **0.017** | 120 |
| Extended to 2023 | −123.9 | 0.224 | 180 |
| **Tightening, excl. Baltic outliers** | **−177.9** | **0.045** | 116 |

The full-sample null is expected: three quarters of all observations have zero rate change, so the interaction is mechanically zero and carries no signal. Restricting to quarters where the ECB actually moved — where the mechanism can operate — yields a significant negative coefficient with the predicted sign. The effect survives dropping Estonia and Latvia, whose banking systems are Swedish-owned and track Riksbank rather than ECB policy.

### Country transmission sensitivities (BLS pp per 100 bps hike)

Strongest pass-through: Greece (−33), Netherlands (−30), Finland (−25). Weakest or perverse: Estonia (+21), Latvia (+16) — the Baltic outliers. Germany (−10) and France (−3) are structurally weak transmitters, consistent with their relationship-banking systems.

### Network dynamics (TERGM)

| Term | Estimate | 95% CI | Reading |
|---|---:|---|---|
| memory | **3.04** | [2.91, 3.19] | Links persist ~21× year to year |
| mutual | **2.66** | [2.33, 3.06] | Strong reciprocity in bilateral chains |
| gwesp (transitivity) | 0.28 | [−0.41, 1.20] | Weak, not significant |
| NCI (node) | −1.03 | [−2.73, 0.70] | Not significant |
| ΔRATE (edge covariate) | 0.015 | [−0.013, 0.063] | **Monetary policy does not move the network** |

The network is overwhelmingly persistent and does not respond to rate changes. In 2022 — the most aggressive ECB tightening on record (+450 bps) — the bilateral network was completely unchanged (Jaccard = 1.000). This is the finding that lets the project claim a direction of causality.

---

## Interpretation

The production network insulates credit transmission during tightening, the mechanism is nonlinear (it weakens in the extreme 2022–23 cycle, as if the shock was large enough to overwhelm the insulation), and the network is exogenous to monetary policy over this horizon. Concentrated economies — those whose output depends on a few highly central sectors — are partially shielded from monetary policy in exactly the way the network mechanism predicts.

---

## Repository Structure

```
production-network-monetary-transmission/
├── data/
│   ├── raw/          # TiVA and BLS source files (not committed — see below)
│   └── processed/    # auto-generated parquet cache (not committed)
├── notebooks/
│   ├── 01_production_network.ipynb       # network construction + metrics
│   ├── 02_transmission_data.ipynb        # BLS analysis + hypothesis preview
│   ├── 03_network_topology_tergm.ipynb   # topology evolution + TERGM
│   └── 04_transmission_tests.ipynb       # panel regressions + robustness
├── src/
│   ├── tiva_loader.py          # TiVA loading + network construction
│   ├── network_metrics.py      # centrality, NCI, dispersion, topology
│   ├── transmission_data.py    # BLS + ECB rate pipeline
│   └── panel_regression.py     # merge + regression framework
├── results/
│   ├── figures/
│   └── tables/                 # regression_results.csv, tergm_results.txt, etc.
├── requirements.txt
└── README.md
```

---

## Reproducing the Analysis

```bash
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute notebooks/01_production_network.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_transmission_data.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_network_topology_tergm.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_transmission_tests.ipynb
```

Set `USE_SYNTHETIC = True` in notebook 1 to run end-to-end on calibrated synthetic data without downloading the 1.7 GB TiVA file. The TERGM section of notebook 3 requires R with the `btergm` package; a Python fallback runs automatically if R is unavailable.

**Getting the data:**
- TiVA 2025: `https://data-explorer.oecd.org` → "TiVA 2025 Principal Indicators (levels)" → download `EXGR_INT.zip`, `PROD.zip` → extract to `data/raw/tiva/`.
- BLS: `https://data.ecb.europa.eu` → Bank Lending Survey → Supply → Enterprises → download all-country CSV → save to `data/raw/bls/bls_enterprises.csv`.

---

## Limitations

- **TiVA ends in 2022.** The 2022–23 tightening cycle, the strongest in the sample, is only partly covered; the extended-panel result carries 2022 network values forward to 2023.
- **Annual network, quarterly BLS.** Within-year network dynamics are invisible; each quarter in a year inherits that year's topology.
- **Country-level BLS only.** The ECB does not publish credit standards by sector and country jointly, so the test is the country-level corollary of a sector-level theory rather than the theory itself.
- **120 tightening observations.** The main result rests on a modest sample; it is significant and robust to outlier removal but would benefit from more tightening episodes.
- **NCI is elevated (0.55–0.84)** relative to typical IO studies, a consequence of using forward linkage on a 43-sector individual-industry panel; cross-country ordering is reliable, absolute levels are not directly comparable to other papers.

---

## Key References

- Acemoglu, D., Carvalho, V., Ozdaglar, A., Tahbaz-Salehi, A. (2012). The network origins of aggregate fluctuations. *Econometrica* 80(5).
- Antràs, P., Chor, D., Fally, T., Hillberry, R. (2012). Measuring the upstreamness of production and trade flows. *AER P&P* 102(3).
- Baqaee, D., Farhi, E. (2019). The macroeconomic impact of microeconomic shocks. *Econometrica* 87(4).
- Boehm, C., Flaaen, A., Pandalai-Nayar, N. (2019). Input linkages and the transmission of shocks. *AER* 109(1).
- Hristov, N., Hülsewig, O., Wollmershäuser, T. (2012). Loan supply shocks during the financial crisis. *JIMF* 31(3).
- Leifeld, P., Cranmer, S., Desmarais, B. (2018). Temporal exponential random graph models with btergm. *Journal of Statistical Software* 83(6).

---

*All data open access. Findings are preliminary and do not represent an institutional view.*
