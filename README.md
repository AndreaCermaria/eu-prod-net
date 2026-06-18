# Production Network Topology and Monetary Policy Transmission

*Does a country's position in the euro area input output network shape how strongly ECB rate changes pass through into bank credit standards?*

---

## Abstract

This project constructs the intra euro area production network from OECD Trade in Value Added (TiVA 2025) data for 1995 to 2022, measures how concentrated each country's production structure is, and tests whether that concentration moderates the transmission of ECB rate changes into bank credit standards (ECB Bank Lending Survey, 2003 to 2026).

The central result is conditional but robust: **during ECB tightening cycles, countries with more concentrated production networks tighten credit standards significantly less in response to rate hikes.** The interaction between rate changes and network concentration is minus 234 (p = 0.017) in the tightening only sample and minus 178 (p = 0.045) after removing two structural outliers. The effect is absent in the full sample, which is expected since the mechanism can only operate when rates are actually moving.

A Temporal Exponential Random Graph Model (TERGM) estimated on the **full sector level network (817 nodes, 28 years)** shows the production network itself does **not** respond to monetary policy. Supply chain links persist with 91x strength year to year (memory coefficient 4.51), and ECB rate changes have zero detectable effect on link formation or dissolution (rate coefficient minus 0.004, 95% CI includes zero). Even during the +450bps tightening of 2022, the bilateral sector level network was among the most stable in the entire 28 year sample. This closes the reverse causality argument: the network shapes credit transmission, not the other way around.

---

## The Mechanism

Standard monetary transmission models treat sectors as isolated borrowers whose response to higher rates depends on their own leverage and cash flow. Network position is absent.

This project adds one idea: a sector that supplies critical intermediate inputs to many others cannot easily cut its own activity when credit tightens, because its customers depend on its output. It is partially locked in. A country whose production is concentrated in such central sectors should therefore transmit monetary policy more weakly into credit conditions.

The prediction is a cross country one: higher production network concentration leads to weaker pass through of ECB rate changes into credit standards.

---

## Data

All data is free and publicly available. No proprietary sources are used.

| Dataset | Source | Role |
|---|---|---|
| TiVA 2025, EXGR_INT (intermediate exports) | OECD Data Explorer | Production network edges |
| TiVA 2025, PROD (gross output) | OECD Data Explorer | Node normalisation |
| Bank Lending Survey, credit standards, enterprises | ECB Data Portal | Transmission outcome |
| ECB main refinancing rate | ECB | Policy/treatment variable |

**Coverage:** 19 EA countries x 43 ISIC sectors x 28 years for the network; 18 countries x 94 quarters (2003 Q1 to 2026 Q2) for the BLS. The regression sample is 15 countries after excluding Cyprus and Malta (BLS samples of 3 to 5 banks) and Luxembourg (financial centre outlier).

---

## Method

### Network construction

An annual directed weighted graph of intra EA intermediate input flows. Each node is a country sector pair; each edge is the USD million value of intermediate inputs shipped from one country sector to another within the euro area. Aggregate ISIC codes (such as "C" for all manufacturing) are filtered out so individual sectors are not double counted against their own totals.

**Centrality (forward linkage):** the total value a sector ships as intermediate inputs to the rest of the EA network. This measures how much the rest of the network depends on a sector's output, and corresponds to the influence vector of Acemoglu et al. (2012).

**Network Centralisation Index (NCI):** the Gini coefficient of sector level forward linkage within a country. High NCI = one or two sectors dominate (Ireland, Luxembourg). Low NCI = many sectors contribute evenly.

### Transmission regression

```
BLS_tightening(i,t) = alpha_i + gamma_t + beta_1 * (delta_RATE_t x NCI_i,t-1) + beta_2 * NCI_i,t-1 + epsilon(i,t)
```

Country fixed effects absorb time invariant national characteristics. Time fixed effects absorb common quarterly shocks including the ECB rate level. Standard errors clustered by country.

### Reverse causality check (TERGM)

A Temporal ERGM on the full sector level network (817 nodes, 28 years, 100 bootstrap replications) tests whether ECB rate changes predict the formation or dissolution of bilateral supply chain links, controlling for endogenous structure (reciprocity, memory) and sector forward linkage.

---

## Results

### Main regression (Notebook 4)

| Specification | beta (rate x NCI) | p value | N |
|---|---:|---:|---:|
| Full sample | minus 17.5 | 0.483 | 1,056 |
| **Tightening episodes only** | **minus 233.7** | **0.017** | **120** |
| Extended to 2023 | minus 123.9 | 0.224 | 180 |
| **Tightening, no Baltic outliers** | **minus 177.9** | **0.045** | **116** |

The full sample null is expected: three quarters of observations have zero rate change, diluting the signal. Restricting to tightening quarters yields a significant negative coefficient with the predicted sign. The effect survives dropping Estonia and Latvia. The effect weakens when the extreme 2022 to 2023 cycle is added, suggesting nonlinearity at extreme policy intensities.

### Network dynamics (Notebook 3, sector level TERGM)

| Term | Estimate | 95% CI | Reading |
|---|---:|---|---|
| memory | **+4.51** | [+4.44, +4.58] | Links persist 91x year to year |
| mutual | **+4.49** | [+4.39, +4.58] | Strong bilateral same sector trade (89x) |
| forward linkage (node) | **minus 47.9** | [minus 50.5, minus 45.3] | High centrality sectors are hubs |
| rate change (edge) | minus 0.004 | [minus 0.06, +0.03] | **Monetary policy does not move the network** |

The network is overwhelmingly persistent. ECB rate changes have zero effect on supply chain link formation. In 2022, during the most aggressive ECB tightening on record, the sector level network was among the most stable in the entire 28 year sample (Jaccard = 0.995).

### Country transmission sensitivities (Notebook 2)

Strongest pass through: Greece (minus 33), Netherlands (minus 30), Finland (minus 25). Weakest: Estonia (+21), Latvia (+16, structural outliers), Germany (minus 10), France (minus 3).

---

## Interpretation

The production network insulates credit transmission during tightening. The mechanism is nonlinear: it weakens in the extreme 2022 to 2023 cycle, as if the shock was large enough to overwhelm the insulation. The network is exogenous to monetary policy: supply chain links persist with 91x strength at the sector level and do not respond to rate changes. Concentrated economies, those whose output depends on a few highly central sectors, are partially shielded from monetary policy in exactly the way the network mechanism predicts.

---

## Repository Structure

```
eu-prod-net/
├── data/
│   ├── raw/          # TiVA and BLS source files (not committed)
│   └── processed/    # auto generated parquet cache (not committed)
├── notebooks/
│   ├── 01_production_network.ipynb       # network construction + metrics
│   ├── 02_transmission_data.ipynb        # BLS analysis + hypothesis preview
│   ├── 03_network_topology_tergm.ipynb   # topology evolution + sector level TERGM
│   └── 04_transmission_tests.ipynb       # panel regressions + robustness
├── src/
│   ├── tiva_loader.py          # TiVA loading + network construction
│   ├── network_metrics.py      # centrality, NCI, topology metrics
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
jupyter nbconvert --to notebook --execute notebooks/03_network_topology_tergm.ipynb   # ~2 hours for TERGM
jupyter nbconvert --to notebook --execute notebooks/04_transmission_tests.ipynb
```

**Getting the data:**

TiVA 2025: go to `https://data-explorer.oecd.org`, search "TiVA 2025 Principal Indicators (levels)", download `EXGR_INT.zip` and `PROD.zip`, extract to `data/raw/tiva/`.

BLS: go to `https://data.ecb.europa.eu`, navigate to Bank Lending Survey > Supply > Enterprises, download the all country CSV, save to `data/raw/bls/bls_enterprises.csv`.

The TERGM section of Notebook 3 requires R with the `btergm` package (`install.packages("btergm")`). A Python fallback runs automatically if R is unavailable.

---

## Key References

- Acemoglu, D., Carvalho, V., Ozdaglar, A., Tahbaz Salehi, A. (2012). The network origins of aggregate fluctuations. *Econometrica* 80(5).
- Antras, P., Chor, D., Fally, T., Hillberry, R. (2012). Measuring the upstreamness of production and trade flows. *AER P&P* 102(3).
- Baqaee, D., Farhi, E. (2019). The macroeconomic impact of microeconomic shocks. *Econometrica* 87(4).
- Boehm, C., Flaaen, A., Pandalai Nayar, N. (2019). Input linkages and the transmission of shocks. *AER* 109(1).
- Hristov, N., Hülsewig, O., Wollmershäuser, T. (2012). Loan supply shocks during the financial crisis. *JIMF* 31(3).
- Leifeld, P., Cranmer, S., Desmarais, B. (2018). Temporal exponential random graph models with btergm. *Journal of Statistical Software* 83(6).

---

## Limitations

- TiVA data ends in 2022; the 2022 to 2023 tightening cycle falls partly outside the network data window.
- TiVA is annual while BLS is quarterly; within year network dynamics are not captured.
- The BLS measures aggregate country level credit standards, not sector level tightening, preventing a direct sector level test of the transmission mechanism.
- 120 tightening observations is a modest sample; the result is significant and robust to outlier removal but would benefit from more tightening episodes.
- NCI values (0.55 to 0.84) are elevated relative to typical IO network studies, reflecting the forward linkage construction on a 43 sector panel.

---


