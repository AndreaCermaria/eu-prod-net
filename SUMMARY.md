# Project Summary — Findings and Narrative

*A standalone companion to the README. This document explains what the project found, why it matters, and how confident one should be in each claim.*

---

## One-paragraph version

Using OECD input–output data, I build the euro area production network and measure how concentrated each country's production structure is. I then test whether that concentration explains why ECB rate changes hit some countries' credit conditions harder than others. The answer is yes, but only during tightening cycles: when the ECB raises rates, countries whose output depends on a few highly central sectors tighten bank credit standards markedly less than countries with diffuse production. A network model confirms that the production network itself is essentially fixed over this horizon and does not react to monetary policy — so the causation runs from network structure to credit transmission, not the other way around.

---

## What each part of the project establishes

**1. The network is real and economically sensible.** The most central sectors by forward linkage are basic metals, chemicals, automotive, and machinery — exactly the upstream input suppliers that network theory predicts. Ireland and Luxembourg are the most concentrated economies; the Baltics are the least. The maximum sector centrality has fallen ~50% since 2007, a genuine finding about the diversification of EA supply chains.

**2. Transmission is highly heterogeneous across countries.** A 100 bps ECB hike moves credit standards by −33 points in Greece but +21 in Estonia. This dispersion is the puzzle the project sets out to explain.

**3. Network concentration explains part of that heterogeneity — during tightening.** The interaction between rate changes and concentration is −234 (p = 0.017) in tightening quarters, −178 (p = 0.045) after removing the two Baltic outliers, and statistically zero in the full sample. The conditional pattern is exactly what the mechanism predicts: insulation only matters when there is a shock to be insulated from.

**4. The effect weakens in the 2022–23 cycle.** Extending the panel to include the +450 bps cycle halves the coefficient and removes significance. The natural reading is nonlinearity — the insulation works at moderate intensity but is overwhelmed by an extreme shock.

**5. The network does not respond to monetary policy.** The TERGM memory coefficient (3.04) shows links persist roughly 21× year to year; the rate coefficient is effectively zero. In 2022 the bilateral network did not change at all. This is what licenses a causal interpretation rather than a mere correlation.

---

## How confident to be

| Claim | Confidence | Why |
|---|---|---|
| Network construction is correct | High | Sector rankings match theory; results stable across specifications |
| Transmission is heterogeneous | High | Directly measured, large and significant cross-country spread |
| Concentration dampens tightening | Moderate | Significant and robust to outliers, but rests on 120 observations |
| Effect is nonlinear in 2022–23 | Low–moderate | Suggestive; confounded by TiVA coverage ending in 2022 |
| Network is exogenous to policy | Moderate–high | Clean TERGM null, strong memory coefficient, 2022 Jaccard = 1.0 |

---

## Why this is relevant to a central bank

The result speaks directly to transmission heterogeneity — the question of why a single policy rate produces uneven real effects across the currency union. It reframes production structure, not just financial structure, as a source of that unevenness, and it does so with a transparent, reproducible, open-data pipeline. The honest treatment of the conditional and sample-limited nature of the finding is itself the point: it demonstrates the ability to extract a real signal, stress-test it, and state precisely how far it can be trusted.

---

## Honest weaknesses

The main result rests on 120 tightening-quarter observations. The BLS is only available at country level, so this tests the country-level corollary of a sector-level theory. NCI levels are higher than comparable studies because of the forward-linkage construction. And the most important policy episode — 2022–23 — sits at the edge of the network data. None of these is fatal, but all of them belong in any honest presentation of the work.
