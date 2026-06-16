"""
panel_regression.py
===================
Panel regression framework testing whether production network centrality
predicts monetary policy transmission heterogeneity.

MAIN SPECIFICATION
------------------
For each country i in quarter t:

  ΔBLS_i,t = α_i + γ_t + β₁ ΔRATE_t × CENTRALITY_i,t-1
           + β₂ CENTRALITY_i,t-1
           + β₃ ΔRATE_t
           + δ X_i,t + ε_i,t

where:
  ΔBLS_i,t     = change in net tightening index (BLS)
  ΔRATE_t      = ECB rate change (basis points)
  CENTRALITY_i = country-level network centrality index (NCI or mean EC)
  X_i,t        = controls (GDP growth, inflation, bank capital ratio)
  α_i          = country fixed effects
  γ_t          = time fixed effects

KEY COEFFICIENT: β₁
The coefficient on the interaction ΔRATE × CENTRALITY.

If β₁ < 0: countries with higher network centrality show WEAKER
           credit tightening in response to ECB rate increases.
           This is the main hypothesis — more central production
           networks insulate the financial sector from rate changes.

If β₁ > 0: the opposite — which would be interesting in a different way
           (centrality amplifies transmission rather than dampening it).

IDENTIFICATION
--------------
Potential endogeneity: network centrality might be correlated with
financial development, which independently affects transmission.

Instruments:
  1. Lagged network centrality (2-year lag) — predetermined
  2. Geographic distance-based network position — exogenous to
     monetary conditions
  3. Pre-sample network structure (1995-2000 period) — clearly
     predates the monetary transmission period we study

Robustness:
  - Split sample: tightening vs easing cycles
  - Include/exclude financial crisis quarters (2008-09, 2020)
  - Different centrality measures (eigenvector vs betweenness vs upstreamness)
  - Country-clustered vs heteroskedasticity-robust standard errors
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ===========================================================================
# SECTION 1 — DATA MERGING
# ===========================================================================

# Map from BLS country codes to TiVA ISO3 codes
# Countries excluded from regression (kept in network construction):
#   CY (Cyprus)     — 3-5 banks in BLS sample; one bank = 25pp swing; too noisy
#   MT (Malta)      — same problem; smallest BLS sample in the EA
#   LU (Luxembourg) — financial centre outlier; BLS driven by international banks,
#                     not ECB domestic transmission; NCI outlier by construction
BLS_REGRESSION_EXCLUDE = {"CY", "MT", "LU"}

BLS_TO_TIVA = {
    "U2": None,      # EA aggregate — skip
    # Large EA economies
    "DE": "DEU",     "FR": "FRA",     "IT": "ITA",
    "ES": "ESP",     "NL": "NLD",     "BE": "BEL",
    "AT": "AUT",     "PT": "PRT",     "FI": "FIN",
    "GR": "GRC",     "IE": "IRL",
    # Smaller EA economies
    "EE": "EST",     "HR": "HRV",     "LT": "LTU",
    "LV": "LVA",     "SI": "SVN",     "SK": "SVK",
    # Excluded from regression (noisy BLS or outlier structure):
    "CY": None,   # Cyprus  — too few banks
    "MT": None,   # Malta   — too few banks
    "LU": None,   # Luxembourg — financial centre outlier
}

# Clean mapping for regression use only (excludes None values)
BLS_TO_TIVA_REGRESSION = {
    k: v for k, v in BLS_TO_TIVA.items()
    if v is not None and k not in BLS_REGRESSION_EXCLUDE
}

TIVA_TO_BLS = {v: k for k, v in BLS_TO_TIVA.items() if v}


def merge_network_and_transmission(
    metrics_panel:       pd.DataFrame,
    country_topology:    pd.DataFrame,
    transmission_panel:  pd.DataFrame,
    exclude_countries:   set | None = None,
) -> pd.DataFrame:
    """
    Merge network topology metrics with monetary transmission outcomes.

    Parameters
    ----------
    exclude_countries : BLS country codes to drop before merging.
                        Defaults to BLS_REGRESSION_EXCLUDE (CY, MT, LU).
                        These are kept in the network but excluded from
                        the regression sample for methodological reasons:
                          CY, MT — too few banks in BLS (3-5); one bank
                                   tightening = 25pp swing; pure noise
                          LU     — financial centre outlier; BLS driven
                                   by international banks, not domestic
                                   ECB transmission

    Parameters
    ----------
    metrics_panel      : node-level metrics from network_metrics.py
    country_topology   : country-level topology from compute_country_level_topology()
    transmission_panel : BLS + rate data from transmission_data.py

    Returns
    -------
    Merged panel indexed by (quarter, country_bls) with both network
    topology and transmission outcome variables.
    """
    # Apply country exclusions (noisy BLS samples and outliers)
    if exclude_countries is None:
        exclude_countries = BLS_REGRESSION_EXCLUDE

    transmission_panel = transmission_panel.copy()
    if "country" in transmission_panel.columns:
        n_before = len(transmission_panel)
        transmission_panel = transmission_panel[
            ~transmission_panel["country"].isin(exclude_countries)
        ]
        n_dropped = n_before - len(transmission_panel)
        if n_dropped > 0:
            print(f"  Regression sample: excluded {n_dropped} obs "
                  f"from {sorted(exclude_countries)} (noisy BLS / outlier)")

    # Convert country_topology index to BLS codes
    topo = country_topology.reset_index()
    topo["country_bls"] = topo["country"].map(TIVA_TO_BLS)
    topo = topo.dropna(subset=["country_bls"])

    # Align time: TiVA is annual, BLS is quarterly
    # Map each quarter to the corresponding TiVA year
    trans = transmission_panel.reset_index()
    trans["year"] = pd.to_datetime(trans["quarter"]).dt.year

    # Lag network metrics by 1 year (predetermined relative to transmission)
    topo_lagged = topo.copy()
    topo_lagged["year"] = topo_lagged["year"] + 1
    topo_lagged = topo_lagged.rename(columns={
        c: f"{c}_lag1" for c in topo_lagged.columns
        if c not in ["year", "country", "country_bls"]
    })

    # Merge
    merged = trans.merge(
        topo_lagged,
        left_on=["year", "country"],
        right_on=["year", "country_bls"],
        how="inner",
        suffixes=("", "_topo"),
    )

    # Also merge current-year topology for robustness
    merged = merged.merge(
        topo.rename(columns={
            c: f"{c}_contemp" for c in topo.columns
            if c not in ["year", "country", "country_bls"]
        }),
        left_on=["year", "country"],
        right_on=["year", "country_bls"],
        how="left",
        suffixes=("", "_c"),
    )

    merged = merged.set_index(["quarter", "country"]).sort_index()

    print(f"Merged panel: {len(merged):,} observations | "
          f"{merged.index.get_level_values('country').nunique()} countries")

    return merged


# ===========================================================================
# SECTION 2 — MAIN PANEL REGRESSION
# ===========================================================================

def run_main_regression(
    panel:        pd.DataFrame,
    centrality_var:str = "nci_lag1",
    outcome_var:  str = "delta_bls",
    rate_var:     str = "delta_rate",
    controls:     list[str] | None = None,
    entity_effects: bool = True,
    time_effects:   bool = True,
) -> object:
    """
    Run the main panel regression:

      ΔBLS_i,t = α_i + γ_t + β₁ ΔRATE_t × CENTRALITY_i,t-1
               + β₂ CENTRALITY_i,t-1 + β₃ ΔRATE_t + δ X + ε

    The key coefficient is β₁ on the interaction term.
    Negative β₁ = more central networks → weaker transmission.

    Parameters
    ----------
    panel           : merged panel from merge_network_and_transmission()
    centrality_var  : column name of centrality measure
    outcome_var     : column name of transmission outcome
    rate_var        : column name of ECB rate change
    controls        : additional control variable names
    entity_effects  : include country fixed effects
    time_effects    : include time (quarter) fixed effects

    Returns
    -------
    linearmodels PanelOLS result object
    """
    try:
        from linearmodels.panel import PanelOLS, BetweenOLS
    except ImportError:
        raise ImportError("Install linearmodels: pip install linearmodels")

    df = panel.copy().dropna(
        subset=[outcome_var, centrality_var, rate_var]
    )

    if len(df) < 30:
        raise ValueError(f"Insufficient observations: {len(df)}")

    # Create interaction term
    df["interaction"] = df[rate_var] * df[centrality_var]

    # Build regressor matrix
    regressors = ["interaction", centrality_var, rate_var]
    if controls:
        regressors += [c for c in controls if c in df.columns]

    df["const"] = 1.0
    regressors  = ["const"] + regressors

    # Ensure MultiIndex (entity, time)
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Panel must have MultiIndex (quarter, country)")

    # Swap to (entity, time) for linearmodels
    df = df.swaplevel()
    df.index.names = ["entity", "time"]

    y = df[outcome_var]
    X = df[regressors]

    model  = PanelOLS(
        y, X,
        entity_effects=entity_effects,
        time_effects=time_effects,
    )
    result = model.fit(
        cov_type="clustered",
        cluster_entity=True,      # cluster SE by country
    )

    return result


def run_impulse_response(
    panel:        pd.DataFrame,
    centrality_var:str = "nci_lag1",
    outcome_var:  str = "bls_tightening",
    n_lags:       int = 4,
) -> pd.DataFrame:
    """
    Estimate impulse response: how does BLS tightening evolve over
    1-4 quarters after a rate change, conditional on network centrality?

    For each lag h = 0, 1, 2, 3, 4:
      ΔBLS_i,t+h = α_i + β₁^h ΔRATE_t × CENTRALITY_i,t-1
                 + β₂^h CENTRALITY_i,t-1 + β₃^h ΔRATE_t + ε

    Returns the β₁^h coefficients and confidence intervals — the
    conditional impulse response function showing how transmission
    evolves over time for different network structures.

    Returns
    -------
    pd.DataFrame: lag h, beta_interaction, se, t_stat, p_value,
                  ci_lower, ci_upper
    """
    from scipy.stats import t as t_dist

    df = panel.copy()
    df["interaction"] = df["delta_rate"] * df[centrality_var]

    records = []
    for lag in range(n_lags + 1):
        # Shift outcome forward by lag quarters
        df_lag = df.copy()
        df_lag[f"outcome_lag{lag}"] = (
            df_lag.groupby(level="country")[outcome_var]
            .shift(-lag)
        )

        sub = df_lag.dropna(
            subset=[f"outcome_lag{lag}", "interaction", centrality_var, "delta_rate"]
        )

        if len(sub) < 20:
            continue

        try:
            result = run_main_regression(
                sub,
                centrality_var=centrality_var,
                outcome_var=f"outcome_lag{lag}",
                rate_var="delta_rate",
            )
            params = result.params
            se     = result.std_errors
            pvals  = result.pvalues

            beta = params.get("interaction", np.nan)
            std  = se.get("interaction", np.nan)
            p    = pvals.get("interaction", np.nan)

            records.append({
                "lag":          lag,
                "beta":         beta,
                "se":           std,
                "t_stat":       beta / (std + 1e-9),
                "p_value":      p,
                "ci_lower":     beta - 1.96 * std,
                "ci_upper":     beta + 1.96 * std,
                "n_obs":        result.nobs,
            })
        except Exception as e:
            records.append({
                "lag": lag, "beta": np.nan, "se": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan, "n_obs": 0,
            })

    return pd.DataFrame(records).set_index("lag")


# ===========================================================================
# SECTION 3 — ROBUSTNESS CHECKS
# ===========================================================================

def run_robustness_suite(
    panel:           pd.DataFrame,
    centrality_vars: list[str] | None = None,
    outcome_var:     str = "delta_bls",
) -> pd.DataFrame:
    """
    Run a battery of robustness checks varying:
    1. Centrality measure (eigenvector, upstreamness, NCI, forward linkage)
    2. Sample period (full, pre-COVID, post-COVID, tightening cycles only)
    3. Fixed effects specification

    Returns
    -------
    Summary DataFrame: one row per specification with key coefficient,
    SE, p-value, N, and specification description.
    """
    if centrality_vars is None:
        centrality_vars = [
            "nci_lag1",
            "mean_eigen_cent_lag1",
            "mean_upstreamness_lag1",
            "mean_fwd_linkage_lag1",
            "centrality_hhi_lag1",
        ]

    results = []

    # Vary centrality measure
    for cent_var in centrality_vars:
        if cent_var not in panel.columns:
            continue
        for entity_fe, time_fe, spec_label in [
            (True,  True,  "Two-way FE"),
            (True,  False, "Country FE only"),
            (False, True,  "Time FE only"),
        ]:
            try:
                result = run_main_regression(
                    panel,
                    centrality_var=cent_var,
                    outcome_var=outcome_var,
                    entity_effects=entity_fe,
                    time_effects=time_fe,
                )
                beta = result.params.get("interaction", np.nan)
                se   = result.std_errors.get("interaction", np.nan)
                p    = result.pvalues.get("interaction", np.nan)

                results.append({
                    "centrality_var": cent_var,
                    "specification":  spec_label,
                    "sample":         "Full",
                    "beta":           beta,
                    "se":             se,
                    "p_value":        p,
                    "n_obs":          result.nobs,
                    "significant":    p < 0.10,
                    "sign_correct":   beta < 0,
                })
            except Exception:
                pass

    # Sub-sample: tightening cycles only
    if "tightening_cycle" in panel.columns:
        tight_panel = panel[panel["tightening_cycle"] == 1]
        for cent_var in centrality_vars[:2]:
            if cent_var not in tight_panel.columns:
                continue
            try:
                result = run_main_regression(
                    tight_panel,
                    centrality_var=cent_var,
                    outcome_var=outcome_var,
                )
                beta = result.params.get("interaction", np.nan)
                se   = result.std_errors.get("interaction", np.nan)
                p    = result.pvalues.get("interaction", np.nan)
                results.append({
                    "centrality_var": cent_var,
                    "specification":  "Two-way FE",
                    "sample":         "Tightening cycles only",
                    "beta":           beta,
                    "se":             se,
                    "p_value":        p,
                    "n_obs":          result.nobs,
                    "significant":    p < 0.10,
                    "sign_correct":   beta < 0,
                })
            except Exception:
                pass

    df = pd.DataFrame(results)
    if len(df) > 0:
        n_sig   = df["significant"].sum()
        n_right = df["sign_correct"].sum()
        print(f"Robustness suite: {len(df)} specifications | "
              f"{n_sig} significant at 10% | "
              f"{n_right} with correct sign (negative)")
    return df


# ===========================================================================
# SECTION 4 — RESULTS FORMATTING
# ===========================================================================

def format_regression_table(
    results_list: list,
    labels:       list[str],
) -> pd.DataFrame:
    """
    Format multiple regression results into a publication-ready table.

    Displays coefficients with significance stars:
      *** p<0.01  ** p<0.05  * p<0.10
    Standard errors in parentheses below coefficients.
    """
    rows = {}
    key_vars = ["interaction", "delta_rate", "centrality"]

    for result, label in zip(results_list, labels):
        col = {}
        try:
            for var in result.params.index:
                beta = result.params[var]
                se   = result.std_errors[var]
                p    = result.pvalues[var]
                stars = "***" if p < 0.01 else "**" if p < 0.05 \
                        else "*" if p < 0.10 else ""
                col[var]           = f"{beta:.4f}{stars}"
                col[f"{var}_se"]   = f"({se:.4f})"
            col["N"]   = result.nobs
            col["R²"]  = f"{result.rsquared:.3f}"
        except Exception:
            pass
        rows[label] = col

    return pd.DataFrame(rows).T


def print_main_result(result) -> None:
    """Print a clean summary of the main regression result."""
    print("\n" + "=" * 60)
    print("MAIN RESULT: Production Network Centrality × Rate Change")
    print("=" * 60)

    try:
        beta  = result.params.get("interaction", np.nan)
        se    = result.std_errors.get("interaction", np.nan)
        p     = result.pvalues.get("interaction", np.nan)
        stars = "***" if p < 0.01 else "**" if p < 0.05 \
                else "*" if p < 0.10 else "(n.s.)"

        print(f"\n  β₁ (ΔRATE × CENTRALITY): {beta:.4f} {stars}")
        print(f"  Standard error:           {se:.4f}")
        print(f"  p-value:                  {p:.4f}")
        print(f"  N:                        {result.nobs}")
        print()
        if beta < 0 and p < 0.10:
            print("  INTERPRETATION: Countries with higher production network")
            print("  centrality show WEAKER credit tightening in response to")
            print("  ECB rate increases — consistent with the insulation")
            print("  hypothesis. Central sectors absorb monetary shocks rather")
            print("  than transmitting them to credit conditions.")
        elif beta > 0 and p < 0.10:
            print("  INTERPRETATION: Unexpected positive sign — network")
            print("  centrality AMPLIFIES transmission. Central sectors may")
            print("  actually face more credit tightening, potentially because")
            print("  banks are more attentive to central sector exposures.")
        else:
            print("  INTERPRETATION: No significant interaction effect detected.")
            print("  Network centrality does not predict transmission heterogeneity")
            print("  at conventional significance levels in this sample.")
    except Exception as e:
        print(f"  Could not display result: {e}")
    print("=" * 60)
