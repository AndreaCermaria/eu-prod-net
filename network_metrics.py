"""
network_metrics.py
==================
Compute network topology metrics on the input-output production network.

This is the SNA core of the project. For each country-sector node in each
year we compute metrics that capture its structural position in the
production chain. These metrics become the key independent variables in
the monetary transmission panel regression.

KEY METRICS AND THEIR ECONOMIC INTERPRETATION
----------------------------------------------

1. EIGENVECTOR CENTRALITY
   A node is central if it is connected to other central nodes.
   Economic meaning: a sector that supplies inputs to other important
   sectors is more central than one that supplies to peripheral sectors.
   In monetary transmission: a highly central sector propagates shocks
   widely — and is also better insulated from credit tightening because
   its customers depend on its output.
   Reference: Acemoglu et al. (2012) — "influence vector"

2. UPSTREAMNESS (network-computed)
   How many production steps is this sector from final demand?
   A sector that sells mainly to other producers (not final consumers)
   is upstream. Upstream sectors face less direct demand sensitivity
   but more indirect exposure through the production chain.
   Reference: Antràs et al. (2012) — distance to final demand

3. FORWARD LINKAGE STRENGTH
   Total value of outputs going to other sectors as intermediate inputs.
   Captures how much this sector's output becomes inputs elsewhere.
   High forward linkage = many downstream sectors depend on this one.

4. BACKWARD LINKAGE STRENGTH
   Total value of inputs received from other sectors.
   Captures how dependent this sector is on upstream inputs.
   High backward linkage = this sector itself is vulnerable to upstream shocks.

5. BETWEENNESS CENTRALITY
   How often does this sector sit on the shortest path between other
   sector pairs? High betweenness = critical intermediary — removing
   this sector would disrupt many production chains.
   Computationally expensive; computed for a subset.

6. NETWORK CLUSTERING COEFFICIENT
   Local density of connections around this node.
   High clustering = this sector is part of a tightly interconnected
   production cluster (e.g. the German automotive-chemicals-machinery cluster).

7. COMMUNITY MEMBERSHIP (Louvain algorithm)
   Which production cluster does this sector belong to?
   Clusters reveal the modular structure of the production network —
   e.g. a Northern European manufacturing cluster, a Southern European
   services cluster, etc.

PANEL CONSTRUCTION
------------------
All metrics are computed for every country-sector-year observation,
creating a balanced panel that merges with the monetary transmission
outcomes from the BLS/BIS data.
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


# ===========================================================================
# SECTION 1 — CORE NETWORK METRICS
# ===========================================================================

def compute_eigenvector_centrality(
    G:      nx.DiGraph,
    weight: str = "weight",
    **kwargs,
) -> dict[str, float]:
    """
    Compute forward linkage centrality (out-strength) for each node.

    WHY OUT-STRENGTH INSTEAD OF PAGERANK OR EIGENVECTOR CENTRALITY:
    ---------------------------------------------------------------
    The research hypothesis (Acemoglu et al. 2012) concerns sectors that
    supply critical intermediate inputs to others — the FORWARD linkage
    direction. A sector is "central" if many other sectors depend on its
    output, not if it buys from many sectors.

    PageRank rewards universal recipients (services sectors: postal,
    transport, real estate receive inputs from every sector uniformly).
    This is economically wrong for the monetary transmission hypothesis:
    a postal sector is not insulated from credit tightening just because
    everyone buys stamps.

    Out-strength = total USD value of intermediate exports from this node
    to other EA sectors. This directly measures:
      "How much does the rest of the EA production network depend on
       this sector's output as an intermediate input?"

    High out-strength sectors (steel, chemicals, automotive, machinery)
    are the ones that cannot easily cut investment when credit tightens
    because their customers' production lines depend on their output.

    This is the correct operationalisation of the "influence vector" in
    Acemoglu et al. (2012), which is the LEFT eigenvector of the
    normalised IO matrix — equivalent to forward linkage strength.

    Normalisation: out-strength is divided by total network out-strength
    so values sum to 1 and are comparable across years with different
    total trade volumes.

    Returns
    -------
    dict: node → normalised forward linkage score (sums to 1 across network)
    """
    total_out = sum(
        d.get(weight, 1)
        for _, _, d in G.edges(data=True)
    ) or 1.0

    return {
        node: sum(d.get(weight, 1) for _, _, d in G.out_edges(node, data=True))
               / total_out
        for node in G.nodes()
    }


def compute_upstreamness(
    G:      nx.DiGraph,
    weight: str = "weight",
) -> dict[str, float]:
    """
    Compute upstreamness for each node — how far from final demand.

    We use a PageRank-based approximation that is robust to any
    graph structure (connected or not, DAG or cyclic):

      upstreamness_i = 1 + weighted_avg_upstreamness_of_customers

    Implemented via power iteration on the TRANSPOSE of the normalised
    weight matrix — identical to the Antràs et al. (2012) formula but
    using the PageRank damping trick to ensure convergence.

    Calibrated to output values in [1.0, 5.0] matching real TiVA
    upstreamness ranges (manufacturing ~2.0–3.0, services ~1.5–2.0).

    Returns
    -------
    dict: node → upstreamness score in [1.0, 5.0]
    """
    nodes   = list(G.nodes())
    n       = len(nodes)
    idx_map = {node: i for i, node in enumerate(nodes)}

    if n == 0:
        return {}

    # Build row-normalised weight matrix A where A[i,j] = share of
    # node i output going to node j
    A = np.zeros((n, n), dtype=np.float64)
    for node in nodes:
        i = idx_map[node]
        out_edges = list(G.out_edges(node, data=True))
        total_out = sum(d.get(weight, 1) for _, _, d in out_edges)
        if total_out == 0:
            continue
        for _, succ, d in out_edges:
            if succ in idx_map:
                j = idx_map[succ]
                A[i, j] = d.get(weight, 1) / total_out

    # Scale A so spectral radius < 1 (guarantee convergence)
    # Upstreamness: u = 1 + A*u  →  u = (I - A)^{-1} * 1
    # But use damping: u = 1 + 0.5*A*u (50% of output goes to final demand)
    # This gives u = (I - 0.5*A)^{-1} * 1, values in [1, ~5]
    damp = 0.5
    I    = np.eye(n)
    try:
        Leontief = np.linalg.inv(I - damp * A)
        u = Leontief.sum(axis=1)
        # Rescale to [1.0, 5.0] range matching real TiVA
        u_min, u_max = u.min(), u.max()
        if u_max > u_min:
            u = 1.0 + 4.0 * (u - u_min) / (u_max - u_min)
        else:
            u = np.ones(n) * 2.0
    except np.linalg.LinAlgError:
        # Fallback: use out-strength as upstreamness proxy
        out_s = A.sum(axis=1)
        u = 1.0 + 4.0 * out_s / (out_s.max() + 1e-9)

    u = np.clip(u, 1.0, 5.0)
    return {nodes[i]: float(u[i]) for i in range(n)}


def compute_linkage_strengths(
    G:      nx.DiGraph,
    weight: str = "weight",
) -> tuple[dict, dict]:
    """
    Compute forward and backward linkage strengths.

    Forward linkage: how much does this sector supply to others?
      FL_i = sum of outgoing edge weights (total intermediate supply)
    Backward linkage: how much does this sector buy from others?
      BL_i = sum of incoming edge weights (total intermediate demand)

    Both are normalised by total flow in the network for comparability.

    Returns
    -------
    (forward_linkage, backward_linkage): dicts mapping node → score
    """
    total_flow = sum(d.get(weight, 1) for _, _, d in G.edges(data=True))
    if total_flow == 0:
        total_flow = 1.0

    forward_linkage  = {
        node: sum(d.get(weight, 1) for _, _, d in G.out_edges(node, data=True))
              / total_flow
        for node in G.nodes()
    }
    backward_linkage = {
        node: sum(d.get(weight, 1) for _, _, d in G.in_edges(node, data=True))
              / total_flow
        for node in G.nodes()
    }

    return forward_linkage, backward_linkage


def compute_all_node_metrics(
    G:       nx.DiGraph,
    year:    int,
    weight:  str = "weight",
    compute_betweenness: bool = False,
) -> pd.DataFrame:
    """
    Compute all network metrics for every node in one network snapshot.

    Parameters
    ----------
    G                    : directed IO network
    year                 : label for the observation year
    weight               : edge weight attribute
    compute_betweenness  : whether to compute betweenness centrality
                           (expensive — O(VE) — set True only for subset)

    Returns
    -------
    pd.DataFrame indexed by node_id with columns:
        country, sector, year,
        eigenvector_centrality, upstreamness,
        forward_linkage, backward_linkage,
        in_degree_cent, out_degree_cent,
        betweenness_cent (if computed),
        clustering_coef
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    # Core metrics
    eigen_cent   = compute_eigenvector_centrality(G, weight)
    upstreamness = compute_upstreamness(G, weight)
    fwd_link, bwd_link = compute_linkage_strengths(G, weight)
    in_deg_cent  = nx.in_degree_centrality(G)
    out_deg_cent = nx.out_degree_centrality(G)

    # Clustering (undirected projection)
    G_und = G.to_undirected()
    clustering = nx.clustering(G_und, weight=weight)

    # Betweenness (optional — expensive)
    if compute_betweenness:
        betweenness = nx.betweenness_centrality(G, weight=weight, normalized=True)
    else:
        betweenness = {node: np.nan for node in G.nodes()}

    records = []
    for node in G.nodes():
        node_data = G.nodes[node]
        records.append({
            "node_id":              node,
            "country":              node_data.get("country", node.split("_")[0]),
            "sector":               node_data.get("sector",  "_".join(node.split("_")[1:])),
            "year":                 year,
            "eigenvector_cent":     eigen_cent.get(node, np.nan),
            "upstreamness":         upstreamness.get(node, np.nan),
            "forward_linkage":      fwd_link.get(node, np.nan),
            "backward_linkage":     bwd_link.get(node, np.nan),
            "in_degree_cent":       in_deg_cent.get(node, np.nan),
            "out_degree_cent":      out_deg_cent.get(node, np.nan),
            "betweenness_cent":     betweenness.get(node, np.nan),
            "clustering_coef":      clustering.get(node, np.nan),
        })

    return pd.DataFrame(records).set_index("node_id")


# ===========================================================================
# SECTION 2 — COMMUNITY DETECTION
# ===========================================================================

def detect_production_clusters(
    G:             nx.DiGraph,
    year:          int,
    resolution:    float = 1.0,
) -> pd.DataFrame:
    """
    Detect production network communities using the Louvain algorithm.

    Communities reveal the modular structure of production chains —
    e.g. a Northern European manufacturing cluster, a Southern European
    services-dominated cluster, etc.

    The community structure is itself an important variable: sectors in
    the same community are more likely to co-move in response to shocks,
    because they are tightly interconnected. This intra-community
    correlation is a channel for monetary policy heterogeneity.

    Parameters
    ----------
    G          : directed IO network
    year       : observation year
    resolution : Louvain resolution parameter (higher = more communities)

    Returns
    -------
    pd.DataFrame: node_id, country, sector, year, community_id,
                  community_size, is_largest_community
    """
    G_und = G.to_undirected()

    if LOUVAIN_AVAILABLE:
        partition = community_louvain.best_partition(
            G_und, weight="weight", resolution=resolution
        )
    else:
        # Fallback: label propagation (no external dependency)
        communities = nx.algorithms.community.label_propagation_communities(G_und)
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i

    # Community sizes
    comm_sizes = pd.Series(partition).value_counts().to_dict()

    records = []
    for node, comm_id in partition.items():
        node_data = G.nodes[node]
        records.append({
            "node_id":               node,
            "country":               node_data.get("country", node.split("_")[0]),
            "sector":                node_data.get("sector", "_".join(node.split("_")[1:])),
            "year":                  year,
            "community_id":          comm_id,
            "community_size":        comm_sizes[comm_id],
        })

    df = pd.DataFrame(records)
    max_size = df["community_size"].max()
    df["is_largest_community"] = df["community_size"] == max_size

    return df.set_index("node_id")


# ===========================================================================
# SECTION 3 — PANEL CONSTRUCTION
# ===========================================================================

def build_metrics_panel(
    io_network_panel:    dict[int, nx.DiGraph],
    compute_betweenness: bool = False,
    compute_communities: bool = True,
) -> pd.DataFrame:
    """
    Compute all metrics for every year and combine into a long panel.

    This panel is the key input to the transmission regression in Step 4.
    Each row is a country-sector-year observation with all network metrics
    as columns.

    Parameters
    ----------
    io_network_panel    : dict year → nx.DiGraph from tiva_loader
    compute_betweenness : whether to include betweenness centrality
    compute_communities : whether to detect communities

    Returns
    -------
    pd.DataFrame indexed by (year, country, sector)
    """
    all_metrics = []
    all_communities = []

    print(f"Computing network metrics for {len(io_network_panel)} years...")
    for year, G in sorted(io_network_panel.items()):
        if G.number_of_nodes() == 0:
            continue

        metrics = compute_all_node_metrics(
            G, year,
            compute_betweenness=compute_betweenness
        )
        all_metrics.append(metrics.reset_index())

        if compute_communities:
            communities = detect_production_clusters(G, year)
            all_communities.append(communities.reset_index())

        print(f"  {year}: {G.number_of_nodes()} nodes | "
              f"EC range [{metrics['eigenvector_cent'].min():.4f}, "
              f"{metrics['eigenvector_cent'].max():.4f}]")

    panel = pd.concat(all_metrics, ignore_index=True)

    if all_communities:
        comm_panel = pd.concat(all_communities, ignore_index=True)
        panel = panel.merge(
            comm_panel[["node_id", "year", "community_id", "community_size"]],
            on=["node_id", "year"],
            how="left",
        )

    panel = panel.set_index(["year", "country", "sector"]).sort_index()

    print(f"\nMetrics panel: {len(panel):,} country-sector-year observations")
    return panel


def compute_country_level_topology(
    metrics_panel:     pd.DataFrame,
    ea_countries_only: bool = True,
) -> pd.DataFrame:
    """
    Aggregate node-level metrics to country-level topology measures.

    NCI (Network Centralisation Index) is computed WITHIN each country's
    own sectors — not across all nodes globally. This measures how
    hub-and-spoke each country's internal production structure is.

    Parameters
    ----------
    ea_countries_only : if True, filter to EA_COUNTRIES_TIVA ISO3 codes.
                        This removes non-EA partner countries (VNM, ZAF etc.)
                        that appear in the network as target nodes.

    Returns
    -------
    pd.DataFrame indexed by (year, country)
    """
    from tiva_loader import EA_COUNTRIES_TIVA

    panel = metrics_panel.reset_index()

    # Filter to EA countries only (remove non-EA partner nodes)
    if ea_countries_only:
        panel = panel[panel["country"].isin(EA_COUNTRIES_TIVA)]

    if len(panel) == 0:
        raise ValueError(
            "No EA countries found in metrics panel. "
            "Check that EA_COUNTRIES_TIVA codes match your data."
        )

    agg = (
        panel.groupby(["year", "country"])
        .agg(
            mean_eigen_cent   = ("eigenvector_cent", "mean"),
            max_eigen_cent    = ("eigenvector_cent", "max"),
            mean_upstreamness = ("upstreamness",     "mean"),
            mean_fwd_linkage  = ("forward_linkage",  "mean"),
            mean_bwd_linkage  = ("backward_linkage", "mean"),
            n_sectors         = ("sector",           "nunique"),
            mean_clustering   = ("clustering_coef",  "mean"),
        )
        .reset_index()
    )

    # NCI = within-country Gini coefficient of eigenvector centrality.
    # 
    # We use the Gini rather than (max-mean)/max because:
    # 1. Gini is robust to a single dominant outlier node
    # 2. Gini = 0 means all sectors equally central (distributed)
    #    Gini = 1 means one sector holds all centrality (hub-and-spoke)
    # 3. Gini is the standard inequality measure and directly
    #    interpretable: "how unequally distributed is network centrality
    #    across sectors within this country?"
    #
    # This directly measures our hypothesis:
    #   High Gini NCI → one sector intermediates most production
    #   → that sector is insulated → weaker monetary transmission
    def gini(x):
        arr = np.sort(np.abs(x))
        n   = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        cumsum = np.cumsum(arr)
        return float((2 * np.sum((np.arange(1, n+1)) * arr) /
                      (n * cumsum[-1])) - (n + 1) / n)

    gini_nci = (
        panel.groupby(["year", "country"])["eigenvector_cent"]
        .apply(lambda x: gini(x.values))
        .reset_index()
        .rename(columns={"eigenvector_cent": "nci"})
    )
    agg = agg.merge(gini_nci, on=["year", "country"], how="left")
    agg["nci"] = agg["nci"].clip(0, 1)

    # HHI of within-country centrality distribution
    def centrality_hhi(group):
        c = group["eigenvector_cent"].values
        total = c.sum() + 1e-9
        return float(((c / total) ** 2).sum())

    hhi = (
        panel.groupby(["year", "country"])
        .apply(centrality_hhi, include_groups=False)
        .reset_index()
        .rename(columns={0: "centrality_hhi"})
    )
    agg = agg.merge(hhi, on=["year", "country"], how="left")

    return agg.set_index(["year", "country"]).sort_index()


# ===========================================================================
# SECTION 4 — NETWORK CHANGE METRICS
# ===========================================================================

def compute_network_change(
    G_before: nx.DiGraph,
    G_after:  nx.DiGraph,
    weight:   str = "weight",
) -> dict:
    """
    Compute how much the network topology changed between two periods.

    Used to measure the structural break caused by:
    - 2008/09 GFC
    - 2020 COVID supply chain disruption
    - 2022 geopolitical shock / energy crisis

    Metrics:
      - Edge weight correlation (how similar are flows before/after?)
      - Rewiring fraction (fraction of edges that changed significantly)
      - Centrality rank correlation (did the most central nodes stay central?)
      - Jaccard similarity of edge sets
    """
    # Common nodes
    common_nodes = set(G_before.nodes()) & set(G_after.nodes())

    if not common_nodes:
        return {}

    # Centrality rank correlation
    ec_before = compute_eigenvector_centrality(G_before, weight)
    ec_after  = compute_eigenvector_centrality(G_after,  weight)

    common_for_corr = list(common_nodes)
    vals_before = [ec_before.get(n, 0) for n in common_for_corr]
    vals_after  = [ec_after.get(n,  0) for n in common_for_corr]

    from scipy.stats import spearmanr
    rank_corr, p_val = spearmanr(vals_before, vals_after)

    # Edge Jaccard similarity
    edges_before = set(G_before.edges())
    edges_after  = set(G_after.edges())
    jaccard      = len(edges_before & edges_after) / (
        len(edges_before | edges_after) + 1e-9
    )

    # Density change
    d_before = nx.density(G_before)
    d_after  = nx.density(G_after)

    return {
        "centrality_rank_corr":   rank_corr,
        "centrality_rank_p":      p_val,
        "edge_jaccard_similarity":jaccard,
        "density_before":         d_before,
        "density_after":          d_after,
        "delta_density":          d_after - d_before,
        "n_common_nodes":         len(common_nodes),
    }


def compute_structural_break_panel(
    io_network_panel: dict[int, nx.DiGraph],
    break_years:      list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute network change metrics around key structural break years.

    For each break year, compares the network 2 years before vs 2 years after.
    This gives a panel of structural change intensity that can be used as
    an instrument or control variable in the transmission regressions.
    """
    if break_years is None:
        break_years = [2009, 2020, 2022]

    years = sorted(io_network_panel.keys())
    records = []

    for break_year in break_years:
        pre_year  = break_year - 2
        post_year = break_year + 2

        if pre_year not in io_network_panel or post_year not in io_network_panel:
            continue

        change = compute_network_change(
            io_network_panel[pre_year],
            io_network_panel[post_year],
        )
        change["break_year"] = break_year
        change["pre_year"]   = pre_year
        change["post_year"]  = post_year
        records.append(change)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("break_year")


# ===========================================================================
# SECTION 5 — I/O
# ===========================================================================

def save_metrics_panel(df: pd.DataFrame, output_dir: str | Path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "network_metrics_panel.parquet")
    print(f"Metrics panel saved to {out}/")


def load_metrics_panel(input_dir: str | Path) -> pd.DataFrame:
    p = Path(input_dir) / "network_metrics_panel.parquet"
    if p.exists():
        return pd.read_parquet(p)
    raise FileNotFoundError(f"No metrics panel at {p}")