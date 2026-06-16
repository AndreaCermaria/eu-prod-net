"""
tiva_loader.py
==============
Load and parse OECD Trade in Value Added (TiVA) 2025 data into input-output
networks for euro area production network analysis.

TiVA tracks the origin of value added in goods and services traded
internationally. The EXGR_INT indicator records bilateral intermediate
export flows: how much sector S in country A exports as intermediate inputs
to sector S in country B. This bilateral matrix is the edge data for
the production network.

Data download
-------------
From https://data-explorer.oecd.org (TiVA 2025 Principal Indicators, levels):

  EXGR_INT.zip  — intermediate exports          [required]
  PROD.zip      — gross output by country-sector [required]
  VALU.zip      — value added                    [optional, for controls]

Extract each ZIP to data/raw/tiva/.

TiVA 2025 CSV columns
---------------------
  REF_AREA         country (ISO3 reporter)
  COUNTERPART_AREA partner country (ISO3, or W for world total)
  ACTIVITY         ISIC Rev.4 sector code
  TIME_PERIOD      year
  OBS_VALUE        value in USD millions
  UNIT_MULT        scale exponent (typically 6; value × 10^(UNIT_MULT-6) = USD mn)

Network construction
--------------------
Only intra-EA bilateral flows are retained as directed edges. Flows to
non-EA partners are accumulated as node-level forward linkage attributes.
Aggregate ISIC codes (single-letter totals such as 'C' for all manufacturing,
and compound codes such as 'BTE', 'GTT') are excluded so that individual
sector nodes are not dominated by their own aggregate totals.

Synthetic data
--------------
generate_synthetic_tiva() produces calibrated synthetic data for pipeline
testing without a data download. Set USE_SYNTHETIC = True in the notebooks.
"""

import pandas as pd
import numpy as np
import networkx as nx
import requests
from pathlib import Path
from io import StringIO
import warnings
warnings.filterwarnings("ignore")


# ===========================================================================
# SECTION 1 — CONFIGURATION
# ===========================================================================

# Euro area countries in TiVA (ISO3 codes as used by OECD)
EA_COUNTRIES_TIVA = [
    "AUT", "BEL", "DEU", "ESP", "FIN",
    "FRA", "GRC", "IRL", "ITA", "LUX",
    "NLD", "PRT", "SVK", "SVN", "EST",
    "LVA", "LTU", "CYP", "MLT",
]

# Broader set for network context (add key non-EA trade partners)
EXTENDED_COUNTRIES = EA_COUNTRIES_TIVA + [
    "USA", "CHN", "GBR", "JPN", "KOR",
    "CHE", "NOR", "SWE", "DNK", "POL",
    "CZE", "HUN", "ROU",
]

# ISIC Rev.4 sector code to readable name mapping (TiVA 2025 ACTIVITY codes)
ISIC_SECTOR_NAMES = {
    # Agriculture, forestry, fishing
    "A01": "Agriculture",   "A02": "Forestry",   "A03": "Fishing",
    "A01T03": "Agriculture/Forestry/Fishing",
    # Mining
    "B05T06": "Coal/Oil mining",  "B07T08": "Metal/non-metal mining",
    "B09": "Mining support",      "B": "Mining",
    # Manufacturing
    "C10T12": "Food/Beverages",   "C13T15": "Textiles",
    "C16": "Wood products",       "C17T18": "Paper/Printing",
    "C19": "Petroleum refining",  "C20": "Chemicals",
    "C21": "Pharmaceuticals",     "C20T21": "Chemicals/Pharma",
    "C22T23": "Rubber/Plastics",  "C24": "Basic metals",
    "C25": "Fabricated metals",   "C242_2432": "Steel/Metals",
    "C24T25": "Metals",           "C26": "Electronics",
    "C27": "Electrical equipment","C26T27": "Electronics/Electrical",
    "C28": "Machinery",           "C29": "Motor vehicles",
    "C30": "Other transport",     "C29T30": "Automotive/Transport",
    "C31T33": "Other manufacturing",
    # Utilities
    "D35": "Electricity/Gas",     "E36T39": "Water/Waste",
    "D35T39": "Utilities",
    # Construction
    "F": "Construction",          "F41T43": "Construction",
    # Services
    "G45T47": "Trade",            "G46": "Wholesale",
    "H49T53": "Transport",        "H49": "Land transport",
    "H50": "Water transport",     "H51": "Air transport",
    "I": "Accommodation/Food",    "I55T56": "Accommodation/Food",
    "J58T60": "Publishing/Media", "J61": "Telecommunications",
    "J62T63": "IT services",      "J": "Information/Communication",
    "K64T66": "Finance/Insurance","K": "Finance",
    "L68": "Real estate",         "L": "Real estate",
    "M69T75": "Professional services", "M": "Professional",
    "N77T82": "Administrative services",
    "O84": "Public administration",
    "P85": "Education",
    "Q86T88": "Health/Social work","Q": "Health",
    "R90T92": "Arts/Entertainment",
    "S94T96": "Other services",
    "T97T98": "Household activities",
    # Aggregates used in TiVA 2025 (kept for name display, filtered from network)
    "MANUF": "Total manufacturing",
    "TOTAL": "Total economy",
    "DTOTAL": "Total domestic",
    # TiVA 2025 individual sector codes not in older editions
    "C10T12": "Food and beverages",
    "C13T15": "Textiles and apparel",
    "C16T18": "Wood, paper, printing",
    "C19":    "Petroleum refining",
    "C19T23": "Petroleum, chemicals, plastics",
    "C20_21": "Chemicals and pharmaceuticals",
    "C22_23": "Rubber, plastics, minerals",
    "C24":    "Basic metals",
    "C25":    "Fabricated metals",
    "C26":    "Electronics and computers",
    "C27":    "Electrical equipment",
    "C26T27": "Electronics and electrical",
    "C28":    "Machinery",
    "C29":    "Motor vehicles",
    "C30":    "Other transport equipment",
    "C31T33": "Furniture and other manufacturing",
    "D35":    "Electricity and gas",
    "E36T39": "Water and waste",
    "F41T43": "Construction",
    "G45T47": "Wholesale and retail trade",
    "G46":    "Wholesale trade",
    "H49T53": "Transport and storage",
    "H49":    "Land transport",
    "H50":    "Water transport",
    "H51":    "Air transport",
    "H52":    "Warehousing",
    "H53":    "Postal and courier",
    "I55T56": "Accommodation and food service",
    "J58T63": "Information and communication",
    "J58T60": "Publishing and broadcasting",
    "J61":    "Telecommunications",
    "J62T63": "IT services",
    "K64T66": "Finance and insurance",
    "K64":    "Financial services",
    "L68":    "Real estate",
    "M69T75": "Professional services",
    "M72":    "Research and development",
    "N77T82": "Administrative services",
    "O84":    "Public administration",
    "P85":    "Education",
    "Q86T88": "Health and social work",
    "R90T93": "Arts and recreation",
    "S94T96": "Other personal services",
    "R_S":    "Arts and other services",
    "RTT":    "Real estate activities",
    "OTT":    "Other transport",
    "GTI":    "Trade, transport and ICT",
    "GTN":    "Trade services",
    "JTN":    "ICT services",
    "BTE":    "Mining, utilities and water",
    "FTT":    "Construction total",
    "GTT":    "Trade and transport",
}

def get_sector_name(code: str) -> str:
    """Map ISIC activity code to readable sector name."""
    code = str(code).strip()
    if code in ISIC_SECTOR_NAMES:
        return ISIC_SECTOR_NAMES[code]
    # Try prefix match
    for k, v in ISIC_SECTOR_NAMES.items():
        if code.startswith(k) or k.startswith(code):
            return v
    return code  # return raw code if no match found



# Macro-sector aggregation (45 TiVA sectors → 15 groups)
SECTOR_GROUPS = {
    "Agriculture":      ["D01T03"],
    "Mining":           ["D05T09"],
    "Food":             ["D10T12"],
    "Textiles":         ["D13T15"],
    "Wood_Paper":       ["D16T18"],
    "Chemicals":        ["D19", "D20T21"],
    "Plastics_Metals":  ["D22T23", "D24T25"],
    "Electronics":      ["D26T27"],
    "Machinery":        ["D28"],
    "Automotive":       ["D29T30"],
    "Other_Manuf":      ["D31T33"],
    "Utilities":        ["D35T39"],
    "Construction":     ["D41T43"],
    "Trade_Transport":  ["D45T47", "D49T53"],
    "Finance_Business": ["D58T63", "D64T66", "D69T82"],
}

OECD_API_BASE_NEW = "https://sdmx.oecd.org/public/rest/data"
OECD_API_BASE_OLD = "https://stats.oecd.org/SDMX-JSON/data"
OECD_API_BASE     = OECD_API_BASE_OLD   # try old first, fallback to new

MANUAL_DOWNLOAD_GUIDE = """
TiVA data download: https://data-explorer.oecd.org
Search: TiVA 2025 Principal Indicators (levels)
Download: EXGR_INT.zip, PROD.zip, VALU.zip
Extract to: data/raw/tiva/
"""


# ===========================================================================
# SECTION 2 — OECD API DOWNLOAD
# ===========================================================================

def fetch_tiva_indicator(
    indicator:    str,
    countries:    list[str] | None = None,
    start_year:   int = 2000,
    end_year:     int = 2020,
) -> pd.DataFrame:
    """
    Fetch a single TiVA indicator from the OECD API.

    Parameters
    ----------
    indicator   : TiVA indicator code, e.g.:
                  'EXGR_DVASH' — domestic VA share of gross exports
                  'UPSTR'      — upstreamness
                  'GVC_SHARE'  — GVC participation share
                  'EXGR_INTSHr'— intermediate exports share
    countries   : list of ISO3 codes (None = all available)
    start_year  : first year
    end_year    : last year

    Returns
    -------
    pd.DataFrame: columns = [country, industry, year, value]
    """
    if countries is None:
        countries = EA_COUNTRIES_TIVA

    country_str = "+".join(countries)
    url = (
        f"{OECD_API_BASE}/TIVA_2023_C2/{country_str}.{indicator}."
        f"...?startTime={start_year}&endTime={end_year}"
        f"&contentType=csv&detail=code&separator=comma"
    )

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))

        # Standardise columns
        df.columns = df.columns.str.upper()
        col_map = {}
        for col in df.columns:
            if "LOCATION" in col or "COUNTRY" in col or "COU" == col:
                col_map[col] = "country"
            elif "INDUSTRY" in col or "IND" in col or "SECTOR" in col:
                col_map[col] = "industry"
            elif "TIME" in col or "YEAR" in col or "PERIOD" in col:
                col_map[col] = "year"
            elif "VALUE" in col or "OBS" in col:
                col_map[col] = "value"

        df = df.rename(columns=col_map)
        req_cols = [c for c in ["country","industry","year","value"]
                    if c in df.columns]
        df = df[req_cols].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["year"]  = pd.to_numeric(df["year"],  errors="coerce").astype("Int64")
        df = df.dropna(subset=["value"])

        print(f"  ✓ {indicator}: {len(df):,} observations")
        return df

    except Exception as e:
        print(f"  ⚠ {indicator}: {e}")
        return pd.DataFrame()


def download_tiva_summary_indicators(
    countries:  list[str] | None = None,
    start_year: int = 2000,
    end_year:   int = 2020,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download key TiVA summary indicators for network analysis.

    Indicators downloaded:
      EXGR_DVASH  — domestic value added share of gross exports
                    (proxy for sector's value-chain integration depth)
      UPSTR       — upstreamness (distance from final demand)
                    (key network metric: how far upstream is this sector?)
      GVC_SHARE   — global value chain participation share
                    (how much of output goes through international chains?)
      EXGR_INTSHr — intermediate exports as share of gross exports
                    (how much of output becomes inputs for other sectors?)

    Returns
    -------
    dict mapping indicator name → long-format DataFrame
    """
    if countries is None:
        countries = EA_COUNTRIES_TIVA

    indicators = {
        "EXGR_DVASH":   "Domestic VA share of gross exports",
        "UPSTR":        "Upstreamness (distance from final demand)",
        "GVC_SHARE":    "GVC participation share",
        "EXGR_INTSHr":  "Intermediate exports share",
    }

    print(f"Downloading TiVA summary indicators...")
    print(f"  Countries: {len(countries)} | Years: {start_year}–{end_year}")
    results = {}

    for code, label in indicators.items():
        df = fetch_tiva_indicator(code, countries, start_year, end_year)
        if len(df) > 0:
            results[code] = df

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for code, df in results.items():
            df.to_parquet(out / f"tiva_{code.lower()}.parquet")
        print(f"\nSaved to {out}/")

    return results


def load_tiva_z_matrix(filepath: str | Path) -> pd.DataFrame:
    """
    Load the TiVA Z matrix (intermediate input flows) from a local CSV.

    The Z matrix is the core of the production network:
      Z[from_country_sector, to_country_sector] = intermediate input value (USD mn)

    Download the full Z matrix from:
    https://stats.oecd.org/Index.aspx?DataSetCode=TIVA_2023_C1
    → Export → Related Files → TIVA_2023_C1.zip
    Extract and save the main CSV to data/raw/tiva/

    Expected format: OECD standard CSV with columns:
      REPORTER, PARTNER, INDUSTRY_REPORTER, INDUSTRY_PARTNER, YEAR, Value

    Returns
    -------
    pd.DataFrame: long format with columns
        [from_country, from_sector, to_country, to_sector, year, flow_usd_mn]
    """
    fp = Path(filepath)
    print(f"Loading Z matrix from {fp.name}...")

    if fp.suffix in [".xlsx", ".xls"]:
        raw = pd.read_excel(filepath, low_memory=False)
    else:
        raw = pd.read_csv(filepath, low_memory=False)

    raw.columns = raw.columns.str.upper().str.strip()

    # Map to standard column names
    col_map = {}
    for col in raw.columns:
        if col in ["REPORTER", "COU", "COUNTRY"]:
            col_map[col] = "from_country"
        elif col in ["PARTNER", "PAR"]:
            col_map[col] = "to_country"
        elif "IND" in col and "REPORTER" in col:
            col_map[col] = "from_sector"
        elif "IND" in col and "PARTNER" in col:
            col_map[col] = "to_sector"
        elif col in ["TIME", "YEAR", "PERIOD"]:
            col_map[col] = "year"
        elif col in ["VALUE", "OBS_VALUE", "OBSVALUE"]:
            col_map[col] = "flow_usd_mn"

    raw = raw.rename(columns=col_map)
    required = ["from_country","to_country","from_sector",
                "to_sector","year","flow_usd_mn"]

    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Could not identify columns: {missing}. "
            f"Available: {list(raw.columns)}"
        )

    df = raw[required].copy()
    df["flow_usd_mn"] = pd.to_numeric(df["flow_usd_mn"], errors="coerce")
    df["year"]        = pd.to_numeric(df["year"],        errors="coerce").astype("Int64")
    df = df.dropna(subset=["flow_usd_mn"]).query("flow_usd_mn > 0")

    print(f"  Loaded: {len(df):,} bilateral sector flows")
    print(f"  Countries: {df.from_country.nunique()} × {df.to_country.nunique()}")
    print(f"  Sectors:   {df.from_sector.nunique()} × {df.to_sector.nunique()}")
    print(f"  Years:     {df.year.min()} – {df.year.max()}")

    return df


# ===========================================================================
# SECTION 3 — NETWORK CONSTRUCTION
# ===========================================================================

def build_io_network(
    z_matrix:       pd.DataFrame,
    year:           int,
    countries:      list[str] | None = None,
    min_flow_usd_mn:float = 10.0,
) -> nx.DiGraph:
    """
    Build a directed weighted input-output network for a given year.

    Parameters
    ----------
    z_matrix        : long-format Z matrix from load_tiva_z_matrix()
    year            : which year to build the network for
    countries       : filter to these countries (None = all)
    min_flow_usd_mn : minimum edge weight to include (filters noise)

    Returns
    -------
    nx.DiGraph where:
      nodes = f"{country}_{sector}" strings
      edges = directed intermediate input flows
      edge weight = flow in USD millions
      node attrs  = country, sector, total_output_proxy
    """
    df = z_matrix[z_matrix["year"] == year].copy()

    if countries:
        df = df[
            df["from_country"].isin(countries) &
            df["to_country"].isin(countries)
        ]

    df = df[df["flow_usd_mn"] >= min_flow_usd_mn]

    G = nx.DiGraph()

    # Add edges
    for row in df.itertuples(index=False):
        src  = f"{row.from_country}_{row.from_sector}"
        dst  = f"{row.to_country}_{row.to_sector}"
        flow = row.flow_usd_mn

        G.add_node(src, country=row.from_country, sector=row.from_sector)
        G.add_node(dst, country=row.to_country,   sector=row.to_sector)

        if G.has_edge(src, dst):
            G[src][dst]["weight"] += flow
        else:
            G.add_edge(src, dst, weight=flow)

    # Normalise weights for comparability across years
    max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1.0)
    for u, v in G.edges():
        G[u][v]["weight_norm"] = G[u][v]["weight"] / max_w

    print(f"  {year}: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def build_io_network_panel(
    z_matrix:       pd.DataFrame,
    years:          list[int] | None = None,
    countries:      list[str] | None = None,
    min_flow_usd_mn:float = 10.0,
) -> dict[int, nx.DiGraph]:
    """Build IO network for every year in the Z matrix."""
    if years is None:
        years = sorted(z_matrix["year"].dropna().unique().astype(int))

    if countries is None:
        countries = EA_COUNTRIES_TIVA

    print(f"Building IO network panel ({len(years)} years)...")
    panel = {}
    for y in years:
        panel[y] = build_io_network(z_matrix, y, countries, min_flow_usd_mn)

    return panel


# ===========================================================================
# SECTION 4 — SYNTHETIC DATA GENERATOR
# ===========================================================================

def generate_synthetic_tiva(
    countries:    list[str] | None = None,
    sectors:      list[str] | None = None,
    start_year:   int = 2000,
    end_year:     int = 2020,
    seed:         int = 42,
) -> dict:
    """
    Generate synthetic TiVA-style data for pipeline development.

    Produces:
    1. Z matrix (long format): bilateral sector flows
    2. Summary indicators: upstreamness, GVC share, VA share

    Calibrated to reproduce key structural features of euro area
    production networks:
    - Germany, France, Italy, Netherlands as major hubs
    - Manufacturing sectors more central than services
    - Post-2008 and post-2020 structural breaks in network topology
    - Cross-country linkages denser within EA than with rest of world

    Returns
    -------
    dict with keys: 'z_matrix', 'upstr', 'gvc_share', 'va_share'
    """
    rng = np.random.default_rng(seed)

    if countries is None:
        countries = EA_COUNTRIES_TIVA[:10]  # subset for speed

    if sectors is None:
        sectors = list(SECTOR_GROUPS.keys())

    years = list(range(start_year, end_year + 1))

    # --- Calibrated base flow matrix ---
    # Each country-sector pair has a base production capacity
    # Germany and France are more central (higher weights)
    country_weights = {
        "DEU": 4.0, "FRA": 3.2, "ITA": 2.8, "ESP": 2.3,
        "NLD": 2.5, "BEL": 1.8, "AUT": 1.4, "FIN": 1.2,
        "PRT": 0.9, "IRL": 1.6, "GRC": 0.7, "LUX": 1.1,
    }
    # Manufacturing sectors more central
    sector_weights = {
        "Automotive": 3.5, "Electronics": 3.2, "Machinery": 3.0,
        "Chemicals": 2.8, "Plastics_Metals": 2.5, "Food": 2.0,
        "Trade_Transport": 2.2, "Finance_Business": 2.0,
        "Utilities": 1.8, "Construction": 1.5, "Textiles": 1.2,
        "Agriculture": 1.0, "Mining": 1.0, "Wood_Paper": 1.1,
        "Other_Manuf": 1.3,
    }

    # Structural breaks
    BREAKS = {
        2009: 0.85,  # GFC: 15% reduction in cross-border flows
        2020: 0.80,  # COVID: 20% reduction
        2022: 0.90,  # geopolitical shock: 10% reduction
    }

    z_records   = []
    upstr_records = []

    for year in years:
        # Year-level multiplier
        multiplier = 1.0
        for break_year, factor in BREAKS.items():
            if year >= break_year:
                multiplier *= factor
        # Recovery: partial rebound after shocks
        if year >= 2010:
            multiplier = min(multiplier * 1.08, 1.0)
        if year >= 2021:
            multiplier = min(multiplier * 1.05, 0.95)

        for fc in countries:
            cw_f = country_weights.get(fc, 1.0)
            for fs in sectors:
                sw_f = sector_weights.get(fs, 1.0)
                for tc in countries:
                    cw_t = country_weights.get(tc, 1.0)
                    for ts in sectors:
                        sw_t = sector_weights.get(ts, 1.0)

                        # Same-country flows are larger
                        home_bias = 3.0 if fc == tc else 1.0

                        # Base flow
                        base = (
                            cw_f * sw_f * cw_t * sw_t *
                            home_bias * multiplier * 50
                        )
                        flow = base * rng.lognormal(0, 0.4)

                        if flow > 5.0:  # minimum threshold
                            z_records.append({
                                "from_country": fc,
                                "from_sector":  fs,
                                "to_country":   tc,
                                "to_sector":    ts,
                                "year":         year,
                                "flow_usd_mn":  flow,
                            })

            # Upstreamness: manufacturing is more upstream (further from demand)
            base_upstr = (
                sector_weights.get(fs, 1.0) / max(sector_weights.values()) * 3
                + rng.normal(0, 0.2)
            )
            upstr_records.append({
                "country":  fc,
                "industry": fs,
                "year":     year,
                "value":    max(1.0, base_upstr),
            })

    z_df    = pd.DataFrame(z_records)
    upstr_df= pd.DataFrame(upstr_records)

    # GVC share (correlated with upstreamness)
    gvc_df = upstr_df.copy()
    gvc_df["value"] = (upstr_df["value"] / 4 + rng.normal(0, 0.05, len(upstr_df))).clip(0.1, 0.9)

    # VA share (inversely related to upstreamness)
    va_df = upstr_df.copy()
    va_df["value"] = (1 - upstr_df["value"] / 5 + rng.normal(0, 0.05, len(upstr_df))).clip(0.2, 0.95)

    print(f"Synthetic TiVA:")
    print(f"  Z matrix: {len(z_df):,} flows | "
          f"{len(countries)} countries | {len(sectors)} sectors | "
          f"{start_year}–{end_year}")
    print(f"  Indicators: upstreamness, GVC share, VA share")
    print("NOTE: synthetic data — download real data from stats.oecd.org")

    return {
        "z_matrix":  z_df,
        "UPSTR":     upstr_df,
        "GVC_SHARE": gvc_df,
        "EXGR_DVASH":va_df,
    }


# ===========================================================================
# SECTION 5 — I/O UTILITIES
# ===========================================================================


def load_tiva_2025(
    data_dir: str | Path,
    countries: list[str] | None = None,
    indicators: list[str] | None = None,
) -> dict:
    """
    Load TiVA 2025 edition data from downloaded ZIP/CSV files.

    This is the PRIMARY loader for real data. TiVA 2025 uses a different
    CSV format than the 2023 edition — each indicator is a separate file.

    DOWNLOAD INSTRUCTIONS:
      From https://data-explorer.oecd.org (TiVA 2025 Principal Indicators):
        EXGR_INT.zip  → intermediate exports  [REQUIRED]
        PROD.zip      → gross output           [REQUIRED]
        VALU.zip      → value added            [RECOMMENDED]

      Extract each ZIP to data/raw/tiva/. You will have:
        data/raw/tiva/EXGR_INT.csv
        data/raw/tiva/PROD.csv
        data/raw/tiva/VALU.csv

    TiVA 2025 CSV format:
      Columns include: INDICATOR, REF_AREA, PARTNER, INDUSTRY,
                       TIME_PERIOD, OBS_VALUE
      REF_AREA = reporter country (ISO3)
      PARTNER  = partner country (ISO3 or WLD)
      INDUSTRY = ISIC Rev.4 code
      TIME_PERIOD = year
      OBS_VALUE = value in USD millions

    Parameters
    ----------
    data_dir   : directory containing the extracted CSV files
    countries  : filter to these ISO3 codes (None = all)
    indicators : which files to load (None = auto-detect all CSVs)

    Returns
    -------
    dict mapping indicator name → standardised long-format DataFrame
      Each DataFrame has columns: country, partner, industry, year, value
    """
    data_dir = Path(data_dir)

    if indicators is None:
        # Auto-detect available CSV files
        csv_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.CSV"))
        # Prefer our key indicators if they exist
        key_files  = ["EXGR_INT.csv", "PROD.csv", "VALU.csv",
                      "EXGR_DVA.csv", "EXGR_INT.csv"]
        found      = {fp.stem.upper(): fp for fp in csv_files}
        indicators = list(found.keys())
    else:
        found = {}
        for ind in indicators:
            for ext in [".csv", ".CSV"]:
                fp = data_dir / f"{ind}{ext}"
                if fp.exists():
                    found[ind.upper()] = fp
                    break

    if not found:
        raise FileNotFoundError(
            f"No TiVA CSV files found in {data_dir}. "
            "Download EXGR_INT.zip, PROD.zip from the OECD TiVA 2025 portal "
            "and extract CSVs to this directory."
        )

    result = {}
    for ind_name, filepath in sorted(found.items()):
        df = _load_tiva_2025_csv(filepath, countries)
        if df is not None and len(df) > 0:
            result[ind_name] = df
            print(f"  ✓ {ind_name:<15}: {len(df):,} obs | "
                  f"{df['country'].nunique()} countries | "
                  f"{df['year'].min()}–{df['year'].max()}")
        else:
            print(f"  ✗ {ind_name:<15}: no data loaded")

    return result


def _load_tiva_2025_csv(
    filepath: str | Path,
    countries: list[str] | None = None,
) -> pd.DataFrame | None:
    """
    Parse a single TiVA 2025 CSV file into standardised long format.

    TiVA 2025 actual column structure (confirmed from downloaded files):
      DATAFLOW, MEASURE, REF_AREA, ACTIVITY, COUNTERPART_AREA,
      UNIT_MEASURE, FREQ, TIME_PERIOD, OBS_VALUE, UNIT_MULT

    Key mappings:
      REF_AREA         → country  (reporter, ISO3)
      COUNTERPART_AREA → partner  (destination/partner, ISO3 or W=world)
      ACTIVITY         → industry (ISIC Rev.4 sector code)
      TIME_PERIOD      → year
      OBS_VALUE        → value (in USD millions × 10^UNIT_MULT)
      UNIT_MULT        → exponent: value × 10^UNIT_MULT = actual USD
                         (typically 6, meaning OBS_VALUE is in USD millions
                          so actual value = OBS_VALUE × 10^6 / 10^6 = USD mn
                          BUT: OBS_VALUE=475, UNIT_MULT=6 means
                          475 × 10^6 = 475 million USD — keep as USD mn)

    UNIT_MULT interpretation:
      UNIT_MULT=6, OBS_VALUE=475 → 475 × 10^6 USD → 475 million USD
      We store values as USD millions for consistency with other sources.
      So: stored_value = OBS_VALUE × 10^(UNIT_MULT - 6)
      When UNIT_MULT=6: stored_value = OBS_VALUE × 1 = OBS_VALUE (USD mn)
      When UNIT_MULT=3: stored_value = OBS_VALUE × 0.001 (USD mn)

    For EXGR_INT: bilateral intermediate exports FROM ref_area TO partner.
    Rows where REF_AREA == COUNTERPART_AREA are DOMESTIC flows (intra).
    Cross-border rows have REF_AREA ≠ COUNTERPART_AREA — these are
    the production network EDGES we use.
    """
    fp = Path(filepath)
    try:
        raw = pd.read_csv(filepath, low_memory=False, sep=None, engine='python')
    except Exception as e:
        try:
            raw = pd.read_csv(filepath, low_memory=False)
        except Exception as e2:
            print(f"    Could not read {fp.name}: {e2}")
            return None

    raw.columns = raw.columns.str.upper().str.strip()

    # TiVA 2025 format: REF_AREA + ACTIVITY + COUNTERPART_AREA + TIME_PERIOD
    is_2025 = (
        "REF_AREA" in raw.columns and
        "ACTIVITY" in raw.columns and
        "TIME_PERIOD" in raw.columns
    )
    # Older TiVA format: LOCATION + INDUSTRY + TIME
    is_older = "LOCATION" in raw.columns or "REPORTER" in raw.columns

    if is_2025:
        col_map = {
            "REF_AREA":         "country",
            "COUNTERPART_AREA": "partner",
            "ACTIVITY":         "industry",
            "TIME_PERIOD":      "year",
            "OBS_VALUE":        "value",
            "UNIT_MULT":        "unit_mult",
            "MEASURE":          "indicator",
        }
    elif is_older:
        col_map = {}
        for col in raw.columns:
            if col in ["LOCATION", "COU", "REPORTER", "REF_AREA"]:
                col_map[col] = "country"
            elif col in ["PARTNER", "PAR", "COUNTERPART_AREA"]:
                col_map[col] = "partner"
            elif col in ["INDICATOR", "IND", "VARIABLE", "MEASURE"]:
                col_map[col] = "indicator"
            elif col in ["INDUSTRY", "SECTOR", "IND_CODE", "ACTIVITY"]:
                col_map[col] = "industry"
            elif col in ["TIME", "YEAR", "PERIOD", "TIME_PERIOD"]:
                col_map[col] = "year"
            elif col in ["VALUE", "OBS_VALUE", "OBSVALUE", "VAL"]:
                col_map[col] = "value"
    else:
        # Try generic detection
        col_map = {}
        for col in raw.columns:
            cl = col.strip()
            if "AREA" in cl and "REF" in cl:     col_map[col] = "country"
            elif "AREA" in cl:                    col_map[col] = "partner"
            elif "ACTIV" in cl or "IND" in cl:   col_map[col] = "industry"
            elif "TIME" in cl or "PERIOD" in cl: col_map[col] = "year"
            elif "OBS" in cl or "VALUE" in cl:   col_map[col] = "value"

    df = raw.rename(columns=col_map)

    # Keep essential columns
    keep = [c for c in ["country","partner","industry","year","value",
                        "indicator","unit_mult"]
            if c in df.columns]
    df = df[keep].copy()

    # Numeric conversion
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce").astype("Int64")

    # Apply UNIT_MULT scaling to get consistent USD millions
    # OBS_VALUE × 10^(UNIT_MULT-6) = USD millions
    if "unit_mult" in df.columns:
        df["unit_mult"] = pd.to_numeric(df["unit_mult"], errors="coerce").fillna(6)
        df["value"] = df["value"] * (10 ** (df["unit_mult"] - 6))
        df = df.drop(columns=["unit_mult"])

    df = df.dropna(subset=["value", "year", "country"])
    df = df[df["value"] != 0]

    # Filter to EA countries
    if countries:
        # Keep rows where reporter OR partner is in our country list
        mask_reporter = df["country"].isin(countries)
        if "partner" in df.columns:
            mask_partner = df["partner"].isin(countries + ["W", "WLD", "_Z"])
            df = df[mask_reporter]  # filter reporter; keep all partners
        else:
            df = df[mask_reporter]

    return df.reset_index(drop=True)


def load_tiva_from_csv(
    filepath: str | Path,
    countries: list[str] | None = None,
) -> dict:
    """
    Load a single TiVA CSV file (any edition).

    Wrapper around _load_tiva_2025_csv that handles both
    TiVA 2025 and older TiVA 2023 formats automatically.

    For TiVA 2025 with multiple files, use load_tiva_2025() instead.

    Parameters
    ----------
    filepath  : path to a single TiVA CSV file
    countries : filter to these ISO3 codes (None = all)

    Returns
    -------
    dict: {indicator_name: DataFrame} or {'VALUE': DataFrame}
    """
    fp = Path(filepath)
    print(f"Loading TiVA from {fp.name}...")

    df = _load_tiva_2025_csv(filepath, countries)
    if df is None or len(df) == 0:
        return {}

    print(f"  Loaded: {len(df):,} obs | "
          f"{df['country'].nunique()} countries | "
          f"{df['year'].min()}–{df['year'].max()}")

    # If indicator column present, split by indicator
    if "indicator" in df.columns:
        result = {}
        for ind in df["indicator"].unique():
            result[str(ind)] = df[df["indicator"] == ind].drop(
                columns=["indicator"]
            ).reset_index(drop=True)
        return result
    else:
        # Name by filename stem
        return {fp.stem.upper(): df}


def load_z_matrix_from_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load Z matrix (intermediate input flows) from manually downloaded CSV.

    The full Z matrix is available from OECD TiVA 2023 C1 dataset:
    https://stats.oecd.org/Index.aspx?DataSetCode=TIVA_2023_C1

    This is a large file (~500MB). For initial analysis, the summary
    indicators (C2) are sufficient to construct upstreamness and
    GVC metrics. The full Z matrix is needed for the network edges.

    Parameters
    ----------
    filepath : path to Z matrix CSV

    Returns
    -------
    Long-format DataFrame with columns:
        from_country, from_sector, to_country, to_sector, year, flow_usd_mn
    """
    return load_tiva_z_matrix(filepath)


def build_network_from_tiva2025(
    tiva_data:  dict,
    year:       int,
    countries:  list[str] | None = None,
    min_share:  float = 0.001,
) -> nx.DiGraph:
    """
    Build a production network from TiVA 2025 EXGR_INT and PROD files.

    EXGR_INT gives intermediate exports: how much sector i in country A
    exports as intermediate inputs. This is our edge data.

    PROD gives gross output: the size of each node.

    Edge weight construction:
      We use intermediate export SHARE = EXGR_INT / PROD
      This normalises for country/sector size so that Germany's
      automotive sector is not automatically more "central" just
      because Germany is bigger.

    Parameters
    ----------
    tiva_data  : dict from load_tiva_2025()
    year       : which year to build network for
    countries  : filter to EA countries (None = use EA_COUNTRIES_TIVA)
    min_share  : minimum intermediate export share to include as edge

    Returns
    -------
    nx.DiGraph: nodes = "COUNTRY_SECTOR" strings
    """
    if countries is None:
        countries = EA_COUNTRIES_TIVA

    # Aggregate industry codes to exclude:
    # TOTAL, MANUF, SERV etc. have enormous flow values by construction
    # (they are sums of other rows) and would dominate centrality.
    # We only keep individual ISIC Rev.4 sectors.
    # TiVA 2025 uses TWO types of aggregate codes that must be excluded:
    #
    # Type 1 — Standard aggregates (same as older editions):
    #   TOTAL, MANUF, SERV etc.
    #
    # Type 2 — Single-letter codes (TiVA 2025 specific):
    #   "C" = total manufacturing (sum of C10 through C33)
    #   "B" = total mining, "F" = construction, "G" = trade, etc.
    #   These represent ALL activity in a sector group and have flow
    #   values equal to the SUM of all individual sub-sectors.
    #   If included, DEU_C dominates forward linkage because it equals
    #   the entire German manufacturing sector's intermediate exports.
    #
    # Type 3 — Compound aggregates ending in T/N/I:
    #   "BTE" = mining+utilities+water, "GTT" = trade+transport, etc.
    #
    # Rule: filter any code that is (a) one letter, or (b) in known list
    AGGREGATE_INDUSTRIES = {
        # Standard
        "TOTAL", "DTOTAL", "MANUF", "SERV", "TSERV", "BSERV",
        "TOPROD", "DPROD", "MARKET", "NONMARKET", "D",
        "_T", "_X", "S13", "S1",
        # Single-letter sector totals (TiVA 2025)
        "A", "B", "C", "E", "F", "G", "H", "I", "J", "K",
        "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
        # Compound aggregates (TiVA 2025)
        "BTE", "FTT", "GTT", "GTN", "GTI", "JTN", "MN",
        "BOE", "DOE", "ATOQ", "ATOZ", "A_B", "C_D",
        "GTI", "DTOT",
    }

    # Get EXGR_INT
    exgr_int = tiva_data.get("EXGR_INT")
    if exgr_int is None:
        raise ValueError(
            "EXGR_INT not found in tiva_data. "
            "Download EXGR_INT.zip from OECD TiVA 2025."
        )

    # Get PROD for normalisation
    prod = tiva_data.get("PROD")

    # Filter to year, countries, and individual sectors (no aggregates)
    df = exgr_int[
        (exgr_int["year"] == year) &
        (exgr_int["country"].isin(countries)) &
        (~exgr_int["industry"].astype(str).isin(AGGREGATE_INDUSTRIES))
    ].copy()
    
    # Also filter PROD to individual sectors
    if prod is not None:
        prod = prod[
            (prod["year"] == year) &
            (~prod["industry"].astype(str).isin(AGGREGATE_INDUSTRIES))
        ]

    if len(df) == 0:
        raise ValueError(f"No EXGR_INT data for year {year}")

    # Use raw USD million values as edge weights.
    # PageRank normalises intrinsically so no need for PROD-based normalisation.
    # PROD normalisation was causing tiny weights that filtered out major sectors.
    df["weight"] = df["value"].clip(lower=0)

    # Filter only truly zero flows (keep all positive values)
    df = df[df["weight"] > 0]

    G = nx.DiGraph()

    # EXGR_INT real structure (TiVA 2025):
    #   REF_AREA (country)         = exporting country
    #   COUNTERPART_AREA (partner) = destination country receiving inputs
    #   ACTIVITY (industry)        = sector of the exporting country
    #
    # Each row = "country X in sector S exports intermediate inputs to partner P"
    # Edge direction: country_sector → partner_sector (same industry assumed)
    # Rows where country == partner are domestic flows (keep as node weight)

    has_partner = "partner" in df.columns and df["partner"].notna().any()

    for row in df.itertuples(index=False):
        sector_name = get_sector_name(str(row.industry))
        src    = f"{row.country}_{sector_name}"
        output = getattr(row, "output", np.nan)
        G.add_node(src, country=row.country, sector=row.industry,
                   sector_name=sector_name, output=output)

        if has_partner:
            partner = str(getattr(row, "partner", ""))
            # Only add edges to OTHER EA countries (intra-EA network)
            # Skip: non-EA partners, world aggregates, domestic flows
            # This gives us the true intra-EA production network where
            # centrality measures each sector's importance WITHIN the EA
            skip_partners = {"W", "WLD", "_Z", "OECD", "G20", "EA", "EU",
                             "WORLD", "ROW", "", "_T", "XDC"}
            is_ea_partner = partner in countries  # countries = EA_COUNTRIES_TIVA
            is_domestic   = partner == row.country
            is_aggregate  = partner in skip_partners

            if is_ea_partner and not is_domestic and not is_aggregate:
                # Intra-EA edge: both reporter and partner are EA countries
                dst = f"{partner}_{sector_name}"
                if dst not in G.nodes:
                    G.add_node(dst, country=partner, sector=row.industry,
                               sector_name=sector_name)
                w = float(getattr(row, "weight", row.value))
                if G.has_edge(src, dst):
                    G[src][dst]["weight"]       += w
                    G[src][dst]["value_usd_mn"] += float(row.value)
                else:
                    G.add_edge(src, dst, weight=w,
                               value_usd_mn=float(row.value))
            else:
                # Non-EA partner or domestic: accumulate as forward linkage
                # This captures exports to the world even if we don't track
                # which non-EA country receives them
                cur = G.nodes[src].get("forward_linkage", 0)
                G.nodes[src]["forward_linkage"] = (
                    cur + float(row.value)
                )
        else:
            G.nodes[src]["forward_linkage"] = float(row.value)

    # Fallback: if no cross-country edges, build proxy from forward linkages
    if G.number_of_edges() == 0:
        print(f"    No cross-country edges — building forward-linkage proxy")
        nodes = list(G.nodes())
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                fl1 = G.nodes[n1].get("forward_linkage", 0)
                fl2 = G.nodes[n2].get("forward_linkage", 0)
                w   = (fl1 * fl2) ** 0.5  # geometric mean
                if w > 0.01:
                    G.add_edge(n1, n2, weight=w)
                    G.add_edge(n2, n1, weight=w)

    print(f"  {year}: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges (from EXGR_INT)")
    return G


def build_network_panel_from_tiva2025(
    tiva_data:  dict,
    countries:  list[str] | None = None,
    years:      list[int] | None = None,
    min_share:  float = 0.001,
    ea_only:    bool  = True,
) -> dict[int, nx.DiGraph]:
    """
    Build annual IO network panel from TiVA 2025 data.

    Parameters
    ----------
    ea_only : if True (default), build EA-only subgraph.
              This is critical for meaningful centrality computation:
              in the full network (~3,500 nodes) each EA country-sector
              has near-zero centrality because the network is dominated
              by the sheer number of non-EA destination nodes.
              The EA-only network (~950 nodes = 19 countries × 50 sectors)
              gives centrality that reflects the INTRA-EA production structure —
              which is the economically relevant quantity for ECB transmission.
    """
    exgr_int = tiva_data.get("EXGR_INT")
    if exgr_int is None:
        raise ValueError("EXGR_INT required")

    if countries is None:
        countries = EA_COUNTRIES_TIVA

    if years is None:
        years = sorted(exgr_int["year"].dropna().unique().astype(int))

    print(f"Building network panel from TiVA 2025 ({len(years)} years)...")
    panel = {}
    for y in years:
        try:
            G_full = build_network_from_tiva2025(
                tiva_data, y, countries, min_share
            )

            if ea_only:
                # Keep only nodes where the country is an EA reporter
                ea_nodes = [
                    n for n, d in G_full.nodes(data=True)
                    if d.get("country") in countries
                ]
                G = G_full.subgraph(ea_nodes).copy()
                # Recompute edge weights after subgraph
                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()
                print(f"  {y}: {n_nodes} nodes, {n_edges} edges "
                      f"(EA-only subgraph from {G_full.number_of_nodes()} total nodes)")
            else:
                G = G_full
                print(f"  {y}: {G.number_of_nodes()} nodes, "
                      f"{G.number_of_edges()} edges (full network)")

            panel[y] = G
        except Exception as e:
            print(f"  ⚠ {y}: {e}")

    return panel

def save_tiva(data: dict, output_dir: str | Path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_parquet(out / f"tiva_{name.lower()}.parquet")
    print(f"TiVA data saved to {out}/")


def load_tiva(input_dir: str | Path) -> dict:
    p = Path(input_dir)
    data = {}
    for fp in p.glob("tiva_*.parquet"):
        key = fp.stem.replace("tiva_", "").upper()
        data[key] = pd.read_parquet(fp)
    if not data:
        raise FileNotFoundError(
            f"No TiVA files in {p}. "
            "Run download_tiva_summary_indicators() or "
            "generate_synthetic_tiva() first."
        )
    return data
