"""
transmission_data.py
====================
Download and process monetary policy transmission outcome variables.

THREE DATA SOURCES
-------------------

1. ECB BANK LENDING SURVEY (BLS)
   Quarterly survey of euro area bank loan officers asking whether
   credit standards for loans to enterprises are tightening or easing,
   and why. Published since 2003.

   Key variable: net percentage of banks reporting TIGHTENING
     (positive = tightening, negative = easing)
   Available by: country, loan type (enterprises / households)
   Free download: ecb.europa.eu/stats/ecb_surveys/bank_lending_survey
   Also via ECB SDW API: series starting with 'BLS.'

2. BIS LOCATIONAL BANKING STATISTICS (LBS)
   Quarterly bilateral cross-border bank lending by country.
   Measures how much banks in one country lend to residents of another.
   Used as an alternative transmission outcome: do cross-border credit
   flows contract more for countries with more central production networks?
   Free download: bis.org/statistics/bankstats.htm

3. ECB STATISTICAL DATA WAREHOUSE (SDW)
   ECB policy rates, bank lending rates by sector, and loan volumes.
   Free API: sdw.ecb.europa.eu/quickview.do

ECB POLICY RATE EPISODES
-------------------------
For the transmission regression we need to identify ECB rate change
episodes. Key episodes in our sample (2000–2024):

  TIGHTENING:
    2000-02 to 2000-10: +175bps (dot-com boom)
    2005-12 to 2007-07: +225bps (pre-GFC)
    2011-04 to 2011-07: +50bps (premature tightening)
    2022-07 to 2023-09: +450bps (inflation fighting cycle)

  EASING:
    2001-01 to 2003-06: -275bps (dot-com bust + 9/11)
    2008-10 to 2009-05: -325bps (GFC)
    2011-11 to 2016-03: -150bps (sovereign debt crisis)
    2019-09:             -10bps (pre-COVID)
    2020-03:             Emergency measures (COVID)

The 2022–2023 tightening cycle is our primary identification episode:
it was the most aggressive tightening in ECB history, happened from
a well-defined starting point, and occurred during a period of
significant production network restructuring — making it ideal for
testing heterogeneous transmission.
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ===========================================================================
# SECTION 1 — ECB SDW API
# ===========================================================================

ECB_SDW_BASE = "https://data-api.ecb.europa.eu/service/data"

ECB_SDW_SERIES = {
    # Policy rates
    "MRO_RATE":       "FM.B.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",  # 3M Euribor (policy proxy)
    "DEPOSIT_RATE":   "FM.B.U2.EUR.RT.MM.EURIBOR1YD_.HSTA",  # 1Y Euribor

    # Bank lending rates to NFCs
    "LENDING_RATE_NFC_DE": "MIR.M.DE.B.A2B.A.R.A.2240.EUR.N",  # Germany NFC rate
    "LENDING_RATE_NFC_FR": "MIR.M.FR.B.A2B.A.R.A.2240.EUR.N",  # France NFC rate
    "LENDING_RATE_NFC_IT": "MIR.M.IT.B.A2B.A.R.A.2240.EUR.N",  # Italy NFC rate
    "LENDING_RATE_NFC_ES": "MIR.M.ES.B.A2B.A.R.A.2240.EUR.N",  # Spain NFC rate

    # Loan volumes to NFCs (growth rates)
    "LOAN_GROWTH_NFC_EA": "BSI.M.U2.N.A.A20.A.1.U2.2240.Z01.E",  # EA NFC loans YoY
}

def fetch_ecb_sdw_series(
    series_key:   str,
    start_period: str = "2000-Q1",
    end_period:   str = "2024-Q4",
    frequency:    str = "Q",
) -> pd.Series | None:
    """
    Fetch a time series from the ECB Data Portal API.

    Parameters
    ----------
    series_key   : ECB SDW series key
    start_period : start period in YYYY-QN or YYYY-MM format
    end_period   : end period

    Returns
    -------
    pd.Series indexed by quarter timestamps, or None on failure
    """
    # Determine dataset from key prefix
    dataset = series_key.split(".")[0]
    url = (
        f"{ECB_SDW_BASE}/{dataset}/{series_key}"
        f"?startPeriod={start_period}&endPeriod={end_period}"
        f"&format=csvdata"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        if len(resp.text) < 50:
            return None

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))

        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            return None

        series = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        try:
            series.index = pd.PeriodIndex(
                df["TIME_PERIOD"], freq=frequency
            ).to_timestamp()
        except Exception:
            series.index = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")

        return series.dropna().sort_index()

    except Exception as e:
        return None


# ===========================================================================
# SECTION 2 — BANK LENDING SURVEY
# ===========================================================================

def download_bls_data(
    output_dir:   str | Path | None = None,
    start_period: str = "2003-Q1",
    end_period:   str = "2024-Q4",
) -> pd.DataFrame:
    """
    Download ECB Bank Lending Survey data.

    The BLS net tightening index is our primary transmission outcome:
      positive = banks are tightening credit standards
      negative = banks are easing

    ECB SDW series keys for BLS enterprise lending:
      BLS.Q.{country}.ALL.TE.O.Z.XB.Q.NET.PC_NTOT
      where country ∈ {U2, DE, FR, IT, ES, NL, BE, AT, ...}
      U2 = euro area aggregate

    Returns
    -------
    pd.DataFrame: columns = countries, index = quarters
                  values = net tightening percentage
    """
    BLS_COUNTRIES = {
        "U2": "Euro area",
        "DE": "Germany",
        "FR": "France",
        "IT": "Italy",
        "ES": "Spain",
        "NL": "Netherlands",
        "BE": "Belgium",
        "AT": "Austria",
        "PT": "Portugal",
        "FI": "Finland",
        "GR": "Greece",
        "IE": "Ireland",
    }

    BLS_KEY_TEMPLATE = (
        "BLS.Q.{country}.ALL.TE.O.Z.XB.Q.NET.PC_NTOT"
    )

    print("Downloading ECB Bank Lending Survey...")
    series_dict = {}

    for country_code, country_name in BLS_COUNTRIES.items():
        key    = BLS_KEY_TEMPLATE.format(country=country_code)
        series = fetch_ecb_sdw_series(key, start_period, end_period, frequency="Q")

        if series is not None and len(series) > 0:
            series_dict[country_code] = series
            print(f"  ✓ {country_name:<20}: {len(series)} quarters")
        else:
            print(f"  ✗ {country_name:<20}: no data")

    if not series_dict:
        print("\n  ⚠ BLS download failed. Use synthetic data or manual download.")
        print(f"  Manual: ecb.europa.eu/stats/ecb_surveys/bank_lending_survey")
        return pd.DataFrame()

    df = pd.DataFrame(series_dict)
    df.index.name = "quarter"

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "bls_net_tightening.parquet")
        df.to_csv(out / "bls_net_tightening.csv")

    print(f"\n✓ BLS data: {df.shape[1]} countries, {df.shape[0]} quarters")
    return df


def download_ecb_policy_rate(
    start_period: str = "1999-Q1",
    end_period:   str = "2024-Q4",
) -> pd.Series:
    """
    Download ECB main refinancing rate (or Euribor 3M as proxy).

    Returns quarterly average policy rate.
    """
    # Try ECB MRO rate first
    mro_key = "FM.B.U2.EUR.RT.MM.EURIBOR3MD_.HSTA"
    series  = fetch_ecb_sdw_series(mro_key, start_period, end_period, "M")

    if series is not None and len(series) > 0:
        # Resample to quarterly average
        return series.resample("QS").mean().rename("policy_rate_3m")

    # Fallback: synthetic rate series
    print("  ⚠ ECB rate download failed — using historical policy rate series")
    return _get_ecb_historical_rate(start_period, end_period)


def _get_ecb_historical_rate(
    start_period: str,
    end_period:   str,
) -> pd.Series:
    """
    Hardcoded ECB policy rate series based on public historical record.
    Used as fallback when API is unavailable.
    Source: ECB website historical key interest rates.
    """
    # Key rate change dates and levels (MRO rate)
    rate_changes = [
        ("1999-01-01", 3.00), ("1999-04-09", 2.50), ("1999-11-05", 3.00),
        ("2000-02-04", 3.25), ("2000-03-17", 3.50), ("2000-04-28", 3.75),
        ("2000-06-09", 4.25), ("2000-10-06", 4.75),
        ("2001-05-11", 4.50), ("2001-08-31", 4.25), ("2001-09-18", 3.75),
        ("2001-11-09", 3.25), ("2002-12-06", 2.75),
        ("2003-03-07", 2.50), ("2003-06-06", 2.00),
        ("2005-12-06", 2.25), ("2006-03-08", 2.50), ("2006-06-15", 2.75),
        ("2006-08-09", 3.00), ("2006-10-11", 3.25), ("2006-12-13", 3.50),
        ("2007-03-14", 3.75), ("2007-06-13", 4.00),
        ("2008-10-08", 3.75), ("2008-11-06", 3.25), ("2008-12-04", 2.50),
        ("2009-01-15", 2.00), ("2009-03-05", 1.50), ("2009-04-02", 1.25),
        ("2009-05-07", 1.00),
        ("2011-04-07", 1.25), ("2011-07-07", 1.50),
        ("2011-11-03", 1.25), ("2011-12-08", 1.00),
        ("2012-07-05", 0.75),
        ("2013-05-02", 0.50), ("2013-11-07", 0.25),
        ("2014-06-05", 0.15), ("2014-09-04", 0.05),
        ("2016-03-10", 0.00),
        ("2019-09-18", 0.00),  # deposit rate went negative earlier
        ("2022-07-27", 0.50), ("2022-09-14", 1.25), ("2022-11-02", 2.00),
        ("2022-12-15", 2.50), ("2023-02-02", 3.00), ("2023-03-22", 3.50),
        ("2023-05-04", 3.75), ("2023-06-15", 4.00), ("2023-07-27", 4.25),
        ("2023-09-14", 4.50),
        ("2024-06-06", 4.25), ("2024-09-12", 3.65), ("2024-10-17", 3.40),
        ("2024-12-12", 3.15),
    ]

    dates  = pd.date_range(start_period, end_period, freq="QS")
    rates  = pd.Series(index=dates, dtype=float)

    current_rate = 3.0
    for date_str, rate in rate_changes:
        change_date = pd.Timestamp(date_str)
        current_rate = rate
        mask = rates.index >= change_date
        rates[mask] = rate

    # Fill any remaining NaN (before first change date)
    rates = rates.ffill().bfill()
    rates.name = "ecb_mro_rate"

    return rates


# ===========================================================================
# SECTION 3 — CONSTRUCT TRANSMISSION OUTCOME VARIABLE
# ===========================================================================

def construct_transmission_panel(
    bls_df:       pd.DataFrame,
    policy_rate:  pd.Series,
    n_lags:       int = 4,
) -> pd.DataFrame:
    """
    Construct the quarterly panel of monetary transmission outcomes.

    For each country-quarter, the transmission outcome variable is:
      ΔBLS_i,t = change in net tightening index

    When we condition on ECB rate changes:
      ΔBLS_i,t / ΔRATE_t = transmission sensitivity
      How much does country i's credit standard change per basis point
      of ECB rate change?

    We also compute:
      - Lagged rate changes (lag 1-4 quarters) for impulse response
      - Rate change indicator (was this a tightening quarter?)
      - Tightening cycle dummies

    Parameters
    ----------
    bls_df      : BLS net tightening data from download_bls_data()
    policy_rate : ECB policy rate from download_ecb_policy_rate()
    n_lags      : number of lags to compute for impulse response

    Returns
    -------
    Long-format panel: [quarter, country, bls_tightening, delta_bls,
                        ecb_rate, delta_rate, rate_change_indicator,
                        tightening_cycle, lag1_rate_change, ...]
    """
    bls = bls_df.copy()
    bls.index = pd.to_datetime(bls.index)
    bls = bls.resample("QS").mean()

    rate = policy_rate.copy()
    rate.index = pd.to_datetime(rate.index)
    rate = rate.resample("QS").mean()

    # Rate changes
    delta_rate = rate.diff().rename("delta_rate")
    rate_lags  = {f"rate_lag{i}": delta_rate.shift(i)
                  for i in range(1, n_lags + 1)}

    # Cycle indicators
    tightening_cycle = pd.Series(0, index=rate.index)
    tightening_episodes = [
        ("2000-02-01", "2000-10-31"),
        ("2005-12-01", "2007-07-31"),
        ("2011-04-01", "2011-07-31"),
        ("2022-07-01", "2023-09-30"),
    ]
    for start, end in tightening_episodes:
        mask = (rate.index >= pd.Timestamp(start)) & \
               (rate.index <= pd.Timestamp(end))
        tightening_cycle[mask] = 1

    records = []
    for country in bls.columns:
        bls_series  = bls[country]
        delta_bls   = bls_series.diff().rename("delta_bls")
        common_idx  = bls_series.index.intersection(rate.index)

        for q in common_idx:
            row = {
                "quarter":           q,
                "country":           country,
                "bls_tightening":    bls_series.get(q, np.nan),
                "delta_bls":         delta_bls.get(q, np.nan),
                "ecb_rate":          rate.get(q, np.nan),
                "delta_rate":        delta_rate.get(q, np.nan),
                "tightening_cycle":  int(tightening_cycle.get(q, 0)),
            }
            for lag_name, lag_series in rate_lags.items():
                row[lag_name] = lag_series.get(q, np.nan)

            records.append(row)

    panel = pd.DataFrame(records).set_index(["quarter", "country"])
    print(f"Transmission panel: {len(panel):,} observations "
          f"({panel.index.get_level_values('country').nunique()} countries)")
    return panel


# ===========================================================================
# SECTION 4 — SYNTHETIC TRANSMISSION DATA (fallback)
# ===========================================================================


# ===========================================================================
# MANUAL DOWNLOAD GUIDES
# ===========================================================================

BLS_MANUAL_GUIDE = """
HOW TO DOWNLOAD ECB BANK LENDING SURVEY DATA
=============================================
OPTION A — ECB Data Portal (recommended):
  1. Go to: https://www.ecb.europa.eu/stats/ecb_surveys/bank_lending_survey/html/index.en.html
  2. Click "BLS time series data" → Download (Excel or CSV)
  3. Save to: data/raw/bls/bls_data.xlsx

OPTION B — ECB Statistical Data Warehouse:
  1. Go to: https://sdw.ecb.europa.eu/
  2. Search: "BLS net percentage tightening"
  3. Select enterprise loans, all EA countries
  4. Export → CSV → Save to data/raw/bls/

OPTION C — ECB Data Portal API (when network available):
  Base: https://data-api.ecb.europa.eu/service/data/BLS/
  Key: Q.{COUNTRY}.ALL.TE.O.Z.XB.Q.NET.PC_NTOT
  Countries: U2 DE FR IT ES NL BE AT PT FI GR IE

Expected format: time series with quarters as rows, countries as columns.
Values = net percentage reporting tightening (positive = tightening).
"""

ECB_RATE_MANUAL_GUIDE = """
HOW TO GET ECB POLICY RATE
============================
This module includes a hardcoded historical rate series (1999–2024)
based on ECB public records — NO DOWNLOAD NEEDED for the policy rate.

For the most current rates or verification:
  ECB website: https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html
"""


def load_bls_from_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load ECB Bank Lending Survey data from a manually downloaded file.

    WHAT TO DOWNLOAD:
      From https://data.ecb.europa.eu/data/data-categories/ecb-surveys/
           bank-lending-survey-bls/supply/enterprises
      → Download as CSV or Excel
      → Save to data/raw/bls/bls_enterprises.csv (or .xlsx)

    The ECB BLS Supply > Enterprises series gives the net percentage of
    banks reporting TIGHTENING of credit standards for enterprise loans.
    Positive = tightening, negative = easing.
    Available from 2003 Q1 for all major euro area countries.

    The new ECB Data Portal CSV format (2024+) has columns:
      KEY, TITLE, REF_AREA, ... , TIME_PERIOD, OBS_VALUE
    where each row is one country-quarter observation.

    Older format (ECB SDW export):
      Wide format with TIME_PERIOD as rows and country codes as columns.

    Parameters
    ----------
    filepath : path to downloaded BLS file (.xlsx or .csv)

    Returns
    -------
    pd.DataFrame: columns = country ISO2 codes, index = quarter timestamps,
                  values = net tightening percentage (positive = tightening)
    """
    fp = Path(filepath)
    print(f"Loading BLS from {fp.name}...")

    # First: try the dedicated ECB BLS wide format parser
    # (handles "DATE | TIME PERIOD | series col with BLS.Q.XX. in header")
    try:
        result = parse_ecb_bls_wide(fp)
        if len(result) > 0 and len(result.columns) > 0:
            return result
    except Exception as e:
        print(f"  ECB wide parser: {e} — trying generic parser")

    if fp.suffix in [".xlsx", ".xls"]:
        # Try multiple sheets — ECB Excel files vary in structure
        xl = pd.ExcelFile(filepath)
        raw = None
        for sheet in xl.sheet_names:
            try:
                candidate = pd.read_excel(filepath, sheet_name=sheet)
                # Look for sheet with time series data
                if len(candidate) > 10:
                    raw = candidate
                    print(f"  Using sheet: {sheet}")
                    break
            except Exception:
                continue
        if raw is None:
            raw = pd.read_excel(filepath, index_col=0)
    else:
        raw = pd.read_csv(filepath, low_memory=False)

    raw.columns = raw.columns.str.strip()

    # Detect format: new ECB Data Portal (long) vs old SDW (wide)
    is_long_format = (
        "OBS_VALUE" in raw.columns and
        ("TIME_PERIOD" in raw.columns or "TIME PERIOD" in raw.columns)
    )

    if is_long_format:
        # New ECB Data Portal format — long, one row per country-quarter
        time_col = "TIME_PERIOD" if "TIME_PERIOD" in raw.columns else "TIME PERIOD"
        val_col  = "OBS_VALUE"

        # Find country column
        country_col = None
        for candidate in ["REF_AREA", "COUNTRY", "LOCATION", "Reference area"]:
            if candidate in raw.columns:
                country_col = candidate
                break

        if country_col is None:
            # Try to infer from KEY column
            if "KEY" in raw.columns:
                raw["country_parsed"] = raw["KEY"].str.extract(r'\.([A-Z]{2})\.')
                country_col = "country_parsed"
            else:
                raise ValueError(
                    f"Cannot identify country column. "
                    f"Columns found: {list(raw.columns)}"
                )

        df = raw[[country_col, time_col, val_col]].copy()
        df.columns = ["country", "quarter", "value"]
        df["value"]   = pd.to_numeric(df["value"], errors="coerce")

        # Parse quarters
        try:
            df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp()
        except Exception:
            df["quarter"] = pd.to_datetime(df["quarter"], errors="coerce")

        df = df.dropna(subset=["quarter", "value"])

        # Pivot to wide format: countries as columns
        wide = df.pivot_table(
            index="quarter", columns="country", values="value", aggfunc="mean"
        )
        wide.index.name = "quarter"
        wide = wide.sort_index()

        print(f"  Loaded (long format): {wide.shape[1]} countries, "
              f"{wide.shape[0]} quarters")
        print(f"  Countries: {sorted(wide.columns.tolist())}")
        print(f"  Date range: {wide.index.min().date()} → {wide.index.max().date()}")
        return wide

    else:
        # Wide/pivoted format — quarters as rows, countries as columns
        # Try to use first column as index if it looks like dates
        first_col = raw.columns[0]
        if raw[first_col].astype(str).str.match(r'\d{4}').any() or            raw[first_col].astype(str).str.match(r'\d{4}-Q').any():
            raw = raw.set_index(first_col)

        # Parse index as quarters
        try:
            raw.index = pd.PeriodIndex(raw.index.astype(str), freq="Q").to_timestamp()
        except Exception:
            try:
                raw.index = pd.to_datetime(raw.index, errors="coerce")
            except Exception:
                pass

        raw = raw.apply(pd.to_numeric, errors="coerce")
        raw.index.name = "quarter"
        raw = raw.dropna(how="all").sort_index()

        print(f"  Loaded (wide format): {raw.shape[1]} countries, "
              f"{raw.shape[0]} quarters")
        print(f"  Date range: {raw.index.min().date()} → {raw.index.max().date()}")
        return raw


def parse_ecb_bls_wide(filepath) -> pd.DataFrame:
    """
    Parse the specific ECB BLS download format where:
    - Column 1: DATE (dd/mm/yyyy)
    - Column 2: TIME PERIOD (YYYYQn)
    - Columns 3+: series with name format containing BLS.Q.{COUNTRY}.ALL...

    The country code is extracted from the column header series key.
    This handles both:
    - Single-country files (one series column)
    - Multi-country files (multiple series columns)

    Returns wide DataFrame: index=quarter, columns=country codes.
    """
    fp = Path(filepath)
    if fp.suffix in ['.xlsx', '.xls']:
        raw = pd.read_excel(filepath)
    else:
        raw = pd.read_csv(filepath, sep=None, engine='python')

    raw.columns = raw.columns.str.strip()

    # Find the TIME PERIOD column
    time_col = next((c for c in raw.columns
                     if 'TIME' in c.upper() and 'PERIOD' in c.upper()), None)
    if time_col is None:
        # Fall back to DATE column
        time_col = next((c for c in raw.columns if 'DATE' in c.upper()), None)

    if time_col is None:
        raise ValueError(f"Cannot find time column. Columns: {list(raw.columns)}")

    # Find series columns (contain BLS. in their name)
    series_cols = [c for c in raw.columns if 'BLS.' in c or 'BLS.Q' in c]

    if not series_cols:
        # Try: any column that is not DATE or TIME PERIOD
        series_cols = [c for c in raw.columns
                       if c != time_col and 'DATE' not in c.upper()]

    result = {}
    for col in series_cols:
        # Extract country code from series key like BLS.Q.DE.ALL.O.E...
        # The country is the 3rd dot-separated element (index 2)
        parts = col.replace('(', '').replace(')', '').split('.')
        # Find the 2-letter country code
        country = None
        for part in parts:
            part = part.strip()
            if len(part) == 2 and part.isupper() and part.isalpha():
                country = part
                break
            elif len(part) == 2 and part == 'U2':
                country = 'U2'
                break
        if country is None:
            # Use column index as fallback label
            country = f'col{series_cols.index(col)}'

        values = pd.to_numeric(raw[col], errors='coerce')
        result[country] = values.values  # use numpy array to avoid index misalignment

    # Parse time index
    time_series = raw[time_col]
    try:
        idx = pd.PeriodIndex(time_series.astype(str), freq='Q').to_timestamp()
    except Exception:
        try:
            # Try dd/mm/yyyy format
            idx = pd.to_datetime(time_series, dayfirst=True, errors='coerce')
        except Exception:
            idx = pd.to_datetime(time_series, errors='coerce')

    df = pd.DataFrame(result, index=idx)
    df.index.name = 'quarter'
    df = df.dropna(how='all').sort_index()

    print(f"  Parsed ECB BLS: {df.shape[1]} countries, {df.shape[0]} quarters")
    print(f"  Countries: {sorted(df.columns.tolist())}")
    if df.shape[0] > 0:
        print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")
    return df

def generate_synthetic_transmission_data(
    countries:  list[str] | None = None,
    start_year: int = 2000,
    end_year:   int = 2024,
    seed:       int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic BLS and policy rate data for pipeline development.

    Key properties:
    - BLS tightening correlates with rate changes (transmission)
    - Heterogeneous transmission across countries (the finding we test)
    - Countries with structural network properties (encoded in TIVA data)
      show systematically weaker transmission (the hypothesis)
    - Tightening cycles match historical ECB rate change episodes

    Returns
    -------
    (bls_df, policy_rate): BLS net tightening DataFrame + policy rate Series
    """
    rng = np.random.default_rng(seed)

    if countries is None:
        countries = ["U2", "DE", "FR", "IT", "ES",
                     "NL", "BE", "AT", "PT", "FI"]

    quarters = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-31", freq="QS"
    )

    # ECB historical policy rate
    policy_rate = _get_ecb_historical_rate(
        f"{start_year}-01-01", f"{end_year}-12-31"
    )
    delta_rate = policy_rate.diff()

    # Country-specific transmission sensitivities
    # Calibrated: Germany and Netherlands (more centralised networks)
    # show weaker transmission; Italy, Spain, Portugal show stronger
    TRANSMISSION_SENSITIVITY = {
        "U2": 2.0,   # euro area average
        "DE": 1.4,   # weaker — centralised manufacturing network
        "NL": 1.5,   # weaker — hub economy
        "FR": 1.9,
        "BE": 1.8,
        "AT": 1.7,
        "FI": 1.8,
        "IT": 2.4,   # stronger — more peripheral network structure
        "ES": 2.3,   # stronger
        "PT": 2.6,   # strongest — most peripheral
        "GR": 2.5,
        "IE": 2.0,
    }

    bls_dict = {}
    for country in countries:
        sensitivity = TRANSMISSION_SENSITIVITY.get(country, 2.0)
        bls_values  = []

        for q in quarters:
            # Base: BLS responds to rate changes with some lag
            dr = delta_rate.get(q, 0.0)
            bls_val = (
                sensitivity * dr * 5 +           # transmission channel
                rng.normal(0, 5) +                # idiosyncratic noise
                rng.normal(0, 2) * 0.7            # common EA factor
            )
            bls_values.append(bls_val)

        bls_dict[country] = pd.Series(bls_values, index=quarters)

    bls_df = pd.DataFrame(bls_dict)
    bls_df.index.name = "quarter"

    print(f"Synthetic BLS: {len(bls_df)} quarters, {bls_df.shape[1]} countries")
    print(f"Synthetic policy rate: {len(policy_rate)} quarters "
          f"({policy_rate.min():.2f}% – {policy_rate.max():.2f}%)")
    print("NOTE: synthetic data — download real BLS from ecb.europa.eu")

    return bls_df, policy_rate


# ===========================================================================
# SECTION 5 — I/O
# ===========================================================================

def save_transmission_data(
    bls_df:      pd.DataFrame,
    policy_rate: pd.Series,
    output_dir:  str | Path,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bls_df.to_parquet(out / "bls_net_tightening.parquet")
    bls_df.to_csv(out / "bls_net_tightening.csv")
    policy_rate.to_frame().to_parquet(out / "ecb_policy_rate.parquet")
    print(f"Transmission data saved to {out}/")


def load_transmission_data(
    input_dir: str | Path,
) -> tuple[pd.DataFrame, pd.Series]:
    p = Path(input_dir)
    bls = pd.read_parquet(p / "bls_net_tightening.parquet")
    rate_df = pd.read_parquet(p / "ecb_policy_rate.parquet")
    rate = rate_df.iloc[:, 0]
    return bls, rate
