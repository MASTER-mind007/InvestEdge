from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np
import datetime as dt

try:
    import requests  # For future HTTP-based data fetching
except ImportError:  # Optional
    requests = None

# Paths and basic setup

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"

for _p in [CONFIG_DIR, RAW_DATA_DIR, CLEAN_DATA_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

FUNDS_CONFIG_PATH = CONFIG_DIR / "funds_config.csv"

# Data structures

@dataclass
class FundConfig:
    fund_name: str
    category: str
    code: str  # could be AMFI code, ISIN, or internal code
    data_source_type: str  # e.g., "amfi", "csv_local", "custom_api"
    data_source_url: str   # local path or remote URL as applicable
    benchmark: Optional[str] = None
    inception_date: Optional[dt.date] = None

    @staticmethod
    def from_series(s: pd.Series) -> "FundConfig":
       
        inception = None
        if pd.notna(s.get("inception_date", None)):
            try:
                inception = pd.to_datetime(s["inception_date"]).date()
            except Exception:
                inception = None

        return FundConfig(
            fund_name=str(s.get("fund_name", "")).strip(),
            category=str(s.get("category", "")).strip(),
            code=str(s.get("code", "")).strip(),
            data_source_type=str(s.get("data_source_type", "")).strip(),
            data_source_url=str(s.get("data_source_url", "")).strip(),
            benchmark=str(s.get("benchmark", "")).strip() or None,
            inception_date=inception,
        )


# Config loader

def load_fund_configs(config_path: Path = FUNDS_CONFIG_PATH) -> List[FundConfig]:
  
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}. Please create funds_config.csv.")

    df = pd.read_csv(config_path)
    required_cols = {"fund_name", "category", "code", "data_source_type", "data_source_url"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Config file is missing required columns: {missing}")

    configs: List[FundConfig] = [FundConfig.from_series(row) for _, row in df.iterrows()]
    return configs


# NAV fetching utilities (placeholders where remote data is required)

def fetch_nav_history(fund: FundConfig) -> pd.DataFrame:
  
    dst = fund.data_source_type.lower()

    if dst == "csv_local":
        return _load_nav_from_local_csv(fund)
    elif dst == "amfi":
        return _fetch_nav_from_amfi(fund)
    elif dst == "custom_api":
        return _fetch_nav_from_custom_api(fund)
    else:
        raise ValueError(f"Unsupported data_source_type '{fund.data_source_type}' for fund {fund.fund_name}")


def _load_nav_from_local_csv(fund: FundConfig) -> pd.DataFrame:
    """Load NAV history from a local CSV.
       """
    path = Path(fund.data_source_url)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Local NAV file not found for {fund.fund_name}: {path}")

    def _read_with_semicolon(p: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(p, sep=";", engine="python", on_bad_lines="skip")
        except TypeError:
            # Fallback for older pandas versions without on_bad_lines
            return pd.read_csv(p, sep=";", engine="python", error_bad_lines=False, warn_bad_lines=False)

    df = pd.read_csv(path)
    needs_semicolon = (
        path.suffix.lower() in {".txt", ".tsv"}
        or (df.shape[1] == 1 and ";" in df.columns[0])
    )
    if needs_semicolon:
        df = _read_with_semicolon(path)

    df = _standardise_nav_columns(df)
    df["fund_name"] = fund.fund_name
    return df


def _fetch_nav_from_amfi(fund: FundConfig) -> pd.DataFrame:
    """Placeholder for AMFI-based NAV fetch.
        """
    raise NotImplementedError(
        f"AMFI fetch not implemented for {fund.fund_name}. "
        "Decide on the exact AMFI data source and update _fetch_nav_from_amfi."
    )


def _fetch_nav_from_custom_api(fund: FundConfig) -> pd.DataFrame:
    """Placeholder for any other free, legal API.
    """
    raise NotImplementedError(
        f"Custom API fetch not implemented for {fund.fund_name}. "
        "Update _fetch_nav_from_custom_api with the chosen API logic."
    )


# Standardisation & cleaning


def _standardise_nav_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise raw NAV DataFrame to have columns: ['date', 'nav'].

    This function tries a few common variants and normalises them.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Identify date column
    date_col = None
    for candidate in ["date", "navdate", "nav_date"]:
        if candidate in cols_lower:
            date_col = cols_lower[candidate]
            break
    if date_col is None:
        raise ValueError("Could not identify a date column in NAV data.")

    # Identify NAV column
    nav_col = None
    for candidate in ["nav", "nav_rs", "nav_value", "net asset value"]:
        if candidate in cols_lower:
            nav_col = cols_lower[candidate]
            break
    if nav_col is None:
        raise ValueError("Could not identify a NAV column in NAV data.")

    out = df[[date_col, nav_col]].copy()
    out.columns = ["date", "nav"]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out = out.dropna(subset=["nav"])

    out = out.sort_values("date").reset_index(drop=True)
    return out


def clean_nav_history(df: pd.DataFrame, freq: str = "B") -> pd.DataFrame:
    """Clean a NAV history DataFrame.
     """
    df = df.copy()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Optional: reindex to business days to detect missing dates (NAVs may not exist for all days)
    # Here we just keep the original dates and compute daily returns on available points.
    df["nav_prev"] = df["nav"].shift(1)
    df["daily_return"] = df["nav"] / df["nav_prev"] - 1
    df.loc[df["nav_prev"].isna(), "daily_return"] = np.nan

    return df


# Data quality checks

def check_missing_dates(df: pd.DataFrame) -> Dict[str, int]:
    """Check for missing calendar days in NAV history.
     """
    if df.empty:
        return {"total_days": 0, "observed_days": 0, "missing_days": 0}

    df = df.sort_values("date")
    start, end = df["date"].min(), df["date"].max()
    full_range = pd.date_range(start, end, freq="D")
    observed = df["date"].dt.normalize().unique()

    total_days = len(full_range)
    observed_days = len(observed)
    missing_days = total_days - observed_days

    return {
        "total_days": int(total_days),
        "observed_days": int(observed_days),
        "missing_days": int(missing_days),
    }


def detect_return_outliers(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """Flag days where absolute daily return exceeds `threshold`.
       """
    if "daily_return" not in df.columns:
        raise ValueError("DataFrame must have 'daily_return' column. Run clean_nav_history first.")

    mask = df["daily_return"].abs() > threshold
    return df.loc[mask].copy()


# Save utilities


def save_nav_data(df_raw: pd.DataFrame, df_clean: pd.DataFrame, fund: FundConfig) -> Tuple[Path, Path]:
    """Save raw and cleaned NAV data to disk and return their paths.
       """
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in fund.fund_name.lower()).strip("_")
    raw_path = RAW_DATA_DIR / f"{safe_name}_raw.csv"
    clean_path = CLEAN_DATA_DIR / f"{safe_name}_clean.csv"

    df_raw.to_csv(raw_path, index=False)
    df_clean.to_csv(clean_path, index=False)

    return raw_path, clean_path



# Orchestrator: run Phase 1 pipeline


def run_phase1_data_backbone(config_path: Path = FUNDS_CONFIG_PATH) -> None:
    """End-to-end Phase 1 runner.

       """
    print("[Phase 1] Loading fund configurations...")
    funds = load_fund_configs(config_path)
    print(f"[Phase 1] Found {len(funds)} funds in config.")

    for fund in funds:
        print(f"\n[Phase 1] Processing fund: {fund.fund_name} ({fund.category})")
        try:
            df_raw = fetch_nav_history(fund)
            df_raw = _standardise_nav_columns(df_raw)
            df_clean = clean_nav_history(df_raw)

            # Basic quality checks
            missing_stats = check_missing_dates(df_clean)
            outliers = detect_return_outliers(df_clean, threshold=0.2)

            print(f"  - Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
            print(f"  - Observed days: {missing_stats['observed_days']} / {missing_stats['total_days']} total calendar days")
            if not outliers.empty:
                print(f"  - WARNING: {len(outliers)} potential return outliers (> 20% in a day). Inspect manually.")

            raw_path, clean_path = save_nav_data(df_raw, df_clean, fund)
            print(f"  - Saved raw data to:   {raw_path}")
            print(f"  - Saved clean data to: {clean_path}")

        except NotImplementedError as nie:
            print(f"  - SKIPPED (not implemented): {nie}")
        except FileNotFoundError as fnf:
            print(f"  - SKIPPED (file missing): {fnf}")
        except Exception as e:
            print(f"  - ERROR processing fund {fund.fund_name}: {e}")


if __name__ == "__main__":
    # Running this file directly will execute Phase 1 end-to-end, assuming
    # `config/funds_config.csv` has been created and populated.
    run_phase1_data_backbone()
