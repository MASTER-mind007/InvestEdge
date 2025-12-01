from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np
import datetime as dt

try:
    import requests  # For HTTP-based data fetching (mfapi)
except ImportError:
    requests = None


PROJECT_ROOT = Path(__file__).resolve().parent
FUNDS_CONFIG_PATH = PROJECT_ROOT / "funds_config.csv"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"

for _p in [DATA_DIR, RAW_DATA_DIR, CLEAN_DATA_DIR]:
    _p.mkdir(parents=True, exist_ok=True)


@dataclass
class FundConfig:
    fund_name: str
    category: str
    code: str
    data_source_type: str
    data_source_url: str
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


def fetch_nav_history(fund: FundConfig) -> pd.DataFrame:
    dst = fund.data_source_type.lower()

    if dst == "csv_local":
        return _load_nav_from_local_csv(fund)
    elif dst == "mfapi":
        return _fetch_nav_from_mfapi(fund)
    elif dst == "amfi":
        raise NotImplementedError("AMFI fetch not implemented yet.")
    elif dst == "custom_api":
        raise NotImplementedError("Custom API fetch not implemented yet.")
    else:
        raise ValueError(f"Unsupported data_source_type '{fund.data_source_type}' for fund {fund.fund_name}")


def _load_nav_from_local_csv(fund: FundConfig) -> pd.DataFrame:
    path = Path(fund.data_source_url)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Local NAV file not found for {fund.fund_name}: {path}")

    df = pd.read_csv(path, sep=None, engine="python")
    df = _standardise_nav_columns(df)
    df["fund_name"] = fund.fund_name
    return df


def _fetch_nav_from_mfapi(fund: FundConfig) -> pd.DataFrame:
    if requests is None:
        raise ImportError(
            "The 'requests' library is required for mfapi. "
            "Install it via 'pip install requests' and re-run."
        )

    scheme_code = fund.code.strip()
    if not scheme_code:
        raise ValueError(f"MFAPI scheme_code (fund.code) is empty for fund {fund.fund_name}")

    url = f"https://api.mfapi.in/mf/{scheme_code}"
    resp = requests.get(url, timeout=10)

    if resp.status_code != 200:
        raise RuntimeError(
            f"MFAPI request failed for {fund.fund_name} (code={scheme_code}): HTTP {resp.status_code}"
        )

    payload = resp.json()
    if "data" not in payload:
        raise ValueError(f"Unexpected MFAPI response format for {fund.fund_name}: 'data' field missing")

    nav_rows = payload["data"]
    df = pd.DataFrame(nav_rows)

    if "date" not in df.columns or "nav" not in df.columns:
        raise ValueError(f"MFAPI data for {fund.fund_name} does not contain 'date' and 'nav' columns")

    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    df["fund_name"] = fund.fund_name
    return df


def _standardise_nav_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}

    date_col = None
    for candidate in ["date", "navdate", "nav_date"]:
        if candidate in cols_lower:
            date_col = cols_lower[candidate]
            break
    if date_col is None:
        raise ValueError("Could not identify a date column in NAV data.")

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


def clean_nav_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["nav_prev"] = df["nav"].shift(1)
    df["daily_return"] = df["nav"] / df["nav_prev"] - 1
    df.loc[df["nav_prev"].isna(), "daily_return"] = np.nan
    return df


def check_missing_dates(df: pd.DataFrame) -> Dict[str, int]:
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
    if "daily_return" not in df.columns:
        raise ValueError("DataFrame must have 'daily_return' column. Run clean_nav_history first.")

    mask = df["daily_return"].abs() > threshold
    return df.loc[mask].copy()


def save_nav_data(df_raw: pd.DataFrame, df_clean: pd.DataFrame, fund: FundConfig) -> Tuple[Path, Path]:
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in fund.fund_name.lower()).strip("_")
    raw_path = RAW_DATA_DIR / f"{safe_name}_raw.csv"
    clean_path = CLEAN_DATA_DIR / f"{safe_name}_clean.csv"

    df_raw.to_csv(raw_path, index=False)
    df_clean.to_csv(clean_path, index=False)

    return raw_path, clean_path


def run_phase1_data_backbone(config_path: Path = FUNDS_CONFIG_PATH) -> None:
    print("[Phase 1] Loading fund configurations...")
    funds = load_fund_configs(config_path)
    print(f"[Phase 1] Found {len(funds)} funds in config.")

    for fund in funds:
        print(f"\n[Phase 1] Processing fund: {fund.fund_name} ({fund.category})")
        try:
            df_raw = fetch_nav_history(fund)
            df_raw = _standardise_nav_columns(df_raw)
            df_clean = clean_nav_history(df_raw)

            missing_stats = check_missing_dates(df_clean)
            outliers = detect_return_outliers(df_clean, threshold=0.2)

            print(f"  - Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
            print(f"  - Observed days: {missing_stats['observed_days']} / {missing_stats['total_days']} total calendar days")
            if not outliers.empty:
                print(f"  - WARNING: {len(outliers)} potential return outliers (> 20% in a day). Inspect manually.")

            raw_path, clean_path = save_nav_data(df_raw, df_clean, fund)
            print(f"  - Saved raw data to:   {raw_path}")
            print(f"  - Saved clean data to: {clean_path}")

        except Exception as e:
            print(f"  - ERROR processing fund {fund.fund_name}: {e}")


if __name__ == "__main__":
    run_phase1_data_backbone()
