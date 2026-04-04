"""
data.loader
-----------
Responsible for loading the raw Excel file and producing a clean
transaction-level DataFrame ready for feature engineering.

All business rules for what constitutes a valid transaction are
contained here — nothing upstream should re-implement them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_REQUIRED_COLS = {
    "Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID",
}


def load_raw(path: str | Path, sheets: list[str] | None = None) -> pd.DataFrame:
    """Load the raw dataset from either a Parquet or Excel file.

    Parquet is the preferred format (~20-50× faster than Excel).
    Run :func:`excel_to_parquet` once to convert, then update
    ``config.yaml → data.path`` to point at the ``.parquet`` file.

    Parameters
    ----------
    path:
        Path to a ``.parquet`` or ``.xlsx`` file.
    sheets:
        Sheet names to load (Excel only). Ignored for Parquet.

    Returns
    -------
    Raw concatenated DataFrame with original dtypes.
    """
    path = Path(path)

    if path.suffix == ".parquet":
        if not path.exists():
            # Parquet not found — auto-convert from the xlsx counterpart
            xlsx = path.with_suffix(".xlsx")
            if not xlsx.exists():
                raise FileNotFoundError(
                    f"Neither {path} nor {xlsx} found. "
                    "Place the source .xlsx in the data/ directory."
                )
            print(f"Parquet not found — converting {xlsx.name} → {path.name} (one-time, ~2 min)...")
            excel_to_parquet(xlsx, sheets=sheets, out_path=path)
        df = pd.read_parquet(path)
    else:
        xl = pd.ExcelFile(path)
        if sheets is None:
            sheets = xl.sheet_names
        frames = [pd.read_excel(xl, sheet_name=s) for s in sheets]
        df = pd.concat(frames, ignore_index=True)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")

    return df


def excel_to_parquet(
    xlsx_path: str | Path,
    sheets: list[str] | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """One-time conversion of the Excel source to Parquet.

    Run once from the repo root::

        python -c "
        from anomaly_detection.data.loader import excel_to_parquet
        excel_to_parquet('data/online_retail_II.xlsx')
        "

    Then update ``config.yaml``::

        data:
          path: data/online_retail_II.parquet

    Parameters
    ----------
    xlsx_path:
        Path to the source ``.xlsx`` file.
    sheets:
        Sheet names to include. Defaults to all sheets.
    out_path:
        Output ``.parquet`` path. Defaults to same directory as xlsx,
        same stem, ``.parquet`` extension.

    Returns
    -------
    Path to the written Parquet file.
    """
    xlsx_path = Path(xlsx_path)
    out_path  = Path(out_path) if out_path else xlsx_path.with_suffix(".parquet")

    xl = pd.ExcelFile(xlsx_path)
    if sheets is None:
        sheets = xl.sheet_names

    frames = [pd.read_excel(xl, sheet_name=s) for s in sheets]
    df = pd.concat(frames, ignore_index=True)

    # Parquet requires uniform column types — cast mixed object cols to str
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype(str)

    df.to_parquet(out_path, index=False)
    print(f"Converted → {out_path}  ({len(df):,} rows)")
    return out_path


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules and return a transaction-level DataFrame.

    Rules applied (in order):
    1. Drop rows with no Customer ID — cannot build a behavioral profile.
    2. Remove cancellations: invoices prefixed with 'C'.
    3. Remove returns / adjustments: Quantity <= 0 or Price <= 0.
    4. Drop exact duplicate rows.
    5. Cast Customer ID to string, parse InvoiceDate, add Revenue column.

    Parameters
    ----------
    df:
        Raw DataFrame from :func:`load_raw`.

    Returns
    -------
    Cleaned DataFrame with an additional ``Revenue`` column.
    """
    df = df.copy()

    df = df.dropna(subset=["Customer ID"])
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df.drop_duplicates()

    df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]

    return df.reset_index(drop=True)
