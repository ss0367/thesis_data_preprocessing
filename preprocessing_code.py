import pandas as pd
import numpy as np

# Configuration
INPUT_FILE = "Preqin_deals_export-08_Dec_25cad878d2-38e1-4f10-93ae-077257ead3d1.xlsx"
SNAPSHOT_DATE = pd.Timestamp("2025-12-08")
OUTPUT_FILE_LONG = "preqin_company_sequences_long.csv"


def tokenize_stage(raw_stage: str) -> str:
    """Map raw Preqin stage labels to observation tokens."""
    if pd.isna(raw_stage):
        return "OTHER"

    s = str(raw_stage).strip().lower()

    # Exit-like labels
    if "merger" in s:
        return "EXITLIKE"
    if "pre-ipo" in s or "pre ipo" in s:
        return "EXITLIKE"
    if "pipe" in s:
        return "EXITLIKE"

    # Canonical stage labels
    if "series a" in s:
        return "Series A"
    if "series b" in s:
        return "Series B"
    if "series c" in s:
        return "Series C"
    if "series d" in s:
        return "Series D"

    # Late-stage labels
    if ("series e" in s or "series f" in s or "series g" in s or
        "series h" in s or "series i" in s):
        return "Series E"

    if "seed" in s:
        return "Seed"

    # Other financing labels
    if "unspecified" in s:
        return "UNSPEC"
    if "venture debt" in s:
        return "DEBT"
    if "grant" in s:
        return "GRANT"
    if "angel" in s:
        return "ANGEL"
    if "add-on" in s or "add on" in s:
        return "ADDON"
    if "secondary stock purchase" in s:
        return "SECONDARY"

    # Growth/expansion treated as late stage
    if "growth capital" in s or "expansion" in s:
        return "Series E"

    return "OTHER"


def preprocess_preqin_to_long(
    input_file: str = INPUT_FILE,
    output_file: str = OUTPUT_FILE_LONG,
) -> pd.DataFrame:
    """Deduplicate, sort, tokenize, and export a long-format company timeline."""
    cols_keep = [
        "DEAL ID",
        "DEAL DATE",
        "STAGE",
        "PORTFOLIO COMPANY",
        "PORTFOLIO COMPANY ID",
        "DEAL SIZE (USD MN)",
        "COMPANY REVENUE (CURR. MN)",
    ]

    # Load data
    raw = pd.read_excel(input_file)

    # Check required columns
    missing = [c for c in cols_keep if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in input file: {missing}")

    df = raw[cols_keep].copy()

    # Keep one row per deal
    df = df.drop_duplicates(subset=["DEAL ID"]).copy()

    # Parse dates
    df["DEAL DATE"] = pd.to_datetime(df["DEAL DATE"], errors="coerce")

    # Sort deals within each company
    df = df.sort_values(
        by=["PORTFOLIO COMPANY ID", "DEAL DATE", "DEAL ID"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    # Event index within company
    df["t"] = df.groupby("PORTFOLIO COMPANY ID").cumcount() + 1

    # Flag final observed event
    last_t = df.groupby("PORTFOLIO COMPANY ID")["t"].transform("max")
    df["is_last_event"] = (df["t"] == last_t).astype(int)

    # Days from final event to snapshot date
    df["delta_days"] = np.nan
    mask_last = df["is_last_event"].eq(1) & df["DEAL DATE"].notna()
    df.loc[mask_last, "delta_days"] = (
        SNAPSHOT_DATE - df.loc[mask_last, "DEAL DATE"]
    ).dt.days

    # Clip negative gaps
    df["delta_days"] = df["delta_days"].clip(lower=0)

    # Observation tokens
    df["raw_stage"] = df["STAGE"]
    df["obs_token"] = df["raw_stage"].apply(tokenize_stage)

    # Deal size features
    df["deal_size_usd_mn"] = pd.to_numeric(df["DEAL SIZE (USD MN)"], errors="coerce")
    df["log_raise"] = np.log1p(df["deal_size_usd_mn"])

    # Revenue
    df["company_revenue_curr_mn"] = pd.to_numeric(
        df["COMPANY REVENUE (CURR. MN)"], errors="coerce"
    )

    # Rename columns
    df = df.rename(
        columns={
            "DEAL ID": "deal_id",
            "DEAL DATE": "deal_date",
            "PORTFOLIO COMPANY": "company_name",
            "PORTFOLIO COMPANY ID": "company_id",
        }
    )

    # Final output columns
    out_cols = [
        "company_id",
        "company_name",
        "t",
        "deal_id",
        "deal_date",
        "raw_stage",
        "obs_token",
        "deal_size_usd_mn",
        "log_raise",
        "company_revenue_curr_mn",
        "delta_days",
        "is_last_event",
    ]
    out = df[out_cols].copy()

    # Export CSV
    out.to_csv(output_file, index=False)

    return out


if __name__ == "__main__":
    processed = preprocess_preqin_to_long(INPUT_FILE, OUTPUT_FILE_LONG)
    print(f"Wrote LONG-format dataset to: {OUTPUT_FILE_LONG}")
    print(f"Rows (deals): {len(processed):,}")
    print("obs_token value counts (top 15):")
    print(processed["obs_token"].value_counts().head(15))
