import pandas as pd
import numpy as np

# 1) Tokenization (raw label -> observation token)

def tokenize_stage(raw_stage: str) -> str:
   
    if pd.isna(raw_stage):
        return "OTHER"

    s = str(raw_stage).strip().lower()

    # Exit-like observations
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

    # Catch-all late stage token: Series E includes E/F/G/H/I + Growth/Expansion
    if ("series e" in s or "series f" in s or "series g" in s or
        "series h" in s or "series i" in s):
        return "Series E"

    # Seed
    if "seed" in s:
        return "Seed"

    # Financing mechanism / other labels (kept as observations, not states)
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

    # Growth/Expansion treated as late-stage observation
    if "growth capital" in s or "expansion" in s:
        return "Series E"

    return "OTHER"


# 2) Preprocessing pipeline

def preprocess_preqin_to_long(
    input_file: str = INPUT_FILE,
    output_file: str = OUTPUT_FILE_LONG,
) -> pd.DataFrame:

    # Columns used in your original notebook (same intent)
    cols_keep = [
        "DEAL ID",
        "DEAL DATE",
        "STAGE",
        "PORTFOLIO COMPANY",
        "PORTFOLIO COMPANY ID",
        "DEAL SIZE (USD MN)",
        "COMPANY REVENUE (CURR. MN)",
    ]

    # Load raw data
    raw = pd.read_excel(input_file)

    # Keep needed columns (fail fast if missing)
    missing = [c for c in cols_keep if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in input file: {missing}")

    df = raw[cols_keep].copy()

    # Deduplicate investor-level repetition: one row per DEAL ID
    df = df.drop_duplicates(subset=["DEAL ID"]).copy()

    # Parse dates
    df["DEAL DATE"] = pd.to_datetime(df["DEAL DATE"], errors="coerce")

    # Sort within company timeline
    df = df.sort_values(
        by=["PORTFOLIO COMPANY ID", "DEAL DATE", "DEAL ID"],
        ascending=[True, True, True],
        kind="mergesort",  # stable sort
    )

    # Create event index t (deal number within company)
    df["t"] = df.groupby("PORTFOLIO COMPANY ID").cumcount() + 1

    # Compute time gaps (days since previous deal in that company)
    df["delta_days"] = (
        df.groupby("PORTFOLIO COMPANY ID")["DEAL DATE"]
          .diff()
          .dt.days
    )

    # Tokenize raw labels into observation tokens
    df["raw_stage"] = df["STAGE"]
    df["obs_token"] = df["raw_stage"].apply(tokenize_stage)

    # Deal size features for later HMM emissions
    df["deal_size_usd_mn"] = pd.to_numeric(df["DEAL SIZE (USD MN)"], errors="coerce")
    df["log_raise"] = np.log1p(df["deal_size_usd_mn"])

    # Revenue (keep, numeric)
    df["company_revenue_curr_mn"] = pd.to_numeric(
        df["COMPANY REVENUE (CURR. MN)"], errors="coerce"
    )

    # Rename for clarity / consistency
    df = df.rename(
        columns={
            "DEAL ID": "deal_id",
            "DEAL DATE": "deal_date",
            "PORTFOLIO COMPANY": "company_name",
            "PORTFOLIO COMPANY ID": "company_id",
        }
    )

    # Final long-format schema (one row per deal)
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
    ]
    out = df[out_cols].copy()

    # Export
    out.to_csv(output_file, index=False)

    return out


if __name__ == "__main__":
    processed = preprocess_preqin_to_long(INPUT_FILE, OUTPUT_FILE_LONG)
    print(f"Wrote LONG-format dataset to: {OUTPUT_FILE_LONG}")
    print(f"Rows (deals): {len(processed):,}")
    print("obs_token value counts (top 15):")
    print(processed["obs_token"].value_counts().head(15))
