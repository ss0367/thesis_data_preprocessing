import pandas as pd

# 1. Stage mapping helpers ----------------------------------------------------

def classify_raw_stage(raw_stage: str):
   
    if pd.isna(raw_stage):
        return None

    s = str(raw_stage).strip().lower()

    # ----- Exit events -----
    if "merger" in s:
        return "Exit"
    if "pre-ipo" in s or "pre ipo" in s:
        return "Exit"
    if "pipe" in s:  # PIPE rounds
        return "Exit"

    # ----- Canonical non-exit funding stages -----
    # Series letters (A–I). F–I all collapse to Series E.
    if "series a" in s:
        return "Series A"
    if "series b" in s:
        return "Series B"
    if "series c" in s:
        return "Series C"
    if "series d" in s:
        return "Series D"
    if "series e" in s or "series f" in s or "series g" in s \
       or "series h" in s or "series i" in s:
        return "Series E"

    # Seed (as long as it isn't part of 'series ...')
    if "seed" in s:
        return "Seed"

    # Secondary Stock Purchase, Add-on, Growth Capital/Expansion -> Series E
    if "secondary stock purchase" in s:
        return "Series E"
    if "add-on" in s or "add on" in s:
        return "Series E"
    if "growth capital" in s or "expansion" in s:
        # Growth Capital/Expansion → treat as late stage
        return "Series E"

    # Everything else -> None (no direct mapping; use sequential logic)
    return None


def assign_canonical_stages_and_exit(group: pd.DataFrame) -> pd.DataFrame:
    
    stages_order = ["Seed", "Series A", "Series B", "Series C", "Series D", "Series E"]
    stage_to_idx = {s: i for i, s in enumerate(stages_order)}
    max_idx = len(stages_order) - 1

    assigned_stages = []
    last_idx = None
    exited = False

    for _, row in group.iterrows():
        if exited:
            # We ignore any deals after the first exit event
            continue

        raw_stage = row["STAGE"]
        mapped = classify_raw_stage(raw_stage)

        if mapped == "Exit":
            assigned_stages.append("Exit")
            exited = True
            continue

        # Non-exit deal
        if last_idx is None:
            # First non-exit deal
            if mapped is not None:
                idx = stage_to_idx[mapped]
            else:
                idx = 0  # Seed
        else:
            # Base forward step from previous non-exit state
            candidate_idx = min(last_idx + 1, max_idx)

            if mapped is not None:
                raw_idx = stage_to_idx[mapped]
                if raw_idx >= last_idx:
                    # Accept raw mapped stage (can jump forward)
                    idx = min(raw_idx, max_idx)
                else:
                    # Raw suggests going backward -> ignore, just step forward
                    idx = candidate_idx
            else:
                # No informative mapping -> follow sequential progression
                idx = candidate_idx

        assigned_stages.append(stages_order[idx])
        last_idx = idx

    # Build output group (only rows up to first Exit if any)
    n = len(assigned_stages)
    truncated = group.iloc[:n].copy()
    truncated["CANONICAL_STAGE"] = assigned_stages
    return truncated


# 2. Main preprocessing pipeline ---------------------------------------------

def main():
    # 2.1 Load raw data
    df_raw = pd.read_excel(INPUT_FILE)

    # 2.2 Keep only relevant columns
    cols_to_keep = [
        "DEAL ID",
        "DEAL DATE",
        "STAGE",
        "PORTFOLIO COMPANY",
        "PORTFOLIO COMPANY ID",
        "DEAL SIZE (USD MN)",
        "COMPANY REVENUE (CURR. MN)",
    ]
    df = df_raw[cols_to_keep].copy()

    # 2.3 Deduplicate: one row per deal ID (since raw file repeats per investor)
    df = df.drop_duplicates(subset=["DEAL ID"])

    # 2.4 Parse dates and sort by company + date + deal ID
    df["DEAL DATE"] = pd.to_datetime(df["DEAL DATE"], errors="coerce")
    df = df.sort_values(["PORTFOLIO COMPANY ID", "DEAL DATE", "DEAL ID"])

    # 2.5 Assign canonical stages and handle Exit/truncation per company
    df_assigned = (
        df.groupby("PORTFOLIO COMPANY ID", group_keys=False)
          .apply(assign_canonical_stages_and_exit)
    )

    # 2.6 Assign deal_number = 1,2,3,... in chronological order (post-truncation)
    df_assigned["deal_number"] = (
        df_assigned.groupby("PORTFOLIO COMPANY ID").cumcount() + 1
    )

    # 2.7 Set multi-index for pivoting
    df_idx = df_assigned.set_index(
        ["PORTFOLIO COMPANY", "PORTFOLIO COMPANY ID", "deal_number"]
    )

    # 2.8 Pivot to wide format: stage_k, deal_date_k, company_revenue_k, deal_size_usd_k
    var_map = {
        "CANONICAL_STAGE": "stage",
        "DEAL DATE": "deal_date",
        "COMPANY REVENUE (CURR. MN)": "company_revenue",
        "DEAL SIZE (USD MN)": "deal_size_usd",
    }

    wide_parts = []
    for col, prefix in var_map.items():
        tmp = df_idx[col].unstack("deal_number")  # columns are 1,2,3,...
        tmp.columns = [f"{prefix}_{int(c)}" for c in tmp.columns]
        wide_parts.append(tmp)

    df_wide = pd.concat(wide_parts, axis=1).reset_index()

    # 2.9 Reorder columns: company, company_id, then stage_1, date_1, revenue_1, size_1, ...
    base_cols = ["PORTFOLIO COMPANY", "PORTFOLIO COMPANY ID"]
    max_deals = df_assigned["deal_number"].max()

    ordered_deal_cols = []
    for k in range(1, max_deals + 1):
        for prefix in ["stage", "deal_date", "company_revenue", "deal_size_usd"]:
            col_name = f"{prefix}_{k}"
            if col_name in df_wide.columns:
                ordered_deal_cols.append(col_name)

    final_cols = base_cols + ordered_deal_cols
    df_final = df_wide[final_cols]

    # 2.10 Export to Excel
    df_final.to_excel(OUTPUT_FILE, index=False)
    print(f"Done. Markov-ready company sequence table written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
