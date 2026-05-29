import pandas as pd

df = pd.read_parquet("filtered_articles_1920_1940.parquet")
print(f"Original dataset: {len(df):,} rows")

# ── Dataset 1: Stratified 10% sample by year ──────────────────────────────────
sampled = (
    df.groupby("year", group_keys=False)
    .apply(lambda g: g.sample(frac=0.1, random_state=42))
    .reset_index(drop=True)
)
print(f"\nDataset 1 (stratified_10pct_by_year): {len(sampled):,} rows")
print("Year breakdown:")
print(sampled["year"].value_counts().sort_index().to_string())

out1 = "stratified_10pct_by_year.parquet"
sampled.to_parquet(out1, index=False)
print(f"Saved → {out1}")


# ── Dataset 2: Northeast vs South regional filter ────────────────────────────
#
# Northeast: New England + core Mid-Atlantic (definitive Union states)
#   - Excludes DC: federal capital, not a state; its newspapers cover federal
#     government from an inside-the-beltway perspective, not a regional one.
#   - Excludes Delaware & Maryland: border slave states with divided loyalties;
#     keeping the comparison clean means sticking to unambiguous Union territory.
#
# South: the 11 former Confederate states (those present in the dataset)
#   - Excludes West Virginia: formed by breaking away from Virginia *to stay
#     in the Union*, so it is constitutionally and politically anti-Confederate.
#   - Excludes Kentucky: never formally seceded; claimed neutrality and was
#     governed by a Union-loyal state government throughout the war.
#
# Rationale: the hypothesis is about deep-rooted attitudes toward federal power
# (states-rights ideology vs. Hamiltonian federalism). Using only the clearest
# historical proxies for each ideology avoids muddying the regional signal.

NORTHEAST = {
    "Connecticut",   # strong industrial / Federalist tradition
    "New York",      # Hamiltonian heartland, finance capital
    "Maine",         # solidly Republican / pro-Union New England
    "New Jersey",    # core Mid-Atlantic Union state
    "Pennsylvania",  # industrial Union state, pro-federal tariffs
    "Massachusetts", # abolitionist / Federalist stronghold
    "Vermont",       # most consistently Republican New England state
}

SOUTH = {
    "Alabama",        # deep Confederate core
    "Arkansas",       # Confederate state, strong states-rights tradition
    "North Carolina", # Confederate state
    "Georgia",        # Confederate core (Sherman's march territory)
    "Florida",        # Confederate state
    "Mississippi",    # Confederate core, strongest states-rights rhetoric
    "Louisiana",      # Confederate state
    "South Carolina", # first state to secede
    # Virginia, Tennessee, Texas not present in this dataset
}

df_regional = df[df["state"].isin(NORTHEAST | SOUTH)].copy()
df_regional["region"] = df_regional["state"].apply(
    lambda s: "Northeast" if s in NORTHEAST else "South"
)

print(f"\nDataset 2 (northeast_vs_south_regions): {len(df_regional):,} rows")
print("\nState breakdown:")
print(df_regional.groupby(["region", "state"]).size().to_string())
print("\nRegion totals:")
print(df_regional["region"].value_counts().to_string())
print("\nRegion × Year counts:")
print(df_regional.groupby(["region", "year"]).size().unstack(fill_value=0).to_string())

out2 = "northeast_vs_south_regions.parquet"
df_regional.to_parquet(out2, index=False)
print(f"\nSaved → {out2}")
