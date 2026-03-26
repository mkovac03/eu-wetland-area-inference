#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Country-level calibrated hierarchical wetland area estimation.

This script estimates country-level wetland class areas from a continental
stratified reference sample. It combines:
    1) stratified base weights from known stratum areas,
    2) calibration by iterative proportional fitting (raking) to known
       country and stratum totals,
    3) weighted country reference compositions,
    4) empirical-Bayes partial pooling within macro-regions, and
    5) posterior area summaries from Dirichlet draws.

Core steps
----------
Base weights from the continental stratified sample:

    w_h^(0) = A_h / n_h

where:
    A_h = known total area of stratum h
    n_h = number of validation samples in stratum h

Calibration by raking enforces:

    sum_{s in country d} w_s = A_d
    sum_{s in stratum h} w_s = A_h

where:
    A_d = known frame area of country d

Within each country, calibrated weights are normalized to preserve an
effective sample size:

    n_eff,d = (sum_s w_ds)^2 / sum_s w_ds^2

    w_ds^norm = w_ds^cal / sum_s w_ds^cal * n_eff,d

Weighted country reference pseudo-counts are then:

    n_tilde_{d,i} = sum_{s in d} w_ds^norm I(y_s = i)

For country d in region r, the pooled class composition is modeled as:

    pi_d | data ~ Dirichlet(kappa * mu_r + n_tilde_d)

where:
    mu_r  = macro-regional mean class composition
    kappa = global shrinkage parameter estimated by empirical Bayes

Country-level class areas are obtained by:

    A_{d,i} = A_d * pi_{d,i}

References
----------
Särndal, Swensson, and Wretman (1992), Model Assisted Survey Sampling,
Chapter 10, for domain estimation in design-based survey inference.

Deville and Särndal (1992), Journal of the American Statistical Association,
for calibration estimators.

Deville, Särndal, and Sautory (1993), Journal of the American Statistical
Association, for generalized raking / iterative proportional fitting.

Gelman et al. (2014), Bayesian Data Analysis, Chapter 5, for hierarchical
pooling and shrinkage.

Notes
-----
- Weighted country class vectors are treated as pseudo-counts.
- Posterior intervals reported here are model-based credible intervals.
- Region definitions are specified in REGION_COUNTRIES below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.special import gammaln


# -----------------------------
# Region mapping (same as source)
# -----------------------------
REGION_COUNTRIES = {
    "Boreal": ["DK", "NO", "SE", "FI", "IS", "EE", "LV", "LT"],
    "Western": ["FR", "BE", "NL", "IE", "UK", "LU", "AD"],
    "Central": ["DE", "AT", "CH", "CZ", "PL", "SK", "HU", "LI", "SI", "RO"],
    "SW_Europe": ["ES", "PT", "IT"],
    "SE_Europe": ["EL", "CY", "TR", "AL", "BA", "HR", "ME", "MK", "RS", "BG"],
}
CNTR_TO_REGION = {c: r for r, cc in REGION_COUNTRIES.items() for c in cc}

# -----------------------------
# Classes / settings (same as source)
# -----------------------------
REF_CLASSES = list(range(0, 8))   # 0..7
COLLAPSE_URBAN_1XX = True
DROP_UNKNOWN_999 = True

RAKE_MAX_ITER = 200
RAKE_TOL = 1e-10

ALPHA0 = 0.5
KAPPA_GRID = np.geomspace(0.3, 300.0, 80)
N_DRAWS_DEFAULT = 5000
RNG_SEED_DEFAULT = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Country-level pooled wetland area estimation."
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        required=True,
        help="Validation sample file with columns: CNTR_ID, Code_18, reference_final.",
    )
    parser.add_argument(
        "--strata-file",
        type=Path,
        required=True,
        help="CSV with country-by-stratum areas.",
    )
    parser.add_argument(
        "--country-totals-file",
        type=Path,
        required=True,
        help="CSV with known country frame totals.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=N_DRAWS_DEFAULT,
        help=f"Number of posterior Dirichlet draws per country. Default: {N_DRAWS_DEFAULT}.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=RNG_SEED_DEFAULT,
        help=f"Random seed for posterior simulation. Default: {RNG_SEED_DEFAULT}.",
    )
    return parser.parse_args()


# =============================
# Helper functions
# =============================
def collapse_code18(code: int) -> int:
    code = int(code)
    if DROP_UNKNOWN_999 and code == 999:
        return 999
    if COLLAPSE_URBAN_1XX and (code // 100) == 1:
        return 100
    return code


def pick_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_ha(series: pd.Series) -> pd.Series:
    med = float(np.nanmedian(series.values))
    if med > 1e6:   # likely m²
        return series / 10000.0
    return series


def dm_loglik(counts_vec: np.ndarray, alpha_vec: np.ndarray) -> float:
    """
    Dirichlet-multinomial log-likelihood up to a normalizing constant:

        p(n | alpha) = Gamma(a0)/Gamma(a0+N) *
                       prod_i Gamma(alpha_i+n_i)/Gamma(alpha_i)
    """
    counts_vec = np.asarray(counts_vec, dtype=float)
    alpha_vec = np.asarray(alpha_vec, dtype=float)
    N = counts_vec.sum()
    a0 = alpha_vec.sum()
    if N <= 0:
        return 0.0
    if np.any(alpha_vec <= 0) or a0 <= 0:
        return -np.inf
    return (
        gammaln(a0) - gammaln(a0 + N)
        + np.sum(gammaln(alpha_vec + counts_vec) - gammaln(alpha_vec))
    )


def ensure_all_classes(vec: np.ndarray, K: int) -> np.ndarray:
    vec = np.asarray(vec, dtype=float).reshape(-1)
    if vec.size != K:
        out = np.zeros(K, dtype=float)
        out[: min(K, vec.size)] = vec[: min(K, vec.size)]
        return out
    return vec


def max_rel_margin_error(dfw: pd.DataFrame, A_d: dict, A_h: dict) -> float:
    sum_country = dfw.groupby("CNTR_ID")["w"].sum()
    sum_code = dfw.groupby("Code_18_est")["w"].sum()

    ce = [
        abs(sum_country[c] - A_d[c]) / A_d[c]
        for c in sum_country.index if c in A_d and A_d[c] > 0
    ]
    he = [
        abs(sum_code[h] - A_h[h]) / A_h[h]
        for h in sum_code.index if h in A_h and A_h[h] > 0
    ]

    return max(ce + he) if (ce or he) else 0.0


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_fp = args.sample_file
    country_strata_fp = args.strata_file
    country_totals_fp = args.country_totals_file
    out_dir = args.out_dir
    n_draws = args.n_draws
    rng_seed = args.rng_seed

    # =============================
    # 1) Load validation samples, assign regions, and collapse strata
    # =============================
    gdf = gpd.read_file(sample_fp)

    needed = ["CNTR_ID", "Code_18", "reference_final"]
    missing = [c for c in needed if c not in gdf.columns]
    if missing:
        raise RuntimeError(f"Sample file missing columns: {missing}\nColumns: {list(gdf.columns)}")

    df = gdf[needed].copy().dropna(subset=needed)

    df["CNTR_ID"] = df["CNTR_ID"].astype(str)
    df["region"] = df["CNTR_ID"].map(CNTR_TO_REGION).fillna("Unassigned")

    df["Code_18_est"] = df["Code_18"].astype(int).map(collapse_code18)
    if DROP_UNKNOWN_999:
        df = df[df["Code_18_est"] != 999].copy()

    df["ref_class"] = pd.to_numeric(df["reference_final"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ref_class"]).copy()
    df["ref_class"] = df["ref_class"].astype(int)

    df = df[df["ref_class"].isin(REF_CLASSES)].copy()

    print("Loaded samples:", len(df))
    print("Region counts:\n", df["region"].value_counts().to_string())

    # =============================
    # 2) EU-wide stratum totals A_h for Code_18_est and base weights w0
    # =============================
    strata = pd.read_csv(country_strata_fp)

    if "CNTR_ID" not in strata.columns:
        if "country" in strata.columns:
            strata = strata.rename(columns={"country": "CNTR_ID"})
        else:
            raise RuntimeError(f"Strata file missing CNTR_ID/country. Columns: {list(strata.columns)}")

    code_col = pick_first_existing(strata, ["Code_18", "code_18", "CODE_18"])
    area_col = pick_first_existing(
        strata,
        ["fty_zero_ha", "fty_zero_ha_total", "A_ha", "area_ha",
         "fty_zero_m2", "area_m2", "A_m2"]
    )
    if code_col is None or area_col is None:
        raise RuntimeError(f"Strata file missing code/area columns. Columns: {list(strata.columns)}")

    strata = strata[["CNTR_ID", code_col, area_col]].copy()
    strata["CNTR_ID"] = strata["CNTR_ID"].astype(str)
    strata["Code_18_est"] = strata[code_col].astype(int).map(collapse_code18)
    if DROP_UNKNOWN_999:
        strata = strata[strata["Code_18_est"] != 999].copy()

    strata["A_country_ha"] = to_ha(pd.to_numeric(strata[area_col], errors="coerce"))
    strata = strata.dropna(subset=["A_country_ha"]).copy()

    Ah = strata.groupby("Code_18_est", as_index=False).agg(A_h_ha=("A_country_ha", "sum"))

    nh = df.groupby("Code_18_est").size().reset_index(name="n_h")
    Ah = Ah.merge(nh, on="Code_18_est", how="left")

    bad = Ah[(Ah["A_h_ha"] > 0) & (Ah["n_h"].isna())]
    if not bad.empty:
        raise RuntimeError(
            "Some Code_18_est strata have area but no samples. Need different collapsing.\n"
            f"{bad[['Code_18_est','A_h_ha']].head(50)}"
        )

    Ah["n_h"] = Ah["n_h"].astype(int)
    Ah["w0"] = Ah["A_h_ha"] / Ah["n_h"]

    df = df.merge(Ah[["Code_18_est", "w0"]], on="Code_18_est", how="left")
    if df["w0"].isna().any():
        raise RuntimeError("Some samples did not receive w0. Check strata join / collapse rules.")

    # =============================
    # 3) Load country totals A_d and rake weights to match (country + strata) margins
    # =============================
    tot = pd.read_csv(country_totals_fp)

    if "CNTR_ID" not in tot.columns:
        if "country" in tot.columns:
            tot = tot.rename(columns={"country": "CNTR_ID"})
        else:
            raise RuntimeError(f"Country totals missing CNTR_ID/country. Columns: {list(tot.columns)}")

    tot_col = pick_first_existing(tot, ["fty_zero_ha_total", "fty_zero_ha", "A_d_ha", "A_ha", "total_ha"])
    if tot_col is None:
        raise RuntimeError(f"Country totals missing total area column. Columns: {list(tot.columns)}")

    tot = tot[["CNTR_ID", tot_col]].copy()
    tot["CNTR_ID"] = tot["CNTR_ID"].astype(str)
    tot["A_d_ha"] = pd.to_numeric(tot[tot_col], errors="coerce")
    tot = tot.dropna(subset=["A_d_ha"]).copy()

    A_d = tot.set_index("CNTR_ID")["A_d_ha"].to_dict()
    A_h = Ah.set_index("Code_18_est")["A_h_ha"].to_dict()

    df["w"] = df["w0"].astype(float)

    print("\nRaking weights (match country totals + stratum totals)...")
    for it in range(1, RAKE_MAX_ITER + 1):
        sum_country = df.groupby("CNTR_ID")["w"].sum()
        g_country = {
            c: (A_d[c] / sum_country[c])
            for c in sum_country.index if c in A_d and sum_country[c] > 0
        }
        df["w"] *= df["CNTR_ID"].map(g_country).fillna(1.0)

        sum_code = df.groupby("Code_18_est")["w"].sum()
        g_code = {
            h: (A_h[h] / sum_code[h])
            for h in sum_code.index if h in A_h and sum_code[h] > 0
        }
        df["w"] *= df["Code_18_est"].map(g_code).fillna(1.0)

        if it in (1, 5, 10, 20, 50, 100, RAKE_MAX_ITER):
            err = max_rel_margin_error(df, A_d=A_d, A_h=A_h)
            print(f"  iter {it:03d} | max rel err = {err:.3e}")
        if max_rel_margin_error(df, A_d=A_d, A_h=A_h) < RAKE_TOL:
            print(f"Converged at iter {it}")
            break

    df = df.rename(columns={"w": "w_cal"})

    # =============================
    # 4) Normalize calibrated weights within country
    #    This preserves a country-level effective sample size scale.
    # =============================
    grp = df.groupby("CNTR_ID")
    sum_w = grp["w_cal"].transform("sum")
    n_d = grp["w_cal"].transform("size").astype(float)
    df["w_norm"] = np.where(sum_w > 0, df["w_cal"] / sum_w * n_d, 1.0)

    # =============================
    # 5) Weighted country reference counts  ñ_{d,i}
    # =============================
    country_counts = (
        df.groupby(["CNTR_ID", "region", "ref_class"])["w_norm"]
          .sum()
          .reset_index(name="ntilde")
    )

    # region totals and region mean proportions μ_r
    region_counts = (
        country_counts.groupby(["region", "ref_class"], as_index=False)["ntilde"]
                     .sum()
    )

    mu_dict = {}
    mu_rows = []
    for r in sorted(region_counts["region"].unique()):
        vec = np.zeros(len(REF_CLASSES), dtype=float)
        sub = region_counts[region_counts["region"] == r]
        for _, row in sub.iterrows():
            vec[int(row["ref_class"])] = float(row["ntilde"])
        alpha = vec + ALPHA0
        mu = alpha / alpha.sum()
        mu_dict[r] = mu
        for i in REF_CLASSES:
            mu_rows.append({"region": r, "ref_class": i, "mu": float(mu[i])})

    mu_region = pd.DataFrame(mu_rows)
    mu_region.to_csv(out_dir / "region_ref_mu.csv", index=False)

    # Build country vectors ñ_{d,·}
    country_vecs = {}
    for (d, r), sub in country_counts.groupby(["CNTR_ID", "region"]):
        vec = np.zeros(len(REF_CLASSES), dtype=float)
        for _, row in sub.iterrows():
            vec[int(row["ref_class"])] = float(row["ntilde"])
        country_vecs[(d, r)] = vec

    # =============================
    # 6) Estimate the global shrinkage parameter kappa by empirical Bayes
    # =============================
    kappa_best = float(KAPPA_GRID[0])
    ll_best = -np.inf

    for kappa in KAPPA_GRID:
        ll = 0.0
        for (d, r), vec in country_vecs.items():
            if vec.sum() <= 0:
                continue
            mu = mu_dict.get(r, np.ones(len(REF_CLASSES)) / len(REF_CLASSES))
            ll += dm_loglik(vec, kappa * mu)
        if ll > ll_best:
            ll_best, kappa_best = ll, float(kappa)

    print(f"\nSelected global kappa = {kappa_best:.3f} (EB grid search)")

    # =============================
    # 7) Draw posterior country compositions pi_d and convert them to areas
    # =============================
    rng = np.random.default_rng(rng_seed)

    rows = []
    for d in sorted(df["CNTR_ID"].unique()):
        r = CNTR_TO_REGION.get(d, "Unassigned")
        mu = mu_dict.get(r, np.ones(len(REF_CLASSES)) / len(REF_CLASSES))
        vec = country_vecs.get((d, r), np.zeros(len(REF_CLASSES), dtype=float))

        # Posterior over proportions in the frame
        alpha_post = np.clip(kappa_best * mu + vec, 1e-8, None)

        pis = rng.dirichlet(alpha_post, size=n_draws)

        Ad = float(A_d.get(d, np.nan))
        if not np.isfinite(Ad) or Ad <= 0:
            Ad = np.nan

        for i in REF_CLASSES:
            p_draw = pis[:, i]
            p_lo, p_hi = np.quantile(p_draw, [0.025, 0.975])

            if np.isfinite(Ad):
                a_draw = Ad * p_draw
                a_lo, a_hi = np.quantile(a_draw, [0.025, 0.975])
                a_mean = float(np.mean(a_draw))
                a_med = float(np.median(a_draw))
            else:
                a_mean = a_med = a_lo = a_hi = np.nan

            rows.append({
                "CNTR_ID": d,
                "region": r,
                "ref_class": i,
                "A_frame_country_ha": float(A_d.get(d, np.nan)),
                "pi_post_mean": float(np.mean(p_draw)),
                "pi_post_median": float(np.median(p_draw)),
                "pi_ci_lo": float(p_lo),
                "pi_ci_hi": float(p_hi),
                "A_post_mean_ha": a_mean,
                "A_post_median_ha": a_med,
                "A_ci_lo": float(a_lo),
                "A_ci_hi": float(a_hi),
                "kappa_global": kappa_best,
                "alpha0": ALPHA0,
                "n_eff_country": float(vec.sum()),
            })

    out = pd.DataFrame(rows)

    # =============================
    # 8) Save outputs (same content, more public-facing filenames)
    # =============================
    eu_totals_fp = out_dir / "eu_collapsed_stratum_totals.csv"
    region_mu_fp = out_dir / "regional_reference_composition.csv"
    country_counts_fp = out_dir / "country_weighted_reference_pseudocounts.csv"
    out_fp = out_dir / "country_pooled_wetland_area_estimates.csv"
    country_region_fp = out_dir / "country_to_region_lookup.csv"
    diag_ref_fp = out_dir / "diagnostic_region_reference_counts_raw.csv"
    kappa_fp = out_dir / "global_shrinkage_parameter.csv"

    Ah.to_csv(eu_totals_fp, index=False)
    mu_region.to_csv(region_mu_fp, index=False)
    country_counts.to_csv(country_counts_fp, index=False)
    out.to_csv(out_fp, index=False)

    membership = pd.DataFrame({"CNTR_ID": sorted(df["CNTR_ID"].unique())})
    membership["region"] = membership["CNTR_ID"].map(CNTR_TO_REGION).fillna("Unassigned")
    membership.to_csv(country_region_fp, index=False)

    diag_ref = df.groupby(["region", "ref_class"]).size().unstack(fill_value=0)
    diag_ref.to_csv(diag_ref_fp)

    pd.DataFrame(
        {"kappa_global": [kappa_best], "alpha0": [ALPHA0], "n_draws": [n_draws]}
    ).to_csv(kappa_fp, index=False)

    print("\nSaved outputs to:", out_dir)
    print("Main result:", out_fp)

if __name__ == "__main__":
    main()
