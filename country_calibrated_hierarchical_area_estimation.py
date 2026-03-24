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
from typing import Dict, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.special import gammaln


M2_PER_HA = 10_000.0
URBAN_STRATUM_CODE = 100
EPS = 1e-12

CLASS_LABELS: Dict[int, str] = {
    0: "Non-wetland",
    1: "Inland marshes",
    2: "Peatbogs",
    3: "Salt marshes",
    4: "Salines",
    5: "Intertidal flats",
    6: "Moors & heathland",
    7: "Surface water",
}

REGION_COUNTRIES: Dict[str, list[str]] = {
    "Northern": ["DK", "NO", "SE", "FI", "IS", "EE", "LV", "LT"],
    "Western": ["FR", "BE", "NL", "IE", "UK", "LU", "AD"],
    "Central": ["DE", "AT", "CH", "CZ", "PL", "SK", "HU", "LI", "SI", "RO"],
    "SW_Europe": ["ES", "PT", "IT"],
    "SE_Europe": ["EL", "CY", "TR", "AL", "BA", "HR", "ME", "MK", "RS", "BG"],
}
COUNTRY_TO_REGION = {
    country: region
    for region, countries in REGION_COUNTRIES.items()
    for country in countries
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Country-level calibrated hierarchical wetland area estimation."
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        required=True,
        help="Vector file with validation points and columns: CNTR_ID, Code_18, reference.",
    )
    parser.add_argument(
        "--strata-file",
        type=Path,
        required=True,
        help="CSV file with country-by-stratum areas and columns: CNTR_ID, Code_18, and one area column.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for summary and diagnostic CSV files.",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=8,
        help="Number of reporting classes. Default: 8.",
    )
    parser.add_argument(
        "--exclude-strata",
        type=int,
        nargs="*",
        default=[],
        help="Optional strata to exclude from the analysis.",
    )
    parser.add_argument(
        "--nsim",
        type=int,
        default=5000,
        help="Number of Dirichlet posterior draws per country. Default: 5000.",
    )
    parser.add_argument(
        "--kappa-min",
        type=float,
        default=0.1,
        help="Minimum kappa value for empirical-Bayes grid search. Default: 0.1.",
    )
    parser.add_argument(
        "--kappa-max",
        type=float,
        default=500.0,
        help="Maximum kappa value for empirical-Bayes grid search. Default: 500.",
    )
    parser.add_argument(
        "--kappa-grid-size",
        type=int,
        default=80,
        help="Number of kappa values in the empirical-Bayes grid. Default: 80.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for posterior simulation. Default: 42.",
    )
    return parser.parse_args()


def recode_code18_to_stratum(code: int) -> int:
    """Recode all CLC 1xx codes to stratum 100."""
    code = int(code)
    return URBAN_STRATUM_CODE if (code // 100) == 1 else code


def to_int_series(series: pd.Series, name: str) -> pd.Series:
    """Coerce a series to integer values; invalid rows become missing."""
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().any():
        n_bad = int(out.isna().sum())
        print(f"[WARN] {n_bad} rows contain invalid values in '{name}' and will be dropped.")
    return out.astype("Int64")


def infer_area_m2(df: pd.DataFrame) -> pd.Series:
    """
    Infer an area column and return area in m^2.

    Accepted columns:
        - fty_zero_m2
        - fty_zero_ha
        - area_m2
    """
    cols = {c.strip(): c for c in df.columns}

    if "fty_zero_m2" in cols:
        return pd.to_numeric(df[cols["fty_zero_m2"]], errors="coerce")
    if "fty_zero_ha" in cols:
        return pd.to_numeric(df[cols["fty_zero_ha"]], errors="coerce") * M2_PER_HA
    if "area_m2" in cols:
        return pd.to_numeric(df[cols["area_m2"]], errors="coerce")

    raise ValueError(
        "No supported area column found. Expected one of: "
        "'fty_zero_m2', 'fty_zero_ha', or 'area_m2'."
    )


def load_samples(
    sample_file: Path,
    n_classes: int,
    exclude_strata: Iterable[int],
) -> pd.DataFrame:
    """
    Load validation points and prepare:

        country = country code
        h       = stratum
        ref     = reference class
    """
    gdf = gpd.read_file(sample_file)

    required = {"CNTR_ID", "Code_18", "reference"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"Sample file is missing required columns: {sorted(missing)}")

    gdf["country"] = gdf["CNTR_ID"].astype(str).str.strip()
    gdf["h"] = to_int_series(gdf["Code_18"], "Code_18")
    gdf["ref"] = pd.to_numeric(gdf["reference"], errors="coerce")

    gdf = gdf.dropna(subset=["country", "h", "ref"]).copy()
    gdf["h"] = gdf["h"].astype(int).map(recode_code18_to_stratum).astype(int)
    gdf["ref"] = gdf["ref"].astype(int)

    exclude_strata = set(exclude_strata)
    if exclude_strata:
        gdf = gdf.loc[~gdf["h"].isin(exclude_strata)].copy()

    if not gdf["ref"].between(0, n_classes - 1).all():
        bad = np.sort(gdf.loc[~gdf["ref"].between(0, n_classes - 1), "ref"].unique())
        raise ValueError(f"Reference labels outside 0..{n_classes - 1}: {bad}")

    return pd.DataFrame(gdf[["country", "h", "ref"]])


def load_country_strata(
    strata_file: Path,
    exclude_strata: Iterable[int],
) -> pd.DataFrame:
    """
    Load known country-by-stratum areas and return:

        country = country code
        h       = stratum
        A_dh_m2 = known area for country d and stratum h
    """
    df = pd.read_csv(strata_file)
    df.columns = [c.strip() for c in df.columns]

    required = {"CNTR_ID", "Code_18"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Strata file is missing required columns: {sorted(missing)}")

    df["country"] = df["CNTR_ID"].astype(str).str.strip()
    df["h"] = pd.to_numeric(df["Code_18"], errors="coerce")
    df["A_dh_m2"] = infer_area_m2(df)

    df = df.dropna(subset=["country", "h", "A_dh_m2"]).copy()
    df["h"] = df["h"].astype(int).map(recode_code18_to_stratum).astype(int)
    df["A_dh_m2"] = df["A_dh_m2"].astype(float)

    exclude_strata = set(exclude_strata)
    if exclude_strata:
        df = df.loc[~df["h"].isin(exclude_strata)].copy()

    # Needed because multiple original codes may be reassigned to the same stratum.
    df = df.groupby(["country", "h"], as_index=False)["A_dh_m2"].sum()

    return df


def build_population_margins(country_strata: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Return known margins:

        A_d = sum_h A_dh
        A_h = sum_d A_dh
    """
    A_d = country_strata.groupby("country")["A_dh_m2"].sum().sort_index()
    A_h = country_strata.groupby("h")["A_dh_m2"].sum().sort_index()
    return A_d, A_h


def compute_base_weights(samples: pd.DataFrame, A_h: pd.Series) -> pd.DataFrame:
    """
    Base weights for the continental stratified sample:

        w_h^(0) = A_h / n_h

    where A_h is known stratum area and n_h is realized sample size
    in stratum h; see Särndal et al. (1992), Ch. 10.
    """
    n_h = samples.groupby("h").size().rename("n_h")
    weight_table = pd.concat([A_h.rename("A_h_m2"), n_h], axis=1)

    if weight_table["n_h"].isna().any():
        missing = weight_table.index[weight_table["n_h"].isna()].tolist()
        raise ValueError(f"No sampled units found for strata: {missing}")

    weight_table["w0"] = weight_table["A_h_m2"] / weight_table["n_h"]
    out = samples.merge(weight_table.reset_index()[["h", "w0"]], on="h", how="left")

    if out["w0"].isna().any():
        missing = np.sort(out.loc[out["w0"].isna(), "h"].unique())
        raise ValueError(f"Base weights could not be assigned for strata: {missing}")

    return out


def rake_weights(
    df: pd.DataFrame,
    target_country_m2: pd.Series,
    target_stratum_m2: pd.Series,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> pd.DataFrame:
    """
    Calibrate weights by iterative proportional fitting (raking) so that:

        sum_{s in country d} w_s = A_d
        sum_{s in stratum h} w_s = A_h

    This is a standard calibration / generalized raking procedure;
    see Deville and Särndal (1992) and Deville et al. (1993).
    """
    out = df.copy()
    out["w_cal"] = out["w0"].astype(float)

    countries_in_sample = sorted(out["country"].unique())
    strata_in_sample = sorted(out["h"].unique())

    missing_country = sorted(set(countries_in_sample) - set(target_country_m2.index))
    if missing_country:
        raise ValueError(f"Missing country totals for sampled countries: {missing_country}")

    missing_stratum = sorted(set(strata_in_sample) - set(target_stratum_m2.index))
    if missing_stratum:
        raise ValueError(f"Missing stratum totals for sampled strata: {missing_stratum}")

    for _ in range(max_iter):
        w_prev = out["w_cal"].to_numpy(copy=True)

        # Match country totals.
        current_country = out.groupby("country")["w_cal"].sum()
        for country, current in current_country.items():
            target = float(target_country_m2.loc[country])
            if current > 0:
                out.loc[out["country"] == country, "w_cal"] *= target / current

        # Match stratum totals.
        current_stratum = out.groupby("h")["w_cal"].sum()
        for h, current in current_stratum.items():
            target = float(target_stratum_m2.loc[h])
            if current > 0:
                out.loc[out["h"] == h, "w_cal"] *= target / current

        max_rel_change = np.max(np.abs(out["w_cal"].to_numpy() - w_prev) / np.maximum(w_prev, EPS))
        if max_rel_change < tol:
            break

    # Final diagnostics.
    chk_country = out.groupby("country")["w_cal"].sum().reindex(target_country_m2.index).dropna()
    chk_stratum = out.groupby("h")["w_cal"].sum().reindex(target_stratum_m2.index).dropna()

    max_country_err = np.max(np.abs(chk_country - target_country_m2.loc[chk_country.index]))
    max_stratum_err = np.max(np.abs(chk_stratum - target_stratum_m2.loc[chk_stratum.index]))

    print(f"Max absolute country calibration error (m^2): {max_country_err:.6f}")
    print(f"Max absolute stratum calibration error (m^2): {max_stratum_err:.6f}")

    return out


def add_normalized_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize calibrated weights within country to preserve the Kish
    effective sample size:

        n_eff,d = (sum_s w_ds)^2 / sum_s w_ds^2
        w_ds^norm = w_ds^cal / sum_s w_ds^cal * n_eff,d
    """
    out = df.copy()

    grouped = out.groupby("country")["w_cal"]
    sum_w = grouped.transform("sum")
    sum_w2 = out.assign(w2=out["w_cal"] ** 2).groupby("country")["w2"].transform("sum")
    out["n_eff_country"] = (sum_w ** 2) / np.maximum(sum_w2, EPS)
    out["w_norm"] = out["w_cal"] / np.maximum(sum_w, EPS) * out["n_eff_country"]

    return out


def build_country_pseudocounts(
    df: pd.DataFrame,
    n_classes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build weighted country reference pseudo-counts:

        n_tilde_{d,i} = sum_{s in d} w_ds^norm I(y_s = i)
    """
    counts = []
    country_diag = []

    for country, sub in df.groupby("country", sort=True):
        row = {"country": country}
        for k in range(n_classes):
            row[f"class_{k}"] = float(sub.loc[sub["ref"] == k, "w_norm"].sum())
        counts.append(row)

        sum_w = float(sub["w_cal"].sum())
        sum_w2 = float((sub["w_cal"] ** 2).sum())
        n_eff = (sum_w ** 2) / max(sum_w2, EPS)
        country_diag.append(
            {
                "country": country,
                "n_country_samples": int(len(sub)),
                "n_eff_country": n_eff,
            }
        )

    count_df = pd.DataFrame(counts).sort_values("country").reset_index(drop=True)
    diag_df = pd.DataFrame(country_diag).sort_values("country").reset_index(drop=True)

    return count_df, diag_df


def composition_from_counts(count_vec: np.ndarray) -> np.ndarray:
    """Convert a non-negative count vector to a composition."""
    total = float(np.sum(count_vec))
    if total <= 0:
        return np.full_like(count_vec, 1.0 / len(count_vec), dtype=float)
    return count_vec / total


def compute_region_means(
    count_df: pd.DataFrame,
    n_classes: int,
) -> pd.DataFrame:
    """
    Compute macro-regional mean class compositions from pooled
    country pseudo-counts. Countries not matched to a region are
    assigned to 'Other'.
    """
    rows = []
    class_cols = [f"class_{k}" for k in range(n_classes)]

    temp = count_df.copy()
    temp["region"] = temp["country"].map(COUNTRY_TO_REGION).fillna("Other")

    global_counts = temp[class_cols].sum().to_numpy(dtype=float)
    global_mu = composition_from_counts(global_counts)

    for region, sub in temp.groupby("region", sort=True):
        region_counts = sub[class_cols].sum().to_numpy(dtype=float)
        mu = composition_from_counts(region_counts)
        row = {"region": region}
        for k in range(n_classes):
            row[f"mu_{k}"] = mu[k]
        rows.append(row)

    region_df = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)

    if "Other" not in set(region_df["region"]):
        row = {"region": "Other"}
        for k in range(n_classes):
            row[f"mu_{k}"] = global_mu[k]
        region_df = pd.concat([region_df, pd.DataFrame([row])], ignore_index=True)

    return region_df


def dirichlet_multinomial_log_marginal(alpha: np.ndarray, x: np.ndarray) -> float:
    """
    Dirichlet-multinomial log marginal likelihood:

        log p(x | alpha) =
            log Gamma(sum alpha)
          - log Gamma(sum alpha + sum x)
          + sum_i [log Gamma(alpha_i + x_i) - log Gamma(alpha_i)]

    This form extends naturally to non-integer pseudo-counts through
    the gamma function.
    """
    alpha = np.asarray(alpha, dtype=float)
    x = np.asarray(x, dtype=float)

    if np.any(alpha <= 0) or np.any(x < 0):
        return -np.inf

    return (
        gammaln(np.sum(alpha))
        - gammaln(np.sum(alpha) + np.sum(x))
        + np.sum(gammaln(alpha + x) - gammaln(alpha))
    )


def estimate_kappa_empirical_bayes(
    count_df: pd.DataFrame,
    region_df: pd.DataFrame,
    n_classes: int,
    kappa_min: float,
    kappa_max: float,
    grid_size: int,
) -> float:
    """
    Estimate the global shrinkage parameter kappa by grid search
    over the Dirichlet-multinomial marginal likelihood.
    """
    class_cols = [f"class_{k}" for k in range(n_classes)]
    mu_cols = [f"mu_{k}" for k in range(n_classes)]

    region_lookup = region_df.set_index("region")[mu_cols]

    temp = count_df.copy()
    temp["region"] = temp["country"].map(COUNTRY_TO_REGION).fillna("Other")

    grid = np.logspace(np.log10(kappa_min), np.log10(kappa_max), grid_size)
    scores = []

    for kappa in grid:
        score = 0.0
        for _, row in temp.iterrows():
            x = row[class_cols].to_numpy(dtype=float)
            mu = region_lookup.loc[row["region"]].to_numpy(dtype=float)
            alpha = np.maximum(kappa * mu, EPS)
            score += dirichlet_multinomial_log_marginal(alpha, x)
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return float(grid[best_idx])


def simulate_country_posteriors(
    count_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    region_df: pd.DataFrame,
    A_d_m2: pd.Series,
    n_classes: int,
    kappa: float,
    nsim: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Simulate posterior country class compositions and convert to areas:

        pi_d | data ~ Dirichlet(kappa * mu_r + n_tilde_d)
        A_{d,i} = A_d * pi_{d,i}
    """
    rng = np.random.default_rng(random_seed)
    class_cols = [f"class_{k}" for k in range(n_classes)]
    mu_cols = [f"mu_{k}" for k in range(n_classes)]

    region_lookup = region_df.set_index("region")[mu_cols]
    diag_lookup = diag_df.set_index("country")

    rows = []

    for _, row in count_df.iterrows():
        country = row["country"]
        region = COUNTRY_TO_REGION.get(country, "Other")

        x = row[class_cols].to_numpy(dtype=float)
        mu = region_lookup.loc[region].to_numpy(dtype=float)
        alpha_post = np.maximum(kappa * mu + x, EPS)

        draws = rng.dirichlet(alpha_post, size=nsim)

        if country not in A_d_m2.index:
            raise ValueError(f"Country total area not found for {country}")

        A_country_ha = float(A_d_m2.loc[country] / M2_PER_HA)

        p_direct = composition_from_counts(x)
        area_direct_ha = A_country_ha * p_direct
        area_draws_ha = draws * A_country_ha

        for k in range(n_classes):
            rows.append(
                {
                    "country": country,
                    "region": region,
                    "ref_class": k,
                    "class_label": CLASS_LABELS.get(k, f"class_{k}"),
                    "n_country_samples": int(diag_lookup.loc[country, "n_country_samples"]),
                    "n_eff_country": float(diag_lookup.loc[country, "n_eff_country"]),
                    "n_tilde": float(x[k]),
                    "p_direct": float(p_direct[k]),
                    "pi_pooled": float(np.mean(draws[:, k])),
                    "A_direct_ha": float(area_direct_ha[k]),
                    "A_post_mean_ha": float(np.mean(area_draws_ha[:, k])),
                    "A_ci_lo_ha": float(np.quantile(area_draws_ha[:, k], 0.025)),
                    "A_ci_hi_ha": float(np.quantile(area_draws_ha[:, k], 0.975)),
                    "kappa": float(kappa),
                }
            )

    return pd.DataFrame(rows).sort_values(["country", "ref_class"]).reset_index(drop=True)


def build_weight_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize base, calibrated, and normalized weights by country."""
    rows = []
    for country, sub in df.groupby("country", sort=True):
        rows.append(
            {
                "country": country,
                "n_country_samples": int(len(sub)),
                "w0_sum": float(sub["w0"].sum()),
                "w_cal_sum": float(sub["w_cal"].sum()),
                "w_norm_sum": float(sub["w_norm"].sum()),
                "w_cal_min": float(sub["w_cal"].min()),
                "w_cal_max": float(sub["w_cal"].max()),
                "w_norm_min": float(sub["w_norm"].min()),
                "w_norm_max": float(sub["w_norm"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(
        sample_file=args.sample_file,
        n_classes=args.n_classes,
        exclude_strata=args.exclude_strata,
    )
    country_strata = load_country_strata(
        strata_file=args.strata_file,
        exclude_strata=args.exclude_strata,
    )

    A_d_m2, A_h_m2 = build_population_margins(country_strata)

    sampled_countries = set(samples["country"].unique())
    known_countries = set(A_d_m2.index)
    missing_countries = sorted(sampled_countries - known_countries)
    if missing_countries:
        raise ValueError(f"Missing country totals for sampled countries: {missing_countries}")

    sampled_strata = set(samples["h"].unique())
    known_strata = set(A_h_m2.index)
    missing_strata = sorted(sampled_strata - known_strata)
    if missing_strata:
        raise ValueError(f"Missing stratum totals for sampled strata: {missing_strata}")

    weighted_samples = compute_base_weights(samples=samples, A_h=A_h_m2)
    weighted_samples = rake_weights(
        df=weighted_samples,
        target_country_m2=A_d_m2,
        target_stratum_m2=A_h_m2,
    )
    weighted_samples = add_normalized_weights(weighted_samples)

    count_df, diag_df = build_country_pseudocounts(
        df=weighted_samples,
        n_classes=args.n_classes,
    )
    region_df = compute_region_means(
        count_df=count_df,
        n_classes=args.n_classes,
    )

    kappa = estimate_kappa_empirical_bayes(
        count_df=count_df,
        region_df=region_df,
        n_classes=args.n_classes,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        grid_size=args.kappa_grid_size,
    )

    summary_df = simulate_country_posteriors(
        count_df=count_df,
        diag_df=diag_df,
        region_df=region_df,
        A_d_m2=A_d_m2,
        n_classes=args.n_classes,
        kappa=kappa,
        nsim=args.nsim,
        random_seed=args.random_seed,
    )

    weight_diag_df = build_weight_diagnostics(weighted_samples)

    summary_fp = args.out_dir / "country_calibrated_hierarchical_area_estimates.csv"
    counts_fp = args.out_dir / "country_weighted_pseudocounts.csv"
    region_fp = args.out_dir / "region_mean_compositions.csv"
    weight_fp = args.out_dir / "country_weight_diagnostics.csv"

    summary_df.to_csv(summary_fp, index=False)
    count_df.to_csv(counts_fp, index=False)
    region_df.to_csv(region_fp, index=False)
    weight_diag_df.to_csv(weight_fp, index=False)

    print(f"Saved: {summary_fp}")
    print(f"Saved: {counts_fp}")
    print(f"Saved: {region_fp}")
    print(f"Saved: {weight_fp}")
    print()
    print(f"Estimated kappa: {kappa:.6f}")
    print(f"Countries estimated: {summary_df['country'].nunique()}")
    print(f"Total samples used: {len(samples):,d}")


if __name__ == "__main__":
    main()
