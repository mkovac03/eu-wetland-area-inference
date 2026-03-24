#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Country-level calibrated hierarchical disturbance allocation.

This script allocates pooled country-level wetland class areas into
country × reference-class × disturbance cells using:
    1) country-level calibrated hierarchical class totals,
    2) EU design-based class × disturbance estimates, and
    3) mapped within-country disturbance shares as auxiliary information.

The allocation is calibrated within each wetland class by iterative
proportional fitting (IPF) so that:

    sum_d A_tilde[c, k, d] = A[c, k]
    sum_c A_tilde[c, k, d] = (sum_c A[c, k]) * q[k, d]

where:
    A[c, k]      = pooled country-level area for reference class k in country c
    q[k, d]      = EU disturbance composition within class k
                 = A_hat_EU[k, d] / sum_d A_hat_EU[k, d]

Mapped within-country disturbance shares are used only to form the
initial seed:

    A^(0)[c, k, d] = A[c, k] * p_map[c, k, d]

The calibrated allocation is repeated across Monte Carlo draws to
propagate uncertainty from:
    - country-level pooled class totals, and
    - EU design-based class × disturbance totals.

Core references
---------------
Särndal, Swensson, and Wretman (1992), Model Assisted Survey Sampling,
Chapter 10, for domain estimation concepts.

Deville and Särndal (1992), JASA, for calibration estimators.

Deville, Särndal, and Sautory (1993), JASA, for generalized raking and
iterative proportional fitting.

Notes
-----
- This script works with aggregated inputs and does not read point samples.
- Reference classes are the wetland classes 1..6.
- Disturbance labels are:
      0 = least disturbed
      1 = intermediately disturbed
      2 = most disturbed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


M2_PER_HA = 10_000.0
EPS = 1e-12

WET_CLASSES = [1, 2, 3, 4, 5, 6]
DISTURBANCE_LEVELS = [0, 1, 2]

DISTURBANCE_LABELS: Dict[int, str] = {
    0: "least disturbed",
    1: "intermediately disturbed",
    2: "most disturbed",
}

CLASS_LABELS: Dict[int, str] = {
    1: "Inland marshes",
    2: "Peatbogs",
    3: "Salt marshes",
    4: "Salines",
    5: "Intertidal flats",
    6: "Moors & heathland",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Country-level calibrated hierarchical disturbance allocation."
    )
    parser.add_argument(
        "--mapped-file",
        type=Path,
        required=True,
        help=(
            "CSV with mapped country × reference-class × disturbance areas. "
            "Expected columns include country, class, disturbance, and area."
        ),
    )
    parser.add_argument(
        "--eu-design-file",
        type=Path,
        required=True,
        help=(
            "CSV with EU design-based reference-class × disturbance estimates "
            "and optional uncertainty columns."
        ),
    )
    parser.add_argument(
        "--country-class-file",
        type=Path,
        required=True,
        help=(
            "CSV with country-level calibrated hierarchical class totals "
            "and optional uncertainty columns."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for CSV files.",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=2000,
        help="Number of Monte Carlo draws. Default: 2000.",
    )
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=0.30,
        help="Target fraction of disturbed area for reporting. Default: 0.30.",
    )
    parser.add_argument(
        "--peat-class",
        type=int,
        default=2,
        help="Reference class ID for peatbogs. Default: 2.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=2000,
        help="Maximum iterations for IPF. Default: 2000.",
    )
    parser.add_argument(
        "--tol-rel",
        type=float,
        default=1e-10,
        help="Relative convergence tolerance for IPF. Default: 1e-10.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for Monte Carlo sampling. Default: 42.",
    )
    return parser.parse_args()


def as_num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def pick_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def to_ha(series: pd.Series, source_name: str) -> pd.Series:
    """
    Convert an area-like column to hectares based on its column name.
    """
    name = str(source_name).lower()
    x = pd.to_numeric(series, errors="coerce")

    if "m2" in name:
        return x / M2_PER_HA
    if "km2" in name or "km^2" in name:
        return x * 100.0
    return x


def infer_uncertainty_se(
    df: pd.DataFrame,
    se_candidates: list[str],
    sd_candidates: list[str],
    var_candidates: list[str],
    lo_candidates: list[str],
    hi_candidates: list[str],
) -> pd.Series:
    """
    Infer a standard-error column from direct SE/SD/variance/CI columns.
    """
    se_col = pick_col(df.columns, se_candidates)
    sd_col = pick_col(df.columns, sd_candidates)
    var_col = pick_col(df.columns, var_candidates)
    lo_col = pick_col(df.columns, lo_candidates)
    hi_col = pick_col(df.columns, hi_candidates)

    if se_col is not None:
        return as_num(df[se_col])

    if sd_col is not None:
        return as_num(df[sd_col])

    if var_col is not None:
        return np.sqrt(np.maximum(as_num(df[var_col]), 0.0))

    if lo_col is not None and hi_col is not None:
        lo = as_num(df[lo_col])
        hi = as_num(df[hi_col])
        return (hi - lo).abs() / (2.0 * 1.96)

    return pd.Series(np.nan, index=df.index, dtype=float)


def lognormal_params_from_mean_se(mean: float, se: float, eps: float = EPS) -> tuple[float, float]:
    mean = float(max(mean, eps))
    se = float(max(se, 0.0))

    if se <= 0.0:
        return np.log(mean), 0.0

    var = se * se
    sigma2 = np.log(1.0 + var / (mean * mean))
    sigma = np.sqrt(max(sigma2, 0.0))
    mu = np.log(mean) - 0.5 * sigma2

    return mu, sigma


def sample_positive(
    mean: float,
    se: float | None,
    rng: np.random.Generator,
    size: int | None = None,
    eps: float = EPS,
) -> np.ndarray | float:
    """
    Sample a strictly positive quantity using a lognormal approximation.
    If uncertainty is unavailable, return the mean as fixed.
    """
    mean = float(mean)

    if se is None or (not np.isfinite(se)) or float(se) <= 0.0:
        value = max(mean, eps)
        if size is None:
            return value
        return np.full(size, value, dtype=float)

    mu, sigma = lognormal_params_from_mean_se(mean, float(se), eps=eps)

    if size is None:
        return float(np.exp(mu + sigma * rng.standard_normal()))

    return np.exp(mu + sigma * rng.standard_normal(size))


def quantile_ci(x: np.ndarray, q_lo: float = 0.025, q_hi: float = 0.975) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(np.nanquantile(x, q_lo)), float(np.nanquantile(x, q_hi))


def raking_ipf(
    seed: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    max_iters: int = 2000,
    tol_rel: float = 1e-10,
    eps: float = EPS,
) -> np.ndarray:
    """
    Iterative proportional fitting for a non-negative matrix X satisfying:

        sum_d X[c, d] = row_targets[c]
        sum_c X[c, d] = col_targets[d]

    This is the classical raking ratio / IPF algorithm used for
    multiplicative calibration; see Deville et al. (1993).
    """
    X = np.array(seed, dtype=float, copy=True)
    X = np.clip(X, 0.0, None)

    if X.sum() <= eps and row_targets.sum() > eps and col_targets.sum() > eps:
        X = np.outer(row_targets, col_targets) / max(col_targets.sum(), eps)

    X = X + eps

    r = np.array(row_targets, dtype=float)
    c = np.array(col_targets, dtype=float)

    rt = r.sum()
    ct = c.sum()
    if rt > eps and ct > eps and abs(rt - ct) / max(rt, ct) > 1e-12:
        c = c * (rt / ct)

    for _ in range(max_iters):
        X_prev = X.copy()

        row_sums = X.sum(axis=1)
        row_factor = np.ones_like(r)
        mask = row_sums > eps
        row_factor[mask] = r[mask] / row_sums[mask]
        X = (X.T * row_factor).T

        col_sums = X.sum(axis=0)
        col_factor = np.ones_like(c)
        mask = col_sums > eps
        col_factor[mask] = c[mask] / col_sums[mask]
        X = X * col_factor

        denom = np.maximum(np.abs(X_prev), eps)
        rel_change = np.max(np.abs(X - X_prev) / denom)
        if rel_change < tol_rel:
            break

    return np.clip(X, 0.0, None)


def load_eu_design_class_disturbance(
    filepath: Path,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Load EU design-based reference-class × disturbance estimates.
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    class_col = pick_col(df.columns, ["ref_class", "class", "class_id", "DN"])
    dist_col = pick_col(df.columns, ["distur", "distur_1", "disturbance"])
    area_col = pick_col(df.columns, ["est_ha", "area_ha", "A_hat_ha", "mean_ha", "est"])

    if None in [class_col, dist_col, area_col]:
        raise ValueError(
            f"EU design file missing required columns. Found: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            class_col: "class",
            dist_col: "disturbance",
            area_col: "A_kd_mean",
        }
    ).copy()

    df["class"] = as_num(df["class"]).astype("Int64")
    df["disturbance"] = as_num(df["disturbance"]).astype("Int64")
    df["A_kd_mean"] = as_num(df["A_kd_mean"])

    df["A_kd_se"] = infer_uncertainty_se(
        df=df,
        se_candidates=["se_ha", "SE_ha", "se", "SE"],
        sd_candidates=["sd_ha", "SD_ha", "sd", "SD"],
        var_candidates=["var_ha2", "Var_ha2", "var", "Var"],
        lo_candidates=["ci_lo_ha", "ci_lo", "CI_lo", "lo_ha", "lower_ha"],
        hi_candidates=["ci_hi_ha", "ci_hi", "CI_hi", "hi_ha", "upper_ha"],
    )

    df = df.dropna(subset=["class", "disturbance", "A_kd_mean"]).copy()
    df["class"] = df["class"].astype(int)
    df["disturbance"] = df["disturbance"].astype(int)

    df = df[
        df["class"].isin(WET_CLASSES) &
        df["disturbance"].isin(DISTURBANCE_LEVELS)
    ].copy()

    df["A_kd_mean"] = df["A_kd_mean"].clip(lower=0.0)

    A_kd_mean = df.pivot_table(
        index="class",
        columns="disturbance",
        values="A_kd_mean",
        aggfunc="sum",
        fill_value=0.0,
    )
    A_kd_se = df.pivot_table(
        index="class",
        columns="disturbance",
        values="A_kd_se",
        aggfunc="mean",
    )

    for d in DISTURBANCE_LEVELS:
        if d not in A_kd_mean.columns:
            A_kd_mean[d] = 0.0
        if d not in A_kd_se.columns:
            A_kd_se[d] = np.nan

    A_kd_mean = A_kd_mean[DISTURBANCE_LEVELS].reindex(WET_CLASSES, fill_value=0.0)
    A_kd_se = A_kd_se[DISTURBANCE_LEVELS].reindex(WET_CLASSES)

    mean_lookup = {
        wet_class: A_kd_mean.loc[wet_class, DISTURBANCE_LEVELS].to_numpy(dtype=float)
        for wet_class in WET_CLASSES
    }
    se_lookup = {
        wet_class: A_kd_se.loc[wet_class, DISTURBANCE_LEVELS].to_numpy(dtype=float)
        for wet_class in WET_CLASSES
    }

    return mean_lookup, se_lookup


def load_country_class_totals(
    filepath: Path,
) -> tuple[list[str], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Load country-level calibrated hierarchical class totals.
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    country_col = pick_col(df.columns, ["CNTR_ID", "country", "country_code"])
    class_col = pick_col(df.columns, ["ref_class", "class", "class_id", "DN"])
    mean_col = pick_col(df.columns, ["A_post_mean_ha", "A_mean_ha", "est_mean_ha", "est_ha", "area_ha", "Ahat_ha"])

    if None in [country_col, class_col, mean_col]:
        raise ValueError(
            f"Country-class totals file missing required columns. Found: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            country_col: "country",
            class_col: "class",
            mean_col: "A_ck_mean",
        }
    ).copy()

    df["country"] = df["country"].astype(str).str.strip()
    df["class"] = as_num(df["class"]).astype("Int64")
    df["A_ck_mean"] = as_num(df["A_ck_mean"])

    df["A_ck_se"] = infer_uncertainty_se(
        df=df,
        se_candidates=["A_post_se_ha", "A_se_ha", "se_ha", "SE_ha", "A_post_se", "A_se", "se", "SE"],
        sd_candidates=["A_post_sd_ha", "A_sd_ha", "sd_ha", "SD_ha", "A_post_sd", "A_sd", "sd", "SD"],
        var_candidates=["A_post_var_ha2", "A_var_ha2", "var_ha2", "Var_ha2", "A_post_var", "A_var", "var", "Var"],
        lo_candidates=["A_post_ci_lo_ha", "A_ci_lo_ha", "A_post_ci_lo", "A_ci_lo", "ci_lo_ha", "ci_lo", "CI_lo", "lower_ha", "lo_ha"],
        hi_candidates=["A_post_ci_hi_ha", "A_ci_hi_ha", "A_post_ci_hi", "A_ci_hi", "ci_hi_ha", "ci_hi", "CI_hi", "upper_ha", "hi_ha"],
    )

    df = df.dropna(subset=["country", "class", "A_ck_mean"]).copy()
    df["class"] = df["class"].astype(int)
    df = df[df["class"].isin(WET_CLASSES)].copy()
    df["A_ck_mean"] = df["A_ck_mean"].clip(lower=0.0)

    countries = sorted(df["country"].unique().tolist())
    index_df = df.set_index(["country", "class"])[["A_ck_mean", "A_ck_se"]].sort_index()

    mean_lookup = {wet_class: np.zeros(len(countries), dtype=float) for wet_class in WET_CLASSES}
    se_lookup = {wet_class: np.full(len(countries), np.nan, dtype=float) for wet_class in WET_CLASSES}

    for wet_class in WET_CLASSES:
        for i, country in enumerate(countries):
            key = (country, wet_class)
            if key in index_df.index:
                mean_lookup[wet_class][i] = float(index_df.loc[key, "A_ck_mean"])
                se_lookup[wet_class][i] = float(index_df.loc[key, "A_ck_se"])

    return countries, mean_lookup, se_lookup


def load_mapped_country_class_disturbance(
    filepath: Path,
    countries: list[str],
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    """
    Load mapped country × reference-class × disturbance areas and convert them
    to within-country disturbance shares used as auxiliary seed information.
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    country_col = pick_col(df.columns, ["CNTR_ID", "country", "country_code", "iso3", "CNTR_CODE", "CNTR"])
    class_col = pick_col(df.columns, ["ref_class", "class", "class_id", "DN"])
    dist_col = pick_col(df.columns, ["distur", "distur_1", "disturbance"])
    area_col = pick_col(df.columns, ["area_ha", "mapped_area_ha", "area", "A_ha", "ha", "area_m2", "A_m2"])

    if None in [country_col, class_col, dist_col, area_col]:
        raise ValueError(
            f"Mapped disturbance file missing required columns. Found: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            country_col: "country",
            class_col: "class",
            dist_col: "disturbance",
            area_col: "area_raw",
        }
    ).copy()

    df["country"] = df["country"].astype(str).str.strip()
    df["class"] = as_num(df["class"]).astype("Int64")
    df["disturbance"] = as_num(df["disturbance"]).astype("Int64")
    df["area_raw"] = as_num(df["area_raw"])

    df = df.dropna(subset=["country", "class", "disturbance", "area_raw"]).copy()
    df["class"] = df["class"].astype(int)
    df["disturbance"] = df["disturbance"].astype(int)

    df = df[
        df["country"].isin(countries) &
        df["class"].isin(WET_CLASSES) &
        df["disturbance"].isin(DISTURBANCE_LEVELS)
    ].copy()

    df["area_ha"] = to_ha(df["area_raw"], area_col).clip(lower=0.0)

    area_table = df.pivot_table(
        index=["country", "class"],
        columns="disturbance",
        values="area_ha",
        aggfunc="sum",
        fill_value=0.0,
    )

    for d in DISTURBANCE_LEVELS:
        if d not in area_table.columns:
            area_table[d] = 0.0

    area_table = area_table[DISTURBANCE_LEVELS]

    row_sums = area_table.sum(axis=1).replace(0.0, np.nan)
    share_table = area_table.div(row_sums, axis=0).fillna(1.0 / len(DISTURBANCE_LEVELS))

    share_lookup: dict[tuple[str, int], np.ndarray] = {}
    for wet_class in WET_CLASSES:
        for country in countries:
            key = (country, wet_class)
            if key in share_table.index:
                share_lookup[key] = share_table.loc[key, DISTURBANCE_LEVELS].to_numpy(dtype=float)
            else:
                share_lookup[key] = np.full(len(DISTURBANCE_LEVELS), 1.0 / len(DISTURBANCE_LEVELS), dtype=float)

    return df, share_lookup


def simulate_allocations(
    countries: list[str],
    country_class_mean: dict[int, np.ndarray],
    country_class_se: dict[int, np.ndarray],
    eu_class_dist_mean: dict[int, np.ndarray],
    eu_class_dist_se: dict[int, np.ndarray],
    mapped_share_lookup: dict[tuple[str, int], np.ndarray],
    mapped_df: pd.DataFrame,
    n_draws: int,
    peat_class: int,
    max_iters: int,
    tol_rel: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate Monte Carlo allocations for country × class × disturbance areas.
    """
    rng = np.random.default_rng(random_seed)

    n_countries = len(countries)
    n_classes = len(WET_CLASSES)
    n_dist = len(DISTURBANCE_LEVELS)

    X_draws = np.zeros((n_draws, n_countries, n_classes, n_dist), dtype=np.float32)
    disturbed_draws = np.zeros((n_draws, n_countries), dtype=np.float32)
    peat_disturbed_draws = np.zeros((n_draws, n_countries), dtype=np.float32)

    j1 = DISTURBANCE_LEVELS.index(1)
    j2 = DISTURBANCE_LEVELS.index(2)

    for t in range(n_draws):
        for kk, wet_class in enumerate(WET_CLASSES):
            # Country row targets A[c, k]
            mean_vec = country_class_mean[wet_class]
            se_vec = country_class_se[wet_class]

            row_targets = np.empty(n_countries, dtype=float)
            for i in range(n_countries):
                row_targets[i] = sample_positive(mean_vec[i], se_vec[i], rng=rng, size=None, eps=EPS)

            pooled_total = float(row_targets.sum())

            # EU design class × disturbance composition q[k, d]
            design_mean = eu_class_dist_mean[wet_class]
            design_se = eu_class_dist_se[wet_class]

            design_col = np.empty(n_dist, dtype=float)
            for j in range(n_dist):
                design_col[j] = sample_positive(design_mean[j], design_se[j], rng=rng, size=None, eps=EPS)

            design_total = float(design_col.sum())

            if pooled_total > EPS and design_total > EPS:
                col_targets = design_col * (pooled_total / design_total)
            elif pooled_total <= EPS:
                col_targets = np.zeros(n_dist, dtype=float)
            else:
                # Fallback to EU mapped composition for this class.
                eu_map = (
                    mapped_df.loc[mapped_df["class"] == wet_class]
                    .groupby("disturbance")["area_ha"]
                    .sum()
                    .reindex(DISTURBANCE_LEVELS)
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                s = eu_map.sum()
                p = (eu_map / s) if s > EPS else np.full(n_dist, 1.0 / n_dist, dtype=float)
                col_targets = pooled_total * p

            # Seed from mapped within-country shares.
            seed = np.zeros((n_countries, n_dist), dtype=float)
            for i, country in enumerate(countries):
                p = mapped_share_lookup[(country, wet_class)].copy()
                p = np.clip(p, 0.0, None)
                s = p.sum()
                p = (p / s) if s > EPS else np.full(n_dist, 1.0 / n_dist, dtype=float)
                seed[i, :] = row_targets[i] * p

            X = raking_ipf(
                seed=seed,
                row_targets=row_targets,
                col_targets=col_targets,
                max_iters=max_iters,
                tol_rel=tol_rel,
                eps=EPS,
            )

            X_draws[t, :, kk, :] = X.astype(np.float32)

            disturbed_k = (X[:, j1] + X[:, j2]).astype(np.float32)
            disturbed_draws[t, :] += disturbed_k

            if wet_class == peat_class:
                peat_disturbed_draws[t, :] += disturbed_k

    return X_draws, disturbed_draws, peat_disturbed_draws


def summarize_allocations(
    X_draws: np.ndarray,
    countries: list[str],
) -> pd.DataFrame:
    """
    Summarize country × class × disturbance posterior draws.
    """
    rows = []

    for i, country in enumerate(countries):
        for kk, wet_class in enumerate(WET_CLASSES):
            A_ck_draw = X_draws[:, i, kk, :].sum(axis=1).astype(float)

            for j, disturbance in enumerate(DISTURBANCE_LEVELS):
                x = X_draws[:, i, kk, j].astype(float)
                mean = float(np.mean(x))
                lo, hi = quantile_ci(x)

                ratio = np.divide(x, A_ck_draw, out=np.zeros_like(x), where=(A_ck_draw > 0))
                r_mean = float(np.mean(ratio))
                r_lo, r_hi = quantile_ci(ratio)

                rows.append(
                    {
                        "country": country,
                        "ref_class": int(wet_class),
                        "class_label": CLASS_LABELS.get(int(wet_class), str(int(wet_class))),
                        "disturbance": int(disturbance),
                        "disturbance_label": DISTURBANCE_LABELS[int(disturbance)],
                        "est_ha_mean": mean,
                        "est_ha_ci_lo": lo,
                        "est_ha_ci_hi": hi,
                        "share_within_class_mean": r_mean,
                        "share_within_class_ci_lo": r_lo,
                        "share_within_class_ci_hi": r_hi,
                    }
                )

    return pd.DataFrame(rows).sort_values(["country", "ref_class", "disturbance"]).reset_index(drop=True)


def summarize_targets(
    disturbed_draws: np.ndarray,
    peat_disturbed_draws: np.ndarray,
    countries: list[str],
    target_fraction: float,
) -> pd.DataFrame:
    """
    Summarize disturbed area and target area by country.
    """
    target_draws = target_fraction * disturbed_draws
    peat_target_draws = target_fraction * peat_disturbed_draws

    rows = []

    for i, country in enumerate(countries):
        disturbed = disturbed_draws[:, i].astype(float)
        target = target_draws[:, i].astype(float)
        peat_disturbed = peat_disturbed_draws[:, i].astype(float)
        peat_target = peat_target_draws[:, i].astype(float)

        rows.append(
            {
                "country": country,
                "disturbed_ha_mean": float(np.mean(disturbed)),
                "disturbed_ha_ci_lo": float(np.quantile(disturbed, 0.025)),
                "disturbed_ha_ci_hi": float(np.quantile(disturbed, 0.975)),
                "target_ha_mean": float(np.mean(target)),
                "target_ha_ci_lo": float(np.quantile(target, 0.025)),
                "target_ha_ci_hi": float(np.quantile(target, 0.975)),
                "peatbog_disturbed_ha_mean": float(np.mean(peat_disturbed)),
                "peatbog_disturbed_ha_ci_lo": float(np.quantile(peat_disturbed, 0.025)),
                "peatbog_disturbed_ha_ci_hi": float(np.quantile(peat_disturbed, 0.975)),
                "peatbog_target_ha_mean": float(np.mean(peat_target)),
                "peatbog_target_ha_ci_lo": float(np.quantile(peat_target, 0.025)),
                "peatbog_target_ha_ci_hi": float(np.quantile(peat_target, 0.975)),
            }
        )

    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def summarize_eu_ratio_sanity(X_draws: np.ndarray) -> pd.DataFrame:
    """
    Summarize calibrated EU class × disturbance shares from the posterior draws.
    """
    rows = []

    EU_kd_draws = X_draws.sum(axis=1)  # sum countries -> (draw, class, dist)

    for kk, wet_class in enumerate(WET_CLASSES):
        class_total = EU_kd_draws[:, kk, :].sum(axis=1)

        for j, disturbance in enumerate(DISTURBANCE_LEVELS):
            x = EU_kd_draws[:, kk, j]
            ratio = np.divide(x, class_total, out=np.zeros_like(x), where=(class_total > 0))

            rows.append(
                {
                    "ref_class": int(wet_class),
                    "class_label": CLASS_LABELS.get(int(wet_class), str(int(wet_class))),
                    "disturbance": int(disturbance),
                    "disturbance_label": DISTURBANCE_LABELS[int(disturbance)],
                    "eu_share_mean": float(np.mean(ratio)),
                    "eu_share_ci_lo": float(np.quantile(ratio, 0.025)),
                    "eu_share_ci_hi": float(np.quantile(ratio, 0.975)),
                }
            )

    return pd.DataFrame(rows).sort_values(["ref_class", "disturbance"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eu_class_dist_mean, eu_class_dist_se = load_eu_design_class_disturbance(args.eu_design_file)

    countries, country_class_mean, country_class_se = load_country_class_totals(args.country_class_file)

    mapped_df, mapped_share_lookup = load_mapped_country_class_disturbance(
        filepath=args.mapped_file,
        countries=countries,
    )

    X_draws, disturbed_draws, peat_disturbed_draws = simulate_allocations(
        countries=countries,
        country_class_mean=country_class_mean,
        country_class_se=country_class_se,
        eu_class_dist_mean=eu_class_dist_mean,
        eu_class_dist_se=eu_class_dist_se,
        mapped_share_lookup=mapped_share_lookup,
        mapped_df=mapped_df,
        n_draws=args.n_draws,
        peat_class=args.peat_class,
        max_iters=args.max_iters,
        tol_rel=args.tol_rel,
        random_seed=args.random_seed,
    )

    allocation_df = summarize_allocations(
        X_draws=X_draws,
        countries=countries,
    )

    target_df = summarize_targets(
        disturbed_draws=disturbed_draws,
        peat_disturbed_draws=peat_disturbed_draws,
        countries=countries,
        target_fraction=args.target_fraction,
    )

    eu_ratio_df = summarize_eu_ratio_sanity(X_draws)

    allocation_fp = args.out_dir / "country_class_disturbance_allocations.csv"
    target_fp = args.out_dir / "country_disturbance_targets.csv"
    eu_ratio_fp = args.out_dir / "eu_class_disturbance_ratio_sanity.csv"

    allocation_df.to_csv(allocation_fp, index=False)
    target_df.to_csv(target_fp, index=False)
    eu_ratio_df.to_csv(eu_ratio_fp, index=False)

    print(f"Saved: {allocation_fp}")
    print(f"Saved: {target_fp}")
    print(f"Saved: {eu_ratio_fp}")
    print()
    print(f"Countries allocated: {len(countries)}")
    print(f"Monte Carlo draws: {args.n_draws}")
    print(f"Target fraction: {args.target_fraction:.2f}")


if __name__ == "__main__":
    main()
