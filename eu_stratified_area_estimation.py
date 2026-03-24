#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EU-wide design-based area and accuracy estimation.

This script estimates class areas and accuracy metrics from an
independent stratified validation sample under design-based inference.

The implementation follows indicator-function estimators for stratified
sampling when the sampling strata are not necessarily identical to the
final map classes.

Core estimators
---------------
For reference class k:

    p_hat_hk = (1 / n_h) * sum_{i in h} I(y_i = k)

    A_hat_k = sum_h A_h * p_hat_hk

    Var(A_hat_k) = sum_h A_h^2 * (1 - n_h / N_h) * s_hk^2 / n_h

    s_hk^2 = [n_h / (n_h - 1)] * p_hat_hk * (1 - p_hat_hk)

For estimated area-matrix cell (reference class r, map class c):

    p_hat_hrc = (1 / n_h) * sum_{i in h} I(y_i = r, m_i = c)

    A_hat_rc = sum_h A_h * p_hat_hrc

Accuracy metrics are derived from the estimated area matrix:

    OA   = sum_k A_hat_kk / sum_r sum_c A_hat_rc
    PA_k = A_hat_kk / sum_c A_hat_kc
    UA_k = A_hat_kk / sum_r A_hat_rk

References
----------
Stehman, S.V. (2014). Estimating area and map accuracy for stratified
random sampling when the strata are different from the map classes.
International Journal of Remote Sensing, 35.

Stehman, S.V. (2009). Sampling designs for accuracy assessment of land
cover. Remote Sensing of Environment, 113, 2455-2462.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd


PIXEL_AREA_M2 = 100.0
M2_PER_HA = 10_000.0
Z_95 = 1.96
URBAN_COLLAPSE_CODE = 100

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EU-wide design-based area and accuracy estimation from a stratified validation sample."
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        required=True,
        help="Vector file with validation points and columns: Code_18, reference, reference_new, Class.",
    )
    parser.add_argument(
        "--strata-file",
        type=Path,
        required=True,
        help="CSV file with known stratum areas by Code_18.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for CSV tables.",
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
    return parser.parse_args()


def collapse_code18(code: int) -> int:
    """Collapse all CLC 1xx strata to a single design stratum."""
    code = int(code)
    return URBAN_COLLAPSE_CODE if (code // 100) == 1 else code


def to_int_series(series: pd.Series, name: str) -> pd.Series:
    """Coerce a series to integer values; invalid rows become missing."""
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().any():
        n_bad = int(out.isna().sum())
        print(f"[WARN] {n_bad} rows contain invalid values in '{name}' and will be dropped.")
    return out.astype("Int64")


def infer_area_m2(df: pd.DataFrame) -> pd.Series:
    """
    Infer the stratum area column and return area in m^2.

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


def sample_variance_bernoulli(p_hat: np.ndarray, n_h: np.ndarray) -> np.ndarray:
    """
    Sample variance of a 0/1 indicator within stratum h:

        s_h^2 = [n_h / (n_h - 1)] * p_hat_h * (1 - p_hat_h)

    Used in the stratified variance estimator with finite population correction;
    see Stehman (2014), Eqs. 25-26.
    """
    out = np.zeros_like(p_hat, dtype=float)
    mask = n_h > 1
    out[mask] = (n_h[mask] / (n_h[mask] - 1.0)) * p_hat[mask] * (1.0 - p_hat[mask])
    return out


def finite_population_correction(n_h: np.ndarray, N_h: np.ndarray) -> np.ndarray:
    """Compute 1 - n_h / N_h, clipped at zero for numerical safety."""
    return np.clip(1.0 - (n_h / N_h), a_min=0.0, a_max=None)


def load_samples(
    sample_file: Path,
    n_classes: int,
    exclude_strata: Iterable[int],
) -> pd.DataFrame:
    """
    Load validation points and prepare:

        h   = design stratum
        ref = reference label
        map = map label

    If reference_new is present, it takes precedence over reference.
    """
    gdf = gpd.read_file(sample_file)

    required = {"Code_18", "reference", "reference_new", "Class"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"Sample file is missing required columns: {sorted(missing)}")

    gdf["h"] = to_int_series(gdf["Code_18"], "Code_18")
    gdf = gdf.dropna(subset=["h"]).copy()
    gdf["h"] = gdf["h"].astype(int).map(collapse_code18).astype(int)

    exclude_strata = set(exclude_strata)
    if exclude_strata:
        gdf = gdf.loc[~gdf["h"].isin(exclude_strata)].copy()

    ref_new = pd.to_numeric(gdf["reference_new"], errors="coerce")
    ref_old = pd.to_numeric(gdf["reference"], errors="coerce")
    gdf["ref"] = ref_new.where(ref_new.notna(), ref_old)

    gdf["map"] = pd.to_numeric(gdf["Class"], errors="coerce")

    gdf = gdf.dropna(subset=["ref", "map"]).copy()
    gdf["ref"] = gdf["ref"].astype(int)
    gdf["map"] = gdf["map"].astype(int)

    if not gdf["ref"].between(0, n_classes - 1).all():
        bad = np.sort(gdf.loc[~gdf["ref"].between(0, n_classes - 1), "ref"].unique())
        raise ValueError(f"Reference labels outside 0..{n_classes - 1}: {bad}")

    if not gdf["map"].between(0, n_classes - 1).all():
        bad = np.sort(gdf.loc[~gdf["map"].between(0, n_classes - 1), "map"].unique())
        raise ValueError(f"Map labels outside 0..{n_classes - 1}: {bad}")

    return pd.DataFrame(gdf)


def load_strata(
    strata_file: Path,
    exclude_strata: Iterable[int],
) -> pd.DataFrame:
    """
    Load known stratum areas and return:

        h      = design stratum
        A_h_m2 = known stratum area
        N_h    = stratum population size in pixels
    """
    df = pd.read_csv(strata_file)
    df.columns = [c.strip() for c in df.columns]

    if "Code_18" not in df.columns:
        raise ValueError("Strata file must contain a 'Code_18' column.")

    df["A_h_m2"] = infer_area_m2(df)
    df["h"] = pd.to_numeric(df["Code_18"], errors="coerce")

    df = df.dropna(subset=["h", "A_h_m2"]).copy()
    df["h"] = df["h"].astype(int).map(collapse_code18).astype(int)
    df["A_h_m2"] = df["A_h_m2"].astype(float)

    exclude_strata = set(exclude_strata)
    if exclude_strata:
        df = df.loc[~df["h"].isin(exclude_strata)].copy()

    # Needed because multiple original 1xx classes collapse into stratum 100.
    df = df.groupby("h", as_index=False)["A_h_m2"].sum()
    df["N_h"] = df["A_h_m2"] / PIXEL_AREA_M2

    return df


def estimate_reference_areas(
    samples: pd.DataFrame,
    strata: pd.DataFrame,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate reference-class area and variance.

    Indicator-function estimator for class area:
        A_hat_k = sum_h A_h * p_hat_hk

    with variance:
        Var(A_hat_k) = sum_h A_h^2 * (1 - n_h / N_h) * s_hk^2 / n_h

    This is the stratified mean estimator applied to a 0/1 indicator
    for membership in reference class k; see Stehman (2014).
    """
    df = samples.merge(strata[["h", "A_h_m2", "N_h"]], on="h", how="left")

    if df["N_h"].isna().any():
        missing_h = np.sort(df.loc[df["N_h"].isna(), "h"].unique())
        raise ValueError(f"Known stratum areas missing for sampled strata: {missing_h}")

    grp = df.groupby("h", sort=False)
    n_h = grp.size().rename("n_h")
    first = grp[["A_h_m2", "N_h"]].first()
    design_df = pd.concat([n_h, first], axis=1).reset_index()

    Ahat_ref_m2 = np.zeros(n_classes, dtype=float)
    Var_Ahat_ref_m2 = np.zeros(n_classes, dtype=float)

    for k in range(n_classes):
        p_hk = grp.apply(lambda d: (d["ref"] == k).mean()).rename("p_hk").reset_index()
        tmp = design_df.merge(p_hk, on="h", how="left")

        n = tmp["n_h"].to_numpy(dtype=float)
        A = tmp["A_h_m2"].to_numpy(dtype=float)
        N = tmp["N_h"].to_numpy(dtype=float)
        p = tmp["p_hk"].to_numpy(dtype=float)

        s2 = sample_variance_bernoulli(p, n)
        fpc = finite_population_correction(n, N)

        Ahat_ref_m2[k] = np.sum(A * p)
        Var_Ahat_ref_m2[k] = np.sum((A ** 2) * fpc * (s2 / n))

    return Ahat_ref_m2, Var_Ahat_ref_m2


def estimate_area_matrix(
    samples: pd.DataFrame,
    strata: pd.DataFrame,
    n_classes: int,
) -> np.ndarray:
    """
    Estimate the population area matrix A_hat_rc.

    For each cell (reference class r, map class c):

        p_hat_hrc = (1 / n_h) * sum_{i in h} I(y_i = r, m_i = c)
        A_hat_rc  = sum_h A_h * p_hat_hrc

    This follows the indicator-based formulation for error-matrix cell
    proportions under stratified sampling; see Stehman (2014).
    """
    df = samples.merge(strata[["h", "A_h_m2", "N_h"]], on="h", how="left")
    grp = df.groupby("h", sort=False)

    n_h = grp.size().rename("n_h")
    first = grp[["A_h_m2", "N_h"]].first()
    design_df = pd.concat([n_h, first], axis=1).reset_index()

    Ahat_rc_m2 = np.zeros((n_classes, n_classes), dtype=float)

    for r in range(n_classes):
        for c in range(n_classes):
            p_hrc = grp.apply(
                lambda d: ((d["ref"] == r) & (d["map"] == c)).mean()
            ).rename("p_hrc").reset_index()

            tmp = design_df.merge(p_hrc, on="h", how="left")
            Ahat_rc_m2[r, c] = np.sum(tmp["A_h_m2"] * tmp["p_hrc"])

    return Ahat_rc_m2


def derive_accuracy_metrics(Ahat_rc_ha: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Derive producer's accuracy, user's accuracy, and overall accuracy
    from the estimated area matrix.

    Convention:
        rows    = reference classes
        columns = map classes

        PA_k = A_hat_kk / sum_c A_hat_kc
        UA_k = A_hat_kk / sum_r A_hat_rk

    This row/column convention matches the direct area-estimation framing
    commonly used in land-cover accuracy assessment; see Stehman (2009).
    """
    ref_marginal = Ahat_rc_ha.sum(axis=1)
    map_marginal = Ahat_rc_ha.sum(axis=0)
    diag = np.diag(Ahat_rc_ha)

    PA = np.divide(diag, ref_marginal, out=np.full_like(diag, np.nan), where=ref_marginal > 0)
    UA = np.divide(diag, map_marginal, out=np.full_like(diag, np.nan), where=map_marginal > 0)

    total_area = Ahat_rc_ha.sum()
    OA = float(diag.sum() / total_area) if total_area > 0 else np.nan

    return PA, UA, OA


def build_summary_table(
    Ahat_ref_m2: np.ndarray,
    Var_Ahat_ref_m2: np.ndarray,
    PA: np.ndarray,
    UA: np.ndarray,
    class_labels: Dict[int, str],
) -> pd.DataFrame:
    """Build class-level summary table with area, SE, CI, PA, and UA."""
    Ahat_ref_ha = Ahat_ref_m2 / M2_PER_HA
    SE_ref_ha = np.sqrt(Var_Ahat_ref_m2) / M2_PER_HA
    CI95_lo = Ahat_ref_ha - Z_95 * SE_ref_ha
    CI95_hi = Ahat_ref_ha + Z_95 * SE_ref_ha

    rows = []
    for k in range(len(Ahat_ref_ha)):
        rows.append(
            {
                "class_id": k,
                "class_label": class_labels.get(k, f"class_{k}"),
                "sample_based_area_ha": Ahat_ref_ha[k],
                "se_ha": SE_ref_ha[k],
                "ci95_lo_ha": CI95_lo[k],
                "ci95_hi_ha": CI95_hi[k],
                "producer_accuracy": PA[k],
                "user_accuracy": UA[k],
            }
        )

    return pd.DataFrame(rows)


def build_matrix_table(Ahat_rc_ha: np.ndarray, class_labels: Dict[int, str]) -> pd.DataFrame:
    """Flatten the estimated area matrix to long format."""
    rows = []
    n_classes = Ahat_rc_ha.shape[0]

    for r in range(n_classes):
        for c in range(n_classes):
            rows.append(
                {
                    "ref_class": r,
                    "map_class": c,
                    "ref_label": class_labels.get(r, f"class_{r}"),
                    "map_label": class_labels.get(c, f"class_{c}"),
                    "area_ha": Ahat_rc_ha[r, c],
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(
        sample_file=args.sample_file,
        n_classes=args.n_classes,
        exclude_strata=args.exclude_strata,
    )
    strata = load_strata(
        strata_file=args.strata_file,
        exclude_strata=args.exclude_strata,
    )

    Ahat_ref_m2, Var_Ahat_ref_m2 = estimate_reference_areas(
        samples=samples,
        strata=strata,
        n_classes=args.n_classes,
    )

    Ahat_rc_m2 = estimate_area_matrix(
        samples=samples,
        strata=strata,
        n_classes=args.n_classes,
    )
    Ahat_rc_ha = Ahat_rc_m2 / M2_PER_HA

    PA, UA, OA = derive_accuracy_metrics(Ahat_rc_ha)

    df_summary = build_summary_table(
        Ahat_ref_m2=Ahat_ref_m2,
        Var_Ahat_ref_m2=Var_Ahat_ref_m2,
        PA=PA,
        UA=UA,
        class_labels=CLASS_LABELS,
    )

    df_matrix = build_matrix_table(
        Ahat_rc_ha=Ahat_rc_ha,
        class_labels=CLASS_LABELS,
    )

    summary_fp = args.out_dir / "eu_sample_based_areas.csv"
    matrix_fp = args.out_dir / "eu_sample_based_area_matrix.csv"

    df_summary.to_csv(summary_fp, index=False)
    df_matrix.to_csv(matrix_fp, index=False)

    print(f"Saved: {summary_fp}")
    print(f"Saved: {matrix_fp}")
    print()
    print(df_summary.to_string(index=False))
    print()
    print(f"Overall accuracy (OA): {OA:.4f}")
    print(f"Total sample size: {len(samples):,d}")
    print(f"Design strata represented: {samples['h'].nunique()}")


if __name__ == "__main__":
    main()
