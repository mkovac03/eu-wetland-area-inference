#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EU-wide design-based disturbance estimation.

This script estimates disturbance-domain areas within the wetland domain
from an independent stratified validation sample under design-based inference.

The implementation treats disturbance levels as reporting domains within
wetlands and applies the stratified indicator estimator to:
    1) wetland total area,
    2) wetland class × disturbance areas, and
    3) disturbance-only totals within wetlands.

Core estimators
---------------
For any binary indicator y_i defining a reporting domain:

    p_hat_h = (1 / n_h) * sum_{i in h} y_i

    A_hat = sum_h A_h * p_hat_h

    Var(A_hat) = sum_h A_h^2 * (1 - n_h / N_h) * s_h^2 / n_h

    s_h^2 = [n_h / (n_h - 1)] * p_hat_h * (1 - p_hat_h)

This is the standard stratified estimator for totals of indicator variables.
Here it is applied to disturbance domains within the wetland population.

References
----------
Stehman, S.V. (2014). Estimating area and map accuracy for stratified
random sampling when the strata are different from the map classes.
International Journal of Remote Sensing, 35.

Stehman, S.V. (2009). Sampling designs for accuracy assessment of land
cover. Remote Sensing of Environment, 113, 2455-2462.

Särndal, Swensson, and Wretman (1992), Model Assisted Survey Sampling,
Chapter 10, for domain estimation concepts.
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
URBAN_STRATUM_CODE = 100

WET_CLASSES = [1, 2, 3, 4, 5, 6]
DISTUR_LEVELS = [0, 1, 2]
UNKNOWN_DISTUR_CODE = -1

DISTUR_LABELS: Dict[int, str] = {
    0: "least disturbed",
    1: "intermediately disturbed",
    2: "most disturbed",
}

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
        description="EU-wide design-based disturbance estimation within the wetland domain."
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        required=True,
        help="Vector file with validation points and columns: Code_18, reference, distur.",
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
        "--include-unknown",
        action="store_true",
        help="If set, include disturbance = -1 as a separate diagnostic output.",
    )
    parser.add_argument(
        "--exclude-strata",
        type=int,
        nargs="*",
        default=[],
        help="Optional strata to exclude from the analysis.",
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


def stratified_indicator_total(df: pd.DataFrame, y: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Estimate the total area for a binary reporting-domain indicator y.

        A_hat   = sum_h A_h * p_hat_h
        Var(A_hat) = sum_h A_h^2 * (1 - n_h / N_h) * s_h^2 / n_h

    This is the stratified indicator estimator for totals; see Stehman (2014).
    """
    g = df.copy()
    g["y"] = y.astype(int)

    grp = g.groupby("h", sort=False)
    n_h = grp.size().rename("n_h")
    p_h = grp["y"].mean().rename("p_h")
    first = grp[["A_h_m2", "N_h"]].first()

    tmp = pd.concat([n_h, p_h, first], axis=1).reset_index()

    Ahat_m2 = np.sum(tmp["A_h_m2"] * tmp["p_h"])

    n = tmp["n_h"].to_numpy(dtype=float)
    A = tmp["A_h_m2"].to_numpy(dtype=float)
    N = tmp["N_h"].to_numpy(dtype=float)
    p = tmp["p_h"].to_numpy(dtype=float)

    s2 = sample_variance_bernoulli(p, n)
    fpc = finite_population_correction(n, N)
    Var_m2 = np.sum((A ** 2) * fpc * (s2 / n))

    Ahat_ha = Ahat_m2 / M2_PER_HA
    Var_ha2 = Var_m2 / (M2_PER_HA ** 2)
    SE_ha = np.sqrt(Var_ha2)
    ci_lo = Ahat_ha - Z_95 * SE_ha
    ci_hi = Ahat_ha + Z_95 * SE_ha

    return float(Ahat_ha), float(Var_ha2), float(SE_ha), float(ci_lo), float(ci_hi)


def load_samples(
    sample_file: Path,
    exclude_strata: Iterable[int],
) -> pd.DataFrame:
    """
    Load validation points and prepare:

        h      = stratum
        ref    = reference label
        distur = disturbance label
    """
    gdf = gpd.read_file(sample_file)

    required = {"Code_18", "reference", "distur"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"Sample file is missing required columns: {sorted(missing)}")

    gdf["h"] = to_int_series(gdf["Code_18"], "Code_18")
    gdf["ref"] = pd.to_numeric(gdf["reference"], errors="coerce")
    gdf["distur"] = pd.to_numeric(gdf["distur"], errors="coerce")

    gdf = gdf.dropna(subset=["h", "ref", "distur"]).copy()
    gdf["h"] = gdf["h"].astype(int).map(recode_code18_to_stratum).astype(int)
    gdf["ref"] = gdf["ref"].astype(int)
    gdf["distur"] = gdf["distur"].astype(int)

    exclude_strata = set(exclude_strata)
    if exclude_strata:
        gdf = gdf.loc[~gdf["h"].isin(exclude_strata)].copy()

    return pd.DataFrame(gdf[["h", "ref", "distur"]])


def load_strata(
    strata_file: Path,
    exclude_strata: Iterable[int],
) -> pd.DataFrame:
    """
    Load known stratum areas and return:

        h      = stratum
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
    df["h"] = df["h"].astype(int).map(recode_code18_to_stratum).astype(int)
    df["A_h_m2"] = df["A_h_m2"].astype(float)

    exclude_strata = set(exclude_strata)
    if exclude_strata:
        df = df.loc[~df["h"].isin(exclude_strata)].copy()

    # Needed because multiple original codes may be reassigned to the same stratum.
    df = df.groupby("h", as_index=False)["A_h_m2"].sum()
    df["N_h"] = df["A_h_m2"] / PIXEL_AREA_M2

    return df


def build_class_disturbance_table(
    df: pd.DataFrame,
    include_unknown: bool,
) -> pd.DataFrame:
    """
    Estimate wetland class × disturbance areas within the wetland domain.
    """
    rows = []

    is_wet = df["ref"].isin(WET_CLASSES)

    A_wet, Var_wet, SE_wet, lo_wet, hi_wet = stratified_indicator_total(df, is_wet.to_numpy())
    rows.append({
        "ref_class": -999,
        "class_label": "Wetlands total",
        "distur": -999,
        "distur_label": "all",
        "est_ha": A_wet,
        "var_ha2": Var_wet,
        "se_ha": SE_wet,
        "ci_lo_ha": lo_wet,
        "ci_hi_ha": hi_wet,
        "n_points_total": int(len(df)),
        "n_points_in_cell": int(is_wet.sum()),
    })

    for k in WET_CLASSES:
        for d in DISTUR_LEVELS:
            y = (df["ref"].eq(k) & df["distur"].eq(d)).to_numpy()
            Ahat, Varhat, SEhat, lo, hi = stratified_indicator_total(df, y)

            rows.append({
                "ref_class": k,
                "class_label": CLASS_LABELS.get(k, str(k)),
                "distur": d,
                "distur_label": DISTUR_LABELS.get(d, str(d)),
                "est_ha": Ahat,
                "var_ha2": Varhat,
                "se_ha": SEhat,
                "ci_lo_ha": lo,
                "ci_hi_ha": hi,
                "n_points_total": int(len(df)),
                "n_points_in_cell": int(y.sum()),
            })

    if include_unknown:
        n_wet_unknown = int((is_wet & (df["distur"] == UNKNOWN_DISTUR_CODE)).sum())
        if n_wet_unknown > 0:
            y = (is_wet & df["distur"].eq(UNKNOWN_DISTUR_CODE)).to_numpy()
            Ahat, Varhat, SEhat, lo, hi = stratified_indicator_total(df, y)

            rows.append({
                "ref_class": -998,
                "class_label": "Wetlands",
                "distur": UNKNOWN_DISTUR_CODE,
                "distur_label": "unknown disturbance (-1)",
                "est_ha": Ahat,
                "var_ha2": Varhat,
                "se_ha": SEhat,
                "ci_lo_ha": lo,
                "ci_hi_ha": hi,
                "n_points_total": int(len(df)),
                "n_points_in_cell": int(y.sum()),
            })

    return pd.DataFrame(rows)


def build_disturbance_totals_table(
    df: pd.DataFrame,
    include_unknown: bool,
) -> pd.DataFrame:
    """
    Estimate disturbance-only totals within wetlands.
    """
    rows = []
    is_wet = df["ref"].isin(WET_CLASSES)

    A_wet, Var_wet, SE_wet, lo_wet, hi_wet = stratified_indicator_total(df, is_wet.to_numpy())
    rows.append({
        "distur": -999,
        "distur_label": "all wetlands",
        "est_ha": A_wet,
        "var_ha2": Var_wet,
        "se_ha": SE_wet,
        "ci_lo_ha": lo_wet,
        "ci_hi_ha": hi_wet,
        "n_points_in_cell": int(is_wet.sum()),
    })

    for d in DISTUR_LEVELS:
        y = (is_wet & df["distur"].eq(d)).to_numpy()
        Ahat, Varhat, SEhat, lo, hi = stratified_indicator_total(df, y)
        rows.append({
            "distur": d,
            "distur_label": DISTUR_LABELS.get(d, str(d)),
            "est_ha": Ahat,
            "var_ha2": Varhat,
            "se_ha": SEhat,
            "ci_lo_ha": lo,
            "ci_hi_ha": hi,
            "n_points_in_cell": int(y.sum()),
        })

    if include_unknown:
        n_wet_unknown = int((is_wet & (df["distur"] == UNKNOWN_DISTUR_CODE)).sum())
        if n_wet_unknown > 0:
            y = (is_wet & df["distur"].eq(UNKNOWN_DISTUR_CODE)).to_numpy()
            Ahat, Varhat, SEhat, lo, hi = stratified_indicator_total(df, y)
            rows.append({
                "distur": UNKNOWN_DISTUR_CODE,
                "distur_label": "unknown disturbance (-1)",
                "est_ha": Ahat,
                "var_ha2": Varhat,
                "se_ha": SEhat,
                "ci_lo_ha": lo,
                "ci_hi_ha": hi,
                "n_points_in_cell": int(y.sum()),
            })

    return pd.DataFrame(rows).sort_values("distur").reset_index(drop=True)


def print_diagnostics(df: pd.DataFrame) -> None:
    """Print basic wetland-domain disturbance diagnostics."""
    is_wet = df["ref"].isin(WET_CLASSES)
    n_wet = int(is_wet.sum())
    n_wet_unknown = int((is_wet & (df["distur"] == UNKNOWN_DISTUR_CODE)).sum())

    print("\n[DIAG] Wetland domain = ref in 1..6")
    print(f"  wet_points = {n_wet:,d}")
    print(f"  wet_points_with_unknown_disturbance = {n_wet_unknown:,d}")

    ct = (
        df.groupby(["ref", "distur"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["ref", "distur"])
    )
    print("\n[DIAG] Counts by (ref, distur):")
    print(ct.to_string(index=False))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(
        sample_file=args.sample_file,
        exclude_strata=args.exclude_strata,
    )
    strata = load_strata(
        strata_file=args.strata_file,
        exclude_strata=args.exclude_strata,
    )

    df = samples.merge(strata[["h", "A_h_m2", "N_h"]], on="h", how="left")
    if df["N_h"].isna().any():
        missing_h = np.sort(df.loc[df["N_h"].isna(), "h"].unique())
        raise ValueError(f"Known stratum areas missing for sampled strata: {missing_h}")

    print_diagnostics(df)

    class_dist_df = build_class_disturbance_table(
        df=df,
        include_unknown=args.include_unknown,
    )
    dist_tot_df = build_disturbance_totals_table(
        df=df,
        include_unknown=args.include_unknown,
    )

    class_dist_fp = args.out_dir / "eu_class_disturbance_estimates.csv"
    dist_tot_fp = args.out_dir / "eu_disturbance_totals.csv"

    class_dist_df.to_csv(class_dist_fp, index=False)
    dist_tot_df.to_csv(dist_tot_fp, index=False)

    print(f"\nSaved: {class_dist_fp}")
    print(f"Saved: {dist_tot_fp}")

    print("\nEU class × disturbance estimates (ha):")
    print(
        class_dist_df.loc[class_dist_df["ref_class"].isin(WET_CLASSES), [
            "ref_class", "class_label", "distur", "distur_label",
            "est_ha", "ci_lo_ha", "ci_hi_ha", "n_points_in_cell"
        ]].to_string(index=False)
    )

    print("\nEU disturbance-only totals within wetlands (ha):")
    print(
        dist_tot_df[[
            "distur", "distur_label", "est_ha", "ci_lo_ha", "ci_hi_ha", "n_points_in_cell"
        ]].to_string(index=False)
    )

    print(f"\nTotal sample size: {len(df):,d}")
    print(f"Strata represented: {df['h'].nunique()}")


if __name__ == "__main__":
    main()
