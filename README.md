# Wetland Area Inference

This repository contains the analysis scripts used to implement the methods described in:

**Highly fragmented European wetlands with uneven restoration needs**  
Gyula Mate Kovács, Xiaoye Tong, Dimitri Gominski, Stefan Oehmcke, Stéphanie Horion, Christin Abel, Eva Ivits, Guy Schurgers, Bo Elberling, Alexander Prishchepov, Sebastian van der Linden, Susan Page, Alexandra Barthelmes, Franziska Tanneberger, and Rasmus Fensholt.  


The scripts correspond to the following methodological components of the paper:

- European-scale design-based area and accuracy estimation
- Country-level wetland area estimation
- Wetland disturbance area estimation

---

## Repository contents

### 1. `eu_design_based_area_estimation.py`
Implements the European-scale design-based area and accuracy estimation.

This script estimates:
- EU-wide reference-class areas
- the estimated area matrix
- producer’s accuracy, user’s accuracy, and overall accuracy

It uses an independent stratified validation sample and known stratum areas.

#### Inputs
- `--sample-file`
  Vector file (e.g. `.gpkg`, `.shp`) containing validation points with columns:
  - `Code_18`
  - `reference`
  - `Class`

- `--strata-file`
  CSV file containing known stratum areas with columns:
  - `Code_18`
  - one of:
    - `fty_zero_m2`
    - `fty_zero_ha`
    - `area_m2`

- `--out-dir`
  Output directory

#### Run
python eu_design_based_area_estimation.py \
  --sample-file /path/to/validation_points.gpkg \
  --strata-file /path/to/stratum_areas.csv \
  --out-dir /path/to/output_dir 

#### Outputs
- `eu_sample_based_areas.csv`
- `eu_sample_based_area_matrix.csv`

---

### 2. `country_calibrated_hierarchical_area_estimation.py`
Implements the country-level wetland area estimation.

This script estimates country-level wetland class areas from the continental stratified reference sample using:
- stratified base weights
- calibration by raking to country and stratum totals
- weighted country pseudo-counts
- hierarchical partial pooling within macro-regions
- posterior area summaries from Dirichlet draws

#### Inputs
- `--sample-file`
  Vector file containing validation points with columns:
  - `CNTR_ID`
  - `Code_18`
  - `reference`

- `--strata-file`
  CSV file containing country-by-stratum areas with columns:
  - `CNTR_ID`
  - `Code_18`
  - one of:
    - `fty_zero_m2`
    - `fty_zero_ha`
    - `area_m2`

- `--out-dir`
  Output directory

#### Run
python country_calibrated_hierarchical_area_estimation.py \
  --sample-file /path/to/validation_points_with_country.gpkg \
  --strata-file /path/to/country_stratum_areas.csv \
  --out-dir /path/to/output_dir 

#### Outputs
- `country_calibrated_hierarchical_area_estimates.csv`
- `country_weighted_pseudocounts.csv`
- `region_mean_compositions.csv`
- `country_weight_diagnostics.csv`

---

### 3. `eu_design_based_disturbance_estimation.py`
Implements the European-scale wetland disturbance area estimation.

This script estimates:
- total wetland area
- wetland reference class × disturbance areas
- disturbance totals within the wetland domain

The estimator is design-based and uses the same stratified indicator framework as the EU-wide area estimation.

#### Inputs
- `--sample-file`
  Vector file containing validation points with columns:
  - `Code_18`
  - `reference`
  - `distur`

- `--strata-file`
  CSV file containing known stratum areas with columns:
  - `Code_18`
  - one of:
    - `fty_zero_m2`
    - `fty_zero_ha`
    - `area_m2`

- `--out-dir`
  Output directory

#### Run
python eu_design_based_disturbance_estimation.py \
  --sample-file /path/to/validation_points_with_disturbance.gpkg \
  --strata-file /path/to/stratum_areas.csv \
  --out-dir /path/to/output_dir 

#### Outputs
- `eu_class_disturbance_estimates.csv`
- `eu_disturbance_totals.csv`

---

### 4. `country_calibrated_hierarchical_disturbance_allocation.py`
Implements the country-level wetland disturbance area estimation.

This script allocates country-level wetland class totals into country × reference-class × disturbance cells using:
- country-level calibrated hierarchical class totals
- EU design-based class × disturbance estimates
- mapped within-country disturbance shares as auxiliary information
- iterative proportional fitting for calibrated allocation
- Monte Carlo simulation to propagate uncertainty

#### Inputs
- `--mapped-file`
  CSV containing mapped country × class × disturbance areas with columns including:
  - `country` or `CNTR_ID`
  - `class` or `ref_class`
  - `disturbance` or `distur`
  - mapped area column such as:
    - `area_ha`
    - `mapped_area_ha`
    - `area`
    - `A_ha`
    - `ha`
    - `area_m2`
    - `A_m2`

- `--eu-design-file`
  CSV with EU design-based reference-class × disturbance estimates, typically:
  - `eu_class_disturbance_estimates.csv`

- `--country-class-file`
  CSV with country-level class totals, typically:
  - `country_calibrated_hierarchical_area_estimates.csv`

- `--out-dir`
  Output directory

#### Run
python country_calibrated_hierarchical_disturbance_allocation.py \
  --mapped-file /path/to/country_class_disturbance_mapped.csv \
  --eu-design-file /path/to/eu_class_disturbance_estimates.csv \
  --country-class-file /path/to/country_calibrated_hierarchical_area_estimates.csv \
  --out-dir /path/to/output_dir 

#### Outputs
- `country_class_disturbance_allocations.csv`
- `country_disturbance_targets.csv`
- `eu_class_disturbance_ratio_sanity.csv`

---

## Input data summary

### Validation point files
Vector files used by the EU-wide and country-level estimation scripts should contain the fields required by the relevant script, including some subset of:

- `Code_18`
- `reference`
- `Class`
- `distur`
- `CNTR_ID`

### Stratum area files
CSV files containing known stratum areas should include:
- `Code_18`
- one area field:
  - `fty_zero_m2`
  - `fty_zero_ha`
  - `area_m2`

For country-level wetland area estimation, the stratum area file must also include:
- `CNTR_ID`

### Disturbance allocation inputs
The country disturbance allocation script requires three aggregated CSV inputs:
1. mapped country × class × disturbance areas
2. EU design-based class × disturbance estimates
3. country-level class totals from the calibrated hierarchical area estimation

---

## Python dependencies

The scripts use:

- `numpy`
- `pandas`
- `geopandas`
- `scipy`

Install them with:

pip install numpy pandas geopandas scipy

---

## Methodological scope

These scripts implement the estimation workflows described in the paper’s Methods section for:
- European-scale design-based wetland area and accuracy estimation
- Country-level wetland area estimation
- Wetland disturbance area estimation

The repository is intended to document the estimation framework used in the study. It is not a full reproduction package for all manuscript figures, manuscript text, or downstream carbon analyses.

---

## Citation

Kovács, G. M., Tong, X., Gominski, D., Oehmcke, S., Horion, S., Abel, C., Ivits, E., Schurgers, G., Elberling, B., Prishchepov, A., van der Linden, S., Page, S., Barthelmes, A., Tanneberger, F., & Fensholt, R. (2026). Highly fragmented European wetlands with uneven restoration needs. Nature.

## References

Deville, J.-C., Särndal, C.-E., & Sautory, O. (1993). Generalized Raking Procedures in Survey Sampling. Journal of the American Statistical Association, 88(423), 1013–1020.

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2014). Bayesian Data Analysis (3rd ed.). CRC Press.

Särndal, C.-E., Swensson, B., & Wretman, J. (1992). Model Assisted Survey Sampling. Springer.

Stehman, S. V. (2014). Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes. International Journal of Remote Sensing, 35, 4923–4939.
