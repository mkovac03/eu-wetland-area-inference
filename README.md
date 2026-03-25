# EU Wetland Area Inference

This repository contains the analysis scripts used to implement the area-inference workflows described in:

**Highly fragmented European wetlands with uneven restoration needs**  
Gyula Mate Kovács, Xiaoye Tong, Dimitri Gominski, Stefan Oehmcke, Stéphanie Horion, Christin Abel, Eva Ivits, Guy Schurgers, Bo Elberling, Alexander Prishchepov, Sebastian van der Linden, Susan Page, Alexandra Barthelmes, Franziska Tanneberger, and Rasmus Fensholt.

The repository contains four main scripts covering:

- EU-wide design-based wetland area and accuracy estimation
- Country-level pooled wetland area estimation
- EU-wide design-based wetland disturbance estimation
- Country-level disturbance allocation and restoration target estimation

## Repository structure

```text
eu-wetland-area-inference/
├── environment.yml
├── eu_design_based_area_estimation.py
├── country_calibrated_hierarchical_area_estimation.py
├── eu_design_based_disturbance_estimation.py
├── country_calibrated_hierarchical_disturbance_allocation.py
├── source_data/
│   ├── eu_design_based_area_estimation_samples.gpkg
│   ├── eu_design_based_area_estimation_strata.csv
│   ├── country_calibrated_hierarchical_area_estimation_samples.gpkg
│   ├── country_calibrated_hierarchical_area_estimation_strata.csv
│   ├── country_calibrated_hierarchical_area_estimation_country_totals.csv
│   ├── eu_design_based_disturbance_estimation_samples.gpkg
│   ├── eu_design_based_disturbance_estimation_strata.csv
│   └── country_calibrated_hierarchical_disturbance_allocation_mapped.csv
└── outputs/
    ├── eu_design_based_area_estimation/
    ├── country_calibrated_hierarchical_area_estimation/
    ├── eu_design_based_disturbance_estimation/
    └── country_calibrated_hierarchical_disturbance_allocation/
```

## Environment setup

Create the Conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate eu-wetland-area
```

## Recommended execution order

Run the scripts in this order:

1. `eu_design_based_area_estimation.py`
2. `country_calibrated_hierarchical_area_estimation.py`
3. `eu_design_based_disturbance_estimation.py`
4. `country_calibrated_hierarchical_disturbance_allocation.py`

The fourth script depends on outputs produced by scripts 2 and 3.

## 1. EU-wide design-based area estimation

### Script
`eu_design_based_area_estimation.py`

### Purpose
Estimates:

- EU-wide reference-class areas
- the estimated area matrix
- producer's accuracy
- user's accuracy
- overall accuracy

This script uses an independent stratified validation sample and known stratum areas.

### Required inputs
- `--sample-file`  
  Vector file with:
  - `Code_18`
  - `reference`
  - `Class`

- `--strata-file`  
  CSV with:
  - `Code_18`
  - one area column among:
    - `fty_zero_m2`
    - `fty_zero_ha`
    - `area_m2`

- `--out-dir`  
  Output directory

### Example run
```bash
python eu_design_based_area_estimation.py \
  --sample-file source_data/eu_design_based_area_estimation_samples.gpkg \
  --strata-file source_data/eu_design_based_area_estimation_strata.csv \
  --out-dir outputs/eu_design_based_area_estimation
```

### Outputs
- `eu_sample_based_areas.csv`
- `eu_sample_based_area_matrix.csv`

---

## 2. Country-level pooled wetland area estimation

### Script
`country_calibrated_hierarchical_area_estimation.py`

### Purpose
Estimates country-level wetland class areas using:

- stratified base weights
- calibration by iterative proportional fitting (raking)
- weighted country reference pseudo-counts
- hierarchical pooling within macro-regions
- posterior area summaries from Dirichlet draws

### Required inputs
- `--sample-file`  
  Vector file with:
  - `CNTR_ID`
  - `Code_18`
  - `reference_final`

- `--strata-file`  
  CSV with:
  - `CNTR_ID`
  - `Code_18`
  - one area column among:
    - `fty_zero_m2`
    - `fty_zero_ha`
    - `area_m2`
    - `A_m2`
    - `A_ha`

- `--country-totals-file`  
  CSV with:
  - `CNTR_ID`
  - one country-total area column among:
    - `fty_zero_ha_total`
    - `fty_zero_ha`
    - `A_d_ha`
    - `A_ha`
    - `total_ha`

- `--out-dir`  
  Output directory

### Example run
```bash
python country_calibrated_hierarchical_area_estimation.py \
  --sample-file source_data/country_calibrated_hierarchical_area_estimation_samples.gpkg \
  --strata-file source_data/country_calibrated_hierarchical_area_estimation_strata.csv \
  --country-totals-file source_data/country_calibrated_hierarchical_area_estimation_country_totals.csv \
  --out-dir outputs/country_calibrated_hierarchical_area_estimation
```

### Outputs
- `country_pooled_wetland_area_estimates.csv`
- `eu_collapsed_stratum_totals.csv`
- `regional_reference_composition.csv`
- `country_weighted_reference_pseudocounts.csv`
- `country_to_region_lookup.csv`
- `diagnostic_region_reference_counts_raw.csv`
- `global_shrinkage_parameter.csv`

---

## 3. EU-wide design-based disturbance estimation

### Script
`eu_design_based_disturbance_estimation.py`

### Purpose
Estimates:

- total wetland area
- wetland class × disturbance areas
- disturbance totals within the wetland domain

The estimator is design-based and uses the same stratified indicator framework as the EU-wide area estimation.

### Required inputs
- `--sample-file`  
  Vector file with:
  - `Code_18`
  - `reference_final`
  - `distur`

- `--strata-file`  
  CSV with:
  - `Code_18`
  - one area column among:
    - `fty_zero_m2`
    - `fty_zero_ha`
    - `area_m2`

- `--out-dir`  
  Output directory

- `--include-unknown`  
  Optional flag to also report disturbance `-1` as a separate diagnostic category

### Example run
```bash
python eu_design_based_disturbance_estimation.py \
  --sample-file source_data/eu_design_based_disturbance_estimation_samples.gpkg \
  --strata-file source_data/eu_design_based_disturbance_estimation_strata.csv \
  --out-dir outputs/eu_design_based_disturbance_estimation \
  --include-unknown
```

### Outputs
- `eu_class_disturbance_estimates.csv`
- `eu_disturbance_totals.csv`

---

## 4. Country-level disturbance allocation and restoration targets

### Script
`country_calibrated_hierarchical_disturbance_allocation.py`

### Purpose
Allocates country-level wetland class totals into country × class × disturbance cells using:

- country-level pooled wetland class totals
- EU design-based class × disturbance estimates
- mapped within-country disturbance shares as auxiliary information
- iterative proportional fitting
- Monte Carlo uncertainty propagation

This script also produces country-level disturbed-area targets, including peatbog-specific targets.

### Required inputs
- `--mapped-file`  
  CSV containing mapped country × class × disturbance areas with columns including:
  - `CNTR_ID` or equivalent country field
  - `class` or `ref_class`
  - `distur` / `distur_1` / `disturbance`
  - one mapped area column such as:
    - `area_ha`
    - `mapped_area_ha`
    - `area`
    - `A_ha`
    - `ha`
    - `area_m2`
    - `A_m2`

- `--eu-design-file`  
  Typically:
  - `outputs/eu_design_based_disturbance_estimation/eu_class_disturbance_estimates.csv`

- `--country-class-file`  
  Typically:
  - `outputs/country_calibrated_hierarchical_area_estimation/country_pooled_wetland_area_estimates.csv`

- `--out-dir`  
  Output directory

### Example run
```bash
python country_calibrated_hierarchical_disturbance_allocation.py \
  --mapped-file source_data/country_calibrated_hierarchical_disturbance_allocation_mapped.csv \
  --eu-design-file outputs/eu_design_based_disturbance_estimation/eu_class_disturbance_estimates.csv \
  --country-class-file outputs/country_calibrated_hierarchical_area_estimation/country_pooled_wetland_area_estimates.csv \
  --out-dir outputs/country_calibrated_hierarchical_disturbance_allocation
```

### Outputs
- `country_class_disturbance_allocations.csv`
- `country_disturbance_targets.csv`
- `eu_class_disturbance_ratio_sanity.csv`

### Restoration targets
`country_disturbance_targets.csv` includes Monte Carlo summaries for:

- total disturbed wetland area
- 30% restoration target area
- disturbed peatbog area
- 30% peatbog restoration target area

## One-shot execution

From the repository root:

```bash
mkdir -p outputs/eu_design_based_area_estimation \
         outputs/eu_design_based_disturbance_estimation \
         outputs/country_calibrated_hierarchical_area_estimation \
         outputs/country_calibrated_hierarchical_disturbance_allocation

python eu_design_based_area_estimation.py \
  --sample-file source_data/eu_design_based_area_estimation_samples.gpkg \
  --strata-file source_data/eu_design_based_area_estimation_strata.csv \
  --out-dir outputs/eu_design_based_area_estimation

python country_calibrated_hierarchical_area_estimation.py \
  --sample-file source_data/country_calibrated_hierarchical_area_estimation_samples.gpkg \
  --strata-file source_data/country_calibrated_hierarchical_area_estimation_strata.csv \
  --country-totals-file source_data/country_calibrated_hierarchical_area_estimation_country_totals.csv \
  --out-dir outputs/country_calibrated_hierarchical_area_estimation

python eu_design_based_disturbance_estimation.py \
  --sample-file source_data/eu_design_based_disturbance_estimation_samples.gpkg \
  --strata-file source_data/eu_design_based_disturbance_estimation_strata.csv \
  --out-dir outputs/eu_design_based_disturbance_estimation \
  --include-unknown

python country_calibrated_hierarchical_disturbance_allocation.py \
  --mapped-file source_data/country_calibrated_hierarchical_disturbance_allocation_mapped.csv \
  --eu-design-file outputs/eu_design_based_disturbance_estimation/eu_class_disturbance_estimates.csv \
  --country-class-file outputs/country_calibrated_hierarchical_area_estimation/country_pooled_wetland_area_estimates.csv \
  --out-dir outputs/country_calibrated_hierarchical_disturbance_allocation
```

## Notes on scope

This repository contains the public inference workflows for wetland area and disturbance estimation. It is not a complete reproduction package for:

- manuscript figures beyond these inference outputs
- manuscript text
- downstream carbon-loss analyses
- all intermediate preprocessing steps used during development

## Citation

Kovács, G. M., Tong, X., Gominski, D., Oehmcke, S., Horion, S., Abel, C., Ivits, E., Schurgers, G., Elberling, B., Prishchepov, A., van der Linden, S., Page, S., Barthelmes, A., Tanneberger, F., & Fensholt, R. (2026). *Highly fragmented European wetlands with uneven restoration needs*. Nature.

## References

Deville, J.-C., Särndal, C.-E., & Sautory, O. (1993). Generalized Raking Procedures in Survey Sampling. *Journal of the American Statistical Association*, 88(423), 1013–1020.

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2014). *Bayesian Data Analysis* (3rd ed.). CRC Press.

Särndal, C.-E., Swensson, B., & Wretman, J. (1992). *Model Assisted Survey Sampling*. Springer.

Stehman, S. V. (2014). Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes. *International Journal of Remote Sensing*, 35, 4923–4939.
