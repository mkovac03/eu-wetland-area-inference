# Wetland Area Inference

This repository contains the analysis scripts used to implement the methods described in:

**Highly fragmented European wetlands with uneven restoration needs**  
Gyula Mate Kovács, Xiaoye Tong, Dimitri Gominski, Stefan Oehmcke, Stéphanie Horion, Christin Abel, Eva Ivits, Guy Schurgers, Bo Elberling, Alexander Prishchepov, Sebastian van der Linden, Susan Page, Alexandra Barthelmes, Franziska Tanneberger, and Rasmus Fensholt.  
**Nature (2026)**

The scripts correspond to the following methodological components of the paper:

- **European-scale design-based area and accuracy estimation**
- **Country-level wetland area estimation**
- **Wetland disturbance area estimation**

These scripts are written as public-facing implementations of the methods and use simplified, explicit variable naming:
- `reference` = reference class label
- `Class` = mapped class label
- `Code_18` = stratum code
- `CNTR_ID` = country code

---

## Repository contents

### 1. `eu_design_based_area_estimation.py`
Implements the **European-scale design-based area and accuracy estimation**.

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
```bash
python eu_design_based_area_estimation.py \
  --sample-file /path/to/validation_points.gpkg \
  --strata-file /path/to/stratum_areas.csv \
  --out-dir /path/to/output_dir \
  --n-classes 8 \
  --exclude-strata 999
