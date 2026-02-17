# Changes: `features.csv` vs `source.csv`

## Summary
- Source file: `backend/data/source.csv` (56 patient rows used, 36 columns)
- Features file: `backend/data/features.csv` (56 rows, 16 columns)
- Rows were kept for patient records.

## Added Column
- `adverse_outcome`: derived binary target for Task 2.
  Defined as `1` if any of: `hospital_death`, `stent_thrombosis`, `hospital_mi`, `stroke_tia`, `stroke`, `repeated_hospitalization`, `repeated_revascularization`, `myocardial_infarction_followup` is `1`; otherwise `0`.
  Positive cases in features dataset: `5`.

## Removed Columns
- `patient_id`
- `angina_one_year_followup`
- `lad_proximal_stenosis_percent`
- `lad_mid_stenosis_percent`
- `lcx_proximal_stenosis_percent`
- `lcx_mid_stenosis_percent`
- `rca_proximal_stenosis_percent`
- `rca_mid_stenosis_percent`
- `dfr_ifr`
- `stent_diameter`
- `stent_length`
- `copd_asthma` (constant value in dataset)
- `blood_flow_type` (near-constant value in dataset)
- `hospital_death`
- `stent_thrombosis`
- `hospital_mi`
- `stroke_tia`
- `stroke`
- `repeated_hospitalization`
- `repeated_revascularization`
- `myocardial_infarction_followup`

## Kept Columns (for training)
- `gender`
- `age`
- `angina_functional_class`
- `post_infarction_cardiosclerosis`
- `multifocal_atherosclerosis`
- `diabetes_mellitus`
- `hypertension`
- `cholesterol_level`
- `bmi`
- `lvef_percent`
- `syntax_score`
- `ffr`
- `plaque_volume_percent`
- `lumen_area`
- `unstable_plaque`
