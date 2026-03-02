# ROCIT Manuscript Analysis

Code accompanying the ROCIT manuscript. This repository contains the processing, training, and analysis scripts used to evaluate [ROCIT](https://github.com/tobybaker/rocit)- a tool for classifying tumor-derived reads from PacBio long-read sequencing using CpG methylation, somatic variants, copy number, and phasing information.

## Structure

| Directory | Contents |
|-----------|----------|
| `code/processing/` | Build training datasets from BAMs, VCFs, ASCAT, DPClust, and phasing output |
| `code/training/` | Train ROCIT transformer and XGBoost models; optimise methylation probabilities |
| `code/analysis/` | Generate manuscript figures and statistics |
| `code/methylation_region_plotter/` | Visualise CpG methylation in genomic regions |

## Dependencies

Requires the [ROCIT package](https://github.com/tobybaker/rocit) plus `polars`, `pytorch-lightning`, `xgboost`, `pysam`, `scikit-learn`, and `matplotlib`.

## License

BSD 3-Clause — see [LICENSE](LICENSE).
