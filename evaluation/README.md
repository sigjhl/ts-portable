# Evaluation code

This folder holds the code that produced the accuracy, runtime, and spleen volumetry results of the
paper *Quantized TotalSegmentator for Fast Abdominal Organ Segmentation on CPUs* (TS-Portable is the
program in this repository), together with the small per-case result files the tables and figures
are computed from.

There are two ways to use it:

1. **From the included result files** (no images needed, a few minutes on a laptop): rerun every
   statistical analysis and the data figures, and check them against the reported values.
2. **From the public datasets**: segment the scans with TS-Portable and the reference
   implementation, score the masks, and then run step 1 on your own per-case files.

The reference implementation is the distributed PyTorch TotalSegmentator package, version 2.12.0
(2.4.0 on the reading station). "Full" means the 1.5-mm models and "fast" the 3-mm models. Paired
differences are always reference implementation minus TS-Portable, so positive values favor the
reference implementation; the noninferiority margin is 0.01.

## 1. Rerun the analyses from the included files

Requirements: Python 3.10 or later, R with `boot`, `boot.pval` 0.7.0, `jsonlite`, `ggplot2` and
`patchwork` (see `requirements.txt`). NumPy is needed for one script and Matplotlib for Figures 1
and 2. If `boot.pval` does not install, download its CRAN source
(`https://cran.r-project.org/src/contrib/boot.pval_0.7.0.tar.gz`), unpack it, and point
`BOOT_PVAL_R_FILE` at `boot.pval/R/boot.pval.R`; the scripts then load that one function.

Run from this folder:

```bash
Rscript noninferiority/bootstrap_endpoints.R                                  # Table 2 (DSC), Tables S5, S9, S10
SEGMENTATION_ENDPOINT=nsd3 Rscript noninferiority/bootstrap_endpoints.R       # Table 2 (NSD@3 mm), Tables S9, S10
Rscript noninferiority/bootstrap_nsd_tolerances.R                             # Table S8
Rscript noninferiority/bootstrap_nsd_tolerance_structures.R                   # Table S10, NSD@1/2 mm
Rscript noninferiority/spleen_only_noninferiority.R                           # Table 2, spleen rows
python3 noninferiority/summarize_accuracy.py                                  # empty predictions, Tables S5-S7, Holm
Rscript noninferiority/volume_strata_analysis.R                               # Table S11
Rscript runtime/paired_runtime_uncertainty.R                                  # Table 3 ratios
python3 runtime/summarize_runtime.py                                          # Table 3 medians and memory, Table 4, Table S13 timing
Rscript spleen_volumetry/amos_spleen_agreement.R                              # Table 5 (AMOS), Table S12
python3 spleen_volumetry/amos_bland_altman_ci.py                              # Table 5 (AMOS), bootstrap CIs of bias and limits
Rscript spleen_volumetry/duke_agreement.R                                     # Table 5 (Duke), Table S13
Rscript figures/fig4_runtime_dots.R
Rscript figures/fig5_spleen_volumetry.R
Rscript figures/figS3_volume_strata.R                                         # after volume_strata_analysis.R
python3 figures/fig1_flowchart.py
python3 figures/fig2_study_design.py
python3 compare_results.py                                                    # regenerated vs reported
```

Outputs go to `results/` (created on first run). `bootstrap_endpoints.R` takes about 1.5 minutes per
endpoint; everything else takes seconds.

`compare_results.py` compares each regenerated JSON file with its counterpart in `data/reported/`
and prints the largest absolute difference. On R 4.5.2 every reported value is reproduced; the
differences are of the order of 1e-15 (floating-point arithmetic) with two expected exceptions,
which it marks CHECK:

- the per-structure intervals of the right kidney cyst at 3 mm in the development set
  (`*_r_boot_structure_ci_ts_fast.json`): six cases, so the bias-corrected interval of the paired
  difference is sensitive to the R version (reported analyses ran on R 4.1.2); descriptive only and
  not shown in the paper;
- `spleen_only_mode_noninferiority.json`: the reported file stores the one-sided p value as 0.001
  (reported as p < .001), the script stores the grid value 0.0005 that boot.pval returns.

## 2. Map of tables and figures

| Paper item | Script | Input (included) | Output |
|---|---|---|---|
| Table 2, all-structure rows | `noninferiority/bootstrap_endpoints.R` (DSC; NSD@3 mm with `SEGMENTATION_ENDPOINT=nsd3`) | `data/accuracy/percase_{amos,ts}_{full,fast}.json` | `dsc_bootstrap_results.json`, `r_bootstrap_results.json` |
| Table 2, spleen rows and footnote d | `noninferiority/spleen_only_noninferiority.R` | `data/spleen/amos_spleen_only_mode_metrics.csv` | `spleen_only_mode_noninferiority.json` |
| Holm adjustment, empty predictions | `noninferiority/summarize_accuracy.py` | per-case files, bootstrap results | `accuracy_summary.json` |
| Table 3 | `runtime/paired_runtime_uncertainty.R`, `runtime/summarize_runtime.py` | `data/runtime/workstation_timing_amos.jsonl` | `paired_runtime_uncertainty.json`, `runtime_summary.json` |
| Table 4 | `runtime/summarize_runtime.py` | `data/runtime/station_amos100_spleen.csv` | `runtime_summary.json` |
| Table 5, AMOS | `spleen_volumetry/amos_spleen_agreement.R` (CCC), `spleen_volumetry/amos_bland_altman_ci.py` (bias and limit CIs) | `data/spleen/amos_spleen_volumes_percase.csv` | `amos_spleen_agreement.json`, `amos_spleen_ba_ci.json` |
| Table 5, Duke | `spleen_volumetry/duke_agreement.R` | `duke_percase.csv` (not included: per-series values derived from the Duke Spleen Data Set, whose terms are CC BY-NC-ND; produce it with `scoring/duke/duke_percase.py` from the data set), `data/runtime/station_duke_spleen.csv` | `duke_agreement.json` |
| Table S4 | `spleen_volumetry/version_comparison.py` (masks needed) | `data/runtime/workstation_version_exp_timing.jsonl` | `version_summary.json` (reported copy in `data/reported/`) |
| Table S5 | `bootstrap_endpoints.R` (`onnx_fp32_sensitivity`, DSC), `summarize_accuracy.py` (`fp32_fidelity`) | `data/accuracy/fp32/` | as above |
| Table S6, S7 | `noninferiority/summarize_accuracy.py` (`direct_dsc`, `per_case_tails`, `per_organ_case`) | per-case files | `accuracy_summary.json` |
| Table S8 | `noninferiority/bootstrap_nsd_tolerances.R` | `data/accuracy/nsd_tolerance/nsd_tolerance_percase_*.json` | `nsd_tolerance_bootstrap_results.json` |
| Tables S9, S10 | `bootstrap_endpoints.R` (per-structure intervals), `bootstrap_nsd_tolerance_structures.R` | per-case files, `nsd_tolerance/raw_amos_*.jsonl` | `*r_boot_structure_ci_amos_*.json`, `nsd_tolerance_structure_ci_amos_*.json` |
| Table S11, Figure S3 | `noninferiority/volume_strata_analysis.R`, `figures/figS3_volume_strata.R` | `data/volume_strata/reference_volumes.csv`, per-case files | `results/volume_strata/` |
| Table S12 | `spleen_volumetry/amos_spleen_agreement.R` | `data/spleen/amos_spleen_volumes_percase.csv` | `amos_spleen_agreement.json` |
| Table S13 | `spleen_volumetry/duke_agreement.R`, `runtime/summarize_runtime.py` | `duke_percase.csv` (not included; see Table 5, Duke), `data/runtime/station_duke_spleen.csv` | `duke_agreement.json`, `runtime_summary.json` |
| Figures 1, 2 | `figures/fig1_flowchart.py`, `figures/fig2_study_design.py` | none (diagrams) | `results/figures/` |
| Figure 3, Figure S2 | `figures/fig3_spleen_liver_example.py`, `figures/figS2_segmentation_examples.py` | AMOS images and masks (not included) | `results/figures/` |
| Figure 4 | `figures/fig4_runtime_dots.R` | `data/runtime/workstation_timing_amos.jsonl` | `results/figures/` |
| Figure 5 | `figures/fig5_spleen_volumetry.R` | `data/spleen/amos_spleen_volumes_percase.csv` | `results/figures/` |
| Figure S5 | `figures/figS5_small_spleen.py` | AMOS case amos_0167 and its masks (not included) | `results/figures/` |

Tables 1 and S1 summarize the public dataset metadata; Tables S2 and S3 and Figure S1 describe the
quantization (see `scripts/` and `docs/METHOD.md` in this repository); Figure S4 is an
image render without a script.

## 3. Included result files

All are small text files. Case identifiers are those of the public datasets; lists are in
`data/cases/`, with `amos_sex_reviewed.csv` (sex of each AMOS scan as assigned from the pelvic
anatomy, the recorded sex, and the basis).

| File | Contents |
|---|---|
| `data/accuracy/percase_<cohort>_<tier>.json` | One record per case: per-case mean DSC and NSD@3 mm of both implementations, and per structure the DSC and NSD at 1, 2, and 3 mm against the reference mask, direct DSC between the implementations' masks, and empty-prediction flags. `amos` = 100 external AMOS CT scans (15 organs), `ts` = 102 TotalSegmentator development-set scans (117 labels). |
| `data/accuracy/nsd_tolerance/` | NSD@1 mm and NSD@2 mm case means (`nsd_tolerance_percase_*`) and per-organ values for AMOS (`raw_amos_*.jsonl`). |
| `data/accuracy/fp32/` | TS-Portable with the unquantized fp32 ONNX models: per-case mean DSC (`dsc_percase_*`) and per-structure NSD@3 mm (`f3`; `i8` is the INT8 value). |
| `data/volume_strata/reference_volumes.csv` | Volume of every reference-mask structure instance (mL). |
| `data/runtime/workstation_timing_amos.jsonl` | Workstation, 100 AMOS scans, five configurations, both resolutions: `label` = configuration\|tier\|case, `proc_wall_s` = cold end-to-end wall time, `peak_pss_mib` = peak proportional set size of the process tree. |
| `data/runtime/workstation_version_exp_timing.jsonl` | The same for reference implementation versions 2.4.0 and 2.12.0 (Table S4). |
| `data/runtime/station_amos100_spleen.csv`, `station_duke_spleen.csv` | PACS reading station, spleen-only mode, one row per run (warm-up rows flagged): wall time, peak working set of the process tree, GPU memory, CPU fallback. |
| `data/spleen/amos_spleen_volumes_percase.csv` | Spleen volume (mL) per AMOS case: manual reference, full mode on the workstation (reference implementation, TS-Portable INT8 and fp32), and spleen-only mode on the reading station (TS-Portable, reference implementation on CPU and GPU). |
| `data/spleen/amos_spleen_only_mode_metrics.csv` | Spleen DSC and NSD@3 mm against the manual spleen in spleen-only mode: TS-Portable on the station (`tsp_pacs`), reference implementation 2.4.0 on the station (`ref240_pacs`), 2.12.0 and 2.4.0 on the workstation (`ref2120_ws`, `ref240_ws`). |
| (Duke per-series files) | Not included: spleen volumes, DSC and NSD per Duke CT series are derived from the Duke Spleen Data Set (CC BY-NC-ND). `scoring/duke/duke_percase.py` regenerates `duke_percase.csv` from the data set; the reported aggregate results are in `data/reported/`. |
| `data/total_label_map.json` | TotalSegmentator's label ids (structure name to id). |
| `data/reported/` | The result files of the reported analyses, for `compare_results.py`. |

## 4. From the public datasets

### Data

- **AMOS** validation release (Ji et al., NeurIPS 2022): `imagesVa/`, `labelsVa/` and
  `labeled_data_meta_0000_0599.csv`. The 100 CT scans are the cases with ids below 500
  (`data/cases/amos_ct_validation.txt`); ids from 500 are MRI.
- **TotalSegmentator dataset**, small release of 102 scans (Wasserthal et al., Radiol Artif Intell
  2023), one folder per case with `ct.nii.gz` and `segmentations/<structure>.nii.gz`
  (`data/cases/totalsegmentator_dev_set.txt`).
- **Duke Spleen Data Set** (Zenodo 7636640), CT series DSDS_0001 to DSDS_0029; convert with
  `scoring/duke/convert_duke_ct.py` (DSDS_0021 is excluded; `scoring/duke/check_dsds0021_alignment.py`
  shows the slice offset).

### Segmentation and timing

The INT8 models are distributed with the program (see the repository README). On a Linux
workstation, `runtime/run_bench.sh` runs one configuration over a folder of scans, one case per
process, writes the masks, and appends the cold wall time and peak memory per case:

```bash
export PY=python3 BIN=target/release/onnx-totalseg INT8_MODELS=models \
       ORT_OPENVINO=/path/to/libonnxruntime.so.1.24.1
runtime/run_bench.sh ours_int8_nncf full AMOS/imagesVa timing.jsonl masks/tsportable/full
runtime/run_bench.sh ts_gpu         full AMOS/imagesVa timing.jsonl masks/reference/full
```

The configurations are `ours_int8_nncf`, `ours_fp32`, `ours_fp32_cuda` (TS-Portable) and `ts_cpu`,
`ts_gpu` (reference implementation through `runtime/run_ts.py`); `fast` instead of `full` selects
the 3-mm models. The header of `run_bench.sh` lists the environment variables. The reported runs used
an idle machine (`NIGHT_ONLY=1`) and each configuration once per case.

On the Windows reading station, `runtime/station/Run-PacsAmos100.ps1` (AMOS, three configurations)
and `runtime/station/Run-PacsDuke29.ps1` (Duke, DICOM and NIfTI input) run the spleen-only mode with
one warm-up case, 250-ms process-tree sampling, and a 5-s cooldown; their headers describe the
expected folder layout. `runtime/version_exp.sh` and `runtime/run_ts_roi.py` produce the
reference-implementation masks of both versions for Table S4.

### Scoring

```bash
python scoring/score_accuracy.py --cohort amos --tier full \
    --amos-labels AMOS/labelsVa --amos-meta AMOS/labeled_data_meta_0000_0599.csv \
    --amos-sex data/cases/amos_sex_reviewed.csv \
    --pred-int8 masks/tsportable/full --pred-ref masks/reference/full \
    [--pred-fp32 masks/tsportable_fp32/full] --out my_data/accuracy
python scoring/score_accuracy.py --cohort ts --tier full --ts-root Totalsegmentator_dataset_small \
    --pred-int8 ... --pred-ref ... --out my_data/accuracy
python scoring/reference_volumes.py --amos-labels AMOS/labelsVa --ts-root Totalsegmentator_dataset_small \
    --amos-sex data/cases/amos_sex_reviewed.csv --out my_data/volume_strata/reference_volumes.csv
python scoring/spleen_volumes.py ...        # AMOS spleen volumes (see its header)
python scoring/score_spleen_only.py ...     # spleen-only DSC and NSD@3 mm (see its header)
python scoring/duke/duke_percase.py DUKE_ROOT my_data/spleen/duke_percase.csv
python -m unittest discover -s scoring -p 'test_*.py'
```

`score_accuracy.py` writes the per-case files in the layout of `data/accuracy/`; pass that folder
to the analysis scripts as their first argument (for example
`Rscript noninferiority/bootstrap_endpoints.R my_data/accuracy`). Scoring rules: every structure
present in the reference mask is scored, an empty prediction scores 0 on DSC and NSD, structures are
averaged with equal weight within a case, and AMOS label 15 ("prostate/uterus") is scored as the
prostate in men only, with sex as assigned from the pelvic anatomy of each scan
(`data/cases/amos_sex_reviewed.csv`; it differs from the AMOS metadata in 27 scans).
`version_comparison.py` takes the same `--amos-sex` file. NSD uses the surface-area-weighted surface Dice of the DeepMind `surface-distance`
package, version 0.1 (`scoring/nsd_core.py`). Applied to the masks of the reported runs, these
scripts reproduce the included per-case files exactly.

## What is not included

CT images, reference masks, and predicted masks (obtain the datasets from their publishers; the Duke
data set is licensed CC BY-NC-ND), the model files (distributed with the program), and the logs of
the individual runs. Machine-specific launch scripts were replaced by the configurable drivers above.
