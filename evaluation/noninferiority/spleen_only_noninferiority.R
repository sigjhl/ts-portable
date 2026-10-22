#!/usr/bin/env Rscript
# Spleen in spleen-only mode (Table 2, last two rows, and its footnote): paired case bootstrap
# noninferiority test of TS-Portable (PACS reading station masks) against the reference
# implementation in spleen-only mode, version 2.12.0 on the workstation (test) and version 2.4.0 on
# the reading station (sensitivity analysis). Same procedure as bootstrap_endpoints.R: 10,000
# resamples, BCa intervals, one-sided p by BCa interval inversion against the 0.01 margin.
#
#   Rscript noninferiority/spleen_only_noninferiority.R [METRICS_CSV] [OUT_DIR]
#
# METRICS_CSV defaults to data/spleen/amos_spleen_only_mode_metrics.csv (per-case DSC and NSD@3 mm of
# each arm against the manual AMOS spleen, written by scoring/score_spleen_only.py).
suppressPackageStartupMessages(library(jsonlite))
script_dir <- local({
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
})
EVALUATION_DIR <- normalizePath(file.path(script_dir, ".."))
cli <- commandArgs(trailingOnly = TRUE)
metrics_csv <- if (length(cli) >= 1) cli[[1]] else file.path(EVALUATION_DIR, "data", "spleen", "amos_spleen_only_mode_metrics.csv")
out_dir <- if (length(cli) >= 2) cli[[2]] else file.path(EVALUATION_DIR, "results", "noninferiority")
Sys.setenv(BOOTSTRAP_FUNCTIONS_ONLY = "1")
source(file.path(script_dir, "bootstrap_endpoints.R"), local = globalenv())
Sys.unsetenv("BOOTSTRAP_FUNCTIONS_ONLY")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

d <- read.csv(metrics_csv)
stopifnot(nrow(d) == 100)
tests <- list(
  ref2120_minus_tsp_dsc  = c("dsc_tsp_pacs",  "dsc_ref2120_ws"),
  ref2120_minus_tsp_nsd3 = c("nsd3_tsp_pacs", "nsd3_ref2120_ws"),
  ref240_minus_tsp_dsc   = c("dsc_tsp_pacs",  "dsc_ref240_pacs"),
  ref240_minus_tsp_nsd3  = c("nsd3_tsp_pacs", "nsd3_ref240_pacs"))
SPLEEN_BASE_SEED <- 20260720L
out <- list()
for (i in seq_along(tests)) {
  cols <- tests[[i]]
  r <- run_paired_boot(d[[cols[1]]], d[[cols[2]]], SPLEEN_BASE_SEED + i - 1L, include_bca = TRUE)$summary
  out[[names(tests)[i]]] <- list(
    n = r$n, mean_diff = r$mean_difference_reference_minus_int8, ci95 = r$ci95_two_sided,
    one_sided_ub95 = r$upper95_one_sided, p_one_sided = r$p_value_one_sided_bca,
    p_value_reporting = r$p_value_reporting, interval = r$selected_interval_method, seed = SPLEEN_BASE_SEED + i - 1L)
  cat(sprintf("%-24s diff %+.5f  95%% CI (%+.5f, %+.5f)  one-sided UB %.5f  %s\n", names(tests)[i],
              r$mean_difference_reference_minus_int8, r$ci95_two_sided[1], r$ci95_two_sided[2],
              r$upper95_one_sided, r$p_value_reporting))
}
out$means <- list(tsp_dsc = mean(d$dsc_tsp_pacs), tsp_nsd3 = mean(d$nsd3_tsp_pacs),
                  ref2120_dsc = mean(d$dsc_ref2120_ws), ref2120_nsd3 = mean(d$nsd3_ref2120_ws),
                  ref240_dsc = mean(d$dsc_ref240_pacs), ref240_nsd3 = mean(d$nsd3_ref240_pacs))
write_json(out, file.path(out_dir, "spleen_only_mode_noninferiority.json"), auto_unbox = TRUE, pretty = TRUE, digits = 8)
