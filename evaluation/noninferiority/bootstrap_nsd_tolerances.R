#!/usr/bin/env Rscript

# Endpoint-only NSD@1-mm and NSD@2-mm sensitivity analysis. The bootstrap
# implementation is sourced from the validated DSC/NSD analysis code so the
# BCa selection, percentile fallback, interval-inversion p value, margin, and
# reporting rules remain identical.

suppressPackageStartupMessages(library(jsonlite))

script_dir <- local({
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
})
EVALUATION_DIR <- normalizePath(file.path(script_dir, ".."))
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) normalizePath(args[[1]]) else file.path(EVALUATION_DIR, "data", "accuracy", "nsd_tolerance")
out_dir <- if (length(args) >= 2) args[[2]] else file.path(EVALUATION_DIR, "results", "noninferiority")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
script_path <- file.path(script_dir, "bootstrap_endpoints.R")
Sys.setenv(BOOTSTRAP_FUNCTIONS_ONLY = "1")
source(script_path, local = globalenv())
Sys.unsetenv("BOOTSTRAP_FUNCTIONS_ONLY")

output_json <- file.path(out_dir, "nsd_tolerance_bootstrap_results.json")
output_csv <- file.path(out_dir, "nsd_tolerance_endpoints.csv")
endpoint_keys <- c("amos_full", "amos_fast", "ts_full", "ts_fast")
tolerances <- c(1L, 2L)
TOLERANCE_BASE_SEED <- 20260725L

read_records <- function(key) {
  fromJSON(
    file.path(input_dir, paste0("nsd_tolerance_percase_", key, ".json")),
    simplifyVector = FALSE
  )
}

rename_tolerance_fields <- function(row, tolerance) {
  row[[paste0("mean_nsd", tolerance, "_int8")]] <- row$mean_nsd3_int8
  row$mean_nsd3_int8 <- NULL
  row[[paste0("mean_nsd", tolerance, "_original_ts")]] <- row$mean_nsd3_reference
  row$mean_nsd3_reference <- NULL
  row$mean_difference_original_ts_minus_int8 <- row$mean_difference_reference_minus_int8
  row$mean_difference_reference_minus_int8 <- NULL
  row$metric <- paste0("per-case mean surface-area-weighted NSD@", tolerance, " mm")
  row
}

all_results <- list()
all_diagnostics <- list()
csv_rows <- list()
csv_index <- 0L

for (tolerance in tolerances) {
  tolerance_key <- paste0("nsd", tolerance)
  tolerance_results <- list()
  tolerance_diagnostics <- list()
  for (key_index in seq_along(endpoint_keys)) {
    key <- endpoint_keys[[key_index]]
    rows <- read_records(key)
    int8_field <- paste0("mean_nsd", tolerance, "_int8")
    reference_field <- paste0("mean_nsd", tolerance, "_original_ts")
    int8 <- vapply(rows, function(row) row[[int8_field]], numeric(1))
    reference <- vapply(rows, function(row) row[[reference_field]], numeric(1))
    cases <- vapply(rows, function(row) row$case, character(1))
    seed <- TOLERANCE_BASE_SEED + tolerance * 100L + key_index
    result <- run_paired_boot(int8, reference, seed, include_bca = TRUE)$summary
    result <- rename_tolerance_fields(result, tolerance)
    tolerance_results[[key]] <- result
    tolerance_diagnostics[[key]] <- list(
      case_order = cases,
      bca = result$bca_diagnostic
    )

    csv_index <- csv_index + 1L
    csv_rows[[csv_index]] <- data.frame(
      tolerance_mm = tolerance,
      cohort_tier = key,
      n = result$n,
      mean_int8 = result[[int8_field]],
      mean_original_ts = result[[paste0("mean_nsd", tolerance, "_original_ts")]],
      original_ts_minus_int8 = result$mean_difference_original_ts_minus_int8,
      interval_method = result$selected_interval_method,
      ci_low = result$ci95_two_sided[[1]],
      ci_high = result$ci95_two_sided[[2]],
      upper95_one_sided = result$upper95_one_sided,
      p_value_one_sided_bca = result$p_value_one_sided_bca,
      p_value_reporting = result$p_value_reporting,
      verdict = result$verdict
    )
  }
  all_results[[tolerance_key]] <- tolerance_results
  all_diagnostics[[tolerance_key]] <- tolerance_diagnostics
}

output <- list(
  comparator = "Original TotalSegmentator 2.12.0, resolution matched",
  estimand = "per-case mean NSD at 1-mm or 2-mm tolerance; paired difference Original TotalSegmentator minus INT8-TS",
  noninferiority_margin = MARGIN,
  method = list(
    implementation = "R boot package",
    r_version = R.version.string,
    boot_version = as.character(packageVersion("boot")),
    jsonlite_version = as.character(packageVersion("jsonlite")),
    boot_pval = boot_pval_origin,
    primary_interval = "BCa when finite and warning-free with finite jackknife acceleration; percentile otherwise",
    decision_rule = "selected one-sided 95% upper bound < 0.01",
    paired_resampling_unit = "case",
    replicates = R_REPLICATES,
    base_seed = TOLERANCE_BASE_SEED,
    p_value_method = "boot.pval BCa interval inversion, alternative less, theta_null 0.01",
    pval_precision_argument = PVAL_PRECISION
  ),
  endpoints = all_results,
  diagnostics = all_diagnostics
)

write_json(output, output_json, auto_unbox = TRUE, pretty = TRUE, digits = 16, null = "null")
write.csv(do.call(rbind, csv_rows), output_csv, row.names = FALSE)
cat("Wrote NSD@1/2 bootstrap results for", length(endpoint_keys) * length(tolerances),
    "endpoints.\n")
