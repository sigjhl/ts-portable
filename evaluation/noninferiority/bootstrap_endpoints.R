#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(boot))
suppressPackageStartupMessages(library(jsonlite))

# Paired case bootstrap noninferiority tests for the per-case mean endpoints (Table 2,
# Supplemental Tables S5, S9 and S10).
#
#   Rscript noninferiority/bootstrap_endpoints.R [DATA_DIR] [OUT_DIR]
#
# SEGMENTATION_ENDPOINT=dsc (default) or nsd3 selects the endpoint. DATA_DIR defaults to
# evaluation/data/accuracy and OUT_DIR to evaluation/results/noninferiority.
# Other scripts source this file with BOOTSTRAP_FUNCTIONS_ONLY=1 to reuse the functions.

script_dir <- local({
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
})
if (!exists("EVALUATION_DIR")) EVALUATION_DIR <- normalizePath(file.path(script_dir, ".."))
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) normalizePath(args[[1]]) else file.path(EVALUATION_DIR, "data", "accuracy")
results_dir <- if (length(args) >= 2) args[[2]] else file.path(EVALUATION_DIR, "results", "noninferiority")
historical_dir <- file.path(data_dir, "fp32")

R_REPLICATES <- 10000L
BASE_SEED <- 20260718L
MARGIN <- 0.01
PVAL_PRECISION <- 0.001
ENDPOINT_METRIC <- Sys.getenv("SEGMENTATION_ENDPOINT", "dsc")
if (!(ENDPOINT_METRIC %in% c("nsd3", "dsc"))) {
  stop("SEGMENTATION_ENDPOINT must be 'nsd3' or 'dsc'")
}
IS_DSC <- ENDPOINT_METRIC == "dsc"
OUTPUT_PREFIX <- if (IS_DSC) "dsc_" else ""

boot_pval_origin <- NULL
if (requireNamespace("boot.pval", quietly = TRUE)) {
  boot_pval_function <- getExportedValue("boot.pval", "boot.pval")
  boot_pval_origin <- list(
    implementation = "boot.pval::boot.pval",
    package_version = as.character(packageVersion("boot.pval")),
    source = "installed R package"
  )
} else {
  boot_pval_file <- Sys.getenv("BOOT_PVAL_R_FILE", "")
  if (!nzchar(boot_pval_file) || !file.exists(boot_pval_file)) {
    stop("install boot.pval or set BOOT_PVAL_R_FILE to its official R/boot.pval.R source file")
  }
  boot_pval_environment <- new.env(parent = globalenv())
  sys.source(boot_pval_file, envir = boot_pval_environment)
  boot_pval_function <- get("boot.pval", envir = boot_pval_environment, inherits = FALSE)
  sha_tool <- if (nzchar(Sys.which("sha256sum"))) c("sha256sum") else c("shasum", "-a", "256")
  sha_line <- system2(sha_tool[[1]], c(sha_tool[-1], shQuote(boot_pval_file)), stdout = TRUE)
  boot_pval_origin <- list(
    implementation = "official boot.pval::boot.pval function sourced from CRAN package",
    package_version = Sys.getenv("BOOT_PVAL_VERSION", "0.7.0"),
    source = normalizePath(boot_pval_file),
    source_url = "https://cran.r-project.org/src/contrib/boot.pval_0.7.0.tar.gz",
    source_sha256 = strsplit(sha_line[[1]], "[[:space:]]+")[[1]][[1]]
  )
}

paired_statistic <- function(data, indices) {
  int8 <- data$int8[indices]
  reference <- data$reference[indices]
  c(mean(int8), mean(reference), mean(reference - int8))
}

capture_boot_ci <- function(fit, conf, type, index) {
  captured <- character()
  error <- NULL
  value <- tryCatch(
    withCallingHandlers(
      boot.ci(fit, conf = conf, type = type, index = index),
      warning = function(w) {
        captured <<- c(captured, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      error <<- conditionMessage(e)
      NULL
    }
  )
  list(value = value, warnings = unique(captured), error = error)
}

interval_from_boot_ci <- function(result, type) {
  if (is.null(result$value)) return(c(NA_real_, NA_real_))
  matrix <- result$value[[type]]
  if (is.null(matrix)) return(c(NA_real_, NA_real_))
  as.numeric(matrix[1, 4:5])
}

bca_selection <- function(two_sided, one_sided, acceleration) {
  ci95 <- interval_from_boot_ci(two_sided, "bca")
  ci90 <- interval_from_boot_ci(one_sided, "bca")
  messages <- unique(c(two_sided$warnings, one_sided$warnings))
  errors <- Filter(Negate(is.null), c(two_sided$error, one_sided$error))
  reasons <- character()
  if (length(messages) > 0) reasons <- c(reasons, paste("boot.ci warning:", messages))
  if (length(errors) > 0) reasons <- c(reasons, paste("boot.ci error:", errors))
  if (!is.finite(acceleration)) reasons <- c(reasons, "non-finite jackknife acceleration")
  if (!all(is.finite(c(ci95, ci90[[2]])))) reasons <- c(reasons, "non-finite BCa limit")
  if (is.finite(ci95[[1]]) && is.finite(ci95[[2]]) && ci95[[1]] > ci95[[2]]) {
    reasons <- c(reasons, "unordered BCa limits")
  }
  list(
    eligible = length(reasons) == 0,
    reasons = reasons,
    ci95_two_sided = ci95,
    upper95_one_sided = ci90[[2]],
    warnings = messages,
    errors = errors
  )
}

jackknife_diagnostics <- function(differences) {
  n <- length(differences)
  leave_one_out <- vapply(seq_len(n), function(i) mean(differences[-i]), numeric(1))
  center <- mean(leave_one_out)
  influence <- center - leave_one_out
  denominator <- 6 * (sum(influence^2)^(3 / 2))
  acceleration <- if (denominator == 0) NA_real_ else sum(influence^3) / denominator
  standardized <- if (sum(influence^2) == 0) rep(0, n) else influence / sqrt(sum(influence^2))
  largest <- order(abs(standardized), decreasing = TRUE)[seq_len(min(5L, n))]
  list(
    acceleration = acceleration,
    max_abs_standardized_jackknife_influence = max(abs(standardized)),
    largest_influences = lapply(largest, function(i) list(
      index = i,
      difference = differences[[i]],
      leave_one_out_mean = leave_one_out[[i]],
      standardized_jackknife_influence = standardized[[i]]
    ))
  )
}

run_paired_boot <- function(int8, reference, seed, include_bca = TRUE) {
  stopifnot(length(int8) == length(reference), length(int8) >= 2)
  data <- data.frame(int8 = as.numeric(int8), reference = as.numeric(reference))
  set.seed(seed)
  fit <- boot(data = data, statistic = paired_statistic, R = R_REPLICATES,
              sim = "ordinary", stype = "i")

  percentile_95 <- capture_boot_ci(fit, conf = 0.95, type = "perc", index = 3)
  percentile_90 <- capture_boot_ci(fit, conf = 0.90, type = "perc", index = 3)
  ci95 <- interval_from_boot_ci(percentile_95, "percent")
  ci90 <- interval_from_boot_ci(percentile_90, "percent")
  difference <- mean(data$reference - data$int8)

  jackknife <- jackknife_diagnostics(data$reference - data$int8)
  bca_95 <- capture_boot_ci(fit, conf = 0.95, type = "bca", index = 3)
  bca_90 <- capture_boot_ci(fit, conf = 0.90, type = "bca", index = 3)
  selected <- bca_selection(bca_95, bca_90, jackknife$acceleration)
  selected_method <- if (selected$eligible) "BCa" else "percentile"
  selected_ci95 <- if (selected$eligible) selected$ci95_two_sided else ci95
  selected_upper95 <- if (selected$eligible) selected$upper95_one_sided else ci90[[2]]
  p_value <- boot_pval_function(
    fit, type = "bca", theta_null = MARGIN, alternative = "less",
    pval_precision = PVAL_PRECISION, index = 3
  )
  p_value_at_lower_grid_boundary <- isTRUE(all.equal(p_value, PVAL_PRECISION / 2))
  p_value_reporting <- if (p_value_at_lower_grid_boundary) {
    sprintf("p < %.3f", PVAL_PRECISION)
  } else {
    sprintf("p = %.4f", p_value)
  }
  verdict_from_bound <- selected_upper95 < MARGIN
  verdict_from_p_value <- p_value < 0.05
  stopifnot(verdict_from_bound == verdict_from_p_value)

  output <- list(
    n = nrow(data),
    mean_nsd3_int8 = mean(data$int8),
    mean_nsd3_reference = mean(data$reference),
    mean_difference_reference_minus_int8 = difference,
    percentile_ci95_two_sided = ci95,
    percentile_upper95_one_sided = ci90[[2]],
    selected_interval_method = selected_method,
    ci95_two_sided = selected_ci95,
    upper95_one_sided = selected_upper95,
    p_value_one_sided_bca = p_value,
    p_value_reporting = p_value_reporting,
    p_value_at_lower_grid_boundary = p_value_at_lower_grid_boundary,
    noninferiority_margin = MARGIN,
    null_hypothesis = "mean(reference - INT8) >= 0.01",
    alternative_hypothesis = "mean(reference - INT8) < 0.01",
    verdict = if (verdict_from_bound) "NON-INFERIOR (met)" else "did NOT meet",
    bootstrap = list(
      implementation = "R boot package",
      package_version = as.character(packageVersion("boot")),
      method = paste("ordinary paired case bootstrap;", selected_method, "interval"),
      replicates = R_REPLICATES,
      seed = seed,
      rng = paste(RNGkind(), collapse = "; "),
      percentile_warnings = unique(c(percentile_95$warnings, percentile_90$warnings))
    ),
    p_value = list(
      implementation = boot_pval_origin,
      method = "BCa confidence-interval inversion; alternative='less'; theta_null=0.01",
      pval_precision_argument = PVAL_PRECISION,
      one_sided_grid_resolution = PVAL_PRECISION / 2,
      reporting_rule = "when the result is the first one-sided grid point, report p < pval_precision",
      decision_alpha = 0.05,
      agrees_with_one_sided_bound = verdict_from_bound == verdict_from_p_value
    )
  )

  if (include_bca) {
    output$bca_diagnostic <- list(
      eligible = selected$eligible,
      fallback_reasons = selected$reasons,
      ci95_two_sided = selected$ci95_two_sided,
      upper95_one_sided = selected$upper95_one_sided,
      warnings = selected$warnings,
      errors = selected$errors,
      jackknife = jackknife
    )
  }
  list(summary = output, fit = fit)
}

run_structure_boot <- function(int8, reference, seed) {
  data <- data.frame(int8 = as.numeric(int8), reference = as.numeric(reference))
  set.seed(seed)
  fit <- boot(data = data, statistic = paired_statistic, R = R_REPLICATES,
              sim = "ordinary", stype = "i")
  arm_names <- c("int8", "stock_ts", "stock_ts_minus_int8")
  result <- list(n = nrow(data))
  for (index in seq_along(arm_names)) {
    percentile <- capture_boot_ci(fit, conf = 0.95, type = "perc", index = index)
    percentile_values <- interval_from_boot_ci(percentile, "percent")
    observed <- if (index == 3) data$reference - data$int8 else data[[index]]
    jackknife <- jackknife_diagnostics(observed)
    bca <- capture_boot_ci(fit, conf = 0.95, type = "bca", index = index)
    # Only the displayed two-sided structure CI is needed, so reuse it as the
    # eligibility input rather than manufacturing a one-sided structure test.
    selected <- bca_selection(bca, bca, jackknife$acceleration)
    values <- if (selected$eligible) selected$ci95_two_sided else percentile_values
    estimate <- fit$t0[[index]]
    result[[arm_names[[index]]]] <- list(
      mean = estimate,
      selected_interval_method = if (selected$eligible) "BCa" else "percentile",
      ci95_two_sided = values,
      bca_eligible = selected$eligible,
      bca_fallback_reasons = selected$reasons,
      bca_ci95_two_sided = selected$ci95_two_sided,
      percentile_ci95_two_sided = percentile_values,
      warnings = unique(c(percentile$warnings, selected$warnings))
    )
  }
  result
}

read_json <- function(path) fromJSON(path, simplifyVector = FALSE)

extract_stock_endpoint <- function(key) {
  rows <- read_json(file.path(data_dir, paste0("percase_", key, ".json")))
  int8_field <- if (IS_DSC) "mean_dsc_int8" else "mean_nsd3_int8"
  reference_field <- if (IS_DSC) "mean_dsc_stock_ts" else "mean_nsd3_stock_ts"
  list(
    int8 = vapply(rows, function(row) row[[int8_field]], numeric(1)),
    reference = vapply(rows, function(row) row[[reference_field]], numeric(1)),
    case = vapply(rows, function(row) row$case, character(1)),
    rows = rows
  )
}

case_means_from_organs <- function(rows, reference_field) {
  list(
    int8 = vapply(rows, function(row) mean(vapply(row$organs, function(x) x$i8, numeric(1))), numeric(1)),
    reference = vapply(rows, function(row) mean(vapply(row$organs, function(x) x[[reference_field]], numeric(1))), numeric(1)),
    case = vapply(rows, function(row) row$case, character(1))
  )
}

extract_fp32_endpoint <- function(key) {
  pieces <- strsplit(key, "_", fixed = TRUE)[[1]]
  cohort <- pieces[[1]]
  tier <- pieces[[2]]
  if (IS_DSC) {
    rows <- read_json(file.path(historical_dir, paste0("dsc_percase_", key, ".json")))
    return(list(
      int8 = vapply(rows, function(row) row$mean_dsc_onnx_fp32, numeric(1)),
      reference = vapply(rows, function(row) row$mean_dsc_original_ts, numeric(1)),
      case = vapply(rows, function(row) row$case, character(1))
    ))
  }
  if (cohort == "amos") {
    all_rows <- read_json(file.path(historical_dir, "nsd_amos_percase.json"))
    rows <- all_rows[[tier]]
  } else {
    rows <- read_json(file.path(historical_dir, paste0("nsd_ts_percase_", tier, ".json")))
  }
  case_means_from_organs(rows, "f3")
}

group_structure_values <- function(rows) {
  grouped <- list()
  for (row in rows) {
    for (organ in row$organs) {
      name <- organ$organ
      if (is.null(grouped[[name]])) grouped[[name]] <- list(int8 = numeric(), reference = numeric())
      int8_value <- if (IS_DSC) organ$dsc_int8_vs_manual else organ$nsd3_int8
      reference_value <- if (IS_DSC) organ$dsc_stock_ts_vs_manual else organ$nsd3_stock_ts
      grouped[[name]]$int8 <- c(grouped[[name]]$int8, int8_value)
      grouped[[name]]$reference <- c(grouped[[name]]$reference, reference_value)
    }
  }
  grouped
}

metricize_summary <- function(row) {
  if (!IS_DSC) return(row)
  row$mean_dsc_int8 <- row$mean_nsd3_int8
  row$mean_nsd3_int8 <- NULL
  row$mean_dsc_reference <- row$mean_nsd3_reference
  row$mean_nsd3_reference <- NULL
  row$metric <- "per-case mean Dice similarity coefficient"
  row
}

if (Sys.getenv("BOOTSTRAP_FUNCTIONS_ONLY", "0") != "1") {

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

endpoint_keys <- c("amos_full", "amos_fast", "ts_full", "ts_fast")
primary <- list()
fp32_sensitivity <- list()
diagnostics <- list()
structure_outputs <- list()

for (key_index in seq_along(endpoint_keys)) {
  key <- endpoint_keys[[key_index]]
  stock <- extract_stock_endpoint(key)
  stock_result <- run_paired_boot(stock$int8, stock$reference,
                                  BASE_SEED + key_index, include_bca = TRUE)
  primary[[key]] <- metricize_summary(stock_result$summary)
  diagnostics[[key]] <- list(
    case_order = stock$case,
    bca = stock_result$summary$bca_diagnostic
  )

  fp32 <- extract_fp32_endpoint(key)
  fp32_result <- run_paired_boot(fp32$int8, fp32$reference,
                                 BASE_SEED + 100L + key_index, include_bca = TRUE)
  fp32_sensitivity[[key]] <- metricize_summary(fp32_result$summary)

  grouped <- group_structure_values(stock$rows)
  structure_result <- list()
  structure_names <- sort(names(grouped))
  for (structure_index in seq_along(structure_names)) {
    name <- structure_names[[structure_index]]
    values <- grouped[[name]]
    structure_result[[name]] <- run_structure_boot(
      values$int8, values$reference,
      BASE_SEED + key_index * 1000L + structure_index
    )
  }
  structure_outputs[[key]] <- structure_result
  write_json(structure_result,
             file.path(results_dir, paste0(OUTPUT_PREFIX, "r_boot_structure_ci_", key, ".json")),
             auto_unbox = TRUE, pretty = TRUE, digits = 16, null = "null")
}

# R-boot leave-one-out sensitivity for the known influential case; descriptive only.
ts_fast <- extract_stock_endpoint("ts_fast")
keep <- ts_fast$case != "s1389"
loo <- run_paired_boot(ts_fast$int8[keep], ts_fast$reference[keep],
                       BASE_SEED + 999L, include_bca = TRUE)$summary
loo <- metricize_summary(loo)

output <- list(
  method = list(
    implementation = "R boot package",
    r_version = R.version.string,
    boot_version = as.character(packageVersion("boot")),
    jsonlite_version = as.character(packageVersion("jsonlite")),
    boot_pval = boot_pval_origin,
    primary_interval = "BCa when boot.ci returns finite ordered limits with finite jackknife acceleration and no warning; percentile otherwise",
    decision_rule = "upper endpoint of the selected one-sided 95% interval < 0.01",
    displayed_interval = "two-sided 95% interval using the same selected method",
    bca_role = "authoritative when eligibility checks pass",
    bca_eligibility = "smooth paired-case mean; finite jackknife acceleration; finite ordered limits; no boot.ci warning or error",
    paired_resampling_unit = "case",
    replicates = R_REPLICATES,
    base_seed = BASE_SEED,
    p_value_method = "boot.pval::boot.pval BCa interval inversion, alternative less, theta_null 0.01",
    pval_precision_argument = PVAL_PRECISION,
    p_value_reporting_floor = "p < 0.001 when the returned value is the first one-sided grid point"
  ),
  primary = primary,
  onnx_fp32_sensitivity = fp32_sensitivity,
  influential_case_sensitivity = list(
    primary_includes_all_cases = TRUE,
    excluded_case = "s1389",
    leave_one_out = loo
  )
)

rename_stock <- function(row) {
  if (IS_DSC) {
    row$mean_dsc_stock_ts <- row$mean_dsc_reference
    row$mean_dsc_reference <- NULL
    row$mean_difference_stock_ts_minus_int8 <- row$mean_difference_reference_minus_int8
    row$mean_difference_reference_minus_int8 <- NULL
    return(row)
  }
  row$mean_nsd3_stock_ts <- row$mean_nsd3_reference
  row$mean_nsd3_reference <- NULL
  row$mean_difference_stock_ts_minus_int8 <- row$mean_difference_reference_minus_int8
  row$mean_difference_reference_minus_int8 <- NULL
  row
}
stock_endpoints <- lapply(primary, rename_stock)
stock_output <- list(
  comparator = "stock TotalSegmentator 2.12.0, resolution matched",
  estimand = if (IS_DSC) {
    "per-case mean DSC; paired difference stock TS minus INT8"
  } else {
    "per-case mean NSD@3 mm; paired difference stock TS minus INT8"
  },
  noninferiority_margin = MARGIN,
  method = output$method,
  endpoints = stock_endpoints
)
endpoint_json <- if (IS_DSC) "dsc_comparator_endpoints.json" else "stock_comparator_endpoints.json"
endpoint_csv <- if (IS_DSC) "dsc_comparator_endpoints.csv" else "stock_comparator_endpoints.csv"
bootstrap_json <- if (IS_DSC) "dsc_bootstrap_results.json" else "r_bootstrap_results.json"
diagnostic_json <- if (IS_DSC) "dsc_bootstrap_bca_diagnostics.json" else "r_bootstrap_bca_diagnostics.json"
write_json(output, file.path(results_dir, bootstrap_json),
           auto_unbox = TRUE, pretty = TRUE, digits = 16, null = "null")
write_json(diagnostics, file.path(results_dir, diagnostic_json),
           auto_unbox = TRUE, pretty = TRUE, digits = 16, null = "null")
write_json(stock_output, file.path(results_dir, endpoint_json),
           auto_unbox = TRUE, pretty = TRUE, digits = 16, null = "null")
csv_rows <- do.call(rbind, lapply(names(stock_endpoints), function(key) {
  row <- stock_endpoints[[key]]
  mean_int8 <- if (IS_DSC) row$mean_dsc_int8 else row$mean_nsd3_int8
  mean_stock <- if (IS_DSC) row$mean_dsc_stock_ts else row$mean_nsd3_stock_ts
  data.frame(cohort_tier = key, n = row$n, mean_int8 = mean_int8,
             mean_stock_ts = mean_stock,
             stock_minus_int8 = row$mean_difference_stock_ts_minus_int8,
             interval_method = row$selected_interval_method,
             ci_low = row$ci95_two_sided[[1]], ci_high = row$ci95_two_sided[[2]],
             upper95_one_sided = row$upper95_one_sided,
             p_value_one_sided_bca = row$p_value_one_sided_bca,
             p_value_reporting = row$p_value_reporting, verdict = row$verdict)
}))
write.csv(csv_rows, file.path(results_dir, endpoint_csv), row.names = FALSE)

cat("Wrote", toupper(ENDPOINT_METRIC), "R boot BCa-with-diagnostic-fallback results for",
    length(endpoint_keys), "primary endpoints,",
    length(endpoint_keys), "fp32 sensitivity endpoints, and",
    sum(vapply(structure_outputs, length, integer(1))), "structures.\n")

}
