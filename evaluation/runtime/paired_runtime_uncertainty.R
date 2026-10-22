#!/usr/bin/env Rscript
# Paired runtime bootstrap on the workstation (Table 3): ratio of each comparator configuration's
# median cold wall time to the median of TS-Portable with INT8 models on the CPU, with BCa 95% CIs
# from 10,000 paired case resamples (n = 100 AMOS cases, every configuration).
#
#   Rscript runtime/paired_runtime_uncertainty.R [TIMING_JSONL] [OUT_DIR]
#
# TIMING_JSONL defaults to data/runtime/workstation_timing_amos.jsonl (one line per case and
# configuration, written by runtime/run_bench.sh through runtime/bench_case.py). Labels are
# "<configuration>|<full|fast>|<case>"; full = 1.5 mm, fast = 3 mm. Configurations: ours_int8_nncf
# (TS-Portable, INT8, CPU), ours_fp32 (fp32 ONNX, OpenVINO CPU), ours_fp32_cuda (fp32 ONNX, CUDA GPU),
# ts_cpu and ts_gpu (reference implementation).

suppressPackageStartupMessages(library(boot))
suppressPackageStartupMessages(library(jsonlite))

here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else file.path(here, "..", "data", "runtime", "workstation_timing_amos.jsonl")
out_dir <- if (length(args) >= 2) args[[2]] else file.path(here, "..", "results", "runtime")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
output_path <- file.path(out_dir, "paired_runtime_uncertainty.json")

R_REPLICATES <- 10000L
BASE_SEED <- 20260721L
CANDIDATE <- "ours_int8_nncf"
COMPARATORS <- c(
  total_segmentator_cpu = "ts_cpu",
  total_segmentator_rtx_a6000_gpu = "ts_gpu",
  rust_fp32_openvino_cpu = "ours_fp32",
  rust_fp32_onnxruntime_cuda_a6000 = "ours_fp32_cuda"
)

read_jsonl <- function(path) stream_in(file(path), verbose = FALSE)

parse_labels <- function(rows) {
  parts <- strsplit(rows$label, "|", fixed = TRUE)
  rows$condition <- vapply(parts, `[[`, character(1), 1)
  rows$tier <- vapply(parts, `[[`, character(1), 2)
  rows$case <- vapply(parts, `[[`, character(1), 3)
  rows
}

runtime_statistic <- function(data, indices) {
  candidate <- data$candidate_cpu_s[indices]
  comparator <- data$comparator_s[indices]
  c(
    median_paired_difference = median(comparator - candidate),
    ratio_of_marginal_medians = median(comparator) / median(candidate)
  )
}

capture_ci <- function(fit, type, index) {
  warnings <- character()
  error <- NULL
  value <- tryCatch(
    withCallingHandlers(
      boot.ci(fit, conf = 0.95, type = type, index = index),
      warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }
    ),
    error = function(e) { error <<- conditionMessage(e); NULL }
  )
  list(value = value, warnings = unique(warnings), error = error)
}

extract_limits <- function(captured, type) {
  if (is.null(captured$value)) return(c(NA_real_, NA_real_))
  matrix <- captured$value[[type]]
  if (is.null(matrix)) return(c(NA_real_, NA_real_))
  as.numeric(matrix[1, 4:5])
}

bca_interval <- function(fit, index) {
  captured <- capture_ci(fit, "bca", index)
  limits <- extract_limits(captured, "bca")
  reasons <- character()
  if (!is.null(captured$error)) reasons <- c(reasons, paste("boot.ci error:", captured$error))
  if (!all(is.finite(limits))) reasons <- c(reasons, "non-finite BCa limit")
  if (all(is.finite(limits)) && limits[[1]] > limits[[2]]) reasons <- c(reasons, "unordered BCa limits")
  if (length(reasons) > 0) stop(paste(reasons, collapse = "; "))
  list(method = "BCa", ci95 = limits, eligible = TRUE, warnings = captured$warnings)
}

run_comparison <- function(tier, comparator_condition, rows, seed) {
  comparator <- rows[rows$condition == comparator_condition & rows$tier == tier & rows$rc == 0, ]
  int8 <- rows[rows$condition == CANDIDATE & rows$tier == tier & rows$rc == 0, ]
  pairs <- merge(
    int8[, c("case", "proc_wall_s")], comparator[, c("case", "proc_wall_s")],
    by = "case", suffixes = c("_candidate", "_comparator"), sort = TRUE
  )
  names(pairs)[names(pairs) == "proc_wall_s_candidate"] <- "candidate_cpu_s"
  names(pairs)[names(pairs) == "proc_wall_s_comparator"] <- "comparator_s"
  stopifnot(nrow(pairs) == 100L, !anyDuplicated(pairs$case))

  set.seed(seed)
  fit <- boot(data = pairs, statistic = runtime_statistic, R = R_REPLICATES, sim = "ordinary", stype = "i")
  difference_interval <- bca_interval(fit, 1)
  ratio_interval <- bca_interval(fit, 2)

  list(
    tier = tier,
    resolution_mm = if (tier == "full") 1.5 else 3,
    comparator_condition = comparator_condition,
    n = nrow(pairs),
    candidate_cpu_median_s = median(pairs$candidate_cpu_s),
    comparator_median_s = median(pairs$comparator_s),
    median_paired_difference_comparator_minus_candidate_s = median(pairs$comparator_s - pairs$candidate_cpu_s),
    median_paired_difference = difference_interval,
    ratio_of_marginal_medians_comparator_over_candidate = median(pairs$comparator_s) / median(pairs$candidate_cpu_s),
    ratio_of_marginal_medians = ratio_interval,
    candidate_faster_case_count = sum(pairs$candidate_cpu_s < pairs$comparator_s),
    bootstrap = list(
      implementation = "R boot::boot and boot::boot.ci",
      package_version = as.character(packageVersion("boot")),
      method = "ordinary paired case bootstrap; BCa 95% confidence interval",
      replicates = R_REPLICATES, seed = seed, rng = paste(RNGkind(), collapse = "; ")
    ),
    pairs = pairs
  )
}

rows <- parse_labels(read_jsonl(input_path))
rows <- rows[rows$condition != "ours_int8", ]   # pre-NNCF weights; not an arm of the manuscript

comparisons <- list()
for (name in names(COMPARATORS)) {
  cond <- COMPARATORS[[name]]
  comparisons[[name]] <- list(
    full = run_comparison("full", cond, rows, BASE_SEED),
    fast = run_comparison("fast", cond, rows, BASE_SEED + 1L)
  )
}

result <- list(
  method = "10,000 ordinary paired case-level bootstrap resamples with BCa 95% confidence intervals",
  estimands = c("median paired wall-time difference", "ratio of marginal medians"),
  direction = "comparator minus/divided by INT8-TS CPU; positive differences and ratios above 1 favor INT8-TS CPU speed",
  candidate_condition = CANDIDATE,
  cases = "all 100 AMOS22 validation CT cases (n = 100 per arm and tier)",
  interpretation = paste(
    "Intervals quantify sampling variability across the 100 cases;",
    "one observation per condition and case does not estimate repeated-run system variability."
  ),
  environment = list(
    R_version = R.version.string,
    boot_version = as.character(packageVersion("boot")),
    jsonlite_version = as.character(packageVersion("jsonlite"))
  ),
  comparisons = comparisons
)

# The reported analysis ran on R 4.1.2 with boot 1.3-28 and jsonlite 1.7.3; other versions give
# the same values up to floating-point rounding.
if (!(identical(result$environment$boot_version, "1.3.28") &&
      grepl("R version 4.1.2", result$environment$R_version, fixed = TRUE))) {
  message("note: environment differs from the reported analysis (R 4.1.2, boot 1.3-28)")
}

write_json(result, output_path, pretty = TRUE, auto_unbox = TRUE, digits = 16, na = "null")

for (name in names(comparisons)) for (tier in c("full", "fast")) {
  x <- comparisons[[name]][[tier]]
  cat(sprintf("%-36s %-4s n=%d  ratio %.3f (95%% BCa %.3f, %.3f)  diff %+.1f s (%.1f, %.1f)  INT8 faster %d/100\n",
    name, tier, x$n, x$ratio_of_marginal_medians_comparator_over_candidate,
    x$ratio_of_marginal_medians$ci95[1], x$ratio_of_marginal_medians$ci95[2],
    x$median_paired_difference_comparator_minus_candidate_s,
    x$median_paired_difference$ci95[1], x$median_paired_difference$ci95[2],
    x$candidate_faster_case_count))
}
