#!/usr/bin/env Rscript

# Per-organ paired differences in NSD at 1-mm and 2-mm tolerances on external AMOS
# (Supplemental Table S10). Uses run_structure_boot() from bootstrap_endpoints.R, so the BCa
# selection and percentile fallback are identical to the DSC and NSD@3 mm per-organ intervals.
#
#   Rscript noninferiority/bootstrap_nsd_tolerance_structures.R [INPUT_DIR] [OUT_DIR]

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
Sys.setenv(BOOTSTRAP_FUNCTIONS_ONLY = "1")
source(file.path(script_dir, "bootstrap_endpoints.R"), local = globalenv())
Sys.unsetenv("BOOTSTRAP_FUNCTIONS_ONLY")

tiers <- c(amos_full = "full", amos_fast = "fast")
tolerances <- c(1L, 2L)
STRUCTURE_BASE_SEED <- 20260725L + 1000L

read_raw <- function(tier) {
  lapply(readLines(file.path(input_dir, paste0("raw_amos_", tier, ".jsonl"))),
         function(line) fromJSON(line, simplifyVector = FALSE))
}

group_by_organ <- function(rows, tolerance) {
  int8_field <- paste0("nsd", tolerance, "_int8")
  reference_field <- paste0("nsd", tolerance, "_original_ts")
  grouped <- list()
  for (row in rows) {
    for (organ in row$organs) {
      name <- organ$organ
      if (is.null(grouped[[name]])) grouped[[name]] <- list(int8 = numeric(), reference = numeric())
      grouped[[name]]$int8 <- c(grouped[[name]]$int8, organ[[int8_field]])
      grouped[[name]]$reference <- c(grouped[[name]]$reference, organ[[reference_field]])
    }
  }
  grouped
}

for (key_index in seq_along(tiers)) {
  key <- names(tiers)[[key_index]]
  rows <- read_raw(tiers[[key_index]])
  output <- list(
    source = paste0("raw_amos_", tiers[[key_index]], ".jsonl"),
    estimand = "per-organ mean NSD at 1-mm or 2-mm tolerance over eligible cases; paired difference reference implementation minus INT8-TS",
    method = list(
      implementation = "R boot package, run_structure_boot from bootstrap_endpoints.R",
      r_version = R.version.string,
      boot_version = as.character(packageVersion("boot")),
      primary_interval = "BCa when finite and warning-free with finite jackknife acceleration; percentile otherwise",
      paired_resampling_unit = "case",
      replicates = R_REPLICATES,
      base_seed = STRUCTURE_BASE_SEED
    )
  )
  for (tolerance in tolerances) {
    grouped <- group_by_organ(rows, tolerance)
    organ_names <- sort(names(grouped))
    per_organ <- list()
    for (organ_index in seq_along(organ_names)) {
      name <- organ_names[[organ_index]]
      seed <- STRUCTURE_BASE_SEED + tolerance * 100L + key_index * 1000L + organ_index
      per_organ[[name]] <- run_structure_boot(grouped[[name]]$int8, grouped[[name]]$reference, seed)
    }
    output[[paste0("nsd", tolerance)]] <- per_organ
  }
  out_path <- file.path(out_dir, paste0("nsd_tolerance_structure_ci_", key, ".json"))
  write_json(output, out_path, auto_unbox = TRUE, pretty = TRUE, digits = 16, null = "null")
  cat("wrote", out_path, "\n")
}
