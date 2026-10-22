#!/usr/bin/env Rscript
# Spleen volume agreement, 100 external AMOS cases (Table 5, AMOS rows; Supplemental Table S12).
# Lin's CCC with BCa 95% CI, Bland-Altman relative bias and 1.96-SD limits of agreement (% of the pair
# mean), empirical 2.5th-97.5th percentiles, median absolute difference, and the limits without the
# smallest spleen (amos_0167). The bootstrap CIs of the bias and limits are computed by
# amos_bland_altman_ci.py.
#
#   Rscript spleen_volumetry/amos_spleen_agreement.R [VOLUMES_CSV] [OUT_DIR]
#
# VOLUMES_CSV defaults to data/spleen/amos_spleen_volumes_percase.csv (written by
# scoring/spleen_volumes.py from the manual AMOS masks and each arm's masks).
suppressPackageStartupMessages({ library(jsonlite); library(boot) })
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
cli <- commandArgs(trailingOnly = TRUE)
volumes_csv <- if (length(cli) >= 1) cli[[1]] else file.path(here, "..", "data", "spleen", "amos_spleen_volumes_percase.csv")
out_dir <- if (length(cli) >= 2) cli[[2]] else file.path(here, "..", "results", "spleen_volumetry")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
d <- read.csv(volumes_csv)
THR <- 314.5   # upper limit of normal splenic volume (Prassopoulos 1997)
R <- 10000; set.seed(20260718)

ccc <- function(x, y) 2 * mean((x - mean(x)) * (y - mean(y))) / (var(x) * (length(x) - 1) / length(x) + var(y) * (length(y) - 1) / length(y) + (mean(x) - mean(y))^2)
bca <- function(stat, x, y) {
  b <- boot(data.frame(x, y), function(dd, i) stat(dd$x[i], dd$y[i]), R = R)
  ci <- tryCatch(boot.ci(b, type = "bca")$bca[4:5], error = function(e) boot.ci(b, type = "perc")$percent[4:5])
  list(est = b$t0, lo = ci[1], hi = ci[2])
}
agree <- function(test, ref, label) {
  diff <- test - ref; rel <- 100 * diff / ((test + ref) / 2)
  c_ccc <- bca(ccc, test, ref)
  list(label = label, n = length(test),
       ccc = c_ccc$est, ccc_ci = c(c_ccc$lo, c_ccc$hi),
       abs_bias_ml = mean(diff), abs_loa_ml = mean(diff) + c(-1.96, 1.96) * sd(diff),
       rel_bias_pct = mean(rel), rel_loa_pct = mean(rel) + c(-1.96, 1.96) * sd(rel),
       median_abs_rel_pct = median(abs(rel)), max_abs_rel_pct = max(abs(rel)),
       n_above_test = sum(test > THR), n_above_ref = sum(ref > THR),
       concordant = sum((test > THR) == (ref > THR)),
       discordant_cases = d$case[(test > THR) != (ref > THR)],
       n_rel_gt5 = sum(abs(rel) > 5))
}
pairs <- list(
  agree(d$pacs_int8_openvino_ml, d$manual_ml,             "PACS station, spleen-only mode: TS-Portable vs manual reference"),
  agree(d$pacs_stock_ts_cpu_ml,  d$manual_ml,             "PACS station, spleen-only mode: reference implementation 2.4.0 (CPU) vs manual reference"),
  agree(d$pacs_int8_openvino_ml, d$pacs_stock_ts_cpu_ml,  "PACS station, spleen-only mode: TS-Portable vs reference implementation 2.4.0 (CPU)"),
  agree(d$pacs_stock_ts_cpu_ml,  d$pacs_stock_ts_cuda_ml, "PACS station, spleen-only mode: reference implementation CPU vs CUDA"),
  agree(d$int8_full_ml,          d$manual_ml,             "Workstation, full 117-label mode: TS-Portable vs manual reference"),
  agree(d$reference_full_ml,     d$manual_ml,             "Workstation, full 117-label mode: reference implementation 2.12.0 vs manual reference"),
  agree(d$int8_full_ml,          d$reference_full_ml,     "Workstation, full 117-label mode: TS-Portable vs reference implementation 2.12.0"),
  agree(d$fp32_full_ml,          d$reference_full_ml,     "Workstation, full 117-label mode: fp32 ONNX vs reference implementation 2.12.0"),
  agree(d$pacs_int8_openvino_ml, d$int8_full_ml,          "TS-Portable: spleen-only mode (PACS station) vs full mode (workstation)"),
  agree(d$pacs_stock_ts_cpu_ml,  d$reference_full_ml,     "Reference implementation: spleen-only mode 2.4.0 (PACS station) vs full mode 2.12.0 (workstation)"))
manual <- list(median_ml = median(d$manual_ml), q1 = quantile(d$manual_ml, .25)[[1]], q3 = quantile(d$manual_ml, .75)[[1]],
               min_ml = min(d$manual_ml), max_ml = max(d$manual_ml), n_above = sum(d$manual_ml > THR))

# Bland-Altman detail for the six comparisons of Table 5 and Table S12: empirical percentiles of
# the relative difference, median absolute difference, and the limits without the smallest spleen.
ba_detail <- function(test, ref, label) {
  rel <- 100 * (test - ref) / ((test + ref) / 2); keep <- d$case != "amos_0167"
  list(label = label, rel_bias_pct = mean(rel), rel_loa_pct = mean(rel) + c(-1.96, 1.96) * sd(rel),
       rel_empirical_2.5_97.5_pct = unname(quantile(rel, c(.025, .975))),
       median_abs_ml = median(abs(test - ref)), median_abs_rel_pct = median(abs(rel)),
       n_abs_rel_gt10 = sum(abs(rel) > 10), within_loa_frac = mean(abs(rel - mean(rel)) <= 1.96 * sd(rel)),
       without_amos_0167 = list(rel_bias_pct = mean(rel[keep]), rel_loa_pct = mean(rel[keep]) + c(-1.96, 1.96) * sd(rel[keep])))
}
bland_altman <- list(
  ba_detail(d$pacs_int8_openvino_ml, d$manual_ml,            "PACS spleen-only: TS-Portable vs manual"),
  ba_detail(d$pacs_stock_ts_cpu_ml,  d$manual_ml,            "PACS spleen-only: reference 2.4.0 vs manual"),
  ba_detail(d$pacs_int8_openvino_ml, d$pacs_stock_ts_cpu_ml, "PACS spleen-only: TS-Portable vs reference 2.4.0"),
  ba_detail(d$int8_full_ml,          d$manual_ml,            "Workstation full mode: TS-Portable vs manual"),
  ba_detail(d$reference_full_ml,     d$manual_ml,            "Workstation full mode: reference 2.12.0 vs manual"),
  ba_detail(d$int8_full_ml,          d$reference_full_ml,    "Workstation full mode: TS-Portable vs reference 2.12.0"))
write_json(list(threshold_ml = THR, resamples = R, manual = manual, pairs = pairs, bland_altman = bland_altman),
           file.path(out_dir, "amos_spleen_agreement.json"), auto_unbox = TRUE, pretty = TRUE, digits = 6)
for (p in bland_altman) cat(sprintf("%-55s bias %+.1f%%; LoA %+.1f to %+.1f; empirical %+.1f to %+.1f; median |d| %.1f mL (%.1f%%); >10%%: %d; without amos_0167 LoA %+.1f to %+.1f\n",
  p$label, p$rel_bias_pct, p$rel_loa_pct[1], p$rel_loa_pct[2], p$rel_empirical_2.5_97.5_pct[1], p$rel_empirical_2.5_97.5_pct[2],
  p$median_abs_ml, p$median_abs_rel_pct, p$n_abs_rel_gt10, p$without_amos_0167$rel_loa_pct[1], p$without_amos_0167$rel_loa_pct[2]))
for (p in pairs) cat(sprintf("%-100s CCC %.4f (%.4f, %.4f); rel bias %+.2f%% LoA (%+.1f, %+.1f); concordant %d/100\n",
                             p$label, p$ccc, p$ccc_ci[1], p$ccc_ci[2], p$rel_bias_pct, p$rel_loa_pct[1], p$rel_loa_pct[2], p$concordant))

