#!/usr/bin/env Rscript
# Duke Spleen CT (28 series; DSDS_0021 excluded): spleen volume agreement with the same estimators as
# the AMOS analysis (amos_spleen_agreement.R): CCC with BCa 95% CI (10,000 resamples),
# relative bias and 1.96-SD limits of agreement (% of the pair mean), empirical 2.5th-97.5th
# percentiles of the relative difference, median absolute relative difference. Overlap (DSC, NSD) is
# summarised descriptively. Also summarises the station timing (Table 5, Duke rows; Table S13).
#
#   Rscript spleen_volumetry/duke_agreement.R [PERCASE_CSV] [STATION_TIMING_CSV] [OUT_DIR]
#
# Defaults: data/spleen/duke_percase.csv (scoring/duke_percase.py), data/runtime/station_duke_spleen.csv.
suppressPackageStartupMessages({ library(jsonlite); library(boot) })
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
cli <- commandArgs(trailingOnly = TRUE)
percase_csv <- if (length(cli) >= 1) cli[[1]] else file.path(here, "..", "data", "spleen", "duke_percase.csv")
timing_csv <- if (length(cli) >= 2) cli[[2]] else file.path(here, "..", "data", "runtime", "station_duke_spleen.csv")
out_dir <- if (length(cli) >= 3) cli[[3]] else file.path(here, "..", "results", "spleen_volumetry")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
d <- read.csv(percase_csv); d <- d[d$excluded == 0, ]
R <- 10000; set.seed(20261019)
ccc <- function(x, y) 2 * mean((x - mean(x)) * (y - mean(y))) / (var(x) * (length(x) - 1) / length(x) + var(y) * (length(y) - 1) / length(y) + (mean(x) - mean(y))^2)
bca <- function(stat, x, y) {
  b <- boot(data.frame(x, y), function(dd, i) stat(dd$x[i], dd$y[i]), R = R)
  ci <- tryCatch(boot.ci(b, type = "bca")$bca[4:5], error = function(e) boot.ci(b, type = "perc")$percent[4:5])
  list(est = b$t0, lo = ci[1], hi = ci[2])
}
agree <- function(test, ref, label) {
  diff <- test - ref; rel <- 100 * diff / ((test + ref) / 2); c_ccc <- bca(ccc, test, ref)
  list(label = label, n = length(test), ccc = c_ccc$est, ccc_ci = c(c_ccc$lo, c_ccc$hi),
       abs_bias_ml = mean(diff), abs_loa_ml = mean(diff) + c(-1.96, 1.96) * sd(diff),
       rel_bias_pct = mean(rel), rel_loa_pct = mean(rel) + c(-1.96, 1.96) * sd(rel),
       rel_empirical_2.5_97.5_pct = unname(quantile(rel, c(.025, .975))),
       median_abs_rel_pct = median(abs(rel)), median_abs_ml = median(abs(diff)), max_abs_rel_pct = max(abs(rel)), n_rel_gt5 = sum(abs(rel) > 5),
       ba_percentile_ci = ba_ci(rel))
}
# Percentile bootstrap CIs (10,000 case resamples) for the relative bias and the 1.96-SD limits.
ba_ci <- function(rel) {
  b <- boot(rel, function(x, i) { m <- mean(x[i]); s <- sd(x[i]); c(m, m - 1.96 * s, m + 1.96 * s) }, R = R)
  ci <- function(k) boot.ci(b, type = "perc", index = k)$percent[4:5]
  list(bias = ci(1), loa_low = ci(2), loa_high = ci(3))
}
desc <- function(x) list(median = median(x), q1 = unname(quantile(x, .25)), q3 = unname(quantile(x, .75)), min = min(x), max = max(x))
pairs <- list(
  agree(d$portable_dicom_ml, d$manual_ml,  "TS-Portable (PACS station, DICOM input) vs manual reference"),
  agree(d$ref2120_ml,        d$manual_ml,  "Reference implementation 2.12.0 (workstation) vs manual reference"),
  agree(d$portable_dicom_ml, d$ref2120_ml, "TS-Portable (PACS station, DICOM input) vs reference implementation 2.12.0"))
overlap <- list()
for (p in c("portable_dicom_vs_manual", "ref2120_vs_manual", "portable_dicom_vs_ref2120"))
  overlap[[p]] <- list(dsc = desc(d[[paste0("dsc_", p)]]), nsd1 = desc(d[[paste0("nsd1_", p)]]), nsd2 = desc(d[[paste0("nsd2_", p)]]), nsd3 = desc(d[[paste0("nsd3_", p)]]))
tm <- read.csv(timing_csv)
tm <- tm[tm$case %in% d$case & tm$status == "ok" & tolower(as.character(tm$warmup)) == "false", ]
w <- function(arm) tm$wall_seconds[tm$arm == arm][order(tm$case[tm$arm == arm])]
timing <- list(n = nrow(d), nifti_s = desc(w("duke_nifti")), dicom_s = desc(w("duke_dicom")), dicom_minus_nifti_s = desc(w("duke_dicom") - w("duke_nifti")))
out <- list(n = nrow(d), excluded = "DSDS_0021", resamples = R, manual_ml = desc(d$manual_ml), n_manual_gt_1000ml = sum(d$manual_ml > 1000),
            dicom_input_identical_to_nifti_input = sum(d$dicom_eq_nifti), pairs = pairs, overlap = overlap, timing = timing)
write_json(out, file.path(out_dir, "duke_agreement.json"), auto_unbox = TRUE, pretty = TRUE, digits = 6)
for (p in pairs) cat(sprintf("%-80s CCC %.5f (%.5f, %.5f); bias %+.2f%% (%+.2f, %+.2f); LoA %+.1f (%+.1f, %+.1f) to %+.1f (%+.1f, %+.1f); empirical (%+.1f, %+.1f); median |d| %.1f mL (%.2f%%); max %.1f%%\n",
  p$label, p$ccc, p$ccc_ci[1], p$ccc_ci[2], p$rel_bias_pct, p$ba_percentile_ci$bias[1], p$ba_percentile_ci$bias[2], p$rel_loa_pct[1], p$ba_percentile_ci$loa_low[1], p$ba_percentile_ci$loa_low[2], p$rel_loa_pct[2], p$ba_percentile_ci$loa_high[1], p$ba_percentile_ci$loa_high[2], p$rel_empirical_2.5_97.5_pct[1], p$rel_empirical_2.5_97.5_pct[2], p$median_abs_ml, p$median_abs_rel_pct, p$max_abs_rel_pct))
for (p in names(overlap)) cat(sprintf("%-28s DSC %.3f [%.3f-%.3f]  NSD3 %.3f [%.3f-%.3f]  NSD1 %.3f\n", p, overlap[[p]]$dsc$median, overlap[[p]]$dsc$q1, overlap[[p]]$dsc$q3, overlap[[p]]$nsd3$median, overlap[[p]]$nsd3$q1, overlap[[p]]$nsd3$q3, overlap[[p]]$nsd1$median))
cat(sprintf("timing (s): NIfTI median %.2f [%.2f-%.2f]; DICOM median %.2f [%.2f-%.2f]; DICOM-NIfTI median %.2f [%.2f-%.2f], max %.2f\n",
  timing$nifti_s$median, timing$nifti_s$q1, timing$nifti_s$q3, timing$dicom_s$median, timing$dicom_s$q1, timing$dicom_s$q3, timing$dicom_minus_nifti_s$median, timing$dicom_minus_nifti_s$q1, timing$dicom_minus_nifti_s$q3, timing$dicom_minus_nifti_s$max))
cat(sprintf("manual volume median %.0f mL [%.0f-%.0f], range %.0f-%.0f; >1000 mL: %d\n", out$manual_ml$median, out$manual_ml$q1, out$manual_ml$q3, out$manual_ml$min, out$manual_ml$max, out$n_manual_gt_1000ml))
