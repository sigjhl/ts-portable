#!/usr/bin/env Rscript
# Volume-stratified quantization-effect analysis (Supplemental Table S11).
#
# Strata are of STRUCTURES: each structure is assigned once, by its median reference volume in the
# dataset, to <25 / 25-75 / 75-185 / >185 mL. Within a stratum, the estimand is the mean paired
# difference (original TotalSegmentator minus INT8-TS) over all instances of its structures; the
# bootstrap resamples CASES (all instances of a resampled case travel together). Same engine as the
# submitted analyses: boot::boot, 10,000 ordinary resamples, BCa 95% intervals, 0.01 margin.
# Also: hollow vs non-hollow, per-structure differences with intervals, Spearman of per-structure
# difference vs log10 median volume, and a two-panel figure.
#
#   Rscript noninferiority/volume_strata_analysis.R [DATA_DIR] [OUT_DIR]
#
# DATA_DIR defaults to evaluation/data; it must hold volume_strata/reference_volumes.csv
# (scoring/reference_volumes.py) and accuracy/percase_{ts,amos}_full.json.
suppressPackageStartupMessages(library(boot))
suppressPackageStartupMessages(library(jsonlite))

here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) normalizePath(args[[1]]) else normalizePath(file.path(here, "..", "data"))
out_dir <- if (length(args) >= 2) args[[2]] else file.path(here, "..", "results", "volume_strata")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

CUTS <- c(25, 75, 185)
STRATUM_NAMES <- c("<25 mL", "25-75 mL", "75-185 mL", ">185 mL")
MARGIN <- 0.01
R_REPLICATES <- 10000L
BASE_SEED <- 20260905L
HOLLOW <- c("esophagus", "stomach", "duodenum", "small_bowel", "colon", "gallbladder", "urinary_bladder")

stratum_of <- function(v) STRATUM_NAMES[1 + findInterval(v, CUTS)]

# ---------------------------------------------------------------- inputs
volumes <- read.csv(file.path(data_dir, "volume_strata", "reference_volumes.csv"), stringsAsFactors = FALSE)
volumes <- volumes[volumes$voxels > 0, ]

load_instances <- function(path, dataset) {
  cases <- fromJSON(path, simplifyVector = FALSE)
  rows <- do.call(rbind, lapply(cases, function(cs) {
    do.call(rbind, lapply(cs$organs, function(o) data.frame(
      dataset = dataset, case = cs$case, structure = o$organ,
      d_dsc = o$dsc_stock_ts_vs_manual - o$dsc_int8_vs_manual,
      d_nsd = o$nsd3_stock_ts - o$nsd3_int8,
      stringsAsFactors = FALSE)))
  }))
  rows
}
inst <- rbind(
  load_instances(file.path(data_dir, "accuracy", "percase_ts_full.json"), "ts_dev"),
  load_instances(file.path(data_dir, "accuracy", "percase_amos_full.json"), "amos")
)
stopifnot(!any(is.na(inst$d_dsc)), !any(is.na(inst$d_nsd)))

# structure median volume per dataset -> stratum
med_vol <- aggregate(volume_ml ~ dataset + structure, data = volumes, FUN = median)
names(med_vol)[3] <- "median_volume_ml"
med_vol$stratum <- stratum_of(med_vol$median_volume_ml)
med_vol$hollow <- med_vol$structure %in% HOLLOW
inst <- merge(inst, med_vol, by = c("dataset", "structure"), all.x = TRUE)
stopifnot(!any(is.na(inst$stratum)))

# ---------------------------------------------------------------- bootstrap machinery
# Pooled mean over instances of the resampled cases: statistic = sum(per-case sums)/sum(per-case counts).
pooled_boot <- function(sub, metric, seed) {
  d <- sub[[metric]]
  by_case <- split(d, sub$case)
  sums <- vapply(by_case, sum, numeric(1)); counts <- vapply(by_case, length, numeric(1))
  df <- data.frame(s = sums, n = counts)
  stat <- function(data, idx) sum(data$s[idx]) / sum(data$n[idx])
  set.seed(seed)
  fit <- boot(df, stat, R = R_REPLICATES, sim = "ordinary", stype = "i")
  ci <- tryCatch(boot.ci(fit, conf = 0.95, type = "bca")$bca[1, 4:5], error = function(e) c(NA, NA))
  list(n_cases = nrow(df), n_instances = length(d), estimate = mean(d),
       ci95_low = ci[[1]], ci95_high = ci[[2]], bca_ok = all(is.finite(ci)),
       noninferior = is.finite(ci[[2]]) && ci[[2]] < MARGIN)
}

run_groups <- function(group_col, label) {
  res <- list(); rows <- NULL; k <- 0L
  for (ds in c("ts_dev", "amos")) for (g in sort(unique(inst[[group_col]][inst$dataset == ds]))) {
    sub <- inst[inst$dataset == ds & inst[[group_col]] == g, ]
    for (metric in c("d_dsc", "d_nsd")) {
      k <- k + 1L
      r <- pooled_boot(sub, metric, BASE_SEED + k)
      r$structures <- sort(unique(sub$structure))
      res[[paste(ds, g, metric, sep = "|")]] <- r
      rows <- rbind(rows, data.frame(dataset = ds, group = as.character(g), metric = sub("d_", "", metric),
        n_structures = length(r$structures), n_cases = r$n_cases, n_instances = r$n_instances,
        mean_difference = r$estimate, ci95_low = r$ci95_low, ci95_high = r$ci95_high,
        noninferior_at_0.01 = r$noninferior, stringsAsFactors = FALSE))
    }
  }
  cat(sprintf("\n== %s ==\n", label)); print(rows, digits = 4, row.names = FALSE)
  list(results = res, table = rows)
}

strata <- run_groups("stratum", "strata (structures binned by median reference volume)")
hollow <- run_groups("hollow", "hollow (TRUE) vs non-hollow (FALSE)")

# ---------------------------------------------------------------- per-structure
per_structure <- NULL; k <- 5000L
for (ds in c("ts_dev", "amos")) for (s in sort(unique(inst$structure[inst$dataset == ds]))) {
  sub <- inst[inst$dataset == ds & inst$structure == s, ]
  row <- data.frame(dataset = ds, structure = s, n_cases = nrow(sub),
    median_volume_ml = sub$median_volume_ml[1], stratum = sub$stratum[1], hollow = sub$hollow[1],
    stringsAsFactors = FALSE)
  for (metric in c("d_dsc", "d_nsd")) {
    k <- k + 1L
    r <- if (nrow(sub) >= 5) pooled_boot(sub, metric, BASE_SEED + k) else
      list(estimate = mean(sub[[metric]]), ci95_low = NA, ci95_high = NA, noninferior = NA)
    m <- sub("d_", "", metric)
    row[[paste0(m, "_mean_difference")]] <- r$estimate
    row[[paste0(m, "_ci95_low")]] <- r$ci95_low
    row[[paste0(m, "_ci95_high")]] <- r$ci95_high
    row[[paste0(m, "_upper_exceeds_margin")]] <- is.finite(r$ci95_high) && r$ci95_high >= MARGIN
  }
  per_structure <- rbind(per_structure, row)
}
exceed <- per_structure[per_structure$dsc_upper_exceeds_margin | per_structure$nsd_upper_exceeds_margin, ]
cat("\n== structures whose own 95% BCa upper bound reaches the 0.01 margin ==\n")
print(exceed[, c("dataset", "structure", "n_cases", "median_volume_ml", "dsc_mean_difference", "dsc_ci95_high", "nsd_mean_difference", "nsd_ci95_high")], digits = 4, row.names = FALSE)

# ---------------------------------------------------------------- Spearman
spearman <- NULL
for (ds in c("ts_dev", "amos")) for (m in c("dsc", "nsd")) {
  ps <- per_structure[per_structure$dataset == ds, ]
  ct <- suppressWarnings(cor.test(log10(ps$median_volume_ml), ps[[paste0(m, "_mean_difference")]], method = "spearman"))
  spearman <- rbind(spearman, data.frame(dataset = ds, metric = m, n_structures = nrow(ps),
    rho = unname(ct$estimate), p_value = ct$p.value, stringsAsFactors = FALSE))
}
cat("\n== Spearman: per-structure mean difference vs log10 median volume ==\n"); print(spearman, digits = 4, row.names = FALSE)

# ---------------------------------------------------------------- outputs
write.csv(strata$table, file.path(out_dir, "strata_table.csv"), row.names = FALSE)
write.csv(hollow$table, file.path(out_dir, "hollow_table.csv"), row.names = FALSE)
write.csv(per_structure, file.path(out_dir, "per_structure.csv"), row.names = FALSE)
write.csv(spearman, file.path(out_dir, "spearman.csv"), row.names = FALSE)
write_json(list(
  design = list(unit_of_stratification = "structure, by median reference volume in the dataset",
    cutpoints_ml = CUTS, strata = STRATUM_NAMES, hollow_structures = HOLLOW, margin = MARGIN,
    estimand = "mean paired difference (original TotalSegmentator minus INT8-TS) over all instances of the group's structures",
    bootstrap = list(implementation = "R boot::boot / boot.ci", method = "ordinary case-level resampling; BCa 95%",
      replicates = R_REPLICATES, base_seed = BASE_SEED), tier = "full (1.5 mm)",
    inputs = c("dsc/results/main/stock_comparator_percase_ts_full.json", "dsc/results/main/stock_comparator_percase_amos_full.json",
      "paper_results/volume_strata/reference_volumes.csv")),
  environment = list(R_version = R.version.string, boot_version = as.character(packageVersion("boot"))),
  strata = strata$results, hollow = hollow$results, per_structure = per_structure, spearman = spearman
), file.path(out_dir, "volume_strata_results.json"), pretty = TRUE, auto_unbox = TRUE, digits = 10, na = "null")

# ---------------------------------------------------------------- figure (two panels, DSC)
pdf(file.path(out_dir, "fig_volume_strata_dsc.pdf"), width = 11, height = 5)
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3, 1))
for (ds in c("ts_dev", "amos")) {
  ps <- per_structure[per_structure$dataset == ds, ]
  ylim <- range(c(ps$dsc_ci95_low, ps$dsc_ci95_high, MARGIN, -MARGIN), na.rm = TRUE)
  plot(ps$median_volume_ml, ps$dsc_mean_difference, log = "x", pch = 19, cex = 0.8,
       col = ifelse(ps$hollow, "#d95f02", "#1b7837"), ylim = ylim,
       xlab = "Median reference volume (mL, log scale)", ylab = "DSC difference (original TS minus INT8-TS)",
       main = if (ds == "amos") "AMOS (external, 15 organs)" else "Development set (117 structures)")
  ok <- is.finite(ps$dsc_ci95_low)
  arrows(ps$median_volume_ml[ok], ps$dsc_ci95_low[ok], ps$median_volume_ml[ok], ps$dsc_ci95_high[ok],
         angle = 90, code = 3, length = 0.02, col = "grey50")
  abline(h = c(0, MARGIN), lty = c(1, 2)); abline(v = CUTS, lty = 3, col = "grey70")
  if (ds == "amos") text(ps$median_volume_ml, ps$dsc_mean_difference, ps$structure, cex = 0.55, pos = 4)
  legend("topright", c("hollow", "non-hollow", "0.01 margin"), col = c("#d95f02", "#1b7837", "black"),
         pch = c(19, 19, NA), lty = c(NA, NA, 2), bty = "n", cex = 0.8)
}
dev.off()
cat("\nwritten to", out_dir, "\n")
