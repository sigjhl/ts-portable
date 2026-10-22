#!/usr/bin/env Rscript
# Supplemental Figure S3: per-structure pooled DSC difference (reference minus TS-Portable) vs median reference volume,
# two panels (development set, AMOS), with 95% BCa bars, margin line, and stratum cutpoints.
suppressPackageStartupMessages({ library(ggplot2); library(patchwork) })
#
#   Rscript figures/figS3_volume_strata.R [PER_STRUCTURE_CSV] [OUT_DIR]
#
# Reads results/volume_strata/per_structure.csv, written by noninferiority/volume_strata_analysis.R.
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
cli <- commandArgs(trailingOnly = TRUE)
per_structure <- if (length(cli) >= 1) cli[[1]] else file.path(here, "..", "results", "volume_strata", "per_structure.csv")
out_dir <- if (length(cli) >= 2) cli[[2]] else file.path(here, "..", "results", "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
ps <- read.csv(per_structure, stringsAsFactors = FALSE)
ps$panel <- factor(ps$dataset, levels = c("ts_dev", "amos"),
                   labels = c("A  TotalSegmentator development set (117 structures, 102 scans)", "B  External AMOS (15 organs, 100 scans)"))
ps$flag_dsc <- ifelse(ps$dsc_ci95_high > 0.01, "upper bound above margin", "within margin")
ps$flag_nsd <- ifelse(ps$nsd_ci95_high > 0.01, "upper bound above margin", "within margin")
cuts <- c(25, 75, 185)
mk <- function(d, ylim, metric = "dsc", ylab = "DSC difference, reference minus TS-Portable") { d$flag <- d[[paste0("flag_", metric)]]; ggplot(d, aes(median_volume_ml, .data[[paste0(metric, "_mean_difference")]])) +
  geom_hline(yintercept = 0, linewidth = 0.4) + geom_hline(yintercept = 0.01, linetype = "dashed", linewidth = 0.5) +
  geom_vline(xintercept = cuts, linetype = "dotted", colour = "grey55") +
  geom_errorbar(aes(ymin = .data[[paste0(metric, "_ci95_low")]], ymax = .data[[paste0(metric, "_ci95_high")]], colour = flag), width = 0, linewidth = 0.45, alpha = 0.8) +
  geom_point(aes(colour = flag), size = 1.6) +
  scale_colour_manual(values = c("within margin" = "#1f77b4", "upper bound above margin" = "#d62728"), name = NULL, limits = c("within margin", "upper bound above margin")) +
  scale_x_log10(breaks = c(1, 3, 10, 30, 100, 300, 1000, 3000), labels = function(x) format(x, big.mark = ",")) +
  coord_cartesian(ylim = ylim) +
  annotate("text", x = c(12, 43, 118, 800), y = ylim[2], label = c("<25 mL", "25-75 mL", "75-185 mL", ">185 mL"), size = 3, colour = "grey30", vjust = 1) +
  labs(x = "Median reference volume of the structure (mL, log scale)", y = ylab) +
  theme_classic(base_size = 10) + theme(legend.position = "none", plot.title = element_text(face = "bold", size = 11)) }
yl <- c(-0.008, 0.012)
pa <- mk(ps[ps$dataset == "ts_dev", ], yl) + ggtitle("A  Development set, DSC")
pb <- mk(ps[ps$dataset == "amos", ], yl) + ggtitle("B  External AMOS, DSC")
pc <- mk(ps[ps$dataset == "ts_dev", ], yl, "nsd", "NSD@3 mm difference, reference minus TS-Portable") + ggtitle("C  Development set, NSD@3 mm")
pd <- mk(ps[ps$dataset == "amos", ], yl, "nsd", "NSD@3 mm difference, reference minus TS-Portable") + ggtitle("D  External AMOS, NSD@3 mm")
p <- (pa | pb) / (pc | pd)
out <- file.path(out_dir, "figS3_volume_strata")
ggsave(paste0(out, ".pdf"), p, width = 11, height = 9); ggsave(paste0(out, ".png"), p, width = 11, height = 9, dpi = 300)
cat("wrote", out, "| structures above margin (either metric):", sum(pmax(ps$dsc_ci95_high, ps$nsd_ci95_high) > 0.01), "\n")
