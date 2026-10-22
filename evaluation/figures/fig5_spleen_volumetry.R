#!/usr/bin/env Rscript
# Figure 5: spleen volumetry in spleen-only mode on the PACS reading station, 100 external AMOS cases.
#
#   Rscript figures/fig5_spleen_volumetry.R [VOLUMES_CSV] [OUT_DIR]
#
# Reads data/spleen/amos_spleen_volumes_percase.csv; writes fig5_spleen_volumetry.pdf and .png.
suppressPackageStartupMessages({ library(ggplot2); library(patchwork) })
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
cli <- commandArgs(trailingOnly = TRUE)
volumes_csv <- if (length(cli) >= 1) cli[[1]] else file.path(here, "..", "data", "spleen", "amos_spleen_volumes_percase.csv")
out_dir <- if (length(cli) >= 2) cli[[2]] else file.path(here, "..", "results", "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
d <- read.csv(volumes_csv)

# ---- figure: PACS station, spleen-only mode -------------------------------------------------------
blue <- "#1f77b4"; grey <- "grey45"
BRK <- c(3, 10, 30, 100, 300, 1000); LIM <- c(2.5, 1200)   # shared volume axes so the four panels align
sc <- function(x, y, xl, yl, ttl) {
  ggplot(data.frame(x, y), aes(x, y)) + geom_abline(slope = 1, intercept = 0, colour = grey, linetype = 2) +
    geom_point(size = 1.4, alpha = 0.7, colour = blue) +
    scale_x_log10(breaks = BRK, limits = LIM) + scale_y_log10(breaks = BRK, limits = LIM) +
    labs(x = xl, y = yl, title = ttl) + theme_classic(base_size = 10) +
    theme(plot.title = element_text(face = "bold", size = 10, hjust = 0), plot.title.position = "plot", aspect.ratio = 1)
}
ba_core <- function(m, rel, b, s, lim, brk, base) {
  ggplot(data.frame(m, rel), aes(m, rel)) +
    geom_hline(yintercept = 0, colour = grey) +
    geom_hline(yintercept = b, colour = blue) + geom_hline(yintercept = b + c(-1.96, 1.96) * s, colour = blue, linetype = 2) +
    geom_point(size = if (base > 8) 1.4 else 1.0, alpha = 0.7, colour = blue) +
    scale_x_log10(breaks = BRK, limits = LIM) + scale_y_continuous(breaks = brk) + coord_cartesian(ylim = lim) +
    theme_classic(base_size = base)
}
# Full-range Bland-Altman panel (every case shown) with an inset of the central distribution
# (same data, y-axis zoomed; the bias and limits are those of all 100 cases). `inset_y` is the zoomed
# y-range, `where` the inset position in panel coordinates (left, bottom, right, top).
ba <- function(test, ref, yl, ttl, lim, inset_y, inset_brk, where) {
  m <- (test + ref) / 2; rel <- 100 * (test - ref) / m; b <- mean(rel); s <- sd(rel)
  main <- ba_core(m, rel, b, s, lim, seq(-100, 60, 20), 10) +
    labs(x = "Mean of the two volumes (mL, log scale)", y = yl, title = ttl) +
    theme(plot.title = element_text(face = "bold", size = 10, hjust = 0), plot.title.position = "plot", aspect.ratio = 1)
  ins <- ba_core(m, rel, b, s, inset_y, inset_brk, 7) +
    labs(x = NULL, y = NULL) +
    theme(plot.background = element_rect(fill = "white", colour = "grey60", linewidth = 0.3),
          plot.margin = margin(3, 4, 2, 2), axis.text = element_text(size = 6))
  main + inset_element(ins, left = where[1], bottom = where[2], right = where[3], top = where[4], align_to = "panel")
}
pA <- sc(d$manual_ml, d$pacs_int8_openvino_ml, "Manual reference volume (mL, log scale)", "TS-Portable volume (mL, log scale)", "A  Volumes: TS-Portable vs manual reference")
pB <- ba(d$pacs_int8_openvino_ml, d$manual_ml, "TS-Portable minus manual (% of mean)", "B  Bland-Altman: TS-Portable vs manual reference", c(-115, 60),
         c(-32, 22), seq(-30, 20, 10), c(0.50, 0.03, 0.99, 0.45))
pC <- sc(d$pacs_stock_ts_cpu_ml, d$pacs_int8_openvino_ml, "Reference implementation volume (mL, log scale)", "TS-Portable volume (mL, log scale)", "C  Volumes: TS-Portable vs reference implementation")
pD <- ba(d$pacs_int8_openvino_ml, d$pacs_stock_ts_cpu_ml, "TS-Portable minus reference implementation (% of mean)", "D  Bland-Altman: TS-Portable vs reference implementation", c(-115, 60),
         c(-12, 12), seq(-8, 8, 4), c(0.50, 0.03, 0.99, 0.45))
p <- (pA | pB) / (pC | pD)
out <- file.path(out_dir, "fig5_spleen_volumetry")
ggsave(paste0(out, ".pdf"), p, width = 9, height = 8.4)
ggsave(paste0(out, ".png"), p, width = 9, height = 8.4, dpi = 300)
cat("wrote", out, "\n")
