#!/usr/bin/env Rscript
# Figure 4: per-case cold end-to-end wall time, five arms, 100 AMOS cases, both tiers (log axis).
suppressPackageStartupMessages({ library(ggplot2); library(jsonlite); library(patchwork) })
#
#   Rscript figures/fig4_runtime_dots.R [TIMING_JSONL] [OUT_DIR]
#
# Reads data/runtime/workstation_timing_amos.jsonl; writes fig4_runtime_dots.pdf and .png.
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))
cli <- commandArgs(trailingOnly = TRUE)
timing <- if (length(cli) >= 1) cli[[1]] else file.path(here, "..", "data", "runtime", "workstation_timing_amos.jsonl")
out_dir <- if (length(cli) >= 2) cli[[2]] else file.path(here, "..", "results", "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
lines <- readLines(timing)
rows <- do.call(rbind, lapply(lines[nzchar(lines)], function(l) { r <- fromJSON(l); if (r$rc != 0) return(NULL)
  p <- strsplit(r$label, "|", fixed = TRUE)[[1]]; data.frame(cond = p[1], tier = p[2], case = p[3], wall = r$proc_wall_s) }))
arms <- c(ts_gpu = "Reference\nimplementation\nGPU", ts_cpu = "Reference\nimplementation\nCPU",
          ours_fp32 = "TS-Portable\nfp32 ONNX\nCPU", ours_fp32_cuda = "TS-Portable\nfp32 ONNX\nGPU",
          ours_int8_nncf = "TS-Portable\nINT8\nCPU")
rows <- rows[rows$cond %in% names(arms), ]
rows$arm <- factor(arms[rows$cond], levels = arms)
rows$tier <- factor(rows$tier, levels = c("full", "fast"), labels = c("A  1.5-mm resolution", "B  3-mm resolution"))
med <- aggregate(wall ~ arm + tier, rows, median)
fill <- c("grey75", "grey75", "grey45", "grey45", "#1f77b4"); names(fill) <- arms
p <- ggplot(rows, aes(arm, wall)) +
  geom_jitter(aes(colour = arm), width = 0.18, height = 0, size = 1.1, alpha = 0.55, show.legend = FALSE) +
  geom_crossbar(data = med, aes(arm, wall, ymin = wall, ymax = wall), width = 0.6, middle.linewidth = 0, linewidth = 0.7) +
  geom_label(data = med, aes(arm, wall, label = sprintf("%.1f s", wall)), vjust = -0.35, size = 3, label.size = 0, fill = "white", alpha = 0.85) +
  scale_colour_manual(values = fill) +
  scale_y_log10(breaks = c(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000), labels = function(x) format(x, big.mark = ","), expand = expansion(mult = c(0.05, 0.12))) +
  scale_x_discrete(expand = expansion(add = 0.7)) +
  facet_wrap(~ tier, ncol = 2, scales = "free_y") +
  labs(x = NULL, y = "Cold end-to-end wall time per case (s, log scale)") +
  theme_classic(base_size = 10) +
  theme(strip.background = element_blank(), strip.text = element_text(hjust = 0, face = "bold", size = 11),
        axis.text.x = element_text(size = 8, lineheight = 0.9), panel.spacing.x = unit(1.5, "lines"))
out <- file.path(out_dir, "fig4_runtime_dots")
ggsave(paste0(out, ".pdf"), p, width = 10.5, height = 5)
ggsave(paste0(out, ".png"), p, width = 10.5, height = 5, dpi = 300)
cat("wrote", out, "\n"); print(med)
