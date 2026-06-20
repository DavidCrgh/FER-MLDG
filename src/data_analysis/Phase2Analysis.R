# Phase 2 Analysis: Welch's t-tests by dataset (num_classes)
# Compares APVIT vs MLDG on best_accuracy for each dataset

library(ggplot2)
library(dplyr)

df <- read.csv("phase2-raw.csv", stringsAsFactors = FALSE)

# Parse types: best_accuracy as numeric, num_classes as factor
df$best_accuracy <- as.numeric(df$best_accuracy)
df$num_classes   <- factor(df$num_classes)

# Unique datasets (num_classes)
datasets <- sort(unique(df$num_classes))

for (dataset in datasets) {
  sub <- df[df$num_classes == dataset, ]
  
  apvit <- sub$best_accuracy[sub$label_generator == "APVIT"]
  mldg  <- sub$best_accuracy[sub$label_generator == "MLDG"]
  
  # Welch's t-test (var.equal = FALSE)
  tt <- t.test(apvit, mldg, var.equal = FALSE)
  
  cat("\n")
  cat("========== Dataset (num_classes): ", as.character(dataset), " ==========\n", sep = "")
  cat("  Mean APVIT:  ", round(mean(apvit), 4), "\n")
  cat("  Mean MLDG:   ", round(mean(mldg), 4), "\n")
  cat("  t-statistic: ", round(tt$statistic, 4), "\n")
  cat("  df:          ", round(tt$parameter, 2), "\n")
  cat("  p-value:     ", format.pval(tt$p.value, digits = 4), "\n")
  cat("  95% CI (MLDG - APVIT): [",
      round(-tt$conf.int[2], 4), ", ",
      round(-tt$conf.int[1], 4), "] pp\n", sep = "")
  cat("\n")
}

#############################################
## 2^2 FACTORIAL ANALYSIS VIA lm()        ##
#############################################
# ±1 contrast coding:
#   G: APVIT = -1, MLDG = +1
#   D: 7-class = -1, 8-class = +1

df$G <- ifelse(df$label_generator == "MLDG", 1, -1)
df$D <- ifelse(df$num_classes == "8", 1, -1)

fit <- lm(best_accuracy ~ G * D, data = df)
cf  <- coef(fit)

cat("\n")
cat("========== 2^2 Factorial Effect Estimates (via lm) ==========\n")
cat("  Grand mean:      ", round(cf["(Intercept)"], 4), "\n")
cat("  Main effect G    (MLDG − APVIT):      ", round(cf["G"],   4), "pp\n")
cat("  Main effect D    (8-class − 7-class): ", round(cf["D"],   4), "pp\n")
cat("  G×D interaction:                      ", round(cf["G:D"], 4), "pp\n")
cat("\n")

cat("========== Inferential Summary ==========\n")
print(summary(fit))

#############################################
## PHASE II CELL MEANS GROUPED BAR CHART  ##
#############################################

cell_stats <- group_by(df, label_generator, num_classes) %>%
  summarise(
    mean_acc = mean(best_accuracy, na.rm = TRUE),
    ymin     = min(best_accuracy, na.rm = TRUE),
    ymax     = max(best_accuracy, na.rm = TRUE),
    .groups  = "drop"
  )

cell_stats$guidance_label <- factor(
  cell_stats$label_generator,
  levels = c("APVIT", "MLDG"),
  labels = c("APViT", "M-LDG")
)
cell_stats$dataset_label <- factor(
  cell_stats$num_classes,
  levels = c("7", "8"),
  labels = c("AffectNet-7", "AffectNet-8")
)

p2_cell_means <- ggplot(
  cell_stats,
  aes(x = dataset_label, y = mean_acc, fill = guidance_label)
) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(
    aes(ymin = ymin, ymax = ymax),
    position = position_dodge(width = 0.8),
    width = 0.25,
    linewidth = 0.5
  ) +
  scale_fill_manual(
    values = c("APViT" = "#2166AC", "M-LDG" = "#D6604D")
  ) +
  scale_y_continuous(
    breaks = seq(50, 65, by = 2),
    expand = expansion(mult = c(0, 0.02))
  ) +
  coord_cartesian(ylim = c(50, 65)) +
  labs(
    x     = "Dataset",
    y     = "Best Validation Accuracy (%)",
    fill  = "Guidance Network"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.major.x = element_blank()
  )

print(p2_cell_means)

output_dir <- "../../Documento_Final___FER/tex/images/chap_results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
ggsave(
  file.path(output_dir, "p2_cell_means_barplot.pdf"),
  plot      = p2_cell_means,
  width     = 6,
  height    = 4,
  device    = "pdf",
  create.dir = TRUE
)