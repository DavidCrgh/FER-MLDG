library(unrepx)
library(car)
library(ggplot2)
library(ggpubr)
library(dplyr)

df <- read.csv("phase1-raw.csv")

###############################################
## Analyze significance of effect estimates ##
##############################################

# Compute effect estimates using Yate's algorithm
(effectEstimates = yates(df$best_accuracy, labels = c("A", "B", "C", "D")))

print(effectEstimates)

hnplot(effectEstimates, ID = 6)

# Since effects A and B are negligible, discard them from the data frame
# this way we prepare the data for a two-way ANOVA with 4 replicates per design

df <- subset(df, select = c(use_apvit, num_classes, best_accuracy))

df$use_apvit <- factor(df$use_apvit)
df$num_classes <- factor(df$num_classes)

print(head(df))
str(df)
print(summary(df))


################################
## CHECK ASSUMPTIONS OF ANOVA ##
################################

# Check the assumptions of ANOVA
# Normality
model <- aov(best_accuracy ~ use_apvit * num_classes, data = df)

qqPlot(model$residuals, id = FALSE)
print(shapiro.test(model$residuals))

# Homogeneity of variance
plot(model, which = 3)
print(leveneTest(model))

# Outliers
# For APVIT groups
p <- ggplot(df) +
  aes(x = use_apvit, y = best_accuracy) +
  geom_boxplot()
print(p)

# For dataset groups
p <- ggplot(df) +
  aes(x = num_classes, y = best_accuracy) +
  geom_boxplot()
print(p)

##################################
## PERFORM PRELIMINARY ANALYSIS ##
##################################

# Descriptive statistics
descriptiveStats <- group_by(df, use_apvit, num_classes) %>%
  summarise(
    mean = mean(best_accuracy, na.rm = TRUE),
    sd = sd(best_accuracy, na.rm = TRUE)
  )

print(descriptiveStats)

# Boxplots

#p <- ggplot(df) +
#  aes(x = num_classes, y = best_accuracy, fill = use_apvit) +
#  geom_boxplot()
#print(p)

# Line plot

#p <- ggline(df,
#            x = "use_apvit",
#            y = "best_accuracy",
#            color = "num_classes",
#            add = c("mean_se") # add mean and std error
#            ) +
#  labs(y = "Mean of best accuracy (%)")
# print(p)

###############################
## INTERACTION PLOT (C x D)  ##
###############################

# Build a summary data frame with the cell means and SE pooled over A and B
interaction_stats <- group_by(df, use_apvit, num_classes) %>%
  summarise(
    mean_acc = mean(best_accuracy, na.rm = TRUE),
    se_acc   = sd(best_accuracy, na.rm = TRUE) / sqrt(n()),
    .groups  = "drop"
  )

# Human-readable factor labels
interaction_stats$apvit_label <- factor(
  interaction_stats$use_apvit,
  levels = levels(interaction_stats$use_apvit),
  labels = c("No", "Yes")
)
interaction_stats$dataset_label <- factor(
  interaction_stats$num_classes,
  levels = levels(interaction_stats$num_classes),
  labels = c("AffectNet-7", "AffectNet-8")
)

p_interaction <- ggplot(
  interaction_stats,
  aes(x = apvit_label, y = mean_acc,
      color = dataset_label, group = dataset_label)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = mean_acc - se_acc, ymax = mean_acc + se_acc),
    width = 0.1
  ) +
  scale_color_manual(
    values = c("AffectNet-7" = "#2166AC", "AffectNet-8" = "#D6604D")
  ) +
  labs(
    x     = "APViT Guidance (Factor C)",
    y     = "Mean Accuracy (%)",
    color = "Dataset (Factor D)"
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")

print(p_interaction)


###########################
## PERFORM TWO-WAY ANOVA ##
###########################

print(summary(model))


###############################
## EFFECT SIZES (ETA-SQUARED) ##
###############################

anova_table <- summary(model)[[1]]
ss           <- anova_table[["Sum Sq"]]
total_ss     <- sum(ss)

eta_sq <- data.frame(
  Source      = rownames(anova_table),
  SS          = ss,
  eta_squared = ss / total_ss
)

cat("\nTotal SS:", total_ss, "\n")
print(eta_sq, digits = 4)


##############################################################
## ASSUMPTION PLOTS WITH HUMAN-READABLE LABELS (ggplot2)   ##
##############################################################

resid_df <- data.frame(
  fitted    = fitted(model),
  residuals = residuals(model),
  std_resid = sqrt(abs(rstandard(model)))
)

# Normal Q-Q plot
theoretical_q <- qqnorm(resid_df$residuals, plot.it = FALSE)
qq_df <- data.frame(
  theoretical = theoretical_q$x,
  sample      = theoretical_q$y
)

p_qq <- ggplot(qq_df, aes(x = theoretical, y = sample)) +
  geom_point(shape = 1, size = 2, color = "#2166AC") +
  geom_abline(
    intercept = mean(resid_df$residuals),
    slope     = sd(resid_df$residuals),
    linetype  = "dashed", color = "gray40"
  ) +
  labs(
    title = "Normal Q-Q Plot of Residuals",
    x     = "Theoretical Quantiles",
    y     = "Sample Residuals"
  ) +
  theme_bw(base_size = 12)

print(p_qq)

# Scale-Location plot (sqrt of standardised residuals vs fitted)
p_sl <- ggplot(resid_df, aes(x = fitted, y = std_resid)) +
  geom_point(shape = 1, size = 2, color = "#D6604D") +
  geom_smooth(method = "loess", se = FALSE,
              linetype = "dashed", color = "gray40", linewidth = 0.8) +
  labs(
    title = "Scale-Location: Homogeneity of Variance",
    x     = "Fitted Values (Predicted Accuracy)",
    y     = expression(sqrt("|Standardised Residuals|"))
  ) +
  theme_bw(base_size = 12)

print(p_sl)
