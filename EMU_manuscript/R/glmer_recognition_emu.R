#!/usr/bin/env Rscript
# ==============================================================================
# Manuscript Figure 1G: Binomial mixed-effects regression (glmer) of Day-1
# recognition memory in the neurosurgical (EMU) cohort (n = 19).
#
# Model:
#   hit ~ zrpe_new + zpm + zalpha_new + zbeta_new + (1 | participant)
#
# Inputs:
#   results/hierarchical_pyem/redesign_v2/combined_df_lme4_hit_rpe_new.csv
#     (exported by notebooks/01_behavior/Fig1DEG_S1_TableS1_modeling_EMU.ipynb)
#
# Outputs:
#   results/hierarchical_pyem/redesign_v2/glmer_hit_coefficients.csv
# ==============================================================================

suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1])))
  repo_root <- normalizePath(file.path(script_dir, ".."))
} else {
  repo_root <- normalizePath(".")
}

base <- Sys.getenv(
  "EMU_GLMER_DIR",
  unset = file.path(repo_root, "results", "hierarchical_pyem", "redesign_v2")
)

in_csv <- file.path(base, "combined_df_lme4_hit_rpe_new.csv")
if (!file.exists(in_csv)) {
  stop("Input CSV not found: ", in_csv,
       "\nRun notebooks/01_behavior/Fig1DEG_S1_TableS1_modeling_EMU.ipynb first.")
}

d <- read.csv(in_csv)

# Day-1-only export; zpm must be trial-varying image memorability (DPRIME)
stopifnot(median(tapply(d$zpm, d$participant, function(x) length(unique(x)))) > 1)

d <- subset(
  d,
  complete.cases(hit, zrpe_new, zpm, zalpha_new, zbeta_new, participant)
)

cat(sprintf("Fitting EMU Day-1 glmer: %d trials across %d participants\n",
            nrow(d), length(unique(d$participant))))
stopifnot(length(unique(d$participant)) == 19)

m <- glmer(
  hit ~ zrpe_new + zpm + zalpha_new + zbeta_new + (1 | participant),
  data = d,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa")
)

print(summary(m))

s <- coef(summary(m))
out <- data.frame(
  term = rownames(s),
  estimate = s[, "Estimate"],
  std.error = s[, "Std. Error"],
  statistic = s[, "z value"],
  p.value = s[, "Pr(>|z|)"],
  row.names = NULL,
  check.names = FALSE
)

out_csv <- file.path(base, "glmer_hit_coefficients.csv")
write.csv(out, out_csv, row.names = FALSE)
cat("Wrote:", out_csv, "\n")
