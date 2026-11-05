#!/usr/bin/env Rscript
# ==============================================================================
# Manuscript Figure S2D: Binomial mixed-effects regression (glmer) of Day-1
# recognition memory in the healthy online (Prolific) cohort.
#
# Model:
#   hit ~ zrpe_new + zpm + zalpha_new + zbeta_new + zage + zsex + (1 | participant)
#
# Inputs:
#   results/hierarchical_pyem/prolific_v1/combined_df_lme4_hit_rpe_new_day_1.csv
#     (exported by notebooks/01_behavior/FigS2BCD_modeling_online.ipynb)
#   $MEMORYBANDIT_DATA_DIR/work/qasims01/MemoryBanditData/Prolific/DelayedMemoryTest/demographics.csv
#
# Outputs:
#   results/hierarchical_pyem/prolific_v1/glmer_hit_coefficients_day_1.csv
#   results/hierarchical_pyem/prolific_v1/glmer_hit_coefficients_day_1_core.csv
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
  "PROLIFIC_GLMER_DIR",
  unset = file.path(repo_root, "results", "hierarchical_pyem", "prolific_v1")
)
data_root <- Sys.getenv(
  "MEMORYBANDIT_DATA_DIR",
  unset = path.expand("~/MemoryBanditData")
)
data_dir <- file.path(
  data_root, "work", "qasims01", "MemoryBanditData", "Prolific", "DelayedMemoryTest"
)
if (!dir.exists(data_dir)) {
  alt_dir <- file.path(data_root, "projects", "guLab", "Salman", "Prolific", "DelayedMemoryTest")
  if (dir.exists(alt_dir)) data_dir <- alt_dir
}

in_d1 <- file.path(base, "combined_df_lme4_hit_rpe_new_day_1.csv")
dem_csv <- file.path(data_dir, "demographics.csv")
if (!file.exists(in_d1)) {
  stop("Input CSV not found: ", in_d1,
       "\nRun notebooks/01_behavior/FigS2BCD_modeling_online.ipynb first.")
}
if (!file.exists(dem_csv)) {
  stop("Demographics CSV not found: ", dem_csv,
       "\nSet MEMORYBANDIT_DATA_DIR to the root of the data directory.")
}

d1 <- read.csv(in_d1)
dem <- read.csv(dem_csv)

# Exclude participants with chance or below-chance recognition sensitivity (d' < 0; 200 -> 193)
if ("dprime" %in% names(d1)) {
  d1 <- subset(d1, is.na(dprime) | dprime >= 0)
}

# Repair shifted demographic rows (Age/Sex column shift in Prolific export)
age_from_age <- suppressWarnings(as.numeric(as.character(dem$Age)))
age_from_sex <- suppressWarnings(as.numeric(as.character(dem$Sex)))
dem$age <- ifelse(!is.na(age_from_age), age_from_age, age_from_sex)

dem$sex <- ifelse(
  dem$Sex %in% c("Female", "Male"),
  as.character(dem$Sex),
  ifelse(
    dem$Ethnicity.simplified %in% c("Female", "Male"),
    as.character(dem$Ethnicity.simplified),
    NA_character_
  )
)

dem$sex_binary <- ifelse(
  dem$sex == "Male", 1,
  ifelse(dem$sex == "Female", 0, NA_real_)
)

demographics <- data.frame(
  participant = as.character(dem$Participant.id),
  age = dem$age,
  sex_binary = dem$sex_binary
)
demographics <- demographics[!duplicated(demographics$participant), ]

analysis_ids <- unique(as.character(d1$participant))
demographics <- demographics[demographics$participant %in% analysis_ids, ]

gelman_standardize <- function(x) {
  (x - mean(x, na.rm = TRUE)) / (2 * sd(x, na.rm = TRUE))
}

demographics$zage <- gelman_standardize(demographics$age)
demographics$zsex <- gelman_standardize(demographics$sex_binary)

d1 <- merge(d1, demographics, by = "participant", all.x = TRUE)

# QC: zpm should be trial-varying image memorability (~40 unique / subject)
stopifnot(median(tapply(d1$zpm, d1$participant, function(x) length(unique(x)))) > 1)

vars_core <- c("hit", "zrpe_new", "zpm", "zalpha_new", "zbeta_new", "participant")
vars_demo <- c(vars_core, "zage", "zsex")

export_fixed_effects <- function(model, path) {
  s <- coef(summary(model))
  out <- data.frame(
    term = rownames(s),
    estimate = s[, "Estimate"],
    std.error = s[, "Std. Error"],
    statistic = s[, "z value"],
    p.value = s[, "Pr(>|z|)"],
    row.names = NULL,
    check.names = FALSE
  )
  write.csv(out, path, row.names = FALSE)
  cat("Wrote:", path, "\n")
}

# --- Day 1 demographic-adjusted (Figure S2D) ---
d1_model <- d1[complete.cases(d1[, vars_demo]), ]
cat(sprintf("Fitting online Day-1 demographic-adjusted glmer: %d trials across %d participants\n",
            nrow(d1_model), length(unique(d1_model$participant))))
m_day1 <- glmer(
  hit ~ zrpe_new + zpm + zalpha_new + zbeta_new + zage + zsex + (1 | participant),
  data = d1_model,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa")
)
print(summary(m_day1))
export_fixed_effects(m_day1, file.path(base, "glmer_hit_coefficients_day_1.csv"))

# --- Day 1 core (no age/sex) ---
d1_core <- d1[complete.cases(d1[, vars_core]), ]
m_day1_core <- glmer(
  hit ~ zrpe_new + zpm + zalpha_new + zbeta_new + (1 | participant),
  data = d1_core,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa")
)
print(summary(m_day1_core))
export_fixed_effects(m_day1_core, file.path(base, "glmer_hit_coefficients_day_1_core.csv"))
