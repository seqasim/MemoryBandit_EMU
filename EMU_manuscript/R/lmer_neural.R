#!/usr/bin/env Rscript
# ==============================================================================
# Manuscript Neural Mixed-Effects Regressions (lme4::lmer in R)
#
# Covers:
#   1. Figure 5B : 1/f-corrected theta power (3.5-8 Hz), feedback vs recognition
#                  cue, in ACC and HPC (p_ACC < 0.001, p_HPC < 0.001)
#   2. Figure 5E : HPC-ACC theta pairwise phase consistency (PPC) at recognition
#                  cue by RPE tercile (ref = positive; pos vs neg p = 0.002,
#                  pos vs neutral p = 0.09)
#   3. Figure 5F : HPC -> ACC theta phase-slope index (PSI) at recognition cue
#                  by RPE tercile (ref = positive; pos vs neg p = 0.04,
#                  pos vs neutral p = 0.2)
#   4. Figure S12: Multivariate theta-HFA phase-amplitude coupling (GC-PAC) at
#                  recognition cue by RPE tercile for HPC-ACC, ACC-ACC, HPC-HPC
#                  (all ps > 0.05)
# ==============================================================================

suppressPackageStartupMessages(library(lme4))
has_lmertest <- requireNamespace("lmerTest", quietly = TRUE)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1])))
  repo_root <- normalizePath(file.path(script_dir, ".."))
} else {
  repo_root <- normalizePath(".")
}

data_root <- Sys.getenv(
  "MEMORYBANDIT_DATA_DIR",
  unset = path.expand("~/MemoryBanditData")
)
conn_scratch <- file.path(
  data_root, "projects", "guLab", "Salman", "EphysAnalyses", "Connectivity", "scratch"
)
out_dir <- file.path(repo_root, "results", "lmer_neural")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

extract_lmer_table <- function(fit, model_label) {
  s <- coef(summary(fit))
  t_col <- if ("t value" %in% colnames(s)) s[, "t value"] else rep(NA_real_, nrow(s))
  p_col <- if ("Pr(>|t|)" %in% colnames(s)) {
    s[, "Pr(>|t|)"]
  } else {
    2 * pnorm(-abs(t_col))
  }
  data.frame(
    model = model_label,
    term = rownames(s),
    estimate = s[, "Estimate"],
    std.error = s[, "Std. Error"],
    t.value = t_col,
    p.value = p_col,
    row.names = NULL,
    check.names = FALSE
  )
}

all_results <- list()

# ------------------------------------------------------------------------------
# 1. Figure 5B: 1/f-corrected theta power (feedback_start vs cue_on)
# ------------------------------------------------------------------------------
psd_candidates <- c(
  file.path(repo_root, "results", "spectral", "tables", "electrode_psd_df.csv"),
  file.path(data_root, "projects", "guLab", "Salman", "EphysAnalyses", "Plots",
            "roi_oscillation_comparison", "tables", "electrode_psd_df.csv")
)
psd_csv <- psd_candidates[file.exists(psd_candidates)][1]

if (!is.na(psd_csv)) {
  cat("\n=== [1/4] Figure 5B: Theta 1/f-corrected power (feedback vs cue) ===\n")
  psd <- read.csv(psd_csv)
  for (roi_name in c("ACC", "HPC")) {
    sub <- subset(psd, roi == roi_name & frequency >= 3.5 & frequency < 8.0)
    if (nrow(sub) > 0 && length(unique(sub$participant)) >= 2) {
      m <- lmer(psd_corrected_z ~ 1 + factor(event) + (1 | participant), data = sub, REML = TRUE)
      tbl <- extract_lmer_table(m, paste0("Fig5B_theta_power_", roi_name))
      print(tbl)
      all_results[[length(all_results) + 1]] <- tbl
    }
  }
} else {
  cat("\n[skip] Figure 5B: electrode_psd_df.csv not found (run 03_spectral/Fig5B_spectra.ipynb).\n")
}

# ------------------------------------------------------------------------------
# 2. Figure 5E: HPC-ACC theta PPC by RPE tercile at recognition cue
# ------------------------------------------------------------------------------
ppc_csv <- file.path(conn_scratch, "ppc_hpc_acc_rpe_cutrev", "cue_on", "ppc_pairs_cue_on.csv")
if (file.exists(ppc_csv)) {
  cat("\n=== [2/4] Figure 5E: HPC-ACC theta PPC at cue by RPE tercile ===\n")
  df_ppc <- read.csv(ppc_csv)
  df_ppc$rpe_cat <- relevel(factor(df_ppc$rpe_cat), ref = "positive")
  m_ppc <- lmer(ppc_z ~ rpe_cat + (1 | subject) + (1 | pair_uid), data = df_ppc, REML = TRUE)
  tbl_ppc <- extract_lmer_table(m_ppc, "Fig5E_HPC_ACC_theta_PPC_cue")
  print(tbl_ppc)
  all_results[[length(all_results) + 1]] <- tbl_ppc
} else {
  cat("\n[skip] Figure 5E: ppc_pairs_cue_on.csv not found (run 05_theta_sync_pac/Fig5DEF_S13_PPC_PSI.ipynb).\n")
}

# ------------------------------------------------------------------------------
# 3. Figure 5F: HPC -> ACC theta PSI by RPE tercile at recognition cue
# ------------------------------------------------------------------------------
psi_csv <- file.path(
  conn_scratch, "psi_hpc_acc_rpe_hitmiss", "cue_on",
  "all_psi_pairs_cue_on_HPC_to_ACC_from_trial_cache.csv"
)
if (file.exists(psi_csv)) {
  cat("\n=== [3/4] Figure 5F: HPC->ACC theta PSI at cue by RPE tercile ===\n")
  df_psi <- read.csv(psi_csv)
  df_psi$rpe_cat <- relevel(factor(df_psi$rpe_cat), ref = "positive")
  subj_col <- if ("participant" %in% names(df_psi)) "participant" else "subject"
  df_psi$subj_id <- df_psi[[subj_col]]
  m_psi <- lmer(psi_z ~ rpe_cat + (1 | subj_id) + (1 | pair_uid), data = df_psi, REML = TRUE)
  tbl_psi <- extract_lmer_table(m_psi, "Fig5F_HPC_to_ACC_theta_PSI_cue")
  print(tbl_psi)
  all_results[[length(all_results) + 1]] <- tbl_psi
} else {
  cat("\n[skip] Figure 5F: all_psi_pairs_cue_on_HPC_to_ACC_from_trial_cache.csv not found.\n")
}

# ------------------------------------------------------------------------------
# 4. Figure S12: Multivariate theta-HFA GC-PAC by RPE tercile at recognition cue
# ------------------------------------------------------------------------------
pac_dirs <- Sys.glob(file.path(conn_scratch, "mv_gcpac_rpe", "multi_target_dual_event", "*"))
pac_csv <- NA_character_
for (d in pac_dirs) {
  cand <- file.path(d, "all_gcpac_subject_dual_event.csv")
  if (file.exists(cand)) {
    pac_csv <- cand
    break
  }
}

if (!is.na(pac_csv)) {
  cat("\n=== [4/4] Figure S12: Multivariate theta-HFA PAC at cue by RPE tercile ===\n")
  df_pac <- read.csv(pac_csv)
  df_pac <- subset(df_pac, event == "cue_on")
  pairs_s12 <- list(c("HPC", "ACC"), c("ACC", "ACC"), c("HPC", "HPC"))
  for (pr in pairs_s12) {
    sub <- subset(df_pac, phase_roi == pr[1] & amp_roi == pr[2] & !is.na(trial_pac_z) & !is.na(rpe_cat))
    if (nrow(sub) > 0 && length(unique(sub$subject)) >= 2) {
      sub$rpe_cat <- relevel(factor(sub$rpe_cat), ref = "positive")
      m_pac <- lmer(trial_pac_z ~ rpe_cat + (1 | subject), data = sub, REML = FALSE)
      tbl_pac <- extract_lmer_table(m_pac, paste0("FigS12_MV_GCPAC_", pr[1], "_", pr[2]))
      print(tbl_pac)
      all_results[[length(all_results) + 1]] <- tbl_pac
    }
  }
} else {
  cat("\n[skip] Figure S12: all_gcpac_subject_dual_event.csv not found (run 05_theta_sync_pac/FigS12_PAC.ipynb).\n")
}

if (length(all_results) > 0) {
  combined_out <- do.call(rbind, all_results)
  out_csv <- file.path(out_dir, "lmer_neural_summary.csv")
  write.csv(combined_out, out_csv, row.names = FALSE)
  cat("\nWrote combined neural lmer results:", out_csv, "\n")
}
