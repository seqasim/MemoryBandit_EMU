# Load required libraries
library(lme4)    # For mixed-effects models
# library(lmerTest)  # Provides p-values for lmer models
library(dplyr)
library(progress)
library(purrr)

# Step 1: Load your data
# Replace 'path/to/your/file.csv' with the actual path to your CSV file
data <- read.csv('/Volumes/T7\ Shield/work/qasims01/MemoryBanditData/EMU/big_tfr_df_mem')

data$rpe_cat <- factor(data$rpe_cat, levels = c("neutral", "neg", "pos"))

# # Restrict to fband =='hfa' and region =='OFC'
test_df <- data[data$fband == 'hfa' & data$region == 'OFC',]
# & data$region == 'OFC',

mixed_model1 <- lmer(
  tfr ~ 1 + C(rpe_cat)*phit+# Fixed effects
  (1|unique_label), 
  data = test_df,
  REML = FALSE   # Use maximum likelihood for model comparison
)

mixed_model2 <- lmer(
  tfr ~ 1 + zrpe*phit+# Fixed effects
  (1|unique_label), 
  data = test_df,
  REML = FALSE   # Use maximum likelihood for model comparison
)

# mixed_model2 <- lmer(
#   tfr ~ 1 + zrpe*phit+# Fixed effects
#   (1|unique_label),  # Random intercepts and slopes for unique_label
#   data = test_df,
#   REML = FALSE   # Use maximum likelihood for model comparison
# )

# mixed_model2 <- lmer(
#   tfr ~ 1 + C(region)*zrpe*phit +# Fixed effects
#   (1|unique_label),  # Random intercepts and slopes for unique_label
#   data = test_df,
#   REML = FALSE   # Use maximum likelihood for model comparison
# )


# mixed_model2 <- lmer(
#   tfr ~ 1 + zrpe + # Fixed effects
#   (1|unique_label),  # Random intercepts and slopes for unique_label
#   data = test_df,
#   REML = FALSE   # Use maximum likelihood for model comparison
# )

# # Let's shuffle the order of the tfr data by building a participant-wise mapping that is applied within each unique_label
# # This will allow us to test the null hypothesis that the order of the tfr data does not matter

# # Shuffle function applied to the entire dataframe
# n_shuffles <- 5  # Number of shuffles to perform
# results <- vector("list", n_shuffles)  # To store the model outputs

# for (i in seq_len(n_shuffles)) {
#   # Shuffle the data
#   shuffled_data <- data %>%
#     group_by(participant, unique_label) %>%
#     mutate(
#       shuffled_trial = sample(unique(trial), size = n(), replace = FALSE)[match(trial, unique(trial))]
#     ) %>%
#     ungroup() %>%
#     group_by(participant, unique_label, shuffled_trial) %>%
#     mutate(
#       tfr = tfr[match(trial, shuffled_trial)]
#     ) %>%
#     ungroup()

#   # Fit the mixed model
#   model <- tryCatch({
#     lmer(
#       tfr ~ 1 + C(hemi) + zrpe + phit + zrpe:phit + 
#       (1 | participant) + 
#       (1 | unique_label),
#       data = shuffled_data,
#       REML = FALSE
#     )
#   }, error = function(e) NA)  # Handle errors gracefully
  
#   # Store the model
#   results[[i]] <- model
# }

# # # Step 4: Model summary
# summary(mixed_model)
