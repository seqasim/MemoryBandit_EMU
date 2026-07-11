data {
  int<lower=1> N;
  int<lower=1> S;
  array[N] int<lower=1,upper=S> subj;
  array[N] int<lower=0,upper=1> choice;
  vector[N] rt;
  vector[N] reward;
  array[S] real<lower=0> min_rt_subj;  // min RT per subject (t0 must be < this)
  real max_a;
}

parameters {
  // group-level
  real mu_alpha;
  real mu_beta;
  real mu_a;
  real mu_t0;

  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  real<lower=0> sigma_a;
  real<lower=0> sigma_t0;

  // subject-level
  vector[S] alpha_raw;
  vector[S] beta_raw;
  vector[S] a_raw;
  vector[S] t0_raw;
}

transformed parameters {
  vector<lower=0,upper=1>[S] alpha;
  vector[S] beta;
  vector<lower=0>[S] a;
  vector<lower=0>[S] t0;

  alpha = inv_logit(mu_alpha + sigma_alpha * alpha_raw);
  beta  = mu_beta + sigma_beta * beta_raw;
  real a_min = 1e-6;
  real t0_min = 1e-6;
  real t0_epsilon = 1e-6;   // keep t0 strictly below min_rt_subj so rt > t0 numerically
  a     = a_min + (max_a - a_min) * inv_logit(mu_a + sigma_a * a_raw);   // a in (a_min, max_a)
  t0    = t0_min + (to_vector(min_rt_subj) - t0_min - t0_epsilon) .* inv_logit(mu_t0 + sigma_t0 * t0_raw);  // t0[s] in (t0_min, min_rt_subj[s] - t0_epsilon)
}

model {
  // priors
  mu_alpha ~ normal(0, 1);
  mu_beta  ~ normal(0, 1);
  mu_a     ~ normal(0, 1);
  mu_t0    ~ normal(0, 1);   // proportion of min_rt_subj (logit scale)

  sigma_alpha ~ exponential(1);
  sigma_beta  ~ exponential(1);
  sigma_a     ~ exponential(1);
  sigma_t0    ~ exponential(1);

  alpha_raw ~ normal(0, 1);
  beta_raw  ~ normal(0, 1);
  a_raw     ~ normal(0, 1);
  t0_raw    ~ normal(0, 1);

  // initialize Q-values
  array[S] vector[2] Q;
  for (s in 1:S)
    Q[s] = rep_vector(0.5, 2);

  // likelihood
  for (n in 1:N) {
    int s = subj[n];
    int a_choice = choice[n] + 1;

    real dv = Q[s][2] - Q[s][1];
    real drift = beta[s] * dv;

    target += wiener_lpdf(
      rt[n] |
      a[s],
      t0[s],
      0.5,
      choice[n] == 1 ? drift : -drift
    );

    real rpe = reward[n] - Q[s][a_choice];
    Q[s][a_choice] += alpha[s] * rpe;
  }
}

generated quantities {
  // Pointwise log-likelihood for LOO/WAIC and model comparison
  vector[N] log_lik;
  // Trial-wise drift rate (beta * (Q[2]-Q[1])) for posterior analysis
  vector[N] drift_trial;
  array[S] vector[2] Q;
  for (s in 1:S)
    Q[s] = rep_vector(0.5, 2);
  for (n in 1:N) {
    int s = subj[n];
    int a_choice = choice[n] + 1;
    real dv = Q[s][2] - Q[s][1];
    real drift = beta[s] * dv;
    drift_trial[n] = drift;
    log_lik[n] = wiener_lpdf(
      rt[n] |
      a[s],
      t0[s],
      0.5,
      choice[n] == 1 ? drift : -drift
    );
    real rpe = reward[n] - Q[s][a_choice];
    Q[s][a_choice] += alpha[s] * rpe;
  }
}
