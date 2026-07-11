// Hierarchical vanilla Wiener (DDM) for recognition memory reaction times.
// Same choice convention as rl_ddm.stan: choice == 1 -> drift v[s]; choice == 0 -> -v[s].

data {
  int<lower=1> N;
  int<lower=1> S;
  array[N] int<lower=1,upper=S> subj;
  array[N] int<lower=0,upper=1> choice;
  vector[N] rt;
  array[S] real<lower=0> min_rt_subj;  // min RT per subject (t0 must be < this)
  real max_a;
}

parameters {
  // group-level
  real mu_v;
  real mu_a;
  real mu_t0;

  real<lower=0> sigma_v;
  real<lower=0> sigma_a;
  real<lower=0> sigma_t0;

  // subject-level (non-centered)
  vector[S] v_raw;
  vector[S] a_raw;
  vector[S] t0_raw;
}

transformed parameters {
  vector[S] v;
  vector<lower=0>[S] a;
  vector<lower=0>[S] t0;

  v = mu_v + sigma_v * v_raw;

  real a_min = 1e-6;
  real t0_min = 1e-6;
  real t0_epsilon = 1e-6;   // keep t0 strictly below min_rt_subj so rt > t0 numerically
  a     = a_min + (max_a - a_min) * inv_logit(mu_a + sigma_a * a_raw);   // a in (a_min, max_a)
  t0    = t0_min + (to_vector(min_rt_subj) - t0_min - t0_epsilon) .* inv_logit(mu_t0 + sigma_t0 * t0_raw);  // t0[s] in (t0_min, min_rt_subj[s] - t0_epsilon)
}

model {
  // priors
  mu_v  ~ normal(0, 2);
  mu_a  ~ normal(0, 1);
  mu_t0 ~ normal(0, 1);   // proportion of min_rt_subj (logit scale)

  sigma_v  ~ exponential(1);
  sigma_a  ~ exponential(1);
  sigma_t0 ~ exponential(1);

  v_raw  ~ normal(0, 1);
  a_raw  ~ normal(0, 1);
  t0_raw ~ normal(0, 1);

  // likelihood
  for (n in 1:N) {
    int s = subj[n];
    real drift = choice[n] == 1 ? v[s] : -v[s];
    target += wiener_lpdf(
      rt[n] |
      a[s],
      t0[s],
      0.5,
      drift
    );
  }
}

generated quantities {
  // Pointwise log-likelihood for LOO/WAIC and model comparison
  vector[N] log_lik;
  // Signed drift passed to wiener on each trial (matches likelihood)
  vector[N] drift_trial;

  for (n in 1:N) {
    int s = subj[n];
    real drift = choice[n] == 1 ? v[s] : -v[s];
    drift_trial[n] = drift;
    log_lik[n] = wiener_lpdf(
      rt[n] |
      a[s],
      t0[s],
      0.5,
      drift
    );
  }
}
