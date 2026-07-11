// Hierarchical Wiener DDM for recognition RTs with trial-wise PM (e.g. image d') affecting drift.
// Linear predictor: mu_n = v[s] + b_pm[s] * pm[n]; choice == 1 -> +mu_n, else -> -mu_n.

data {
  int<lower=1> N;
  int<lower=1> S;
  array[N] int<lower=1,upper=S> subj;
  array[N] int<lower=0,upper=1> choice;
  vector[N] rt;
  vector[N] pm;
  array[S] real<lower=0> min_rt_subj;
  real max_a;
}

parameters {
  real mu_v;
  real mu_a;
  real mu_t0;

  real mu_b_pm;

  real<lower=0> sigma_v;
  real<lower=0> sigma_a;
  real<lower=0> sigma_t0;
  real<lower=0> sigma_b_pm;

  vector[S] v_raw;
  vector[S] a_raw;
  vector[S] t0_raw;
  vector[S] b_pm_raw;
}

transformed parameters {
  vector[S] v;
  vector[S] b_pm;
  vector<lower=0>[S] a;
  vector<lower=0>[S] t0;

  v = mu_v + sigma_v * v_raw;
  b_pm = mu_b_pm + sigma_b_pm * b_pm_raw;

  real a_min = 1e-6;
  real t0_min = 1e-6;
  real t0_epsilon = 1e-6;
  a  = a_min + (max_a - a_min) * inv_logit(mu_a + sigma_a * a_raw);
  t0 = t0_min + (to_vector(min_rt_subj) - t0_min - t0_epsilon)
       .* inv_logit(mu_t0 + sigma_t0 * t0_raw);
}

model {
  mu_v ~ normal(0, 2);
  mu_a ~ normal(0, 1);
  mu_t0 ~ normal(0, 1);
  mu_b_pm ~ normal(0, 1);

  sigma_v ~ exponential(1);
  sigma_a ~ exponential(1);
  sigma_t0 ~ exponential(1);
  sigma_b_pm ~ exponential(1);

  v_raw ~ normal(0, 1);
  a_raw ~ normal(0, 1);
  t0_raw ~ normal(0, 1);
  b_pm_raw ~ normal(0, 1);

  for (n in 1:N) {
    int s = subj[n];
    real mu_n = v[s] + b_pm[s] * pm[n];
    real drift = choice[n] == 1 ? mu_n : -mu_n;
    target += wiener_lpdf(rt[n] | a[s], t0[s], 0.5, drift);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] drift_trial;
  vector[N] drift_lin;

  for (n in 1:N) {
    int s = subj[n];
    real mu_n = v[s] + b_pm[s] * pm[n];
    drift_lin[n] = mu_n;
    real drift = choice[n] == 1 ? mu_n : -mu_n;
    drift_trial[n] = drift;
    log_lik[n] = wiener_lpdf(rt[n] | a[s], t0[s], 0.5, drift);
  }
}
