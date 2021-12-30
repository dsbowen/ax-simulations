import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import multivariate_normal

DATA_DIR = "../data/clean"
RESULTS_DIR = "results"
BATCH_SIZE = 10
N_BATCHES = 500
SIZE = 1000
CONTROL = "No Payment"
HAYSTACK = "Very Low Pay"

def make_haystack(i, hay_df):
    df = hay_df.copy()
    df["treatment"] = f"hay_{str(i).zfill(3)}"
    return df

def run_batch(df, experiment_df, n_treatments, use_adaptive_assignment):
    # estimate distribution of treatment effects
    X = pd.get_dummies(experiment_df.treatment)
    X[CONTROL] = 1
    results = sm.OLS(experiment_df.target, X).fit().get_robustcov_results()
    indices = np.array(results.model.exog_names) != CONTROL
    names = np.array(results.model.exog_names)[indices]
    params = results.params[indices]
    if len(params) < n_treatments:
        raise RuntimeError("Model was unable to estimate effects of all treatments")
    cov = results.cov_params()[indices][:, indices]

    if use_adaptive_assignment:
        dist = multivariate_normal(params, cov)

        # compute acquisition function (exploration sampling) weights
        pr_best = np.identity(n_treatments)[dist.rvs(SIZE).argmax(axis=1)].mean(axis=0)
        weights = pr_best * (1 - pr_best)

        # draw the sample for the next batch
        weights_mapping = {name: weight for name, weight in zip(names, weights)}
        weights_mapping[CONTROL] = weights.max()
        weights = df.treatment.map(weights_mapping)
    else:
        weights = None

    sample = df.sample(BATCH_SIZE, weights=weights, replace=True)
    return sample, {"names": names, "params": params, "cov": cov}

if __name__ == "__main__":
    sim_no, needle, n_treatments, strategy = sys.argv[1:]
    sim_no = int(sim_no)
    n_treatments = int(n_treatments)
    np.random.seed(sim_no)

    results = []

    df = pd.read_csv(os.path.join(DATA_DIR, "DellaVigna_Pope.csv"))
    hay_df = df[df.treatment == HAYSTACK]
    haystacks = [make_haystack(i, hay_df) for i in range(n_treatments - 1)]
    df = pd.concat([df[df.treatment.isin((CONTROL, needle))]] + haystacks).reset_index(drop=True)
    experiment_df = pd.DataFrame()

    for _ in range(N_BATCHES):
        try:
            sample, result = run_batch(df, experiment_df, n_treatments, strategy=="adaptive")
            result["n_users"] = len(experiment_df)
            results.append(result)
        except:
            sample = df.sample(BATCH_SIZE, replace=True)
        experiment_df = experiment_df.append(sample)

    results_df = pd.DataFrame(results)
    results_df["sim_no"] = sim_no
    results_df["needle"] = needle
    results_df["n_treatments"] = n_treatments
    filename = f"sim_no={sim_no}-needle={needle}-n_treatments={n_treatments}-strategy={strategy}.csv"
    results_df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
