import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from conditional_inference.rqu import RQU
from scipy.stats import multivariate_normal

DATA_DIR = "../data/clean"
RESULTS_DIR = "results"
BATCH_SIZE = 10
N_BATCHES = 500
COLLECT_RESULTS_FREQUENCY = 50
SIZE = 1000
CONTROL = "No_Payment"
HAYSTACK = "Very_Low_Pay"

def make_haystack(i, hay_df):
    df = hay_df.copy()
    df["treatment"] = f"hay_{str(i).zfill(3)}"
    return df

def run_batch(i, needle, df, experiment_df, n_treatments, use_adaptive_assignment):
    if use_adaptive_assignment or i % COLLECT_RESULTS_FREQUENCY == 0:
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

    weights = None
    if use_adaptive_assignment:
        dist = multivariate_normal(params, cov)

        # compute acquisition function (exploration sampling) weights
        pr_best = np.identity(n_treatments)[dist.rvs(SIZE).argmax(axis=1)].mean(axis=0)
        weights = pr_best * (1 - pr_best)

        # draw the sample for the next batch
        weights_mapping = {name: weight for name, weight in zip(names, weights)}
        weights_mapping[CONTROL] = weights.max()
        weights = df.treatment.map(weights_mapping)
    sample = df.sample(BATCH_SIZE, weights=weights, replace=True)

    metrics = None
    if i % COLLECT_RESULTS_FREQUENCY == 0:
        best_index = params.argmax()
        rqu_result = RQU(params, cov).fit(cols=[best_index], beta=.005)
        conf_int = rqu_result.conf_int()
        metrics = {
            "pr_best_arm": names[best_index] == needle,
            "p_value": rqu_result.pvalues[0],
            "len_ci": float(conf_int[:, 1] - conf_int[:, 0])
        }

    return sample, metrics

if __name__ == "__main__":
    sim_no, needle, n_treatments, strategy = sys.argv[1:]
    sim_no = int(sim_no)
    n_treatments = int(n_treatments)
    np.random.seed(sim_no)

    records = []

    df = pd.read_csv(os.path.join(DATA_DIR, "DellaVigna_Pope.csv"))
    hay_df = df[df.treatment == HAYSTACK]
    haystacks = [make_haystack(i, hay_df) for i in range(n_treatments - 1)]
    df = pd.concat([df[df.treatment.isin((CONTROL, needle))]] + haystacks).reset_index(drop=True)
    experiment_df = pd.DataFrame()

    for i in range(N_BATCHES):
        try:
            sample, record = run_batch(i, needle, df, experiment_df, n_treatments, strategy=="adaptive")
            if record is not None:
                record["n_users"] = len(experiment_df)
                records.append(record)
        except:
            sample = df.sample(BATCH_SIZE, replace=True)
        experiment_df = experiment_df.append(sample)

    results_df = pd.DataFrame(records)
    results_df["sim_no"] = sim_no
    results_df["needle"] = needle
    results_df["n_treatments"] = n_treatments
    results_df["strategy"] = strategy
    filename = f"sim_no={sim_no}-needle={needle}-n_treatments={n_treatments}-strategy={strategy}.csv"
    results_df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
