"""Run "needle-in-a-haystack" simulations without a control arm.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from conditional_inference.bayes.empirical import LinearEmpiricalBayes
from conditional_inference.rqu import RQU
from scipy.stats import multivariate_normal

# location of the DellaVigna and Pope effort study data
DATA_DIR = "../data/clean"
# location to store the results
RESULTS_DIR = "results"
BATCH_SIZE = 10
N_BATCHES = 501
COLLECT_RESULTS_FREQUENCY = 50  # frequency (in batches) to record metrics
SIZE = 1000  # number of samples from the posterior distribution of effects
HAYSTACK = "Very_Low_Pay"

def make_haystack(i, hay_df):
    df = hay_df.copy()
    df["treatment"] = f"hay_{str(i).zfill(3)}"
    return df

def run_batch(i, needle, df, experiment_df, n_treatments, use_adaptive_assignment):
    """Run a batch.

    Args:
        i (int): Batch number.
        needle (str): Name of the needle arm.
        df (pd.DataFrame): Data from the original DellaVigna and Pope experiment.
        experiment_df (pd.DataFrame): Data from the current (simulated) experiment.
        n_treatments (int): Number of treatment arms.
        use_adaptive_assignment (bool): Use adaptive (as opposed to random) assignment.

    Raises:
        RuntimeError: If there is not enough data to estimate the effects of all
            treatment arms.

    Returns:
        tuple[pd.DataFrame, Optional[dict[str, Any]]]: Sample of participants in the
            current batch, mapping of metric names to values
    """
    if use_adaptive_assignment or i % COLLECT_RESULTS_FREQUENCY == 0:
        # estimate distribution of treatment effects
        X = pd.get_dummies(experiment_df.treatment)
        results = sm.OLS(experiment_df.target, X).fit().get_robustcov_results()
        names = np.array(results.model.exog_names)
        if len(results.params) < n_treatments:
            raise RuntimeError("Model was unable to estimate effects of all treatments")
        cov = results.cov_params()

    weights = None
    if use_adaptive_assignment:
        dist = multivariate_normal(results.params, cov)

        # compute acquisition function (exploration sampling) weights
        pr_best = np.identity(n_treatments)[dist.rvs(SIZE).argmax(axis=1)].mean(axis=0)
        weights = pr_best * (1 - pr_best)

        # draw the sample for the next batch
        weights_mapping = {name: weight for name, weight in zip(names, weights)}
        weights = df.treatment.map(weights_mapping)
    sample = df.sample(BATCH_SIZE, weights=weights, replace=True)

    metrics = None
    if i % COLLECT_RESULTS_FREQUENCY == 0:
        best_index = LinearEmpiricalBayes(results.params, cov).fit().params.argmax()
        rqu_result = RQU(results.params, cov).fit(cols=[best_index], beta=.005)
        conf_int = rqu_result.conf_int()
        metrics = {
            "pr_best_arm": names[best_index] == needle,
            "len_ci": float(conf_int[:, 1] - conf_int[:, 0])
        }

    return sample, metrics

if __name__ == "__main__":
    # initialize simulation variables
    sim_no, needle, n_treatments, strategy = sys.argv[1:]
    sim_no = int(sim_no)
    n_treatments = int(n_treatments)
    np.random.seed(sim_no)
    records = []

    # read in data from the DellaVigna and Pope effort study
    df = pd.read_csv(os.path.join(DATA_DIR, "DellaVigna_Pope.csv"))
    hay_df = df[df.treatment == HAYSTACK]
    haystacks = [make_haystack(i, hay_df) for i in range(n_treatments - 1)]
    df = pd.concat([df[df.treatment == needle]] + haystacks).reset_index(drop=True)
    experiment_df = pd.DataFrame()

    # run the simulation
    for i in range(N_BATCHES):
        try:
            sample, record = run_batch(
                i, needle, df, experiment_df, n_treatments, strategy=="adaptive"
            )
            if record is not None:
                record["n_users"] = len(experiment_df)
                records.append(record)
        except:
            # an exception will be raised when there are not enough data to estimate
            # the effects of all treatment arms
            sample = df.sample(BATCH_SIZE, replace=True)
        experiment_df = experiment_df.append(sample)

    # collect and write out results
    results_df = pd.DataFrame(records)
    results_df["sim_no"] = sim_no
    results_df["needle"] = needle
    results_df["n_treatments"] = n_treatments
    results_df["strategy"] = strategy
    filename = f"sim_no={sim_no}-needle={needle}-n_treatments={n_treatments}-strategy={strategy}.csv"
    results_df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
