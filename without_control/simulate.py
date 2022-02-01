"""Run "needle-in-a-haystack" simulations without a control arm.
"""
import os
import sys

import numpy as np
import pandas as pd

from src.simulator import Simulator

# location of the DellaVigna and Pope effort study data
DATA_DIR = "../data/clean"
# location to store the results
RESULTS_DIR = "results"
N_OBS = [500, 1000, 1500, 2000, 2500]
MIN_OBS_PER_ARM = np.arange(2, 6).astype(int)


if __name__ == "__main__":
    def make_hay_df(i):
        hay_df = df[df.arm == haystack]
        hay_df["arm"] = f"haystack_{str(i).zfill(2)}"
        return hay_df

    # initialize simulation variables
    sim_no, needle, haystack, n_arms = sys.argv[1:]
    sim_no = int(sim_no)
    n_arms = int(n_arms)
    np.random.seed(sim_no)
    records = []

    # read in data from the DellaVigna and Pope effort study
    df = pd.read_csv(os.path.join(DATA_DIR, "pilot07.csv")).rename(columns={"treatment": "arm", "target": "y"})
    df = pd.concat([df[df.arm == needle]] + [make_hay_df(i) for i in range(n_arms - 1)], ignore_index=True)
    simulator = Simulator(df, N_OBS, use_bootstrap=False, use_bayesian_weighting=False)
    simulator.run_random_assignment()
    simulator.run_successive_rejects()
    for min_obs_per_arm in MIN_OBS_PER_ARM:
        simulator.run_exploration_sampling(1, min_obs_per_arm)

    # collect and write out results
    simulator.results["sim_no"] = sim_no
    simulator.results["needle"] = needle
    simulator.results["haystack"] = haystack
    simulator.results["n_arms"] = n_arms
    filename = f"sim_no={sim_no}-needle={needle}-haystack={haystack}-n_arms={n_arms}.csv"
    simulator.results.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
