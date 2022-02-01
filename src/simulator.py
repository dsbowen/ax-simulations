import warnings
from typing import Iterable, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from conditional_inference.bayes.classic import LinearClassicBayes
from conditional_inference.bayes.empirical import LinearEmpiricalBayes
from conditional_inference.rqu import RQU
from scipy.stats import multivariate_normal


class Simulator:
    """Convenience class for running simulations.

    Args:
        filename (str): Name of the data file.
        n_obs (Iterable[int], optional): Number of observations to simulate. Defaults to
            None.
        X_policy (np.ndarray, optional): (# treatments, # features) array describing
            policy features. If None, this will use a constant regressor. Defaults to
            None.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_obs: Union[int, Iterable[int]] = None,
        use_bootstrap: bool = True,
        use_bayesian_weighting: bool = True,
        X_policy: np.ndarray = None,
    ):
        self.n_obs = [n_obs] if isinstance(n_obs, int) else n_obs
        self.X_simulated, self.y_simulated = None, None
        self.results = pd.DataFrame()
        df = df[df.arm != "control"]
        if use_bootstrap:
            # create stratified bootstrapped dataset
            bootstrapped_df = df.groupby("arm", group_keys=False).apply(
                lambda x: x.sample(frac=1, replace=True)
            )
            self.X, self.y = pd.get_dummies(bootstrapped_df.arm), bootstrapped_df.y
        else:
            self.X, self.y = pd.get_dummies(df.arm), df.y
        self.n_arms = self.X.shape[1]

        result = sm.OLS(self.y, self.X).fit().get_robustcov_results()
        mean, cov = result.params, result.cov_params()
        if use_bayesian_weighting:
            # use an empirical Bayes estimator to re-weight observations so that the true
            # treatment effects are drawn from the empirical Bayes posterior
            if X_policy is None:
                X_policy = np.ones((self.n_arms, 1))
            projection = X_policy @ np.linalg.inv(X_policy.T @ X_policy) @ X_policy.T
            tau = (
                ((mean - projection @ mean) ** 2).sum()
                / (mean.shape[0] - X_policy.shape[1] - 2)
                * np.identity(mean.shape[0])
            )
            # weight_matrix[i, k] is the weight to put on arm k
            # when assigning a participant
            weight_matrix = cov @ np.linalg.inv(tau) @ (
                projection - np.identity(mean.shape[0])
            ) + np.identity(mean.shape[0])
            sample_weight = (
                weight_matrix.repeat(self.X.sum(axis=0), axis=1).T
                / self.X.sum(axis=0).values
            )
            # (# observations, # treatments) vector of weights
            self.sample_weight = np.clip(sample_weight, 0, np.inf)
        else:
            # use an un-weighted (traditional) bootstrap
            weight_matrix = np.identity(self.n_arms)
            self.sample_weight = pd.get_dummies(df.arm).values

        # compute ground truth treatment effects and rankings
        self.true_effects = weight_matrix @ mean
        self.true_ranks = (-self.true_effects).argsort().argsort()

    def clear_simulated_observations(self) -> None:
        """Clear the simulated observations."""
        self.X_simulated, self.y_simulated = None, None

    def assign_participants(
        self, n_obs: int = None, assignment_weight: np.ndarray = None
    ) -> None:
        """Assign participants to treatments.

        Args:
            n_obs (int, optional): Number of participants to assign. Defaults to None.
            assignment_weight (np.ndarray, optional): (# arms,) array of the proportion
                of participants assigned to each treatment. Defaults to None.
        """
        if n_obs is None:
            n_obs = self.batch_size
        if assignment_weight is None or assignment_weight.sum() == 0:
            assignment_weight = np.ones(self.n_arms)
        assignment_weight = assignment_weight / assignment_weight.sum()
        n_obs_per_arm = np.floor(n_obs * assignment_weight)
        remaining_obs = np.random.choice(
            np.arange(self.n_arms),
            size=int(n_obs - n_obs_per_arm.sum()),
            p=assignment_weight,
        )
        n_obs_per_arm = (
            n_obs_per_arm + np.identity(self.n_arms)[remaining_obs].sum(axis=0)
        ).astype(int)

        y = []
        for k in range(self.n_arms):
            y += self.y.sample(
                n=n_obs_per_arm[k], replace=True, weights=self.sample_weight[:, k]
            ).tolist()
        X = np.identity(self.n_arms).repeat(n_obs_per_arm, axis=0)

        if self.X_simulated is None:
            self.X_simulated, self.y_simulated = X, np.array(y)
        else:
            self.X_simulated = np.concatenate((self.X_simulated, X), axis=0)
            self.y_simulated = np.concatenate((self.y_simulated, y))

    def update_results(
        self, strategy: str, recommended_arm: int = None, **params
    ) -> None:
        """Update simulation results based on the simulated data.

        Args:
            strategy (str): Name of the assignment strategy.
            recommended_arm (int, optional): Index of the recommended treatment.
                Defaults to None.
        """

        def get_results(column):
            def compute_estimates(
                estimator_name, model_cls, init_kwargs=None, fit_kwargs=None
            ):
                if init_kwargs is None:
                    init_kwargs = {}
                if fit_kwargs is None:
                    fit_kwargs = {}

                results = model_cls.from_results(ols_results, **init_kwargs).fit(
                    cols=[column], **fit_kwargs
                )
                conf_int = results.conf_int()[0]
                return {
                    "estimator": estimator_name,
                    "point": results.params[0],
                    "ppf025": conf_int[0],
                    "ppf975": conf_int[1],
                }

            try:
                df = pd.DataFrame(
                    [
                        compute_estimates(
                            "OLS",
                            LinearClassicBayes,
                            init_kwargs=dict(prior_cov=np.inf),
                        ),
                        compute_estimates("Conditional", RQU),
                        compute_estimates("Hybrid", RQU, fit_kwargs=dict(beta=0.005)),
                        compute_estimates(
                            "Projection", RQU, fit_kwargs=dict(projection=True)
                        ),
                        compute_estimates("Bayes", LinearEmpiricalBayes),
                    ]
                )
                df["n_obs_recommended"] = self.X_simulated[:, column].sum()
                df["rank"] = self.true_ranks[column]
                df["effect"] = self.true_effects[column]
                return df
            except np.linalg.LinAlgError:
                warnings.warn("Singular matrix detected.")
                return pd.DataFrame()

        ols_results = (
            sm.OLS(self.y_simulated, self.X_simulated).fit().get_robustcov_results()
        )
        if recommended_arm is None:
            bayes_df = get_results(
                LinearEmpiricalBayes.from_results(ols_results).fit().params.argmax()
            )
            bayes_df["selection"] = "bayes"
            ols_df = get_results(ols_results.params.argmax())
            ols_df["selection"] = "conventional"
            df = pd.concat([bayes_df, ols_df])
        else:
            df = get_results(recommended_arm)
        df["n_obs"] = self.y_simulated.shape[0]
        df["average_effect"] = self.true_effects.mean()
        df["best_effect"] = self.true_effects.max()
        df["strategy"] = strategy
        for key, value in params.items():
            df[key] = value
        self.results = self.results.append(df, ignore_index=True)

    def run_random_assignment(self) -> None:
        """Run simulations with random assignment."""
        for n_obs in self.n_obs:
            self.clear_simulated_observations()
            self.assign_participants(n_obs)
            self.update_results("random")

    def run_exploration_sampling(
        self, batch_size: int = 1, min_obs_per_arm: int = 0
    ) -> None:
        """Run simulations with exploration sampling."""
        self.clear_simulated_observations()
        self.assign_participants(min_obs_per_arm * self.n_arms)
        n_batches = np.round(np.array(self.n_obs) / batch_size).astype(int)
        starting_batch = int(np.floor(len(self.y_simulated) / batch_size))
        for batch_no in range(starting_batch, n_batches.max() + 1):
            if (n_batches == batch_no).any():
                self.update_results(
                    "exploration",
                    batch_size=batch_size,
                    min_obs_per_arm=min_obs_per_arm,
                )

            try:
                results = (
                    sm.OLS(self.y_simulated, self.X_simulated)
                    .fit()
                    .get_robustcov_results()
                )
                rvs = multivariate_normal(results.params, results.cov_params()).rvs(
                    10000
                )
                pr_best = np.identity(self.n_arms)[rvs.argmax(axis=1)].mean(axis=0)
                assignment_weights = pr_best * (1 - pr_best)
            except:
                assignment_weights = None
            n_obs = batch_size - len(self.y_simulated) % batch_size
            self.assign_participants(n_obs, assignment_weights)

    def run_successive_rejects(self) -> None:
        """Run simulations with successive rejects."""
        logK = 0.5 + sum([1 / i for i in range(2, self.n_arms + 1)])
        for n_obs in self.n_obs:
            self.clear_simulated_observations()
            # n_k is an array of number of observations per iteration
            n_k = np.array(
                [
                    np.ceil(1 / logK * (n_obs - self.n_arms) / (self.n_arms + 1 - i))
                    for i in range(1, self.n_arms)
                ]
            )
            n_k = np.arange(self.n_arms, 1, -1) * np.diff(np.insert(n_k, 0, 0))
            # add remaining observations to zeroeth batch
            n_k[0] += n_obs - n_k.sum()
            assignment_weight = np.ones(self.n_arms)
            for n_ki in n_k:
                self.assign_participants(n_ki, assignment_weight)
                params = (
                    sm.OLS(self.y_simulated, self.X_simulated)
                    .fit()
                    .get_robustcov_results()
                    .params
                )
                # indicates that the arm hasn't been rejected yet
                active_arm = assignment_weight == 1
                drop_idx = np.where(active_arm & (params == params[active_arm].min()))[
                    0
                ][0]
                assignment_weight[drop_idx] = 0
            self.update_results(
                "successive_rejects", np.where(assignment_weight == 1)[0][0]
            )
