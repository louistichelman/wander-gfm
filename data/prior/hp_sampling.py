"""Sample ``SCMPrior`` hyperparameters from the distribution specs in prior config.

``HpSamplerList`` draws one value per key (uniform, log-uniform, weighted choice,
truncated normal, or a meta-distribution over those parameters).
"""

from __future__ import annotations

import math
import random

import numpy as np
import scipy.stats as stats
import torch


def choice_sampler(choice_values):
    return lambda: random.choice(choice_values)


def bimodal_truncnorm_sampler(mu1, sigma1, mu2, sigma2, p=0.5, lower_bound=0, upper_bound=1000000):
    s1 = trunc_norm_sampler(mu1, sigma1, lower_bound, upper_bound)
    s2 = trunc_norm_sampler(mu2, sigma2, lower_bound, upper_bound)

    return lambda: s1() if np.random.rand() < p else s2()


def trunc_norm_sampler(mu, sigma, lower_bound=0, upper_bound=1000000):
    """Creates a sampler for truncated normal distribution with given mean and std."""
    return lambda: stats.truncnorm(
        (lower_bound - mu) / sigma, (upper_bound - mu) / sigma, loc=mu, scale=sigma
    ).rvs(1)[0]


def beta_sampler(a, b):
    """Creates a sampler for beta distribution with shape parameters a and b."""
    return lambda: np.random.beta(a, b)


def loc_scaled_beta_sampler(a, b, loc, scale):
    """Creates a sampler for beta distribution with shape parameters a and b."""
    return lambda: loc + scale * np.random.beta(a, b)


def gamma_sampler(a, b):
    """Creates a sampler for gamma distribution with shape parameter a and scale parameter b."""
    return lambda: np.random.gamma(a, b)


def uniform_sampler(a, b):
    """Creates a sampler for uniform distribution between a and b."""
    return lambda: np.random.uniform(a, b)


def log_uniform_sampler(a, b):
    """Creates a sampler for log-uniform distribution between a and b."""
    return lambda: np.exp(np.random.uniform(np.log(a), np.log(b))).item()


def log_uniform_int_sampler(a, b):
    """Creates a sampler for log-uniform integer distribution between a and b."""
    return lambda: round(np.exp(np.random.uniform(np.log(a), np.log(b))))


def gamma_int_sampler(shape, scale, min_val, max_val):
    """Sample a positive integer from Gamma(shape, scale), clamped to [min_val, max_val]."""

    def sampler():
        x = np.random.gamma(shape, scale)
        return int(np.clip(round(x), min_val, max_val))

    return sampler


def mixed_log_uniform_sampler(min_first, max_first, min_second, max_second, p_first):
    """Creates a sampler for log-uniform distribution between a and b."""
    return lambda: (
        np.exp(np.random.uniform(np.log(min_first), np.log(max_first))).item()
        if np.random.random() < p_first
        else np.exp(np.random.uniform(np.log(min_second), np.log(max_second))).item()
    )


# TODO: is this really uniform?
def uniform_int_sampler(a, b):
    """Creates a sampler for uniform integer distribution between a and b."""
    return lambda: round(np.random.uniform(a, b))


def uniform_int_with_default_sampler(
    a: int,
    b: int,
    default: int,
    p_default: float,
):
    """Creates a sampler for uniform int in [a, b) or returns default with probability p_default"""
    return (
        lambda: default if (np.random.rand() < p_default) else np.random.randint(a, b)
    )


def weighted_choice_sampler(choice_values, choice_weights):
    """Creates a sampler that picks from choice_values with given probability weights."""
    weights = np.array(choice_weights, dtype=float)
    weights /= weights.sum()
    return lambda: choice_values[np.random.choice(len(choice_values), p=weights)]


def trunc_norm_log_scaled_sampler(min_mean, max_mean, min_std, max_std, lower_bound, do_round):
    """Creates a sampler that draws one value from a log-scaled truncated normal.

    Equivalent to the meta version but collapses the two-level hierarchy into
    a single draw: sample log_mean and log_std, compute mu/sigma, then draw.
    """
    log_min_mean = math.log(min_mean)
    log_max_mean = math.log(max_mean)
    log_min_std = math.log(min_std)
    log_max_std = math.log(max_std)

    def sampler():
        log_mean = np.random.uniform(log_min_mean, log_max_mean)
        log_std = np.random.uniform(log_min_std, log_max_std)
        mu = math.exp(log_mean)
        sigma = mu * math.exp(log_std)
        sample = trunc_norm_sampler(mu, sigma)()
        if do_round:
            sample = round(sample)
        return lower_bound + sample

    return sampler


class HpSampler:
    """
    A modular hyperparameter sampler that supports both basic and meta-distributions.

    Meta-distributions include:
    - meta_beta: Beta distribution with sampled parameters
    - meta_gamma: Gamma distribution with sampled parameters
    - meta_trunc_norm: Truncated normal with sampled parameters
    - meta_trunc_norm_log_scaled: Log-scaled truncated normal
    - meta_choice: Categorical distribution with sampled probabilities
    - meta_choice_mixed: Mixed categorical with sampled probabilities

    Parameters:
        distribution (str): Name of the distribution to use
        device (str): Device to use for tensor operations
        **kwargs: Distribution-specific parameters such as:
            - min, max: bounds for uniform distributions
            - scale: scaling factor for beta distribution
            - lower_bound: minimum value for truncated distributions
            - choice_values: possible values for categorical distributions
    """

    def __init__(self, distribution, device, **kwargs):
        self.distribution = distribution
        self.device = device
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.initialize_distribution()

    def initialize_distribution(self):
        if self.distribution.startswith("meta"):
            self.initialize_meta_distribution()
        elif self.distribution == "choice":
            self.sampler = choice_sampler(self.choice_values)
        elif self.distribution == "uniform":
            self.sampler = uniform_sampler(self.min, self.max)
        elif self.distribution == "log_uniform":
            self.sampler = log_uniform_sampler(self.min, self.max)
        elif self.distribution == "log_uniform_int":
            self.sampler = log_uniform_int_sampler(self.min, self.max)
        elif self.distribution == "gamma_int":
            self.sampler = gamma_int_sampler(
                self.shape,
                self.scale,
                self.min,
                self.max,
            )
        elif self.distribution == "mixed_log_uniform":
            self.sampler = mixed_log_uniform_sampler(
                self.min_first,
                self.max_first,
                self.min_second,
                self.max_second,
                self.p_first,
            )
        elif self.distribution == "beta":
            self.sampler = beta_sampler(self.a, self.b)
        elif self.distribution == "loc_scaled_beta":
            self.sampler = loc_scaled_beta_sampler(self.a, self.b, self.loc, self.scale)
        elif self.distribution == "uniform_int":
            self.sampler = uniform_int_sampler(self.min, self.max)
        elif self.distribution == "uniform_int_with_default":
            self.sampler = uniform_int_with_default_sampler(
                self.min, self.max, self.default, self.p_default
            )
        elif self.distribution == "weighted_choice":
            self.sampler = weighted_choice_sampler(
                self.choice_values, self.choice_weights
            )
        elif self.distribution == "trunc_norm_log_scaled":
            self.sampler = trunc_norm_log_scaled_sampler(
                self.min_mean,
                self.max_mean,
                self.min_std,
                self.max_std,
                getattr(self, "lower_bound", 0),
                getattr(self, "round", False),
            )
        elif self.distribution == "bimodal_truncnorm":
            self.sampler = bimodal_truncnorm_sampler(
                self.mu1,
                self.sigma1,
                self.mu2,
                self.sigma2,
                self.p,
                getattr(self, "lower_bound", 0),
                getattr(self, "upper_bound", 1000000),
            )
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def initialize_meta_distribution(self):
        if self.distribution == "meta_beta":
            self.sampler = self.setup_meta_beta_sampler()
        elif self.distribution == "meta_gamma":
            self.sampler = self.setup_meta_gamma_sampler()
        elif self.distribution == "meta_trunc_norm":
            self.sampler = self.setup_meta_trunc_norm_sampler()
        elif self.distribution == "meta_trunc_norm_log_scaled":
            self.sampler = self.setup_meta_trunc_norm_log_scaled_sampler()
        elif self.distribution == "meta_choice":
            self.sampler = self.setup_meta_choice_sampler()
        elif self.distribution == "meta_choice_mixed":
            self.sampler = self.setup_meta_choice_mixed_sampler()
        else:
            raise ValueError(f"Unsupported meta distribution: {self.distribution}")

    def ensure_hyperparameter(self, attr_name, distribution, min, max):
        if not hasattr(self, attr_name):
            setattr(
                self,
                attr_name,
                HpSampler(
                    distribution=distribution, device=self.device, min=min, max=max
                ),
            )

    def setup_meta_beta_sampler(self):
        """Sets up a meta-beta distribution sampler.
        Returns a closure that samples beta distribution parameters and then samples from that beta."""
        # Dynamically define b and k if not explicitly provided
        self.ensure_hyperparameter("b", "uniform", self.min, self.max)
        self.ensure_hyperparameter("k", "uniform", self.min, self.max)

        def sampler():
            b = self.b() if callable(self.b) else self.b
            k = self.k() if callable(self.k) else self.k
            return lambda: self.scale * beta_sampler(b, k)()

        return sampler

    def setup_meta_gamma_sampler(self):
        """Sets up a meta-gamma distribution sampler.
        Returns a closure that samples gamma distribution parameters and then samples from that gamma."""
        self.ensure_hyperparameter(
            "alpha", "uniform", self.alpha_min, math.log(self.max_alpha)
        )
        self.ensure_hyperparameter(
            "scale", "uniform", self.scale_min, self.max_scale
        )

        def sampler():
            alpha = self.alpha() if callable(self.alpha) else self.alpha
            scale = self.scale() if callable(self.scale) else self.scale

            def sub_sampler():
                sample = gamma_sampler(math.exp(alpha), scale / math.exp(alpha))()
                return (
                    self.lower_bound + round(sample)
                    if self.round
                    else self.lower_bound + sample
                )

            return sub_sampler

        return sampler

    def setup_meta_trunc_norm_sampler(self):
        """Sets up a meta truncated normal distribution sampler.
        Returns a closure that samples normal distribution parameters and then samples from that normal."""
        self.ensure_hyperparameter("mean", "uniform", self.min_mean, self.max_mean)
        self.ensure_hyperparameter("std", "uniform", self.min_std, self.max_std)

        def sampler():
            mean = self.mean() if callable(self.mean) else self.mean
            std = self.std() if callable(self.std) else self.std

            def sub_sampler():
                sample = trunc_norm_sampler(mean, std)()
                return (
                    self.lower_bound + round(sample)
                    if self.round
                    else self.lower_bound + sample
                )

            return sub_sampler

        return sampler

    def setup_meta_trunc_norm_log_scaled_sampler(self):
        """Sets up a log-scaled meta truncated normal distribution sampler.
        Useful for parameters that vary on logarithmic scales."""
        self.ensure_hyperparameter(
            "log_mean", "uniform", math.log(self.min_mean), math.log(self.max_mean)
        )
        self.ensure_hyperparameter(
            "log_std", "uniform", math.log(self.min_std), math.log(self.max_std)
        )

        def sampler():
            log_mean = self.log_mean() if callable(self.log_mean) else self.log_mean
            log_std = self.log_std() if callable(self.log_std) else self.log_std
            mu = math.exp(log_mean)
            sigma = mu * math.exp(log_std)

            def sub_sampler():
                sample = trunc_norm_sampler(mu, sigma)()
                return (
                    self.lower_bound + round(sample)
                    if self.round
                    else self.lower_bound + sample
                )

            return sub_sampler

        return sampler

    def _choice_weight_bounds(self, i: int):
        """Return (min, max) logit bounds for choice *i*.

        Uses per-choice ``choice_weight_ranges`` when available, otherwise
        falls back to the global ``choice_weight_min`` / ``choice_weight_max``.
        """
        if hasattr(self, "choice_weight_ranges"):
            return self.choice_weight_ranges[i]
        return self.choice_weight_min, self.choice_weight_max

    def setup_meta_choice_sampler(self):
        """Sets up a meta-categorical distribution sampler.
        Returns a closure that samples probabilities and then samples categorical values."""
        for i in range(len(self.choice_values)):
            w_min, w_max = self._choice_weight_bounds(i)
            self.ensure_hyperparameter(
                f"choice_{i}_weight",
                distribution="uniform",
                min=w_min,
                max=w_max,
            )

        def sampler():
            weights = []
            for i in range(len(self.choice_values)):
                attr = getattr(self, f"choice_{i}_weight")
                weights.append(attr() if callable(attr) else attr)
            weights = torch.softmax(torch.tensor(weights, dtype=torch.float), 0)
            choice_idx = torch.multinomial(weights, 1).item()
            return self.choice_values[choice_idx]

        return sampler

    def setup_meta_choice_mixed_sampler(self):
        """Sets up a mixed meta-categorical distribution sampler.
        Similar to meta_choice but with different probability scaling."""
        for i in range(len(self.choice_values)):
            w_min, w_max = self._choice_weight_bounds(i)
            self.ensure_hyperparameter(
                f"choice_{i}_weight",
                distribution="uniform",
                min=w_min,
                max=w_max,
            )

        def sampler():
            weights = []
            for i in range(len(self.choice_values)):
                attr = getattr(self, f"choice_{i}_weight")
                weights.append(attr() if callable(attr) else attr)
            weights = torch.softmax(torch.tensor(weights, dtype=torch.float), 0)

            def sub_sampler():
                choice_idx = torch.multinomial(weights, 1).item()
                return self.choice_values[choice_idx]()

            return lambda: sub_sampler

        return sampler

    def __call__(self):
        return self.sampler()


class HpSamplerList:
    """
    A container for multiple hyperparameter samplers that handles batch sampling.

    Parameters:
        hyperparameters (dict): Dictionary mapping parameter names to their sampling configurations
        device (str): Device to use for tensor operations

    Example:
        hp_config = {
            'learning_rate': {
                'distribution': 'meta_trunc_norm_log_scaled',
                'min_mean': 1e-4,
                'max_mean': 1e-1
            },
            'num_layers': {
                'distribution': 'uniform_int',
                'min': 2,
                'max': 10
            }
        }
        sampler = HpSamplerList(hp_config, device='cuda')
        params = sampler.sample()  # Returns dict with sampled values
    """

    def __init__(self, hyperparameters, device):
        self.device = device
        self.hyperparameters = {
            name: HpSampler(device=device, **params)
            for name, params in hyperparameters.items()
            if params
        }

    def sample(self):
        return {name: hp() for name, hp in self.hyperparameters.items()}
