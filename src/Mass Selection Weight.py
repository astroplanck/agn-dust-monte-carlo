# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
import numpy as np


def compute_mass_weights(n_real, m_mean):
    """
    Simple logic:
    1) Compute total mass of each super particle: M_i = N_i * m_mean_i
    2) Compute raw weight: w_i = M_i^(4/3)
    3) Normalize to probabilities: W_i = w_i / sum(w_i)

    Inputs:
    - n_real : 1D array-like, number of real particles in each super particle (N_i)
    - m_mean : 1D array-like, mean mass of real particles in each super particle (m_mean_i)

    Output:
    - weights : 1D numpy array, normalized selection probability for each super particle
    """
    n_real = np.asarray(n_real, dtype=float)
    m_mean = np.asarray(m_mean, dtype=float)

    if n_real.shape != m_mean.shape:
        raise ValueError("n_real and m_mean must have the same shape.")
    if np.any(n_real < 0) or np.any(m_mean < 0):
        raise ValueError("Inputs must be non-negative.")

    # Total mass of each super particle
    M = n_real * m_mean

    # Raw mass-based weights (proportional to M^(4/3))
    raw_w = M ** (4.0 / 3.0)

    total_w = raw_w.sum()
    if total_w <= 0:
        raise ValueError("Sum of weights is zero. Check input masses.")

    # Normalized probabilities
    weights = raw_w / total_w
    return weights


def sample_super_particle_manual(weights, rng=None):
    """
    Sample one super particle index by manual cumulative-interval method.

    Logic:
    1) Normalize weights
    2) Build cumulative probabilities
    3) Draw a random number u in [0, 1)
    4) Find which interval contains u
    """
    weights = np.asarray(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")
    if weights.sum() <= 0:
        raise ValueError("weights sum must be positive.")

    # Step 1: normalize
    probs = weights / weights.sum()

    # Step 2: cumulative probabilities
    cum_probs = np.cumsum(probs)

    # Step 3: random number in [0, 1)
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random()

    # Step 4: find interval
    # Find first index i such that u < cum_probs[i]
    idx = np.searchsorted(cum_probs, u, side="right")

    return idx


# A Example
if __name__ == "__main__":
    n_sp = 10_000
    rng = np.random.default_rng()

    # Replace these with real simulation data
    n_real = rng.integers(1, 1000, size=n_sp)
    m_mean = rng.uniform(1e-18, 1e-15, size=n_sp)

    weights = compute_mass_weights(n_real, m_mean)
    chosen_idx = sample_super_particle(weights, rng=rng)

    print("weights sum =", weights.sum())
    print("chosen SP index =", chosen_idx)
