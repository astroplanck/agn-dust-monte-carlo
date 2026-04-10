# src/mc_timestep.py
"""
Event-Driven Monte Carlo Timestep
===================================
Week 7 deliverable — Micah Allard

Implements the three core functions of the event-driven MC loop:

    1. sample_dt(Lambda)               — time to next collision
    2. select_particle_i(lambda_array) — primary particle selection
    3. select_particle_j(g_array, i)   — secondary particle selection

Physics background
------------------
The separable collision kernel K_ij ≈ f_i · g_j gives each particle i
a collision activity:

    λ_i = f_i · Σ_j g_j

The total collision rate is:

    Λ = Σ_i λ_i

In event-driven MC we do NOT advance by a fixed Δt.  Instead we draw
the time to the next collision from an exponential distribution:

    Δt = −ln(u) / Λ,   u ~ Uniform(0, 1)

High Λ → small Δt (frequent collisions)
Low  Λ → large Δt (rare collisions)

Particle selection is a two-step weighted draw:
    Step 1: select particle i  with probability ∝ λ_i
    Step 2: select particle j (j ≠ i)  with probability ∝ g_j,
            using a reject-resample loop to guarantee j ≠ i.

NOTE — Brownian motion:
    brownian_velocity() is implemented in kernel.py and is included
    in the relative velocities that build g_array.  No changes needed here.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# 1. Timestep sampling
# ---------------------------------------------------------------------------

def sample_dt(Lambda: float, rng=None) -> float:
    """
    Draw the time interval to the next collision event.

    Uses the standard result for a Poisson process with rate Λ:

        Δt = −ln(u) / Λ,   u ~ Uniform(0, 1)

    Parameters
    ----------
    Lambda : float
        Total collision rate Λ = Σ_i λ_i  (s⁻¹).
        Must be strictly positive.
    rng : numpy Generator, optional
        A numpy.random.Generator for reproducibility.
        If None, a fresh default_rng() is created.

    Returns
    -------
    dt : float  (s, always > 0)

    Raises
    ------
    ValueError
        If Lambda ≤ 0.
    """
    if Lambda <= 0:
        raise ValueError(f"Lambda must be positive, got {Lambda}.")

    if rng is None:
        rng = np.random.default_rng()

    u  = rng.random()
    dt = -math.log(u) / Lambda

    return dt


# ---------------------------------------------------------------------------
# 2. Primary particle selection  (∝ λ_i)
# ---------------------------------------------------------------------------

def select_particle_i(lambda_array, rng=None) -> int:
    """
    Select the primary collision participant, particle i.

    Particle i is drawn with probability proportional to its collision
    activity λ_i:

        P(i) = λ_i / Λ

    Parameters
    ----------
    lambda_array : array-like of float
        1-D array of collision activities λ_i.
        All entries must be non-negative; at least one must be > 0.
    rng : numpy Generator, optional

    Returns
    -------
    i : int  — index of the selected primary particle

    Raises
    ------
    ValueError
        If lambda_array is empty or all-zero.
    """
    lambda_array = np.asarray(lambda_array, dtype=float)

    Lambda = lambda_array.sum()

    if Lambda <= 0:
        raise ValueError("Sum of lambda_array must be positive.")

    probs_i = lambda_array / Lambda

    if rng is None:
        rng = np.random.default_rng()

    i = rng.choice(len(lambda_array), p=probs_i)

    return int(i)


# ---------------------------------------------------------------------------
# 3. Secondary particle selection  (∝ g_j, j ≠ i)
# ---------------------------------------------------------------------------

def select_particle_j(g_array, i: int, rng=None) -> int:
    """
    Select the secondary collision partner, particle j ≠ i.

    Particle j is drawn with probability proportional to g_j, using a
    reject-resample loop to guarantee j ≠ i (per the pseudocode):

        REPEAT:
            j = RANDOM_CHOICE(indices, probability ∝ g_j)
        UNTIL j ≠ i

    Parameters
    ----------
    g_array : array-like of float
        1-D array of dynamical state factors g_j.
        All entries must be non-negative; at least one g_{j≠i} must be > 0.
    i : int
        Index of the already-selected primary particle.
    rng : numpy Generator, optional

    Returns
    -------
    j : int  — index of the selected secondary particle, guaranteed j ≠ i

    Raises
    ------
    ValueError
        If g_array has fewer than 2 entries or sums to zero.
    """
    g_array = np.asarray(g_array, dtype=float)

    sum_g = g_array.sum()

    if sum_g <= 0:
        raise ValueError("Sum of g_array must be positive.")
    if g_array.size < 2:
        raise ValueError("Need at least 2 particles to form a collision pair.")

    if rng is None:
        rng = np.random.default_rng()

    probs_j = g_array / sum_g

    # Reject-resample loop: keep drawing until j ≠ i
    j = i
    while j == i:
        j = rng.choice(len(g_array), p=probs_j)

    return int(j)


# ---------------------------------------------------------------------------
# Quick self-test  (python mc_timestep.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=== mc_timestep.py self-test ===\n")

    # --- sample_dt ---
    Lambda = 1e-10
    dt = sample_dt(Lambda, rng=rng)
    print(f"sample_dt : Λ={Lambda:.1e} s⁻¹ → Δt={dt:.4e} s")
    assert dt > 0

    # Error case
    try:
        sample_dt(0.0)
        assert False
    except ValueError:
        pass

    # --- select_particle_i ---
    lam = np.array([100.0, 1.0, 1.0, 1.0])
    counts = np.zeros(4, dtype=int)
    for _ in range(10_000):
        counts[select_particle_i(lam, rng=rng)] += 1
    print(f"select_particle_i (10k): {counts}  — particle 0 should dominate")
    assert counts[0] == counts.max()

    # --- select_particle_j ---
    g = np.array([1.0, 5.0, 2.0, 3.0])
    j = select_particle_j(g, i=1, rng=rng)
    print(f"select_particle_j : i=1 excluded → j={j}")
    assert j != 1

    # Confirm j ≠ i over many trials
    for _ in range(1000):
        j = select_particle_j(g, i=2, rng=rng)
        assert j != 2, "j must never equal i"

    print("\nAll mc_timestep self-tests passed.")
