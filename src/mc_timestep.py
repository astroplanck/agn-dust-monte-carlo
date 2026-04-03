# src/mc_timestep.py
"""
Event-Driven Monte Carlo Timestep Module
=========================================
Week 7 deliverable — Micah Allard

Implements the three core functions of the event-driven MC loop:

    1. sample_dt(Lambda)          -> time to next collision
    2. select_particle_i(lambda_array) -> primary particle selection
    3. select_particle_j(g_array, i)   -> secondary particle selection

Physics background
------------------
The separable collision kernel approximates:

    K_ij ≈ f_i * g_j

so that the "collision activity" of particle i is:

    λ_i = f_i * Σ_j g_j   (summed over j ≠ i, O(N) instead of O(N²))

The total collision rate is:

    Λ = Σ_i λ_i

In event-driven MC, we do NOT advance by a fixed Δt. Instead we ask:
"when does the next collision happen?" and draw Δt from an exponential
distribution with rate Λ:

    Δt = -ln(u) / Λ,   u ~ Uniform(0, 1)

High Λ  → small Δt  (collisions are frequent)
Low  Λ  → large Δt  (collisions are rare)

Particle selection is a two-step weighted draw:
    Step 1: select particle i with probability ∝ λ_i
    Step 2: select particle j (j ≠ i) with probability ∝ g_j

NOTE — Missing physics (to be added in kernel.py by Chenran/Yuhong):
    Brownian motion relative velocity Δv_BM is not yet implemented in
    kernel.py. Once it is, kernel.compute_kernel_factors() will include
    it automatically and the functions here require NO changes.
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

        Δt = -ln(u) / Λ,   u ~ Uniform(0, 1)

    Parameters
    ----------
    Lambda : float
        Total collision rate Λ = Σ_i λ_i  (units: s⁻¹).
        Must be strictly positive.
    rng : numpy Generator, optional
        A numpy.random.Generator instance for reproducibility.
        If None, a fresh default_rng() is created.

    Returns
    -------
    dt : float
        Sampled time to next collision in seconds.  Always > 0.

    Raises
    ------
    ValueError
        If Lambda is not strictly positive.
    """
    if Lambda <= 0.0:
        raise ValueError(f"Lambda must be positive, got {Lambda}.")

    if rng is None:
        rng = np.random.default_rng()

    u = rng.random()

    # Guard against the astronomically unlikely u == 0 edge case
    # (would give dt = +inf).  In practice this never fires.
    if u == 0.0:
        u = float(np.finfo(float).tiny)

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

    This is equivalent to selecting which Poisson process fires first.

    Parameters
    ----------
    lambda_array : array-like of float
        1-D array of collision activities λ_i for each super-particle.
        All entries must be non-negative, and at least one must be > 0.
    rng : numpy Generator, optional

    Returns
    -------
    i : int
        Index of the selected primary particle.

    Raises
    ------
    ValueError
        If lambda_array is empty or all-zero.
    """
    lambda_array = np.asarray(lambda_array, dtype=float)

    if lambda_array.size == 0:
        raise ValueError("lambda_array is empty.")
    if np.any(lambda_array < 0):
        raise ValueError("All lambda_i must be non-negative.")

    total = lambda_array.sum()
    if total <= 0.0:
        raise ValueError("Sum of lambda_array is zero — no collisions possible.")

    if rng is None:
        rng = np.random.default_rng()

    # Build cumulative probability array and draw once
    probs = lambda_array / total
    cum   = np.cumsum(probs)
    u     = rng.random()

    i = int(np.searchsorted(cum, u, side="right"))

    # Clamp to valid range (handles floating-point edge at u → 1)
    i = min(i, lambda_array.size - 1)
    return i


# ---------------------------------------------------------------------------
# 3. Secondary particle selection  (∝ g_j, j ≠ i)
# ---------------------------------------------------------------------------

def select_particle_j(g_array, i: int, rng=None) -> int:
    """
    Select the secondary collision partner, particle j ≠ i.

    Given that particle i was selected, particle j is drawn with
    probability proportional to g_j, excluding j = i:

        P(j | i) = g_j / (Σ_{k ≠ i} g_k)

    This follows from the separable-kernel approximation K_ij ≈ f_i * g_j.

    Parameters
    ----------
    g_array : array-like of float
        1-D array of the g-factors for each super-particle.
        All entries must be non-negative.
    i : int
        Index of the already-selected primary particle (excluded from draw).
    rng : numpy Generator, optional

    Returns
    -------
    j : int
        Index of the selected secondary particle.  Guaranteed j ≠ i.

    Raises
    ------
    ValueError
        If g_array has fewer than 2 entries, or all g_j (j ≠ i) are zero.
    """
    g_array = np.asarray(g_array, dtype=float)

    if g_array.size < 2:
        raise ValueError("Need at least 2 particles to form a collision pair.")
    if not (0 <= i < g_array.size):
        raise ValueError(f"Index i={i} out of bounds for g_array of size {g_array.size}.")
    if np.any(g_array < 0):
        raise ValueError("All g_j must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    # Build weights with i masked out
    weights = g_array.copy()
    weights[i] = 0.0

    total = weights.sum()
    if total <= 0.0:
        raise ValueError(
            f"All g_j are zero for j ≠ {i} — cannot select a collision partner."
        )

    probs = weights / total
    cum   = np.cumsum(probs)
    u     = rng.random()

    j = int(np.searchsorted(cum, u, side="right"))
    j = min(j, g_array.size - 1)

    # Safety: should never equal i given the mask, but guard just in case
    # of floating-point pathology near the masked bin
    if j == i:
        # Fall back to the next highest-weight particle
        weights[i] = 0.0
        j = int(np.argmax(weights))

    return j


# ---------------------------------------------------------------------------
# Quick self-test (python mc_timestep.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- sample_dt ---
    Lambda = 1e-10   # s^-1, typical for sparse AGN disk
    dt = sample_dt(Lambda, rng=rng)
    print(f"sample_dt : Λ = {Lambda:.2e} s⁻¹  →  Δt = {dt:.4e} s")
    assert dt > 0, "dt must be positive"

    # --- select_particle_i ---
    # Give particle 0 a dominant rate; it should win most of the time
    lam = np.array([100.0, 1.0, 1.0, 1.0])
    counts = np.zeros(4, dtype=int)
    for _ in range(10_000):
        counts[select_particle_i(lam, rng=rng)] += 1
    print(f"select_particle_i distribution (10k trials): {counts}")
    assert counts[0] == counts.max(), "Particle 0 should be selected most often"

    # --- select_particle_j ---
    g = np.array([1.0, 5.0, 2.0, 3.0])
    j = select_particle_j(g, i=1, rng=rng)   # particle 1 excluded
    print(f"select_particle_j : i=1 excluded  →  j = {j}")
    assert j != 1, "j must not equal i"

    print("\nAll self-tests passed.")
