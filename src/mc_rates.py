# src/mc_rates.py
"""
Monte Carlo Collision Rates
============================
Week 7 deliverable — Micah Allard

Computes per-particle collision activities λ_i and total rate Λ
from the separable kernel factors (f, g) produced by kernel.py.

Per the week 7 pseudocode:

    λ_i = f_i · Σ_j g_j   (sum over ALL j, including j = i)

    Λ   = Σ_i λ_i

This is the O(N) reduction: compute sum_g once, multiply per particle.

These values feed directly into mc_timestep.py:
    Λ → sample_dt(Lambda)
    λ → select_particle_i(lambda_array)
    g → select_particle_j(g_array, i)
"""

import numpy as np


def compute_lambda_i(f_array, g_array) -> np.ndarray:
    """
    Compute per-particle collision activities λ_i.

    Per the pseudocode:
        sum_g   = Σ_j g_j
        λ_i     = f_i · sum_g

    The sum runs over ALL particles (including i itself), consistent
    with the pseudocode.  This keeps the implementation strictly O(N).

    Parameters
    ----------
    f_array : array-like, shape (N,)
        Projectile size factors f_i = a_i².  All entries must be ≥ 0.
    g_array : array-like, shape (N,)
        Dynamical state factors g_i = characteristic_velocity(St_i).
        All entries must be ≥ 0.

    Returns
    -------
    lambda_array : numpy array, shape (N,)
        Collision activity λ_i for each super-particle.

    Raises
    ------
    ValueError
        If arrays have mismatched shapes, fewer than 2 elements, or
        contain negative values.
    """
    f_array = np.asarray(f_array, dtype=float)
    g_array = np.asarray(g_array, dtype=float)

    if f_array.shape != g_array.shape:
        raise ValueError(
            f"f_array and g_array must have the same shape, "
            f"got {f_array.shape} and {g_array.shape}."
        )
    if f_array.size < 2:
        raise ValueError("Need at least 2 particles.")
    if np.any(f_array < 0):
        raise ValueError("All f_i must be non-negative.")
    if np.any(g_array < 0):
        raise ValueError("All g_j must be non-negative.")

    sum_g = g_array.sum()

    lambda_array = f_array * sum_g

    return lambda_array


def compute_total_rate(lambda_array) -> float:
    """
    Compute the total collision rate Λ = Σ_i λ_i.

    This is the rate parameter for the exponential waiting-time
    distribution used in the event-driven timestep:

        Δt = −ln(u) / Λ

    Parameters
    ----------
    lambda_array : array-like, shape (N,)
        Per-particle collision activities from compute_lambda_i().
        All entries must be non-negative.

    Returns
    -------
    Lambda : float
        Total collision rate Λ.

    Raises
    ------
    ValueError
        If lambda_array is empty or contains negative values.
    """
    lambda_array = np.asarray(lambda_array, dtype=float)

    if lambda_array.size == 0:
        raise ValueError("lambda_array is empty.")
    if np.any(lambda_array < 0):
        raise ValueError("All λ_i must be non-negative.")

    Lambda = lambda_array.sum()

    return float(Lambda)


# ---------------------------------------------------------------------------
# Quick self-test  (python mc_rates.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=== mc_rates.py self-test ===\n")

    f = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array([0.5, 1.0, 1.5, 2.0])

    lam = compute_lambda_i(f, g)
    sum_g = g.sum()   # = 5.0

    # Expected: λ_i = f_i * sum_g  (pseudocode — no self-exclusion)
    expected = f * sum_g
    assert np.allclose(lam, expected), f"Mismatch: {lam} vs {expected}"
    print(f"sum_g        = {sum_g}")
    print(f"lambda_array = {lam}")
    print(f"expected     = {expected}")

    # All positive
    assert np.all(lam > 0)

    # Total rate
    Lambda = compute_total_rate(lam)
    print(f"Λ            = {Lambda:.4f}  (expected {expected.sum():.4f})")
    assert np.isclose(Lambda, expected.sum())

    # Doubling f_i doubles λ_i
    lam2 = compute_lambda_i(2 * f, g)
    assert np.allclose(lam2, 2 * lam)

    # Error handling
    try:
        compute_lambda_i(np.array([1.0]), np.array([1.0]))
        assert False, "Should raise ValueError for N < 2"
    except ValueError:
        pass

    try:
        compute_total_rate(np.array([]))
        assert False, "Should raise ValueError for empty array"
    except ValueError:
        pass

    print("\nAll mc_rates self-tests passed.")
