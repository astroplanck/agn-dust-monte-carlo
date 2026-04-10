# src/mc_rates.py
"""
Monte Carlo Collision Rates
============================
Week 7 deliverable — Micah Allard

Computes the per-particle collision activities λ_i and the total
collision rate Λ from the separable kernel factors (f, g) built by kernel.py.

    λ_i = f_i · Σ_{j≠i} g_j  =  f_i · (G_total − g_i)

    Λ   = Σ_i λ_i

where G_total = Σ_j g_j.

This achieves O(N) complexity: instead of evaluating all N² pairs,
we precompute G_total once and subtract g_i for each i.

These quantities feed directly into mc_timestep.py:
    - Λ  → sample_dt(Lambda)          time to next collision
    - λ  → select_particle_i(lambda_array)
    - g  → select_particle_j(g_array, i)
"""

import numpy as np


def compute_lambda_i(f_array, g_array) -> np.ndarray:
    """
    Compute per-particle collision activities λ_i.

    Using the separable kernel K_ij ≈ f_i · g_j, the activity of
    particle i (its total collision rate with all partners) is:

        λ_i = Σ_{j≠i} K_ij  ≈  f_i · Σ_{j≠i} g_j
                              =  f_i · (G_total − g_i)

    This is O(N): compute G_total = Σ g_j once, then subtract g_i per particle.

    Parameters
    ----------
    f_array : array-like, shape (N,)
        Projectile factors f_i = m_i^(4/3).  All entries must be ≥ 0.
    g_array : array-like, shape (N,)
        Target collision-rate factors g_j = mean(σ_ij · Δv_ij).
        All entries must be ≥ 0.

    Returns
    -------
    lambda_array : numpy array, shape (N,)
        Collision activity λ_i for each super-particle.  Units are
        whatever units f · g carry (typically cm³/s · g^(4/3)).

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

    G_total = g_array.sum()

    # λ_i = f_i · (G_total − g_i)   — exclude self-interaction
    lambda_array = f_array * (G_total - g_array)

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
        Total collision rate Λ.  Always ≥ 0.

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

    return float(lambda_array.sum())


# ---------------------------------------------------------------------------
# Quick self-test  (python mc_rates.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=== mc_rates.py self-test ===")

    # Simple 4-particle case with known values
    f = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array([0.5, 1.0, 1.5, 2.0])

    lam = compute_lambda_i(f, g)
    G   = g.sum()   # = 5.0

    # Expected: λ_i = f_i * (G - g_i)
    expected = f * (G - g)
    assert np.allclose(lam, expected), f"λ mismatch: {lam} vs {expected}"
    print(f"lambda_array = {lam}")
    print(f"expected     = {expected}")

    # All λ_i should be positive (f > 0, G - g_i > 0 for these values)
    assert np.all(lam > 0), "All activities should be positive"

    # Total rate
    Lambda = compute_total_rate(lam)
    print(f"Λ = {Lambda:.6f}   (expected {expected.sum():.6f})")
    assert np.isclose(Lambda, expected.sum())

    # --- Proportionality check ---
    # Doubling all f_i should double all λ_i and Λ
    lam2 = compute_lambda_i(2 * f, g)
    assert np.allclose(lam2, 2 * lam), "Doubling f should double λ"

    # Equal f, equal g → all λ_i equal
    f_eq = np.ones(5)
    g_eq = np.ones(5)
    lam_eq = compute_lambda_i(f_eq, g_eq)
    assert np.allclose(lam_eq, lam_eq[0]), "Equal f,g should give equal λ"

    # --- Error handling ---
    try:
        compute_lambda_i(np.array([1.0]), np.array([1.0]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        compute_total_rate(np.array([]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("\nAll mc_rates self-tests passed.")
