# src/kernel.py
"""
Collision Kernel Physics
========================
Week 7 deliverable — Micah Allard

Implements the separable collision kernel K_ij ≈ f_i · g_j, where:

    f_i = m_i^(4/3)          (gravitational focusing / mass selection weight)
    g_j = σ_ij · Δv_ij       (geometric cross-section × relative velocity)

The six public functions are:

    1. collision_cross_section(a_i, a_j)
    2. brownian_velocity(m_i, m_j, T_K)
    3. turbulent_velocity(St_i, St_j, alpha, cs_cm_s)
    4. radial_drift_velocity(St_i, St_j, eta, vK_cm_s)
    5. relative_velocity(m_i, m_j, St_i, St_j, T_K, alpha, cs_cm_s, eta, vK_cm_s)
    6. compute_kernel_factors(particles, disk_state, params, alpha, eta, vK_cm_s)

Physics notes
-------------
Brownian motion (from proposal, classical coagulation):
    Δv_BM = sqrt(8 k_B T / π · (1/m_i + 1/m_j))

Turbulent velocity — full Ormel & Cuzzi (2007) three-regime approximation.
Let St_s = min(St_i, St_j), St_l = max(St_i, St_j):

    Regime 1 — both St < 1:
        Δv_t = Vg · |St_i/(1+St_i) − St_j/(1+St_j)|

    Regime 2 — one St ≥ 1, one St < 1:
        Δv_t = Vg · sqrt(St_s / (1 + St_s))

    Regime 3 — both St ≥ 1:
        Δv_t = Vg · sqrt((1/St_i + 1/St_j) / 2)

    where Vg = sqrt(alpha) · cs  is the RMS turbulent gas velocity.

Radial drift (Weidenschilling 1977, via drag_updated.py):
    Δv_drift = eta · vK · |St_i/(1+St_i²) − St_j/(1+St_j²)|

Quadrature sum:
    Δv = sqrt(Δv_BM² + Δv_turb² + Δv_drift²)

Separable decomposition:
    K_ij = f_i · g_j
    f_i  = m_i^(4/3)
    g_j  = (1/(N−1)) · Σ_{i≠j} σ_ij · Δv_ij   (mean over all partners)

    This is the standard least-squares separable approximation: g_j captures
    the mean collision rate of particle j with the rest of the population.
"""

import math
import numpy as np

from params import DiskParams, KB_CGS, PI
from drag_updated import delta_v_drift as _delta_v_drift


# ---------------------------------------------------------------------------
# 1. Collision cross-section
# ---------------------------------------------------------------------------

def collision_cross_section(a_i: float, a_j: float) -> float:
    """
    Geometric collision cross-section (cm²).

        σ_ij = π (a_i + a_j)²

    Gravitational focusing is absorbed into the mass-selection weight f_i = m_i^(4/3),
    so we use the bare geometric cross-section here.

    Parameters
    ----------
    a_i, a_j : float
        Grain radii in cm.  Must be non-negative.

    Returns
    -------
    sigma : float  (cm²)
    """
    if a_i < 0.0 or a_j < 0.0:
        raise ValueError("Grain radii must be non-negative.")
    return PI * (a_i + a_j) ** 2


# ---------------------------------------------------------------------------
# 2. Brownian motion relative velocity
# ---------------------------------------------------------------------------

def brownian_velocity(m_i: float, m_j: float, T_K: float) -> float:
    """
    Relative velocity due to Brownian (thermal) motion (cm/s).

    From the proposal (classical coagulation treatment):
        Δv_BM = sqrt(8 k_B T / π · (1/m_i + 1/m_j))

    Dominates at the earliest growth stage when grains are small
    and Stokes numbers are negligible.

    Parameters
    ----------
    m_i, m_j : float
        Grain masses in grams.  Must be strictly positive.
    T_K : float
        Local gas temperature in Kelvin.  Must be strictly positive.

    Returns
    -------
    dv_BM : float  (cm/s, always ≥ 0)
    """
    if m_i <= 0.0 or m_j <= 0.0:
        raise ValueError("Grain masses must be strictly positive.")
    if T_K <= 0.0:
        raise ValueError("Temperature must be strictly positive.")

    return math.sqrt(8.0 * KB_CGS * T_K / PI * (1.0 / m_i + 1.0 / m_j))


# ---------------------------------------------------------------------------
# 3. Turbulent relative velocity  —  Ormel & Cuzzi (2007), three regimes
# ---------------------------------------------------------------------------

def turbulent_velocity(
    St_i: float,
    St_j: float,
    alpha: float,
    cs_cm_s: float,
) -> float:
    """
    Relative velocity due to turbulence (cm/s).

    Full three-regime Ormel & Cuzzi (2007) closed-form approximation.
    Let St_s = min(St_i, St_j) and St_l = max(St_i, St_j).

    Regime 1 — both St < 1 (both grains coupled to turbulent eddies):
        Δv_t = Vg · |St_i/(1+St_i) − St_j/(1+St_j)|

    Regime 2 — one St ≥ 1, one St < 1 (large grain sweeps through eddies):
        Δv_t = Vg · sqrt(St_s / (1 + St_s))

    Regime 3 — both St ≥ 1 (both grains decouple from gas):
        Δv_t = Vg · sqrt((1/St_i + 1/St_j) / 2)

    where  Vg = sqrt(alpha) · cs  is the RMS turbulent gas velocity.

    Parameters
    ----------
    St_i, St_j : float
        Stokes numbers of the two grains.  Must be non-negative.
    alpha : float
        Turbulent viscosity parameter (dimensionless).  Must be positive.
    cs_cm_s : float
        Isothermal sound speed in cm/s.  Must be positive.

    Returns
    -------
    dv_turb : float  (cm/s, always ≥ 0)
    """
    if St_i < 0.0 or St_j < 0.0:
        raise ValueError("Stokes numbers must be non-negative.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if cs_cm_s <= 0.0:
        raise ValueError("cs must be positive.")

    Vg   = math.sqrt(alpha) * cs_cm_s
    St_s = min(St_i, St_j)
    St_l = max(St_i, St_j)

    if St_l < 1.0:
        # Regime 1: both sub-unity
        dv = Vg * abs(
            St_i / (1.0 + St_i) - St_j / (1.0 + St_j)
        )
    elif St_s < 1.0:
        # Regime 2: mixed — small grain couples to eddies, large grain does not
        dv = Vg * math.sqrt(St_s / (1.0 + St_s))
    else:
        # Regime 3: both super-unity — relative motion set by inertia
        dv = Vg * math.sqrt(0.5 * (1.0 / St_i + 1.0 / St_j))

    return dv


# ---------------------------------------------------------------------------
# 4. Radial drift relative velocity  (thin wrapper around drag_updated)
# ---------------------------------------------------------------------------

def radial_drift_velocity(
    St_i: float,
    St_j: float,
    eta: float,
    vK_cm_s: float,
) -> float:
    """
    Magnitude of the differential radial drift velocity between two grains (cm/s).

    Delegates to drag_updated.delta_v_drift():
        |Δv_drift| = eta · vK · |St_i/(1+St_i²) − St_j/(1+St_j²)|

    Parameters
    ----------
    St_i, St_j : float
        Stokes numbers of the two grains.
    eta : float
        Dimensionless pressure gradient parameter (≥ 0).
    vK_cm_s : float
        Keplerian velocity at the local disk radius (cm/s).

    Returns
    -------
    dv_drift : float  (cm/s, always ≥ 0)
    """
    return _delta_v_drift(St_i, St_j, eta, vK_cm_s)


# ---------------------------------------------------------------------------
# 5. Total relative velocity  (quadrature sum of all three contributions)
# ---------------------------------------------------------------------------

def relative_velocity(
    m_i: float,
    m_j: float,
    St_i: float,
    St_j: float,
    T_K: float,
    alpha: float,
    cs_cm_s: float,
    eta: float,
    vK_cm_s: float,
) -> float:
    """
    Total relative collision velocity (cm/s).

    Quadrature sum of Brownian, turbulent, and radial-drift contributions:

        Δv = sqrt(Δv_BM² + Δv_turb² + Δv_drift²)

    Parameters
    ----------
    m_i, m_j   : grain masses (g)
    St_i, St_j : Stokes numbers (dimensionless)
    T_K        : gas temperature (K)
    alpha      : turbulent viscosity parameter (dimensionless)
    cs_cm_s    : isothermal sound speed (cm/s)
    eta        : dimensionless pressure gradient parameter
    vK_cm_s    : Keplerian velocity (cm/s)

    Returns
    -------
    dv : float  (cm/s, always ≥ 0)
    """
    dv_BM    = brownian_velocity(m_i, m_j, T_K)
    dv_turb  = turbulent_velocity(St_i, St_j, alpha, cs_cm_s)
    dv_drift = radial_drift_velocity(St_i, St_j, eta, vK_cm_s)

    return math.sqrt(dv_BM ** 2 + dv_turb ** 2 + dv_drift ** 2)


# ---------------------------------------------------------------------------
# 6. Compute separable kernel factors f_i and g_j
# ---------------------------------------------------------------------------

def compute_kernel_factors(
    particles,
    disk_state: dict,
    params: DiskParams,
    alpha: float = 1e-3,
    eta: float = None,
    vK_cm_s: float = None,
) -> tuple:
    """
    Build the separable kernel factor arrays f and g for all super-particles.

    Separable decomposition:
        K_ij ≈ f_i · g_j
        f_i  = m_i^(4/3)
        g_j  = (1/(N−1)) · Σ_{i≠j} σ_ij · Δv_ij

    g_j is the mean collision rate of particle j averaged over all partners i.
    This is the standard O(N²) → O(N) reduction: once f and g are built,
    the activity of particle i is  λ_i = f_i · Σ_j g_j  (computed in mc_rates).

    Parameters
    ----------
    particles : list of objects, each with attributes:
                    .mass   (float, g)
                    .radius (float, cm)
                    .St     (float, dimensionless Stokes number)
    disk_state : dict returned by disk.disk_state().  Required keys:
                    'T_K', 'cs_cm_s', 'OmegaK_s^-1'
                 Optional key used if vK_cm_s not supplied:
                    'vK_cm_s'
    params     : DiskParams instance (used for eta estimate if eta is None)
    alpha      : float, turbulence parameter α (default 1e-3)
    eta        : float or None.
                 If None, estimated as  η ≈ 0.5·(cs/vK)²·(p + q + 1.5)
    vK_cm_s    : float or None.
                 Keplerian velocity in cm/s.  If None, taken from disk_state
                 key 'vK_cm_s' if present, otherwise estimated as cs (a
                 rough stand-in; caller should supply the true value).

    Returns
    -------
    f_array : numpy array, shape (N,)  — projectile mass weights  [g^(4/3)]
    g_array : numpy array, shape (N,)  — mean target collision rates  [cm³/s]
    """
    N = len(particles)
    if N < 2:
        raise ValueError("Need at least 2 particles to compute kernel factors.")

    T_K    = disk_state["T_K"]
    cs     = disk_state["cs_cm_s"]
    OmegaK = disk_state["OmegaK_s^-1"]

    # Keplerian velocity: vK = OmegaK · R.  If caller does not supply it
    # directly, we fall back to the value stored in disk_state (if present),
    # or use cs as a rough order-of-magnitude estimate and warn.
    if vK_cm_s is None:
        vK_cm_s = disk_state.get("vK_cm_s", None)
    if vK_cm_s is None:
        import warnings
        warnings.warn(
            "vK_cm_s not supplied and not in disk_state; "
            "falling back to cs as rough estimate. "
            "Pass vK_cm_s = OmegaK * R_cm for accurate drift velocities.",
            RuntimeWarning,
        )
        vK_cm_s = cs

    # Estimate eta from disk power-law indices if not provided
    if eta is None:
        eta = max(0.0, 0.5 * (cs / vK_cm_s) ** 2 * (params.p + params.q + 1.5))

    # Extract per-particle arrays
    masses = np.array([p.mass   for p in particles], dtype=float)
    radii  = np.array([p.radius for p in particles], dtype=float)
    stokes = np.array([p.St     for p in particles], dtype=float)

    # f_i = m_i^(4/3)  — gravitational focusing / mass selection weight
    f_array = masses ** (4.0 / 3.0)

    # g_j = mean over i≠j of σ_ij · Δv_ij
    g_array = np.zeros(N, dtype=float)

    for j in range(N):
        total = 0.0
        for i in range(N):
            if i == j:
                continue
            sigma = collision_cross_section(radii[i], radii[j])
            dv    = relative_velocity(
                m_i     = masses[i],
                m_j     = masses[j],
                St_i    = stokes[i],
                St_j    = stokes[j],
                T_K     = T_K,
                alpha   = alpha,
                cs_cm_s = cs,
                eta     = eta,
                vK_cm_s = vK_cm_s,
            )
            total += sigma * dv
        g_array[j] = total / (N - 1)

    return f_array, g_array


# ---------------------------------------------------------------------------
# Quick self-test  (python kernel.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    print("=== kernel.py self-test ===")

    # --- 1. cross section ---
    sigma = collision_cross_section(1e-4, 2e-4)
    print(f"σ(1µm, 2µm)  = {sigma:.4e} cm²   (expect ~{PI*(3e-4)**2:.4e})")
    assert sigma > 0

    # --- 2. Brownian velocity ---
    m1, m2 = 1e-12, 2e-12   # grams
    T = 100.0                # K
    dv_bm = brownian_velocity(m1, m2, T)
    print(f"Δv_BM        = {dv_bm:.4e} cm/s   (should be positive)")
    assert dv_bm > 0

    # Heavier pair → lower velocity (more inertia)
    dv_bm_heavy = brownian_velocity(1e-6, 2e-6, T)
    assert dv_bm > dv_bm_heavy, "Lighter grains should have higher Brownian velocity"

    # --- 3. Turbulent velocity, all three regimes ---
    alpha, cs = 1e-3, 1e5   # cm/s
    dv_r1 = turbulent_velocity(0.01, 0.1,  alpha, cs)   # both < 1
    dv_r2 = turbulent_velocity(0.1,  2.0,  alpha, cs)   # mixed
    dv_r3 = turbulent_velocity(2.0,  5.0,  alpha, cs)   # both > 1
    print(f"Δv_turb R1   = {dv_r1:.4e} cm/s")
    print(f"Δv_turb R2   = {dv_r2:.4e} cm/s")
    print(f"Δv_turb R3   = {dv_r3:.4e} cm/s")
    assert dv_r1 >= 0 and dv_r2 >= 0 and dv_r3 >= 0

    # Equal St → zero turbulent velocity (regime 1 & 3)
    assert turbulent_velocity(0.1, 0.1, alpha, cs) == 0.0
    assert turbulent_velocity(3.0, 3.0, alpha, cs) == 0.0

    # --- 4. Drift velocity ---
    dv_drift = radial_drift_velocity(0.01, 0.1, eta=0.003, vK_cm_s=1e7)
    print(f"Δv_drift     = {dv_drift:.4e} cm/s")
    assert dv_drift >= 0

    # --- 5. Relative velocity ---
    dv = relative_velocity(
        m_i=1e-12, m_j=2e-12,
        St_i=0.01, St_j=0.1,
        T_K=100.0, alpha=1e-3,
        cs_cm_s=1e5, eta=0.003, vK_cm_s=1e7,
    )
    print(f"Δv_total     = {dv:.4e} cm/s   (≥ each component)")
    assert dv >= max(dv_bm, dv_r1, dv_drift)

    print("\nAll kernel self-tests passed.")
