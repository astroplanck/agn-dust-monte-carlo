# src/kernel.py
"""
Collision Kernel Physics

Implements the separable collision kernel K_ij ≈ f_i · g_j.

Per the week 7 pseudocode:
    f_i = a_i²                             (geometric size factor)
    g_i = characteristic_velocity(St_i)    (dynamical state factor)

Public functions:
    1. collision_cross_section(particle_i, particle_j)
    2. brownian_velocity(m_i, m_j, T_K)
    3. turbulent_velocity(St_i, St_j, cs_cm_s, alpha)
    4. radial_drift_velocity(St, disk_state)
    5. relative_velocity(particle_i, particle_j, disk_state, params)
    6. compute_kernel_factors(particles, disk_state, params)

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
    v_drift(St) = −eta · vK · St / (1 + St²)
    Δv_drift    = |v_drift(St_i) − v_drift(St_j)|

Quadrature sum:
    Δv = sqrt(Δv_BM² + Δv_turb² + Δv_drift²)
"""

import math
import numpy as np

from params import DiskParams, KB_CGS, PI
from drag_updated import radial_drift_velocity as _single_drift_velocity


# ---------------------------------------------------------------------------
# 1. Collision cross-section
# ---------------------------------------------------------------------------

def collision_cross_section(particle_i, particle_j) -> float:
    """
    Geometric collision cross-section σ_ij (cm²).

        σ_ij = π (a_i + a_j)²

    Parameters
    ----------
    particle_i, particle_j : objects with attribute .radius (cm)

    Returns
    -------
    sigma_ij : float  (cm²)
    """
    a_i = particle_i.radius
    a_j = particle_j.radius

    sigma_ij = PI * (a_i + a_j) ** 2

    return sigma_ij


# ---------------------------------------------------------------------------
# 2. Brownian motion relative velocity
# ---------------------------------------------------------------------------

def brownian_velocity(m_i: float, m_j: float, T_K: float) -> float:
    """
    Relative velocity due to Brownian (thermal) motion (cm/s).

    From the proposal (classical coagulation):
        Δv_BM = sqrt(8 k_B T / π · (1/m_i + 1/m_j))

    Dominates in the earliest growth stage when grains are small
    and Stokes numbers are negligible.

    Parameters
    ----------
    m_i, m_j : float  — grain masses (g), must be strictly positive
    T_K      : float  — local gas temperature (K), must be strictly positive

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
    cs_cm_s: float,
    alpha: float,
) -> float:
    """
    Relative velocity due to turbulence (cm/s).

    Full three-regime Ormel & Cuzzi (2007) closed-form approximation.
    Note argument order matches pseudocode: (St_i, St_j, cs, alpha).

    Let St_s = min(St_i, St_j) and St_l = max(St_i, St_j):

    Regime 1 — both St < 1:
        Δv_t = Vg · |St_i/(1+St_i) − St_j/(1+St_j)|

    Regime 2 — one St ≥ 1, one St < 1:
        Δv_t = Vg · sqrt(St_s / (1 + St_s))

    Regime 3 — both St ≥ 1:
        Δv_t = Vg · sqrt((1/St_i + 1/St_j) / 2)

    Parameters
    ----------
    St_i, St_j : float  — Stokes numbers, must be ≥ 0
    cs_cm_s    : float  — isothermal sound speed (cm/s)
    alpha      : float  — turbulent viscosity parameter

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
        # Regime 2: one sub-unity, one super-unity
        dv = Vg * math.sqrt(St_s / (1.0 + St_s))
    else:
        # Regime 3: both super-unity
        dv = Vg * math.sqrt(0.5 * (1.0 / St_i + 1.0 / St_j))

    return dv


# ---------------------------------------------------------------------------
# 4. Single-particle radial drift velocity
# ---------------------------------------------------------------------------

def radial_drift_velocity(St: float, disk_state: dict) -> float:
    """
    Radial drift velocity of a single grain (cm/s).

        v_drift(St) = −eta · vK · St / (1 + St²)

    Delegates to drag_updated.radial_drift_velocity().
    Negative = inward drift toward the SMBH.

    Parameters
    ----------
    St         : float — Stokes number of the grain
    disk_state : dict  — must contain keys 'eta' and 'vK_cm_s'

    Returns
    -------
    v_drift : float  (cm/s, negative = inward)
    """
    eta = disk_state["eta"]
    vK  = disk_state["vK_cm_s"]

    return _single_drift_velocity(St, eta, vK)


# ---------------------------------------------------------------------------
# 5. Total relative velocity  (quadrature sum)
# ---------------------------------------------------------------------------

def relative_velocity(
    particle_i,
    particle_j,
    disk_state: dict,
    params: DiskParams,
) -> float:
    """
    Total relative collision velocity (cm/s).

    Matches pseudocode signature:
        relative_velocity(particle_i, particle_j, disk_state, params)

    Quadrature sum of all three contributions:
        Δv = sqrt(Δv_BM² + Δv_turb² + Δv_drift²)

    Parameters
    ----------
    particle_i, particle_j : objects with attributes .mass (g) and .St
    disk_state : dict with keys 'T_K', 'cs_cm_s', 'eta', 'vK_cm_s'
    params     : DiskParams — provides .alpha

    Returns
    -------
    v_rel : float  (cm/s, always ≥ 0)
    """
    m_i  = particle_i.mass
    m_j  = particle_j.mass
    St_i = particle_i.St
    St_j = particle_j.St

    T  = disk_state["T_K"]
    cs = disk_state["cs_cm_s"]

    # Brownian contribution
    v_brownian = brownian_velocity(m_i, m_j, T)

    # Turbulent contribution
    v_turbulence = turbulent_velocity(St_i, St_j, cs, params.alpha)

    # Drift contribution
    v_drift_i = radial_drift_velocity(St_i, disk_state)
    v_drift_j = radial_drift_velocity(St_j, disk_state)
    v_drift   = abs(v_drift_i - v_drift_j)

    v_rel = math.sqrt(
        v_brownian   ** 2 +
        v_turbulence ** 2 +
        v_drift      ** 2
    )

    return v_rel


# ---------------------------------------------------------------------------
# 6. Compute separable kernel factors f_i and g_i
# ---------------------------------------------------------------------------

def _characteristic_velocity(St: float, disk_state: dict, params: DiskParams) -> float:
    """
    Characteristic velocity for a single particle (cm/s).

    Used as g_i in the separable kernel decomposition K_ij ≈ f_i · g_j.
    Encodes how dynamically active particle i is, based on its Stokes number:

        g_i = Vg · St_i / (1 + St_i)

    where Vg = sqrt(alpha) · cs.  This factor peaks at large St (decoupled
    grains) and goes to zero for perfectly gas-coupled grains (St → 0).

    Parameters
    ----------
    St         : float — Stokes number
    disk_state : dict  — must contain 'cs_cm_s'
    params     : DiskParams — provides .alpha

    Returns
    -------
    v_char : float  (cm/s)
    """
    Vg = math.sqrt(params.alpha) * disk_state["cs_cm_s"]
    return Vg * St / (1.0 + St)


def compute_kernel_factors(
    particles,
    disk_state: dict,
    params: DiskParams,
) -> tuple:
    """
    Build the separable kernel factor arrays f and g for all super-particles.

    Per the pseudocode:
        f_i = a_i²                              (geometric size factor)
        g_i = characteristic_velocity(St_i)     (dynamical state factor)

    Parameters
    ----------
    particles  : list of objects, each with attributes:
                     .radius  (float, cm)
                     .St      (float, dimensionless Stokes number)
    disk_state : dict from disk.disk_state(), must also contain
                 'eta' and 'vK_cm_s'
    params     : DiskParams instance (provides .alpha)

    Returns
    -------
    f_array : numpy array, shape (N,)  — geometric size factors  [cm²]
    g_array : numpy array, shape (N,)  — characteristic velocities  [cm/s]
    """
    f_array = []
    g_array = []

    for particle_i in particles:
        a_i  = particle_i.radius
        St_i = particle_i.St

        # Geometric size factor
        f_i = a_i ** 2

        # Dynamical state factor
        g_i = _characteristic_velocity(St_i, disk_state, params)

        f_array.append(f_i)
        g_array.append(g_i)

    return np.array(f_array, dtype=float), np.array(g_array, dtype=float)


# ---------------------------------------------------------------------------
# Quick self-test  (python kernel.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import types

    print("=== kernel.py self-test ===\n")

    # Mock particle and disk
    def make_particle(radius, mass, St):
        p = types.SimpleNamespace(radius=radius, mass=mass, St=St)
        return p

    p1 = make_particle(radius=1e-4, mass=1e-12, St=0.01)
    p2 = make_particle(radius=2e-4, mass=8e-12, St=0.10)

    mock_disk = {
        "T_K":     100.0,
        "cs_cm_s": 1e5,
        "eta":     3e-3,
        "vK_cm_s": 1e7,
    }

    mock_params = types.SimpleNamespace(alpha=1e-3)

    # 1. Cross-section
    sigma = collision_cross_section(p1, p2)
    print(f"σ_ij         = {sigma:.4e} cm²")
    assert sigma > 0

    # 2. Brownian velocity
    dv_bm = brownian_velocity(p1.mass, p2.mass, mock_disk["T_K"])
    print(f"Δv_BM        = {dv_bm:.4e} cm/s")
    assert dv_bm > 0

    # 3. Turbulent velocity — all three regimes
    dv_r1 = turbulent_velocity(0.01, 0.1, 1e5, 1e-3)
    dv_r2 = turbulent_velocity(0.1,  2.0, 1e5, 1e-3)
    dv_r3 = turbulent_velocity(2.0,  5.0, 1e5, 1e-3)
    print(f"Δv_turb R1   = {dv_r1:.4e} cm/s  (both St < 1)")
    print(f"Δv_turb R2   = {dv_r2:.4e} cm/s  (mixed)")
    print(f"Δv_turb R3   = {dv_r3:.4e} cm/s  (both St > 1)")
    assert turbulent_velocity(0.1, 0.1, 1e5, 1e-3) == 0.0, "Equal St → Δv_turb = 0"

    # 4. Single drift
    vd = radial_drift_velocity(p1.St, mock_disk)
    print(f"v_drift      = {vd:.4e} cm/s")

    # 5. Relative velocity
    vrel = relative_velocity(p1, p2, mock_disk, mock_params)
    print(f"Δv_rel       = {vrel:.4e} cm/s")
    assert vrel > 0

    # 6. Kernel factors
    f_arr, g_arr = compute_kernel_factors([p1, p2], mock_disk, mock_params)
    print(f"f_array      = {f_arr}  (expect [a1², a2²] = [{p1.radius**2:.2e}, {p2.radius**2:.2e}])")
    print(f"g_array      = {g_arr}")
    assert np.allclose(f_arr, [p1.radius**2, p2.radius**2])
    assert np.all(g_arr >= 0)

    print("\nAll kernel self-tests passed.")
