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
import math
import numpy as np

# -----------------------------------------
# Call turbulent relative velocity function
# -----------------------------------------
from .turbulent_relative_velocity import turbulent_relative_velocity_cm_s

# -----------------------------------------
# Call mass weight function
# -----------------------------------------
from .mass_selection_weight import compute_mass_weights

# -----------------------------------------
# These two modules/functions do not exist yet.
# Keep these imports commented out for now.
# -----------------------------------------
# from .brownian_relative_velocity import brownian_relative_velocity_cm_s
# from .radial_drift_relative_velocity import radial_drift_relative_velocity_cm_s


# ============================================================
# Index convention for one superparticle in sp_list:
# sp = [r_sp, rho_sp, m_dust, N_sp, m_sp, ts, St]
#        0      1       2      3     4    5   6
# ============================================================

RADIUS_IDX = 0
M_DUST_IDX = 2
N_REAL_IDX = 3
STOKES_IDX = 6


def collision_cross_section(sp_i, sp_j):
    """
    Compute collision cross section:
        sigma_ij = pi * (a_i + a_j)^2

    Parameters
    ----------
    sp_i, sp_j : list-like
        Superparticles from sp_list.
        We call their radius using index 0.

    Returns
    -------
    sigma_ij : float
        Collision cross section.
    """

    # -----------------------------------------
    # Call particle radius from superparticle i
    # -----------------------------------------
    a_i = float(sp_i[RADIUS_IDX])

    # -----------------------------------------
    # Call particle radius from superparticle j
    # -----------------------------------------
    a_j = float(sp_j[RADIUS_IDX])

    sigma_ij = math.pi * (a_i + a_j) ** 2
    return sigma_ij


def relative_velocity(
    sp_i,
    sp_j,
    cs_cm_s,
    OmegaK,
    params,
    brownian_velocity_func=None,
    radial_drift_velocity_func=None,
):
    """
    Compute total relative velocity:
        Delta v_ij = sqrt(v_B^2 + v_R^2 + v_T^2)

    Here:
    - v_B : Brownian relative velocity contribution
    - v_R : radial drift relative velocity contribution
    - v_T : turbulent relative velocity contribution

    Parameters
    ----------
    sp_i, sp_j : list-like
        Superparticles from sp_list.
    cs_cm_s : float
        Sound speed.
    OmegaK : float
        Keplerian angular frequency.
    params : DiskParams
        Parameter object used by turbulence model.
    brownian_velocity_func : callable or None
        Future Brownian velocity function.
        Expected interface: f(sp_i, sp_j, ...) -> float
    radial_drift_velocity_func : callable or None
        Future radial drift velocity function.
        Expected interface: f(sp_i, sp_j, ...) -> float

    Returns
    -------
    delta_v_ij : float
        Total relative velocity.
    """

    # -----------------------------------------
    # Call Stokes number from superparticle i
    # for turbulent velocity
    # -----------------------------------------
    St_i = float(sp_i[STOKES_IDX])

    # -----------------------------------------
    # Call Stokes number from superparticle j
    # for turbulent velocity
    # -----------------------------------------
    St_j = float(sp_j[STOKES_IDX])

    # -----------------------------------------
    # Call turbulent relative velocity function
    # from turbulent_relative_velocity.py
    # -----------------------------------------
    v_turb = float(
        turbulent_relative_velocity_cm_s(
            St_i,
            St_j,
            cs_cm_s,
            OmegaK,
            params,
        )
    )

    # -----------------------------------------
    # Call Brownian relative velocity function
    # -----------------------------------------
    if brownian_velocity_func is None:
        v_brown = 0.0
    else:
        v_brown = float(brownian_velocity_func(sp_i, sp_j))

    # -----------------------------------------
    # Call radial drift relative velocity function
    # -----------------------------------------
    if radial_drift_velocity_func is None:
        v_radial = 0.0
    else:
        v_radial = float(radial_drift_velocity_func(sp_i, sp_j))

    # -----------------------------------------
    # Combine three relative velocity contributions
    # -----------------------------------------
    delta_v_ij = math.sqrt(v_brown**2 + v_radial**2 + v_turb**2)

    return delta_v_ij


def compute_all_mass_weights(sp_list):
    """
    Compute normalized mass weights for all superparticles.

    Weight logic from mass_selection_weight.py:
        M_i = N_i * m_mean_i
        raw_w_i = M_i^(4/3)
        W_i = raw_w_i / sum(raw_w)

    In your current superparticle structure:
    - N_i     = sp[3]
    - m_mean_i = m_dust = sp[2]

    Parameters
    ----------
    sp_list : list
        List of superparticles.

    Returns
    -------
    weights : np.ndarray
        Normalized mass weights for all superparticles.
    """

    # -----------------------------------------
    # Call N_i from every superparticle
    # -----------------------------------------
    n_real = np.array([sp[N_REAL_IDX] for sp in sp_list], dtype=float)

    # -----------------------------------------
    # Call m_mean_i from every superparticle
    # -----------------------------------------
    m_mean = np.array([sp[M_DUST_IDX] for sp in sp_list], dtype=float)

    # -----------------------------------------
    # Call compute_mass_weights() from
    # mass_selection_weight.py
    # -----------------------------------------
    weights = compute_mass_weights(n_real, m_mean)

    return weights


def compute_kernel_factors(
    sp_i,
    sp_j,
    idx_i,
    idx_j,
    sp_list,
    cs_cm_s,
    OmegaK,
    params,
    brownian_velocity_func=None,
    radial_drift_velocity_func=None,
):
    """
    Compute full collision kernel with mass weights:
        K_ij = sigma_ij * Delta v_ij * W_i * W_j

    Parameters
    ----------
    sp_i, sp_j : list-like
        Superparticles i and j.
    idx_i, idx_j : int
        Indices of i and j in sp_list.
    sp_list : list
        Full superparticle list, needed to compute all mass weights.
    cs_cm_s : float
        Sound speed.
    OmegaK : float
        Keplerian frequency.
    params : DiskParams
        Disk parameter object.
    brownian_velocity_func : callable or None
        Future Brownian velocity function.
    radial_drift_velocity_func : callable or None
        Future radial drift velocity function.

    Returns
    -------
    result : dict
        Dictionary containing sigma_ij, delta_v_ij, W_i, W_j, K_ij
    """

    # -----------------------------------------
    # Call collision_cross_section()
    # to compute sigma_ij
    # -----------------------------------------
    sigma_ij = collision_cross_section(sp_i, sp_j)

    # -----------------------------------------
    # Call relative_velocity()
    # to compute Delta v_ij
    # -----------------------------------------
    delta_v_ij = relative_velocity(
        sp_i,
        sp_j,
        cs_cm_s,
        OmegaK,
        params,
        brownian_velocity_func=brownian_velocity_func,
        radial_drift_velocity_func=radial_drift_velocity_func,
    )

    # -----------------------------------------
    # Call compute_all_mass_weights()
    # to get all normalized mass weights
    # -----------------------------------------
    all_weights = compute_all_mass_weights(sp_list)

    # -----------------------------------------
    # Call mass weight W_i using index idx_i
    # -----------------------------------------
    W_i = float(all_weights[idx_i])

    # -----------------------------------------
    # Call mass weight W_j using index idx_j
    # -----------------------------------------
    W_j = float(all_weights[idx_j])

    # -----------------------------------------
    # Compute final kernel:
    # K_ij = sigma_ij * Delta v_ij * W_i * W_j
    # -----------------------------------------
    K_ij = sigma_ij * delta_v_ij * W_i * W_j

    return {
        "sigma_ij": sigma_ij,
        "delta_v_ij": delta_v_ij,
        "W_i": W_i,
        "W_j": W_j,
        "K_ij": K_ij,
    }
