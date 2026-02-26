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
# src/disk.py
import math
from typing import Dict

from .params import DiskParams, G_CGS, KB_CGS, MH_CGS, PI, PC_TO_CM


def T_of_R(R_cm: float, params: DiskParams) -> float:
    """Temperature profile in Kelvin."""
    R_pc = R_cm / PC_TO_CM
    return params.T0_K * (R_pc ** (-params.q))


def cs_of_T(T_K: float, params: DiskParams) -> float:
    """Isothermal sound speed in cm/s."""
    return math.sqrt(KB_CGS * T_K / (params.mu * MH_CGS))


def OmegaK_of_R(R_cm: float, params: DiskParams) -> float:
    """Keplerian angular frequency in 1/s."""
    return math.sqrt(G_CGS * params.MBH_g / (R_cm ** 3))


def Hg_of_R(R_cm: float, params: DiskParams) -> float:
    """Gas scale height in cm."""
    T_K = T_of_R(R_cm, params)
    cs = cs_of_T(T_K, params)
    OmegaK = OmegaK_of_R(R_cm, params)
    return cs / OmegaK


def Sigma_of_R(R_cm: float, params: DiskParams) -> float:
    """Gas surface density in g/cm^2."""
    R_pc = R_cm / PC_TO_CM
    return params.Sigma0_g_cm2 * (R_pc ** (-params.p))


def rho_mid_of_R(R_cm: float, params: DiskParams) -> float:
    """Midplane gas density in g/cm^3, assuming Gaussian vertical profile."""
    Hg = Hg_of_R(R_cm, params)
    Sigma = Sigma_of_R(R_cm, params)
    return Sigma / (math.sqrt(2.0 * PI) * Hg)


def disk_state(R_cm: float, params: DiskParams) -> Dict[str, float]:
    """
    Convenience bundle of disk quantities at radius R.

    Returns dict with:
      T_K, cs_cm_s, OmegaK_s^-1, Hg_cm, Sigma_g_cm2, rho_g_g_cm3
    """
    T_K = T_of_R(R_cm, params)
    cs = cs_of_T(T_K, params)
    OmegaK = OmegaK_of_R(R_cm, params)
    Hg = cs / OmegaK
    Sigma = Sigma_of_R(R_cm, params)
    rho_g = Sigma / (math.sqrt(2.0 * PI) * Hg)

    return {
        "T_K": T_K,
        "cs_cm_s": cs,
        "OmegaK_s^-1": OmegaK,
        "Hg_cm": Hg,
        "Sigma_g_cm2": Sigma,
        "rho_g_g_cm3": rho_g,
    }
