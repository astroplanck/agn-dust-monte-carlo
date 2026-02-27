# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Julia 1.12
#     language: julia
#     name: julia-1.12
# ---

# %%
# src/params.py
"""
Parameter defaults for Disk Physics + Stokes Number stage.
We use CGS units internally:
- length: cm
- mass: g
- time: s
- temperature: K
- surface density: g/cm^2
- volume density: g/cm^3
"""

from dataclasses import dataclass
import math

# ---------- Physical constants (CGS) ----------
G_CGS = 6.67430e-8         # cm^3 g^-1 s^-2
KB_CGS = 1.380649e-16      # erg/K = g cm^2 s^-2 K^-1
MH_CGS = 1.6735575e-24     # g
PI = math.pi

PC_TO_CM = 3.085677581e18  # cm
MSUN_TO_G = 1.98847e33     # g


@dataclass(frozen=True)
class DiskParams:
    """
    Minimal AGN disk background parameterization.

    T(R) = T0 * (R/1pc)^(-q)
    Sigma(R) = Sigma0 * (R/1pc)^(-p)

    Notes:
      - MBH_g should be in grams.
      - Sigma0_g_cm2 is a surface density normalization at 1 pc.
      - mu is mean molecular weight in units of m_H.
    """
    MBH_g: float = 1.0e7 * MSUN_TO_G

    T0_K: float = 100.0
    q: float = 0.5

    Sigma0_g_cm2: float = 1.0e3
    p: float = 1.0

    mu: float = 2.34

    # Dust internal density (compact grain material density)
    rho_s_g_cm3: float = 1.0

    # Turbulence
    reynold: float = 10 ** 10 # Reynold's number
    alpha: float = 10 ** -3 # Shakura-Sunyaev turbulent velocity parameter
    y_a: float = 1.6
