# a_cm is the dust grain radius (cm)
# rho_s_g_cm3 is the density of the dust grain (g/cm^3)
# rho_g_g_cm3 is the gas mass density at radius R (g/cm^3)
# cs_cm_s is the gas sound speed at radius R (cm/s)
# OmegaK_s_inv is the Keplerian angular frequency at radius R (s^-1)
# ts_s is the stopping time (s)

def stopping_time_epstein(a_cm: float, rho_s_g_cm3: float, rho_g_g_cm3: float, cs_cm_s: float) -> float:
    if a_cm < 0.0:
        raise ValueError("a must be nonnegative.")
    if rho_s_g_cm3 <= 0.0:
        raise ValueError("rho_s must be positive.")
    if rho_g_g_cm3 <= 0.0 or cs_cm_s <= 0.0:
        raise ValueError("rho_g and cs must be positive.")
    return (rho_s_g_cm3 * a_cm) / (rho_g_g_cm3 * cs_cm_s)

def stokes_number(OmegaK_s_inv: float, ts_s: float) -> float:
    if OmegaK_s_inv <= 0.0 or ts_s < 0.0:
        raise ValueError("OmegaK must be positive and ts must be nonnegative.")
    return OmegaK_s_inv * ts_s


# ---------- Radial drift velocity ----------

def _drift_factor(St: float) -> float:
    """
    The dimensionless Stokes-number factor in the drift formula:
        f(St) = St / (1 + St^2)
    This peaks at St = 1 (maximum drift) and falls off on both sides.
    """
    return St / (1.0 + St ** 2)


def radial_drift_velocity(St: float, eta: float, vK_cm_s: float) -> float:
    """
    Radial drift velocity of a single grain (cm/s).

    v_drift = -eta * vK * St / (1 + St^2)

    The negative sign indicates inward (toward the SMBH) motion.

    Parameters
    ----------
    St        : Stokes number of the grain (dimensionless)
    eta       : dimensionless pressure gradient parameter from disk_state()
    vK_cm_s   : Keplerian velocity at the grain's radius (cm/s)

    Returns
    -------
    v_drift in cm/s  (negative = inward)
    """
    if eta < 0.0:
        raise ValueError("eta must be nonnegative.")
    if vK_cm_s <= 0.0:
        raise ValueError("vK must be positive.")
    return -eta * vK_cm_s * _drift_factor(St)


def delta_v_drift(St1: float, St2: float, eta: float, vK_cm_s: float) -> float:
    """
    Magnitude of the differential drift velocity between two particles (cm/s).

    |Delta v_drift| = |v_drift(St1) - v_drift(St2)|
                    = eta * vK * |f(St1) - f(St2)|

    This is the drift contribution to the relative collision velocity that
    feeds into the quadrature sum:
        Delta v^2 = Delta v_BM^2 + Delta v_turb^2 + Delta v_drift^2

    Parameters
    ----------
    St1, St2  : Stokes numbers of the two colliding particles
    eta       : dimensionless pressure gradient parameter from disk_state()
    vK_cm_s   : Keplerian velocity at this disk radius (cm/s)

    Returns
    -------
    |Delta v_drift| in cm/s  (always nonnegative)
    """
    if eta < 0.0:
        raise ValueError("eta must be nonnegative.")
    if vK_cm_s <= 0.0:
        raise ValueError("vK must be positive.")
    return eta * vK_cm_s * abs(_drift_factor(St1) - _drift_factor(St2))