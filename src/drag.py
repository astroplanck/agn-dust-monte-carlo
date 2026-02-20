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