import math
from .params import DiskParams, PC_TO_CM, G_CGS, KB_CGS, MH_CGS, PI
from .disk import disk_state
from .drag_updated import stopping_time_epstein, stokes_number, radial_drift_velocity, delta_v_drift


def _eta_and_vK(R_cm: float, params: DiskParams, dlogR: float = 1e-3):
    """
    Compute eta (dimensionless pressure gradient) and vK (Keplerian velocity)
    locally for use in the sanity check, without modifying disk.py.

    eta = -(1/2) * (cs/vK)^2 * d(log P)/d(log R)
    """
    def log_pressure(R: float) -> float:
        R_pc = R / PC_TO_CM
        T    = params.T0_K * (R_pc ** (-params.q))
        cs   = math.sqrt(KB_CGS * T / (params.mu * MH_CGS))
        OmK  = math.sqrt(G_CGS * params.MBH_g / R**3)
        Hg   = cs / OmK
        Sig  = params.Sigma0_g_cm2 * (R_pc ** (-params.p))
        rho  = Sig / (math.sqrt(2.0 * PI) * Hg)
        return math.log(rho * cs**2)

    R_plus  = R_cm * (1.0 + dlogR)
    R_minus = R_cm * (1.0 - dlogR)
    dlogP_dlogR = (log_pressure(R_plus) - log_pressure(R_minus)) / (2.0 * dlogR)

    T_K    = params.T0_K * ((R_cm / PC_TO_CM) ** (-params.q))
    cs     = math.sqrt(KB_CGS * T_K / (params.mu * MH_CGS))
    OmegaK = math.sqrt(G_CGS * params.MBH_g / R_cm**3)
    vK     = OmegaK * R_cm
    eta    = -0.5 * (cs / vK)**2 * dlogP_dlogR
    return eta, vK


def run():

    params = DiskParams()

    radii_pc = [0.1, 1.0, 10.0]
    sizes_cm = [1e-4, 1e-1]  # ~1 micron, ~1 mm

    print("Disk + Dust Coupling Sanity Check")
    print()

    for R_pc in radii_pc:
        R_cm = R_pc * PC_TO_CM
        st   = disk_state(R_cm, params)
        eta, vK = _eta_and_vK(R_cm, params)

        print(f" [R = {R_pc} pc] ")
        print(f"  eta  = {eta:.4e}  (pressure gradient parameter)")
        print(f"  vK   = {vK:.3e} cm/s")
        print()

        print(f"  {'Grain size':<14} {'ts (s)':<14} {'St':<14} {'v_drift (cm/s)':<18}")
        print(f"  {'-'*60}")

        stokes_by_size = {}
        for a_cm in sizes_cm:
            ts = stopping_time_epstein(
                a_cm=a_cm,
                rho_s_g_cm3=params.rho_s_g_cm3,
                rho_g_g_cm3=st["rho_g_g_cm3"],
                cs_cm_s=st["cs_cm_s"],
            )
            St = stokes_number(st["OmegaK_s^-1"], ts)
            vd = radial_drift_velocity(St, eta, vK)
            stokes_by_size[a_cm] = St
            print(f"  a={a_cm:.1e} cm     {ts:<14.3e} {St:<14.3e} {vd:<18.3e}")

        St_small = stokes_by_size[sizes_cm[0]]
        St_large = stokes_by_size[sizes_cm[1]]
        dv = delta_v_drift(St_small, St_large, eta, vK)
        print()
        print(f"  delta_v_drift (1um vs 1mm) = {dv:.3e} cm/s")

        for a_cm, St in stokes_by_size.items():
            if St < 0.1:
                regime = "St << 1 : slow drift (gas-coupled)"
            elif St < 10.0:
                regime = "St ~ 1  : maximum drift (danger zone)"
            else:
                regime = "St >> 1 : slow drift (decoupled)"
            print(f"  a={a_cm:.1e} cm -> {regime}")

        print()


if __name__ == "__main__":
    run()
