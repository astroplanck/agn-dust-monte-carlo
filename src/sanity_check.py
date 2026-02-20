from .params import DiskParams, PC_TO_CM
from .disk import disk_state
from .drag import stopping_time_epstein, stokes_number

def run():

    params = DiskParams()

    radii_pc = [0.1, 1.0, 10.0]
    sizes_cm = [1e-4, 1e-1]  # ~1 micron, ~1 mm

    print("=== Disk + Dust Coupling Sanity Check ===")
    print(f"Using params: {params}")
    print()

    for R_pc in radii_pc:
        R_cm = R_pc * PC_TO_CM
        st = disk_state(R_cm, params)

        print(f"--- R = {R_pc} pc ---")
        print(f"T = {st['T_K']:.3e} K")
        print(f"cs = {st['cs_cm_s']:.3e} cm/s")
        print(f"OmegaK = {st['OmegaK_s^-1']:.3e} 1/s")
        print(f"Hg = {st['Hg_cm']:.3e} cm")
        print(f"Sigma = {st['Sigma_g_cm2']:.3e} g/cm^2")
        print(f"rho_g(mid) = {st['rho_g_g_cm3']:.3e} g/cm^3")

        for a_cm in sizes_cm:
            ts = stopping_time_epstein(
                a_cm = a_cm,
                rho_s_g_cm3 = params.rho_s_g_cm3,
                rho_g_g_cm3 = st["rho_g_g_cm3"],
                cs_cm_s = st["cs_cm_s"],
            )
            St = stokes_number(st["OmegaK_s^-1"], ts)
            print(f"  a={a_cm:.1e} cm -> ts={ts:.3e} s, St={St:.3e}")
        print()

if __name__ == "__main__":
    run()