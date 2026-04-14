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
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================
# Physical constants (CGS)
# ============================================================
G_CGS = 6.67430e-8         # cm^3 g^-1 s^-2
KB_CGS = 1.380649e-16      # erg/K = g cm^2 s^-2 K^-1
MH_CGS = 1.6735575e-24     # g
PI = math.pi

PC_TO_CM = 3.085677581e18  # cm
MSUN_TO_G = 1.98847e33     # g
YR_TO_S = 3.15576e7        # s


# ============================================================
# Disk parameters
# ============================================================
@dataclass(frozen=True)
class DiskParams:
    """
    Minimal AGN disk background parameterization.

    T(R) = T0 * (R / 1 pc)^(-q)
    Sigma(R) = Sigma0 * (R / 1 pc)^(-p)
    """
    MBH_g: float = 1.0e7 * MSUN_TO_G

    T0_K: float = 100.0
    q: float = 0.5

    Sigma0_g_cm2: float = 1.0e3
    p: float = 1.0

    mu: float = 2.34
    rho_s_g_cm3: float = 1.0

    reynold: float = 1.0e10
    alpha: float = 1.0e-3
    y_a: float = 1.6

    Lx: float = 1.0
    f_dg_avg: float = 0.5


# ============================================================
# Simulation parameters
# ============================================================
@dataclass(frozen=True)
class SimulationParams:
    """
    All important tunable knobs are collected here.
    Change only these first.
    """

    # --------------------------------------------------------
    # Core physical / numerical knobs
    # --------------------------------------------------------
    R0_pc: float = 1.0
    box_length_pc: float = 1.0e-6

    n_superparticles: int = 100
    initial_real_particles_per_swarm: float = 1.0e15

    initial_radius_cm: float = 1.0e-3
    initial_radius_logscatter_dex: float = 0.3

    t_end_yr: float = 1.0e4
    snapshot_every_yr: float = 1.0e3

    v_stick_cm_s: float = 1.0e2
    v_frag_cm_s: float = 1.0e4

    compaction_factor: float = 1.2
    fragment_factor: float = 0.5

    output_dir: str = "snapshots"

    # --------------------------------------------------------
    # Derived quantities (auto-updated)
    # --------------------------------------------------------
    @property
    def R0_cm(self):
        return self.R0_pc * PC_TO_CM

    @property
    def t_end_s(self):
        return self.t_end_yr * YR_TO_S

    @property
    def snapshot_times_s(self):
        if self.snapshot_every_yr <= 0:
            raise ValueError("snapshot_every_yr must be positive.")
        n_snaps = max(1, int(self.t_end_yr / self.snapshot_every_yr))
        return tuple(i * self.snapshot_every_yr * YR_TO_S for i in range(1, n_snaps + 1))

    @property
    def box_length_cm(self):
        return self.box_length_pc * PC_TO_CM

    @property
    def box_volume_cm3(self):
        return self.box_length_cm ** 3


# ============================================================
# Background disk quantities
# ============================================================
def temperature(R_cm, params):
    return params.T0_K * (R_cm / PC_TO_CM) ** (-params.q)


def sound_speed(R_cm, params):
    T_R = temperature(R_cm, params)
    return np.sqrt((KB_CGS * T_R) / (params.mu * MH_CGS))


def omega_k(R_cm, params):
    return np.sqrt((G_CGS * params.MBH_g) / (R_cm ** 3))


def scale_height(R_cm, params):
    return sound_speed(R_cm, params) / omega_k(R_cm, params)


def surface_density(R_cm, params):
    return params.Sigma0_g_cm2 * (R_cm / PC_TO_CM) ** (-params.p)


def midplane_gas_density(R_cm, params):
    return surface_density(R_cm, params) / (np.sqrt(2.0 * PI) * scale_height(R_cm, params))


# ============================================================
# Particle property helpers
# ============================================================
def radius_from_mass(m, rho_int):
    if m <= 0:
        raise ValueError(f"Particle mass must be positive, got {m}")
    if rho_int <= 0:
        raise ValueError(f"Internal density must be positive, got {rho_int}")
    return ((3.0 * m) / (4.0 * PI * rho_int)) ** (1.0 / 3.0)


def stopping_time(a, rho_int, R_cm, params):
    rho_g = midplane_gas_density(R_cm, params)
    c_s = sound_speed(R_cm, params)
    return (rho_int * a) / (rho_g * c_s)


def stokes_number(t_stop, R_cm, params):
    return t_stop * omega_k(R_cm, params)


def compute_stokes(a, rho_int, R_cm, params):
    t_s = stopping_time(a, rho_int, R_cm, params)
    return stokes_number(t_s, R_cm, params)


def stopping_time_from_stokes(St, R_cm, params):
    return St / omega_k(R_cm, params)


# ============================================================
# Relative velocities and collision kernel
# ============================================================
def brownian_velocity(m1, m2, R_cm, params):
    T = temperature(R_cm, params)
    return np.sqrt((8.0 * KB_CGS * T / PI) * (1.0 / m1 + 1.0 / m2))


def turbulent_velocity(St1, St2, R_cm, params):
    c_s = sound_speed(R_cm, params)
    alpha = params.alpha
    return np.sqrt(alpha) * c_s * np.sqrt(abs(St1 - St2))


def v_kepler(R_cm, params):
    return np.sqrt(G_CGS * params.MBH_g / R_cm)


def eta_parameter(R_cm, params):
    c_s = sound_speed(R_cm, params)
    v_k = v_kepler(R_cm, params)
    return (c_s ** 2) / (v_k ** 2)


def radial_velocity(St, R_cm, params):
    v_k = v_kepler(R_cm, params)
    eta = eta_parameter(R_cm, params)
    return -2.0 * eta * v_k * St / (1.0 + St ** 2)


def radial_relative_velocity(St1, St2, R_cm, params):
    v1 = radial_velocity(St1, R_cm, params)
    v2 = radial_velocity(St2, R_cm, params)
    return abs(v1 - v2)


def relative_velocity(m1, m2, St1, St2, R_cm, params):
    v_brown = brownian_velocity(m1, m2, R_cm, params)
    v_turb = turbulent_velocity(St1, St2, R_cm, params)
    v_rad = radial_relative_velocity(St1, St2, R_cm, params)
    return np.sqrt(v_brown ** 2 + v_turb ** 2 + v_rad ** 2)


def collision_cross_section(a1, a2):
    return PI * (a1 + a2) ** 2


def collision_kernel(a1, a2, v_rel):
    return collision_cross_section(a1, a2) * v_rel


# ============================================================
# Super-particle initialization and updates
# ============================================================
def initialize_superparticles(
    n_superparticles=50,
    R_cm=None,
    params=None,
    r_sp_cm=1.0e-3,
    rho_sp_g_cm3=1.0,
    N_real=1.0e3,
    radius_logscatter_dex=0.0,
    rng=None
):
    """
    Each super-particle carries a fixed swarm mass m_sp.
    Its effective real-particle count updates as:
        N_sp = m_sp / m_dust
    """
    if R_cm is None:
        raise ValueError("R_cm must be provided.")
    if params is None:
        params = DiskParams()
    if rng is None:
        rng = np.random.default_rng()

    rho_g_R = midplane_gas_density(R_cm, params)
    c_s_R = sound_speed(R_cm, params)
    omega_k_R = omega_k(R_cm, params)

    sp_list = []

    for sp_id in range(1, n_superparticles + 1):
        if radius_logscatter_dex > 0.0:
            ln_sigma = radius_logscatter_dex * np.log(10.0)
            r_i = r_sp_cm * np.exp(rng.normal(0.0, ln_sigma))
        else:
            r_i = r_sp_cm

        m_dust = (4.0 / 3.0) * PI * (r_i ** 3) * rho_sp_g_cm3
        m_sp = N_real * m_dust

        ts = (rho_sp_g_cm3 * r_i) / (rho_g_R * c_s_R)
        St = ts * omega_k_R

        sp_list.append({
            "id": sp_id,
            "r_sp": r_i,
            "rho_sp": rho_sp_g_cm3,
            "m_dust": m_dust,
            "N_sp": float(N_real),
            "m_sp": m_sp,
            "ts": ts,
            "St": St
        })

    return sp_list


def update_derived_properties(sp, R_cm, params):
    for key in ("m_dust", "rho_sp", "m_sp"):
        if key not in sp:
            raise KeyError(f"sp must contain '{key}'")

    if sp["m_dust"] <= 0:
        raise ValueError("m_dust must stay positive.")
    if sp["rho_sp"] <= 0:
        raise ValueError("rho_sp must stay positive.")
    if sp["m_sp"] <= 0:
        raise ValueError("m_sp must stay positive.")

    sp["r_sp"] = radius_from_mass(sp["m_dust"], sp["rho_sp"])
    sp["ts"] = stopping_time(sp["r_sp"], sp["rho_sp"], R_cm, params)
    sp["St"] = stokes_number(sp["ts"], R_cm, params)
    sp["N_sp"] = sp["m_sp"] / sp["m_dust"]

    return sp


# ============================================================
# Event rates
# ============================================================
def collision_rate(sp_i, sp_j, R_cm, disk_params, sim_params):
    """
    Prototype event rate:
        R_ij = (N_i * N_j / V) * K_ij
    """
    m1 = sp_i["m_dust"]
    m2 = sp_j["m_dust"]
    St1 = sp_i["St"]
    St2 = sp_j["St"]
    a1 = sp_i["r_sp"]
    a2 = sp_j["r_sp"]
    N1 = sp_i["N_sp"]
    N2 = sp_j["N_sp"]

    v_rel = relative_velocity(m1, m2, St1, St2, R_cm, disk_params)
    K_ij = collision_kernel(a1, a2, v_rel)

    return (N1 * N2 / sim_params.box_volume_cm3) * K_ij


def build_rate_matrix(sp_list, R_cm, disk_params, sim_params):
    n = len(sp_list)
    rate_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            Rij = collision_rate(sp_list[i], sp_list[j], R_cm, disk_params, sim_params)
            rate_matrix[i, j] = Rij
            rate_matrix[j, i] = Rij

    return rate_matrix


def total_collision_rate(rate_matrix):
    return np.sum(np.triu(rate_matrix, k=1))


def sample_event_time(R_total, rng=None):
    if R_total <= 0:
        raise ValueError(f"R_total must be positive, got {R_total}")

    if rng is None:
        rng = np.random.default_rng()

    u = rng.random()
    while u == 0.0:
        u = rng.random()

    return -np.log(u) / R_total


def sample_collision_pair(rate_matrix, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if rate_matrix.shape[0] != rate_matrix.shape[1]:
        raise ValueError("rate_matrix must be square.")

    n = rate_matrix.shape[0]
    pairs = []
    rates = []

    for i in range(n):
        for j in range(i + 1, n):
            Rij = rate_matrix[i, j]
            if Rij > 0:
                pairs.append((i, j))
                rates.append(Rij)

    if not rates:
        raise ValueError("No positive collision rates found.")

    rates = np.asarray(rates, dtype=float)
    probs = rates / rates.sum()
    cum_probs = np.cumsum(probs)

    u = rng.random()
    idx = np.searchsorted(cum_probs, u, side="right")
    return pairs[idx]


# ============================================================
# Collision outcomes
# ============================================================
def classify_collision(v_rel, sim_params):
    if v_rel < 0:
        raise ValueError(f"Relative velocity must be non-negative, got {v_rel}")

    if v_rel < sim_params.v_stick_cm_s:
        return "sticking"
    elif v_rel < sim_params.v_frag_cm_s:
        return "compaction"
    else:
        return "fragmentation"


def _pick_target_and_partner(sp_i, sp_j, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < 0.5:
        return sp_i, sp_j
    return sp_j, sp_i


def apply_sticking(sp_i, sp_j, R_cm, disk_params, rng=None):
    """
    Representative-particle sticking update:
    choose one target representative and let it absorb one particle
    of the partner's type.
    """
    target, partner = _pick_target_and_partner(sp_i, sp_j, rng=rng)

    old_m_t = target["m_dust"]
    old_m_p = partner["m_dust"]
    old_rho_t = target["rho_sp"]
    old_rho_p = partner["rho_sp"]

    new_m = old_m_t + old_m_p
    new_rho = (old_m_t * old_rho_t + old_m_p * old_rho_p) / new_m

    target["m_dust"] = new_m
    target["rho_sp"] = new_rho
    update_derived_properties(target, R_cm, disk_params)

    return target, partner


def apply_compaction(sp_i, sp_j, R_cm, disk_params, compaction_factor=1.2):
    if compaction_factor <= 1.0:
        raise ValueError("compaction_factor should be > 1.")

    sp_i["rho_sp"] *= compaction_factor
    sp_j["rho_sp"] *= compaction_factor

    update_derived_properties(sp_i, R_cm, disk_params)
    update_derived_properties(sp_j, R_cm, disk_params)

    return sp_i, sp_j


def apply_fragmentation(sp_i, sp_j, R_cm, disk_params, fragment_factor=0.5, rng=None):
    """
    Representative-particle fragmentation update:
    choose one target representative and reduce its m_dust.
    """
    if not (0.0 < fragment_factor < 1.0):
        raise ValueError("fragment_factor must be between 0 and 1.")

    target, partner = _pick_target_and_partner(sp_i, sp_j, rng=rng)

    target["m_dust"] *= fragment_factor
    update_derived_properties(target, R_cm, disk_params)

    return target, partner


def execute_collision(sp_i, sp_j, v_rel, R_cm, disk_params, sim_params, rng=None):
    outcome = classify_collision(v_rel, sim_params)

    if outcome == "sticking":
        apply_sticking(sp_i, sp_j, R_cm, disk_params, rng=rng)
    elif outcome == "compaction":
        apply_compaction(
            sp_i, sp_j, R_cm, disk_params,
            compaction_factor=sim_params.compaction_factor
        )
    elif outcome == "fragmentation":
        apply_fragmentation(
            sp_i, sp_j, R_cm, disk_params,
            fragment_factor=sim_params.fragment_factor,
            rng=rng
        )
    else:
        raise ValueError(f"Unknown collision outcome: {outcome}")

    return outcome


# ============================================================
# Snapshot utilities
# ============================================================
def snapshot_from_superparticles(sp_list, t, collision_log=None):
    snapshot = {
        "time": t,
        "masses": np.array([sp["m_dust"] for sp in sp_list], dtype=float),
        "radii": np.array([sp["r_sp"] for sp in sp_list], dtype=float),
        "densities": np.array([sp["rho_sp"] for sp in sp_list], dtype=float),
        "numbers": np.array([sp["N_sp"] for sp in sp_list], dtype=float),
        "swarm_masses": np.array([sp["m_sp"] for sp in sp_list], dtype=float),
        "stokes": np.array([sp["St"] for sp in sp_list], dtype=float),
        "stopping_times": np.array([sp["ts"] for sp in sp_list], dtype=float),
        "collision_log": collision_log if collision_log is not None else []
    }
    return snapshot


def save_snapshot(snapshot, output_path):
    np.savez(
        output_path,
        time=snapshot["time"],
        masses=snapshot["masses"],
        radii=snapshot["radii"],
        densities=snapshot["densities"],
        numbers=snapshot["numbers"],
        swarm_masses=snapshot["swarm_masses"],
        stokes=snapshot["stokes"],
        stopping_times=snapshot["stopping_times"],
        collision_log=np.array(snapshot["collision_log"], dtype=object)
    )


def read_snapshot(path):
    with np.load(path, allow_pickle=True) as data:
        return {
            "time": float(data["time"]),
            "masses": np.array(data["masses"], dtype=float),
            "radii": np.array(data["radii"], dtype=float),
            "densities": np.array(data["densities"], dtype=float),
            "numbers": np.array(data["numbers"], dtype=float),
            "swarm_masses": np.array(data["swarm_masses"], dtype=float),
            "stokes": np.array(data["stokes"], dtype=float),
            "stopping_times": np.array(data["stopping_times"], dtype=float),
            "collision_log": list(data["collision_log"])
        }


# ============================================================
# Main Monte Carlo driver
# ============================================================
def run_monte_carlo(disk_params, sim_params, output_dir=None, rng=None):
    """
    Run the AGN dust-growth Monte Carlo simulation.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if output_dir is None:
        output_dir = sim_params.output_dir

    os.makedirs(output_dir, exist_ok=True)

    R_cm = sim_params.R0_cm
    t = 0.0
    step = 0
    collision_log = []

    sp_list = initialize_superparticles(
        n_superparticles=sim_params.n_superparticles,
        R_cm=R_cm,
        params=disk_params,
        r_sp_cm=sim_params.initial_radius_cm,
        rho_sp_g_cm3=disk_params.rho_s_g_cm3,
        N_real=sim_params.initial_real_particles_per_swarm,
        radius_logscatter_dex=sim_params.initial_radius_logscatter_dex,
        rng=rng
    )

    snapshot_idx = 0
    snapshot = snapshot_from_superparticles(sp_list, t, collision_log=collision_log)
    save_snapshot(snapshot, os.path.join(output_dir, f"snapshot_{snapshot_idx:03d}.npz"))
    snapshot_idx += 1

    snapshot_times = list(sim_params.snapshot_times_s)
    next_snapshot_ptr = 0

    while t < sim_params.t_end_s:
        rate_matrix = build_rate_matrix(sp_list, R_cm, disk_params, sim_params)
        R_total = total_collision_rate(rate_matrix)

        if R_total <= 0:
            print("No more positive collision rates. Stopping simulation.")
            break

        dt = sample_event_time(R_total, rng=rng)
        t_new = t + dt

        while next_snapshot_ptr < len(snapshot_times) and t_new >= snapshot_times[next_snapshot_ptr]:
            snapshot_time = snapshot_times[next_snapshot_ptr]
            snapshot = snapshot_from_superparticles(sp_list, snapshot_time, collision_log=collision_log)
            save_snapshot(snapshot, os.path.join(output_dir, f"snapshot_{snapshot_idx:03d}.npz"))
            snapshot_idx += 1
            next_snapshot_ptr += 1

        i, j = sample_collision_pair(rate_matrix, rng=rng)
        sp_i = sp_list[i]
        sp_j = sp_list[j]

        v_rel = relative_velocity(
            sp_i["m_dust"], sp_j["m_dust"],
            sp_i["St"], sp_j["St"],
            R_cm, disk_params
        )

        outcome = execute_collision(
            sp_i, sp_j, v_rel,
            R_cm, disk_params, sim_params,
            rng=rng
        )

        collision_log.append({
            "time": t_new,
            "i": i,
            "j": j,
            "v_rel": float(v_rel),
            "outcome": outcome,
            "m_i": float(sp_i["m_dust"]),
            "m_j": float(sp_j["m_dust"]),
            "St_i": float(sp_i["St"]),
            "St_j": float(sp_j["St"])
        })

        t = t_new
        step += 1

        if step % 500 == 0:
            mean_mass = np.mean([sp["m_dust"] for sp in sp_list])
            mean_st = np.mean([sp["St"] for sp in sp_list])
            print(
                f"step={step}, "
                f"t={t:.3e} s, "
                f"R_total={R_total:.3e} 1/s, "
                f"<m_dust>={mean_mass:.3e} g, "
                f"<St>={mean_st:.3e}"
            )

    snapshot = snapshot_from_superparticles(sp_list, t, collision_log=collision_log)
    save_snapshot(snapshot, os.path.join(output_dir, f"snapshot_{snapshot_idx:03d}.npz"))

    print("Simulation finished.")
    print(f"Final time: {t:.6e} s")
    print(f"Total steps: {step}")
    print(f"Snapshots saved in: {output_dir}")

    return sp_list


# ============================================================
# Plotting / post-processing
# ============================================================
def plot_snapshot_results(snapshot_dir="snapshots"):
    if not os.path.exists(snapshot_dir):
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

    snapshot_files = sorted(
        [f for f in os.listdir(snapshot_dir) if f.endswith(".npz")]
    )

    if len(snapshot_files) == 0:
        raise ValueError(f"No .npz snapshot files found in {snapshot_dir}")

    snapshots = [read_snapshot(os.path.join(snapshot_dir, f)) for f in snapshot_files]

    times_yr = [snap["time"] / YR_TO_S for snap in snapshots]
    mean_masses = [np.mean(snap["masses"]) for snap in snapshots]
    median_masses = [np.median(snap["masses"]) for snap in snapshots]
    mean_radii = [np.mean(snap["radii"]) for snap in snapshots]
    mean_stokes = [np.mean(snap["stokes"]) for snap in snapshots]
    collision_counts = [len(snap["collision_log"]) for snap in snapshots]

    first_data = snapshots[0]
    last_data = snapshots[-1]

    print("========== Snapshot Summary ==========")
    print(f"Number of snapshots: {len(snapshot_files)}")
    print(f"First snapshot: {snapshot_files[0]}")
    print(f"Last snapshot : {snapshot_files[-1]}")
    print(f"Initial mean mass   = {mean_masses[0]:.6e} g")
    print(f"Final mean mass     = {mean_masses[-1]:.6e} g")
    print(f"Initial mean radius = {mean_radii[0]:.6e} cm")
    print(f"Final mean radius   = {mean_radii[-1]:.6e} cm")
    print(f"Initial mean Stokes = {mean_stokes[0]:.6e}")
    print(f"Final mean Stokes   = {mean_stokes[-1]:.6e}")
    print(f"Final collision log length = {collision_counts[-1]}")
    print("======================================")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. mean mass vs time
    axes[0, 0].plot(times_yr, mean_masses, marker="o")
    axes[0, 0].set_xlabel("Time [yr]")
    axes[0, 0].set_ylabel("Mean single-particle mass [g]")
    axes[0, 0].set_title("Mean dust mass vs time")
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale("log")

    # 2. median mass vs time
    axes[0, 1].plot(times_yr, median_masses, marker="o")
    axes[0, 1].set_xlabel("Time [yr]")
    axes[0, 1].set_ylabel("Median single-particle mass [g]")
    axes[0, 1].set_title("Median dust mass vs time")
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale("log")

    # 3. mean radius vs time
    axes[0, 2].plot(times_yr, mean_radii, marker="o")
    axes[0, 2].set_xlabel("Time [yr]")
    axes[0, 2].set_ylabel("Mean particle radius [cm]")
    axes[0, 2].set_title("Mean radius vs time")
    axes[0, 2].grid(True)
    axes[0, 2].set_yscale("log")

    # 4. mean Stokes vs time
    axes[1, 0].plot(times_yr, mean_stokes, marker="o")
    axes[1, 0].set_xlabel("Time [yr]")
    axes[1, 0].set_ylabel("Mean Stokes number")
    axes[1, 0].set_title("Mean Stokes vs time")
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale("log")

    # 5. collision count vs time
    axes[1, 1].plot(times_yr, collision_counts, marker="o")
    axes[1, 1].set_xlabel("Time [yr]")
    axes[1, 1].set_ylabel("Logged collision count")
    axes[1, 1].set_title("Collision log count vs time")
    axes[1, 1].grid(True)

    # 6. initial vs final mass distribution
    axes[1, 2].hist(first_data["masses"], bins=20, alpha=0.7, label="Initial")
    axes[1, 2].hist(last_data["masses"], bins=20, alpha=0.7, label="Final")
    axes[1, 2].set_xlabel("Single-particle mass [g]")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].set_title("Initial vs final mass distribution")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()

    # second figure for radius/Stokes histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(first_data["radii"], bins=20, alpha=0.7, label="Initial")
    axes[0].hist(last_data["radii"], bins=20, alpha=0.7, label="Final")
    axes[0].set_xlabel("Particle radius [cm]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Initial vs final radius distribution")
    axes[0].legend()

    axes[1].hist(first_data["stokes"], bins=20, alpha=0.7, label="Initial")
    axes[1].hist(last_data["stokes"], bins=20, alpha=0.7, label="Final")
    axes[1].set_xlabel("Stokes number")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Initial vs final Stokes distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    disk_params = DiskParams()
    sim_params = SimulationParams()

    t0 = time.time()
    run_monte_carlo(disk_params, sim_params, output_dir=sim_params.output_dir)
    t1 = time.time()

    print(f"Wall-clock runtime = {(t1 - t0):.2f} s = {(t1 - t0)/60:.2f} min")

    plot_snapshot_results(sim_params.output_dir)
