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
from astropy import units as u
from astropy import constants as const

# ============================================================
# Physical constants (CGS)
# ============================================================
G_CGS = const.G.cgs        # cm^3 g^-1 s^-2
KB_CGS = const.k_B.cgs     # erg/K = g cm^2 s^-2 K^-1
MH_CGS = 1.6735575e-24     # g
PI = math.pi

PC_TO_CM = u.cm / u.pc  # cm
MSUN_TO_G = u.g / u.M_sun     # g
YR_TO_S = u.s / u.year        # s


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
    MBH_g: float = (1.0e7 * u.M_sun).to_value(u.g)
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
    R0_pc: float = 1.0 * u.pc
    box_length_pc: float = 1.0e-6 * u.pc

    n_superparticles: int = 1000
    initial_real_particles_per_swarm: float = 1.0e6

    initial_radius_cm: float = 1.0e-3 * u.cm
    initial_radius_logscatter_dex: float = 0.3

    t_end_yr: float = 1.0e4 * u.year
    snapshot_every_yr: float = t_end_yr / 20
    deltat = 10 * u.year

    v_stick_cm_s: float = 1.0e2 * u.cm / u.s
    v_frag_cm_s: float = 1.0e4 * u.cm / u.s 

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
    # If R is large (cm), convert to pc (1.0). If R is small (1.0), it's already pc.
    R_val = R_cm.to_value(u.cm) if hasattr(R_cm, 'unit') else R_cm
    R_pc = R_val / 3.08567758e18 if R_val > 1e10 else R_val
    return params.T0_K * (R_pc ** -params.q)

def surface_density(R_cm, params):
    R_val = R_cm.to_value(u.cm) if hasattr(R_cm, 'unit') else R_cm
    R_pc = R_val / 3.08567758e18 if R_val > 1e10 else R_val
    return params.Sigma0_g_cm2 * (R_pc ** -params.p)

def omega_k(R_cm, params):
    R_val = R_cm.to_value(u.cm) if hasattr(R_cm, 'unit') else R_cm
    # Force R_val to be centimeters (~3e18)
    R_true_cm = R_val if R_val > 1e10 else R_val * 3.08567758e18
    return np.sqrt((G_CGS.value * params.MBH_g) / (R_true_cm ** 3))

def midplane_gas_density(R_cm, params):
    # Ensure R_cm is converted once here
    R_val = R_cm.to_value(u.cm) if hasattr(R_cm, 'unit') else R_cm
    R_true_cm = R_val if R_val > 1e10 else R_val * 3.08567758e18
    
    sigma = surface_density(R_true_cm, params)
    cs = sound_speed(R_true_cm, params)
    ok = omega_k(R_true_cm, params)
    h = cs / ok
    return sigma / (2.506628 * h)

def temperature(R_cm, params):
    r_val = R_cm.to_value(u.cm) if hasattr(R_cm, 'unit') else R_cm
    R_ratio = r_val / 3.08567758e18
    t0 = params.T0_K.value if hasattr(params.T0_K, 'unit') else params.T0_K
    return t0 * (R_ratio ** -params.q)

def sound_speed(R_cm, params):
    T_R = temperature(R_cm, params)
    # Return a raw float in cm/s
    return np.sqrt((KB_CGS.value * T_R) / (params.mu * MH_CGS))

def update_derived_properties(data, R_cm_input, params):
    R = 3.08567758e18 
    
    M = params.MBH_g.value if hasattr(params.MBH_g, 'unit') else params.MBH_g
    G = 6.6743e-8
    KB = 1.3806e-16
    MH = 1.6735e-24
    
    om_k = np.sqrt(G * M / R**3)           # Result: ~6.7e-12
    T = params.T0_K                        # Result: 100
    cs = np.sqrt(KB * T / (params.mu * MH)) # Result: ~5.9e4
    Sigma = params.Sigma0_g_cm2            # Result: 1000
    
    # rho_g = Sigma / (sqrt(2pi) * (cs/om_k))
    rho_g = Sigma / (2.5066 * (cs / om_k)) # Result: ~4.5e-14
    
    # 3. Update the table
    data['m_dust'] = data['m_sp'] / data['N_sp']
    data['r_sp'] = ((3.0 * data['m_dust']) / (4.0 * np.pi * data['rho_sp'])) ** (1/3)
    
    # ts = (rho_int * a) / (rho_g * cs)
    data['ts'] = (data['rho_sp'] * data['r_sp']) / (rho_g * cs)
    data['St'] = data['ts'] * om_k
    
    return data


def scale_height(R_cm, params):
    return sound_speed(R_cm, params) / omega_k(R_cm, params)

# ============================================================
# Particle property helpers
# ============================================================
def radius_from_mass(m, rho_int):
    # NumPy handles the power operation (**) across the whole array at once
    return ((3.0 * m) / (4.0 * np.pi * rho_int)) ** (1.0 / 3.0)

def stopping_time(a, rho_int, R_cm, params):
    rho_g = midplane_gas_density(R_cm, params)
    c_s = sound_speed(R_cm, params)
    return (rho_int * a) / (rho_g * c_s)

def stokes_number(t_stop, R_cm, params):
    return t_stop * omega_k(R_cm, params)

def compute_stokes(a, rho_int, R_cm, params):
    rho_g = midplane_gas_density(R_cm, params) # returns g/cm3
    c_s = sound_speed(R_cm, params)           # returns cm/s
    om_k = omega_k(R_cm, params)              # returns 1/s
    t_s = (rho_int * u.g/u.cm**3 * a * u.cm) / (rho_g * c_s)
    t_s_val = t_s.to_value(u.s)
    
    stokes = t_s_val * om_k.to_value(1/u.s)
    return stokes

def stopping_time_from_stokes(St, R_cm, params):
    return St / omega_k(R_cm, params)

# ============================================================
# Relative velocities and collision kernel
# ============================================================
def brownian_velocity(m1, m2, R_cm, params):
    T = temperature(R_cm, params)
    # Use raw KB_CGS value
    return np.sqrt((8.0 * KB_CGS.value * T / PI) * (1.0 / m1 + 1.0 / m2))

def relative_velocity_vec(m1, m2, St1, St2, R_cm, params):
    """Calculates relative velocity using raw float math."""
    v_brown = brownian_velocity(m1, m2, R_cm, params) # Removed .value
    v_turb = turbulent_velocity(St1, St2, R_cm, params) # Removed .value
    
    v1_rad = radial_velocity(St1, R_cm, params)
    v2_rad = radial_velocity(St2, R_cm, params)
    v_rad = np.abs(v1_rad - v2_rad)
    
    return np.sqrt(v_brown**2 + v_turb**2 + v_rad**2)

def turbulent_velocity(St1, St2, R_cm, params):
    c_s = sound_speed(R_cm, params)
    alpha = params.alpha
    # Relative turbulence between particles
    return np.sqrt(alpha) * c_s * np.sqrt(np.abs(St1 - St2))

def radial_velocity(St, R_cm, params):
    # R_cm is ~3e18
    v_k = np.sqrt(G_CGS.value * params.MBH_g / R_cm)
    # cs/vk is roughly H/R
    eta = (sound_speed(R_cm, params)**2) / (v_k**2)
    return -2.0 * eta * v_k * St / (1.0 + St**2)

def eta_parameter(R_cm, params):
    c_s = sound_speed(R_cm, params)
    v_k = v_kepler(R_cm, params)
    return (c_s ** 2) / (v_k ** 2)

def v_kepler(R_cm, params):
    # R_cm is already a float
    return np.sqrt(G_CGS.value * params.MBH_g / R_cm)

def collision_kernel_vec(a1, a2, v_rel):
    """Calculates kernel for arrays of pairs."""
    sigma = np.pi * (a1 + a2)**2
    return sigma * v_rel


def collision_cross_section(a1, a2):
    return PI * (a1 + a2) ** 2


# ============================================================
# Super-particle initialization and updates
# ============================================================
def initialize_superparticles(sim_params, disk_params, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    n = sim_params.n_superparticles
    # Define the Table Structure
    dtype = [('id', 'i4'), ('r_sp', 'f8'), ('rho_sp', 'f8'), 
             ('m_dust', 'f8'), ('N_sp', 'f8'), ('m_sp', 'f8'), 
             ('ts', 'f8'), ('St', 'f8')]
    
    data = np.zeros(n, dtype=dtype)
    data['id'] = np.arange(1, n + 1)
    data['rho_sp'] = disk_params.rho_s_g_cm3
    data['N_sp'] = sim_params.initial_real_particles_per_swarm
    
    # Vectorized Radius with log-scatter
    r_base = sim_params.initial_radius_cm.value
    if sim_params.initial_radius_logscatter_dex > 0:
        ln_sigma = sim_params.initial_radius_logscatter_dex * np.log(10.0)
        data['r_sp'] = r_base * np.exp(rng.normal(0.0, ln_sigma, n))
    else:
        data['r_sp'] = r_base
        
    # Update physics across the whole table at once
    data['m_dust'] = (4.0/3.0) * np.pi * (data['r_sp']**3) * data['rho_sp']
    data['m_sp'] = data['N_sp'] * data['m_dust']
    
    return data

def update_derived_properties(data, R_cm, params):
    # All these now return raw floats
    rho_g = midplane_gas_density(R_cm, params)
    c_s = sound_speed(R_cm, params)
    om_k = omega_k(R_cm, params)

    # Recalculate physical growth
    data['m_dust'] = data['m_sp'] / data['N_sp']
    data['r_sp'] = ((3.0 * data['m_dust']) / (4.0 * np.pi * data['rho_sp'])) ** (1.0 / 3.0)

    # Update coupling physics
    data['ts'] = (data['rho_sp'] * data['r_sp']) / (rho_g * c_s)
    data['St'] = data['ts'] * om_k
    # --- ADD AT THE END OF update_derived_properties ---
    if np.any(data['St'] < 1e-15):
        print("!!! DATA COLLAPSE DETECTED !!!")
        print(f"rho_g: {rho_g:.2e} | cs: {c_s:.2e} | om_k: {om_k:.2e}")
        print(f"Sample m_dust: {data['m_dust'][:3]}")
        # Stop immediately to catch the error
        raise ValueError("Stokes number dropped to zero-scale.")

    return data


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
            Rij = collision_rate(sp_list[i], sp_list[j], R_cm, disk_params, sim_params).value
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


def resolve_collisions_vectorized(data, dt, box_volume, sim_params, disk_params, rng):
    n = len(data)
    if n < 2: return data

    # Create random pairs
    indices = np.arange(n)
    rng.shuffle(indices)
    mid = n // 2
    idx1, idx2 = indices[:mid], indices[mid:2*mid]

    # Calculate kernel and probability
    v_rel = relative_velocity_vec(data['m_dust'][idx1], data['m_dust'][idx2], 
                                  data['St'][idx1], data['St'][idx2], sim_params.R0_pc.to_value(u.cm), disk_params)
    kernel = np.pi * (data['r_sp'][idx1] + data['r_sp'][idx2])**2 * v_rel
    
    # Prob = (Kernel * dt * N_real_per_swarm) / Volume
    prob = (kernel * dt * sim_params.initial_real_particles_per_swarm) / box_volume/ 3.08e18
    collided = rng.random(mid) < prob
    
    if rng.random() < 0.001:
        print(f"probability: {prob[0]}")

    
    if np.any(collided):
        c_idx1, c_idx2 = idx1[collided], idx2[collided]
        
        # --- THE CRITICAL MASS CONSERVATION STEP ---
        # 1. Add source swarm mass to target swarm mass
        data['m_sp'][c_idx1] += data['m_sp'][c_idx2]
        
        # 2. Kill the source particles (sets mass to 0)
        data['m_sp'][c_idx2] = 0.0
        
        # Note: radius and St will be updated by update_derived_properties in the next step
        
    return data


def snapshot_from_superparticles(data, t, collision_log=None):
    """
    Creates a snapshot dictionary with keys mapped to 
    exactly what save_snapshot expects.
    """
    return {
        "time": t,
        "masses": data['m_dust'].copy(),
        "radii": data['r_sp'].copy(),         # Maps r_sp to radii
        "densities": data['rho_sp'].copy(),   # Maps rho_sp to densities
        "numbers": data['N_sp'].copy(),       # Maps N_sp to numbers
        "swarm_masses": data['m_sp'].copy(),  # Maps m_sp to swarm_masses
        "stokes": data['St'].copy(),          # Maps St to stokes
        "stopping_times": data['ts'].copy(),  # Maps ts to stopping_times
        "collision_log": collision_log if collision_log else []
    }

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
def run_monte_carlo_vectorized(disk_params, sim_params, output_dir=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if output_dir is None:
        output_dir = sim_params.output_dir
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        

        # Check if it is a file (not a subdirectory)
        if os.path.isfile(file_path):
            os.remove(file_path)  # Remove the file
            #print(f"Deleted file: {filename}")
    #Will delete previous run snapshots!!
    #os.rmdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    R_cm = sim_params.R0_cm

    # 1. Initialize as a NumPy Structured Array (The Vectorized Table)
    data = initialize_superparticles(sim_params, disk_params, rng=rng)

    snapshot_idx = 0
    snapshot_times = list(sim_params.snapshot_times_s)
    next_snapshot_ptr = 0

    print(f"Starting vectorized simulation with {len(data)} superparticles...")
    t_s = 0.0
    t_end_s = (sim_params.t_end_yr*YR_TO_S).value
    dt_val_s = (sim_params.deltat).value 
    # Define fixed step (10 , 1000years)
    
    snapshot_times_s = [st.to(u.s).value for st in sim_params.snapshot_times_s]
    next_snapshot_ptr = 0
    snapshot_idx = 0
    snapshot = snapshot_from_superparticles(data, 0.0)
    save_snapshot(snapshot, os.path.join(output_dir, "snapshot_000.npz"))
    snapshot_idx = 1 # Start indices at 1 for the loop

    # --- THE NEW LOOP ---

    while t_s < t_end_s:
        # 1. RESOLVE COLLISIONS FIRST (This merges particles)

        data = resolve_collisions_vectorized(
            data, 
            dt_val_s, 
            sim_params.box_volume_cm3.value, 
            sim_params, 
            disk_params, 
            rng
        )        
        # 2. FILTER IMMEDIATELY (Removes the consumed particles)
        data = data[data['m_sp'] > 0]
        
        # 3. UPDATE PHYSICS (Calculates the new mass and larger radius for survivors)
        data = update_derived_properties(data, R_cm, disk_params)

        # 4. NOW HANDLE SNAPSHOTS (Records the updated growth)
        if next_snapshot_ptr < len(snapshot_times_s) and t_s >= snapshot_times_s[next_snapshot_ptr]:
            # Update properties one last time before saving
            data = update_derived_properties(data, R_cm, disk_params)
            
            snapshot = snapshot_from_superparticles(data, t_s)
            filename = os.path.join(output_dir, f"snapshot_{snapshot_idx:03d}.npz")
            save_snapshot(snapshot, os.path.join(output_dir, f"snapshot_{snapshot_idx:03d}.npz"))
            
            snapshot_idx += 1
            next_snapshot_ptr += 1

        t_s += dt_val_s

    print(f"Simulation finished. Final particle count: {len(data)}, time end: {t_s}, snapshots: {len(snapshot_times)}")
    return data


# ============================================================
# Plotting / post-processing
# ============================================================
def plot_snapshot_results(snapshot_dir="snapshots"):
    if not os.path.exists(snapshot_dir):
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

    snapshot_files = sorted(
        [f for f in os.listdir(snapshot_dir) if f.endswith(".npz")]
    )
    #snapshot_files = snapshot_files[0:int(sim_params.t_end_yr.value / sim_params.snapshot_every_yr.value)]

    if len(snapshot_files) == 0:
        raise ValueError(f"No .npz snapshot files found in {snapshot_dir}")

    snapshots = [read_snapshot(os.path.join(snapshot_dir, f)) for f in snapshot_files]

    times_yr = [snap["time"] for snap in snapshots]
    mean_masses = [np.mean(snap["masses"]) for snap in snapshots]
    median_masses = [np.median(snap["masses"]) for snap in snapshots]
    mean_radii = [np.mean(snap["radii"]) for snap in snapshots]
    mean_stokes = [np.mean(snap["stokes"]) for snap in snapshots]
    collision_counts = [len(snap["collision_log"]) for snap in snapshots]
    number_sp = [len(snap["numbers"]) for snap in snapshots]

    first_data = snapshots[0]
    last_data = snapshots[-1]

    print("========== Snapshot Summary ==========")
    print(f"Number of snapshots in folder: {len(snapshot_files)}")
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

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), facecolor = 'tan')

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

    # 5.mass check vs time
    
    snapshot_dir = SimulationParams().output_dir

    times_m, total_mass, rel_mass, snapshots = mass_conservation_analysis(snapshot_dir)
    axes[1, 1].plot(times_m, rel_mass, marker="o")
    axes[1, 1].set_xlabel("Time [yr]")
    axes[1, 1].set_ylabel("Logged relative mass")
    axes[1, 1].set_title("Total mass vs time")
    axes[1, 1].grid(True)
    
    # 6. initial vs final mass distribution
    axes[1, 2].hist(first_data["masses"], alpha=0.7, label="Initial")
    axes[1, 2].hist(last_data["masses"], alpha=0.7, label="Final")
    axes[1, 2].set_xlabel("Single-particle mass [g]")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].set_title("Initial vs final mass distribution")
    axes[1, 2].legend()
    str = '\n'.join(( r'R0: ' % (SimulationParams.R0_pc),
    r'box length pc: ' % (SimulationParams.box_length_pc),
    r'number of particles: ' %(SimulationParams.n_superparticles),
    r'initial particles per swarm: ' % (SimulationParams.initial_real_particles_per_swarm),

    r'initial particle r: ' % (SimulationParams.initial_radius_cm),

    r'end years' %( SimulationParams.t_end_yr),
    r'years per snapshot : ' %( SimulationParams.snapshot_every_yr),
    r'delta t: ' % (sim_params.deltat) ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1, 2].text(0,0.5,str, fontsize=10, bbox=props)

    #plt.tight_layout()
    #plt.show()

    # second figure for radius/Stokes histograms
    #fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[2,0].plot(times_yr, number_sp, alpha=0.7)
    axes[2,0].set_title("Time evolve")
    axes[2,0].set_xlabel("Time")
    axes[2,0].set_ylabel("Particle Count")
    axes[2,0].set_title("Particles Existing")

    axes[2,1].hist(first_data["radii"],alpha=0.7, label="Initial")
    axes[2,1].hist(last_data["radii"], alpha=0.7, label="Final")
    axes[2,1].set_xlabel("Particle radius [cm]")
    axes[2,1].set_ylabel("Count")
    axes[2,1].set_title("Initial vs final radius distribution")
    axes[2,1].legend()

    axes[2,2].hist(first_data["stokes"], bins = 20, alpha=0.7, label="Initial")
    axes[2,2].hist(last_data["stokes"], bins = 20, alpha=0.7, label="Final")
    axes[2,2].set_xlabel("Stokes number")
    axes[2,2].set_ylabel("Count")
    axes[2,2].set_title("Initial vs final Stokes distribution")
    axes[2,2].set_xlim(-1e-10, 1e-10)
    axes[2,2].legend()

    plt.tight_layout()
    plt.show()

#Mass Conservation

def compute_total_mass(snapshot):
    return np.sum(snapshot["swarm_masses"])

def consistency_check(snapshot):
    m_sp = snapshot["swarm_masses"]
    m_dust = snapshot["masses"]
    N_sp = snapshot["numbers"]

    reconstructed = N_sp * m_dust
    error = np.abs(reconstructed - m_sp)

    print(f"Max consistency error: {np.max(error):.6e}")
    print(f"Mean consistency error: {np.mean(error):.6e}")

def mass_conservation_analysis(snapshot_dir="snapshots"):
    snapshot_files = sorted(
        [f for f in os.listdir(snapshot_dir) if f.endswith(".npz")]
    )

    if not snapshot_files:
        raise ValueError(f"No .npz snapshot files found in {snapshot_dir}")

    snapshots = [read_snapshot(os.path.join(snapshot_dir, f)) for f in snapshot_files]

    times = []
    total_mass = []

    for snap in snapshots:
        times.append(snap["time"] / YR_TO_S)
        total_mass.append(compute_total_mass(snap))

    times = np.array(times)
    total_mass = np.array(total_mass)

    initial_mass = total_mass[0]
    rel_mass = total_mass / initial_mass

    print("Mass Conservation:")
    print(f"Initial total mass = {initial_mass:.6e}")
    print(f"Final total mass   = {total_mass[-1]:.6e}")
    print(f"Absolute change    = {(total_mass[-1] - initial_mass):.6e}")
    print(f"Relative change    = {(rel_mass[-1] - 1):.6e}")

    return times, total_mass, rel_mass, snapshots


#Mass Conservation Check

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    disk_params = DiskParams()
    sim_params = SimulationParams()

    t0 = time.time()
    run_monte_carlo_vectorized(disk_params, sim_params, output_dir=sim_params.output_dir)
    t1 = time.time()

    print(f"Wall-clock runtime = {(t1 - t0):.2f} s = {(t1 - t0)/60:.2f} min")
    

    plot_snapshot_results(sim_params.output_dir)

# %%
