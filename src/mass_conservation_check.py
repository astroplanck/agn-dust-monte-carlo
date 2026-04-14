import os
import numpy as np
import matplotlib.pyplot as plt
import chenran_agn_disk_2 as sim

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

    snapshots = [sim.read_snapshot(os.path.join(snapshot_dir, f)) for f in snapshot_files]

    times = []
    total_mass = []

    for snap in snapshots:
        times.append(snap["time"] / sim.YR_TO_S)
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

def plot_mass_conservation(times, total_mass, rel_mass):
    plt.figure(figsize=(8, 5))
    plt.plot(times, rel_mass, marker="o")
    plt.xlabel("Time [yr]")
    plt.ylabel("Total Mass / Initial Mass")
    plt.title("Mass Conservation Check")
    plt.grid(True)
    plt.show()

snapshot_dir = sim.SimulationParams().output_dir

times, total_mass, rel_mass, snapshots = mass_conservation_analysis(snapshot_dir)
plot_mass_conservation(times, total_mass, rel_mass)
consistency_check(snapshots[-1])