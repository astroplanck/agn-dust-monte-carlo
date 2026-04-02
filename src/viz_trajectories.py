import numpy as np
import matplotlib.pyplot as plt


def track_largest_particle(snapshot_list):

    times = []
    largest_masses = []

    for snapshot in snapshot_list:
        time = snapshot["time"]
        masses = snapshot["masses"]

        largest = np.max(masses)

        times.append(time)
        largest_masses.append(largest)

    return np.array(times), np.array(largest_masses)


def track_median_mass(snapshot_list):

    times = []
    median_masses = []

    for snapshot in snapshot_list:
        time = snapshot["time"]
        masses = snapshot["masses"]

        median = np.median(masses)

        times.append(time)
        median_masses.append(median)

    return np.array(times), np.array(median_masses)


def plot_trajectory(times, values, label):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(times, values, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    ax.set_title("Growth Trajectory")
    ax.legend()

    return fig

if __name__ == "__main__":
    snapshots = [
        {"time": 0.0, "masses": [1e-12, 2e-12, 3e-12]},
        {"time": 1.0, "masses": [2e-12, 4e-12, 6e-12]},
        {"time": 2.0, "masses": [3e-12, 5e-12, 9e-12]},
    ]

    times_l, largest = track_largest_particle(snapshots)
    print("times:", times_l)
    print("largest masses:", largest)

    fig = plot_trajectory(times_l, largest, "Largest Mass")
    plt.show()