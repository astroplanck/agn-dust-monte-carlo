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