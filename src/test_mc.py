import numpy as np

from mc_rates import compute_lambda_i, compute_total_rate
from mc_timestep import sample_dt, select_particle_i, select_particle_j
import kernel

from params import DiskParams
from disk import disk_state

class Particle:
    def __init__(self, radius, mass, St):
        self.radius = radius
        self.mass = mass
        self.St = St

def test_small_particle_list():
    particles = [
        Particle(1e-4, 1e-12, 0.01),
        Particle(2e-4, 2e-12, 0.05),
        Particle(3e-4, 3e-12, 0.1)
    ]
    return particles



# Test 1
def test_lambda_positive():
    particles = test_small_particle_list()
    params = DiskParams()
    R = 1e18
    dstate = disk_state(R, params)

    f_array, g_array = kernel.compute_kernel_factors(
        particles, dstate, params
    )

    lambda_array = compute_lambda_i(f_array, g_array)
    for l in lambda_array:
        assert l > 0

    print("test_lambda_positive passed")

# Test 2
def test_total_rate_positive():
    particles = test_small_particle_list()
    params = DiskParams()
    R = 1e18
    dstate = disk_state(R, params)

    f_array, g_array = kernel.compute_kernel_factors(
        particles, dstate, params
    )

    lambda_array = compute_lambda_i(f_array, g_array)
    Lambda = compute_total_rate(lambda_array)
    assert Lambda > 0

    print("test_total_rate_positive passed")

# Test 3
def test_sample_dt():
    Lambda = 10
    dt = sample_dt(Lambda)
    assert dt > 0

    print("test_sample_dt passed")

# Test 4
def test_particle_selection():
    lambda_array = np.array([1, 3, 2, 4])
    g_array = np.array([2, 1, 5, 3])

    i = select_particle_i(lambda_array)
    j = select_particle_j(g_array, i)

    assert 0 <= i < len(lambda_array)
    assert 0 <= j < len(g_array)
    assert i != j

    print("test_particle_selection passed")

# Test 5
def test_sampling_distribution():
    np.random.seed(0)

    lambda_array = np.array([100, 1, 1, 1])
    counts = np.zeros(len(lambda_array))

    trials = 10000
    for k in range(trials):
        i = select_particle_i(lambda_array)
        counts[i] += 1

    assert counts[0] == max(counts)

    print("test_sampling_distribution passed")



if __name__ == "__main__":

    test_lambda_positive()
    test_total_rate_positive()
    test_sample_dt()
    test_particle_selection()
    test_sampling_distribution()

    print("\nAll tests passed.")