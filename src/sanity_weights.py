import numpy as np

from .weights import sample_pair


def sanity_test_weights(trials=10000):
    masses = np.array([1, 2, 4, 8, 16], dtype=float)
    selection_counts = np.zeros(len(masses), dtype=int)

    for k in range(trials):
        i, j = sample_pair(masses)
        selection_counts[i] += 1
        selection_counts[j] += 1

    expected_weights = masses ** (4 / 3)

    print("masses:")
    print(masses)

    print("\nselection_counts:")
    print(selection_counts)

    print("\nmasses^(4/3) (expected trend):")
    print(expected_weights)

    print("\nnormalized masses^(4/3):")
    print(expected_weights / np.sum(expected_weights))

    print("\nselection frequencies:")
    print(selection_counts / np.sum(selection_counts))



if __name__ == "__main__":
    sanity_test_weights()