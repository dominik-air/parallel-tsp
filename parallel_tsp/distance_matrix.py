import numpy as np


class DistanceMatrix:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def distance(self, city1: int, city2: int) -> float:
        return self.matrix[city1][city2]

    def __len__(self) -> int:
        return len(self.matrix)


def generate_random_distance_matrix(num_cities: int) -> DistanceMatrix:
    return DistanceMatrix(np.random.rand(num_cities, num_cities))
