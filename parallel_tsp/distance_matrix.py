import numpy as np


class DistanceMatrix:
    """Represents a distance matrix for a set of cities.

    Attributes:
        matrix (np.ndarray): A 2D numpy array where matrix[i][j] represents the distance
            between city i and city j.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """Initializes the DistanceMatrix with a given matrix.

        Args:
            matrix (np.ndarray): A 2D numpy array representing the distances between cities.
        """
        self.matrix = matrix

    def distance(self, city1: int, city2: int) -> float:
        """Returns the distance between two cities.

        Args:
            city1 (int): The index of the first city.
            city2 (int): The index of the second city.

        Returns:
            float: The distance between city1 and city2.
        """
        return self.matrix[city1][city2]

    def __len__(self) -> int:
        """Returns the number of cities in the distance matrix.

        Returns:
            int: The number of cities.
        """
        return len(self.matrix)


def generate_random_distance_matrix(num_cities: int) -> DistanceMatrix:
    """Generates a random distance matrix for a specified number of cities.

    Args:
        num_cities (int): The number of cities.

    Returns:
        DistanceMatrix: An instance of DistanceMatrix with random distances between cities.
    """
    return DistanceMatrix(np.random.rand(num_cities, num_cities))
