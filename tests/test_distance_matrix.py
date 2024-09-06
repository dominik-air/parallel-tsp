import numpy as np
import pytest

from parallel_tsp.distance_matrix import DistanceMatrix, generate_random_distance_matrix


@pytest.mark.unit
def test_distance_matrix_initialization():
    matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    distance_matrix = DistanceMatrix(matrix)

    assert isinstance(distance_matrix, DistanceMatrix)
    assert np.array_equal(distance_matrix.matrix, matrix)


@pytest.mark.unit
def test_distance_matrix_distance():
    matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    distance_matrix = DistanceMatrix(matrix)

    assert distance_matrix.distance(0, 1) == 1
    assert distance_matrix.distance(1, 2) == 3
    assert distance_matrix.distance(2, 0) == 2


@pytest.mark.unit
def test_distance_matrix_length():
    matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    distance_matrix = DistanceMatrix(matrix)

    assert len(distance_matrix) == 3


@pytest.mark.unit
def test_random_distance_matrix_initialization():
    num_cities = 4
    distance_matrix = generate_random_distance_matrix(num_cities)

    assert isinstance(distance_matrix, DistanceMatrix)
    assert distance_matrix.matrix.shape == (num_cities, num_cities)
    assert np.all(distance_matrix.matrix >= 0)
    assert np.all(distance_matrix.matrix <= 1)
