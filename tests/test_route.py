import random

import numpy as np
import pytest

from parallel_tsp.distance_matrix import DistanceMatrix
from parallel_tsp.route import Route, crossover


@pytest.fixture
def distance_matrix():
    matrix = np.array([[0, 2, 9, 10], [1, 0, 6, 4], [15, 7, 0, 8], [6, 3, 12, 0]])
    return DistanceMatrix(matrix)


@pytest.fixture
def route(distance_matrix):
    city_indices = [0, 1, 2, 3]
    return Route(city_indices, distance_matrix)


@pytest.mark.unit
def test_route_initialization(route, distance_matrix):
    assert route.city_indices == [0, 1, 2, 3]
    assert route.distance_matrix == distance_matrix


@pytest.mark.unit
def test_route_length(route):
    # Distance calculation should be 6 (3 -> 0) + 2 (0 -> 1) + 6 (1 -> 2) + 8 (2 -> 3) = 22
    assert route.length() == 22


@pytest.mark.unit
def test_route_mutation(route, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.01)
    monkeypatch.setattr(random, "sample", lambda x, k: [1, 3])

    route.mutate(mutation_rate=0.05)

    assert route.city_indices == [0, 3, 2, 1]


@pytest.mark.unit
def test_crossover(route, distance_matrix):
    parent1 = Route([0, 1, 2, 3], distance_matrix)
    parent2 = Route([3, 2, 1, 0], distance_matrix)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(random, "sample", lambda x, k: [1, 3])

        child1, child2 = crossover(parent1, parent2)

    assert child1.city_indices == [3, 1, 2, 0]
    assert child2.city_indices == [0, 2, 1, 3]


@pytest.mark.unit
def test_crossover_helper(route, distance_matrix):
    parent1 = Route([0, 1, 2, 3], distance_matrix)
    parent2 = Route([3, 2, 1, 0], distance_matrix)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(random, "sample", lambda x, k: [1, 3])

        child = crossover(parent1, parent2)[0]

    assert child.city_indices == [3, 1, 2, 0]


@pytest.mark.unit
def test_mutate_no_change(route, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.99)

    original_city_indices = route.city_indices[:]
    route.mutate(mutation_rate=0.05)

    assert route.city_indices == original_city_indices
