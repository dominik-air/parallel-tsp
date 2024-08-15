import random

import numpy as np
import pytest

from parallel_tsp.distance_matrix import DistanceMatrix
from parallel_tsp.genetic_algorithm import GeneticAlgorithm
from parallel_tsp.population import Population
from parallel_tsp.route import Route


@pytest.fixture
def distance_matrix():
    matrix = np.array([[0, 2, 9, 10], [1, 0, 6, 4], [15, 7, 0, 8], [6, 3, 12, 0]])
    return DistanceMatrix(matrix)


@pytest.fixture
def population(distance_matrix):
    return Population(size=5, distance_matrix=distance_matrix)


@pytest.fixture
def genetic_algorithm(population):
    return GeneticAlgorithm(
        population=population, generations=10, mutation_rate=0.05, tournament_size=3
    )


def test_genetic_algorithm_initialization(genetic_algorithm, population):
    assert genetic_algorithm.population == population
    assert genetic_algorithm.generations == 10
    assert genetic_algorithm.mutation_rate == 0.05
    assert genetic_algorithm.tournament_size == 3


def test_genetic_algorithm_run(genetic_algorithm, population, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.01)
    monkeypatch.setattr(random, "sample", lambda x, k: x[:k])

    initial_routes = [route.city_indices for route in population.routes]
    final_best_route, _ = genetic_algorithm.run()
    final_routes = [route.city_indices for route in population.routes]

    assert isinstance(final_best_route, Route)
    assert initial_routes != final_routes


def test_select_best(genetic_algorithm, population, monkeypatch):
    route_lengths = [40, 30, 20, 10]

    for route, length in zip(population.routes, route_lengths):
        monkeypatch.setattr(route, "length", lambda length=length: length)

    best_route = genetic_algorithm.select_best(population)

    assert isinstance(best_route, Route)
    assert best_route.length() == 10
