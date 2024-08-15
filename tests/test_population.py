import random

import numpy as np
import pytest

from parallel_tsp.distance_matrix import DistanceMatrix
from parallel_tsp.population import (
    Population,
    combine_populations,
    generate_populations,
)
from parallel_tsp.route import Route


@pytest.fixture
def distance_matrix():
    matrix = np.array([[0, 2, 9, 10], [1, 0, 6, 4], [15, 7, 0, 8], [6, 3, 12, 0]])
    return DistanceMatrix(matrix)


@pytest.fixture
def population(distance_matrix):
    return Population(size=5, distance_matrix=distance_matrix)


def test_population_initialization(population):
    assert len(population.routes) == 5
    assert all(isinstance(route, Route) for route in population.routes)


def test_random_route_generation(population):
    route = population.random_route(num_cities=4)
    assert len(route) == 4
    assert set(route) == {0, 1, 2, 3}


def test_evolution_changes_population(population, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.01)
    monkeypatch.setattr(random, "sample", lambda x, k: x[:k])

    initial_routes = [route.city_indices.copy() for route in population.routes]
    population.evolve(mutation_rate=0.05, tournament_size=3)
    evolved_routes = [route.city_indices for route in population.routes]

    assert initial_routes != evolved_routes


def test_select_parent(population, monkeypatch):
    monkeypatch.setattr(random, "sample", lambda x, k: x[:k])

    selected_parent = population.select_parent(tournament_size=3)
    assert selected_parent in population.routes


def test_generate_populations(distance_matrix):
    populations = generate_populations(
        num_populations=3, population_size=5, distance_matrix=distance_matrix
    )
    assert len(populations) == 3
    assert all(isinstance(population, Population) for population in populations)


def test_combine_populations(distance_matrix):
    population1 = Population(size=5, distance_matrix=distance_matrix)
    population2 = Population(size=5, distance_matrix=distance_matrix)

    combined_population = combine_populations(population1, population2)

    assert isinstance(combined_population, Population)
    assert len(combined_population.routes) == 5
    assert combined_population.routes[0].distance_matrix == distance_matrix
