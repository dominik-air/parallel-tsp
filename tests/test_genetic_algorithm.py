import random
from functools import partial

import numpy as np
import pytest

from parallel_tsp.distance_matrix import DistanceMatrix
from parallel_tsp.genetic_algorithm import (
    GeneticAlgorithm,
    parametrise_genetic_algorithm,
    select_best,
)
from parallel_tsp.population import Population
from parallel_tsp.route import Route
from parallel_tsp.stop_condition import StopCondition


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
        population=population,
        mutation_rate=0.05,
        tournament_size=3,
        stop_condition=StopCondition(max_generations=10),
    )


def test_genetic_algorithm_initialization(genetic_algorithm, population):
    assert genetic_algorithm.population == population
    assert genetic_algorithm.mutation_rate == 0.05
    assert genetic_algorithm.tournament_size == 3
    assert isinstance(genetic_algorithm.initial_best_route, Route)
    assert genetic_algorithm.best_route is genetic_algorithm.initial_best_route


def test_genetic_algorithm_run_iterations(genetic_algorithm, population, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.01)
    monkeypatch.setattr(random, "sample", lambda x, k: x[:k])

    initial_routes = [route.city_indices for route in population.routes]
    genetic_algorithm.run()
    final_best_route = genetic_algorithm.best_route
    final_routes = [route.city_indices for route in population.routes]

    assert isinstance(final_best_route, Route)
    assert initial_routes != final_routes
    assert final_best_route.length() <= genetic_algorithm.initial_best_route.length()


def test_select_best(genetic_algorithm, population, monkeypatch):
    route_lengths = [40, 30, 20, 10]

    for route, length in zip(population.routes, route_lengths):
        monkeypatch.setattr(route, "length", lambda length=length: length)

    best_route = select_best(population.routes)

    assert isinstance(best_route, Route)
    assert best_route.length() == 10


def test_parametrise_genetic_algorithm():
    parametrized_ga = parametrise_genetic_algorithm(
        mutation_rate=0.1,
        tournament_size=3,
        stop_condition=StopCondition(max_generations=10),
    )

    assert isinstance(parametrized_ga, partial)
    assert parametrized_ga.keywords["mutation_rate"] == 0.1
    assert parametrized_ga.keywords["tournament_size"] == 3


def test_parametrise_genetic_algorithm_invalid_inputs():
    with pytest.raises(ValueError):
        parametrise_genetic_algorithm(
            mutation_rate=1.1,
            tournament_size=3,
            stop_condition=StopCondition(max_generations=10),
        )

    with pytest.raises(ValueError):
        parametrise_genetic_algorithm(
            mutation_rate=-0.1,
            tournament_size=3,
            stop_condition=StopCondition(max_generations=10),
        )

    with pytest.raises(ValueError):
        parametrise_genetic_algorithm(
            mutation_rate=0.1,
            tournament_size=-1,
            stop_condition=StopCondition(max_generations=10),
        )
