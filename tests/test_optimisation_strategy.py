import networkx as nx
import numpy as np
import pytest

from parallel_tsp.distance_matrix import DistanceMatrix
from parallel_tsp.optimisation_strategy import (
    ChristofidesOptimization,
    GreedyTSPOptimization,
    MixedOptimization,
    NoOptimization,
    population_to_graph,
    routes_to_population,
)
from parallel_tsp.population import Population


@pytest.fixture
def distance_matrix():
    matrix = np.array([[0, 2, 9, 10], [2, 0, 6, 4], [9, 6, 0, 8], [10, 4, 8, 0]])
    return DistanceMatrix(matrix)


@pytest.fixture
def population(distance_matrix):
    return Population(size=3, distance_matrix=distance_matrix)


def test_population_to_graph(population):
    G, matrix = population_to_graph(population)

    assert isinstance(G, nx.Graph)
    assert len(G.nodes) == len(matrix)
    assert len(G.edges) == len(matrix) * (len(matrix) - 1) // 2

    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            assert G[i][j]["weight"] == matrix[i][j]


def test_routes_to_population(distance_matrix):
    routes = [[0, 1, 2, 3], [3, 2, 1, 0], [0, 3, 2, 1]]
    population = routes_to_population(routes, distance_matrix)

    assert isinstance(population, Population)
    assert len(population.routes) == len(routes)
    for route, expected_route in zip(population.routes, routes):
        assert route.city_indices == expected_route
        assert np.array_equal(route.distance_matrix, distance_matrix)


def test_no_optimization(population):
    strategy = NoOptimization()
    optimized_population = strategy.optimize(population)

    assert optimized_population == population
    assert optimized_population.routes == population.routes


def test_christofides_optimization(monkeypatch, population):
    def mock_christofides(G):
        return [0, 1, 2, 3, 0]

    monkeypatch.setattr(nx.algorithms.approximation, "christofides", mock_christofides)

    strategy = ChristofidesOptimization()
    optimized_population = strategy.optimize(population)

    assert isinstance(optimized_population, Population)
    assert len(optimized_population.routes) == population.size
    for route in optimized_population.routes:
        assert set(route.city_indices) == {0, 1, 2, 3}
        assert route.city_indices[0] == 0
        assert len(route.city_indices) == 4


def test_greedy_tsp_optimization(monkeypatch, population):
    def mock_greedy_tsp(G):
        return [0, 1, 2, 3, 0]

    monkeypatch.setattr(nx.algorithms.approximation, "greedy_tsp", mock_greedy_tsp)

    strategy = GreedyTSPOptimization()
    optimized_population = strategy.optimize(population)

    assert isinstance(optimized_population, Population)
    assert len(optimized_population.routes) == population.size
    for route in optimized_population.routes:
        assert set(route.city_indices) == {0, 1, 2, 3}
        assert route.city_indices[0] == 0
        assert len(route.city_indices) == 4


def test_mixed_optimization(monkeypatch, population):
    def mock_greedy_tsp(G):
        return [0, 1, 2, 3, 0]

    monkeypatch.setattr(nx.algorithms.approximation, "greedy_tsp", mock_greedy_tsp)

    base_strategy = GreedyTSPOptimization()
    mixed_strategy = MixedOptimization(base_optimization=base_strategy, mix_ratio=0.5)

    optimized_population = mixed_strategy.optimize(population)

    num_optimized = int(0.5 * population.size)
    num_non_optimized = population.size - num_optimized

    assert len(optimized_population.routes) == population.size
    assert len(optimized_population.routes[:num_optimized]) == num_optimized
    assert len(optimized_population.routes[num_optimized:]) == num_non_optimized

    for route in optimized_population.routes[:num_optimized]:
        assert set(route.city_indices) == {0, 1, 2, 3}
        assert route.city_indices[0] == 0
        assert len(route.city_indices) == 4

    for route in optimized_population.routes[num_non_optimized:]:
        assert route.city_indices in [r.city_indices for r in population.routes]
        assert len(route.city_indices) == 4


if __name__ == "__main__":
    pytest.main()
