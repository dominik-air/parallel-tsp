from abc import ABC, abstractmethod
from typing import Tuple

import networkx as nx
import numpy as np
from networkx.algorithms.approximation import christofides, greedy_tsp

from .population import Population, Route
from .route import Route


def population_to_graph(population: Population) -> Tuple[nx.Graph, np.ndarray]:
    distance_matrix = population.distance_matrix.matrix
    G = nx.Graph()

    num_cities = len(distance_matrix)

    G.add_nodes_from(range(num_cities))

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            G.add_edge(i, j, weight=distance_matrix[i][j])

    return G, distance_matrix


def routes_to_population(routes: list, distance_matrix: np.ndarray) -> Population:
    population_size = len(routes)
    route_objects = [Route(route, distance_matrix) for route in routes]
    return Population(
        size=population_size, distance_matrix=distance_matrix, routes=route_objects
    )


class OptimizationStrategy(ABC):
    @abstractmethod
    def optimize(self, population: Population) -> Population:
        """Apply local optimization to the population."""
        pass


class NoOptimization(OptimizationStrategy):
    def optimize(self, population: Population) -> Population:
        return population


class ChristofidesOptimization(OptimizationStrategy):
    def optimize(self, population: Population) -> Population:
        G, _ = population_to_graph(population)
        optimized_routes = [christofides(G)[:-1]] * population.size
        return routes_to_population(optimized_routes, population.distance_matrix)


class GreedyTSPOptimization(OptimizationStrategy):
    def optimize(self, population: Population) -> Population:
        G, _ = population_to_graph(population)
        optimized_routes = [greedy_tsp(G)[:-1]] * population.size
        return routes_to_population(optimized_routes, population.distance_matrix)


class MixedOptimization(OptimizationStrategy):
    def __init__(self, base_optimization: OptimizationStrategy, mix_ratio: float):
        self.base_optimization = base_optimization
        self.mix_ratio = mix_ratio

    def optimize(self, population: Population) -> Population:
        optimized_population = self.base_optimization.optimize(population)

        num_optimized = int(self.mix_ratio * population.size)
        num_non_optimized = population.size - num_optimized

        optimized_routes = optimized_population.routes[:num_optimized]
        non_optimized_routes = population.routes[:num_non_optimized]

        mixed_routes = optimized_routes + non_optimized_routes
        return Population(
            size=population.size,
            distance_matrix=population.distance_matrix,
            routes=mixed_routes,
        )
