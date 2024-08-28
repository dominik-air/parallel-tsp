import logging
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from networkx.algorithms.approximation import christofides, greedy_tsp

from .population import Population, Route
from .route import Route

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def population_to_graph(population: Population) -> tuple[nx.Graph, np.ndarray]:
    """Converts a population into a NetworkX graph representation.

    Each city in the population is represented as a node in the graph, and the
    edges between nodes are weighted by the distances from the distance matrix.

    Args:
        population (Population): The population to be converted.

    Returns:
        tuple[nx.Graph, np.ndarray]: A tuple containing the generated graph and
        the original distance matrix.
    """
    logger.debug("Converting population to graph")
    distance_matrix = population.distance_matrix.matrix
    G = nx.Graph()

    num_cities = len(distance_matrix)

    G.add_nodes_from(range(num_cities))

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            G.add_edge(i, j, weight=distance_matrix[i][j])

    logger.debug(f"Graph created with {num_cities} nodes and {G.size()} edges")
    return G, distance_matrix


def routes_to_population(routes: list, distance_matrix: np.ndarray) -> Population:
    """Converts a list of routes into a Population object.

    Args:
        routes (list): A list of routes where each route is a list of city indices.
        distance_matrix (np.ndarray): The distance matrix associated with the routes.

    Returns:
        Population: The generated Population object.
    """
    logger.debug(f"Converting {len(routes)} routes to Population object")
    population_size = len(routes)
    route_objects = [Route(route, distance_matrix) for route in routes]
    logger.info(f"Converted {population_size} routes into Population object")
    return Population(
        size=population_size, distance_matrix=distance_matrix, routes=route_objects
    )


class OptimizationStrategy(ABC):
    """Abstract base class for all optimization strategies."""

    @abstractmethod
    def optimize(self, population: Population) -> Population:
        """Apply local optimization to the population.

        Args:
            population (Population): The population to be optimized.

        Returns:
            Population: The optimized population.
        """
        pass


class NoOptimization(OptimizationStrategy):
    """A strategy that performs no optimization."""

    def optimize(self, population: Population) -> Population:
        """Returns the population without any modifications.

        Args:
            population (Population): The population to be returned as is.

        Returns:
            Population: The unmodified population.
        """
        logger.info("Applying NoOptimization strategy")
        return population


class ChristofidesOptimization(OptimizationStrategy):
    """Optimization strategy using the Christofides algorithm."""

    def optimize(self, population: Population) -> Population:
        """Applies the Christofides algorithm to optimize the population.

        Args:
            population (Population): The population to be optimized.

        Returns:
            Population: The optimized population with Christofides' routes.
        """
        logger.info("Applying ChristofidesOptimization strategy")
        G, _ = population_to_graph(population)
        optimized_routes = [christofides(G)[:-1]] * population.size
        logger.info("Christofides optimization complete")
        return routes_to_population(optimized_routes, population.distance_matrix)


class GreedyTSPOptimization(OptimizationStrategy):
    """Optimization strategy using a greedy TSP algorithm."""

    def optimize(self, population: Population) -> Population:
        """Applies a greedy TSP algorithm to optimize the population.

        Args:
            population (Population): The population to be optimized.

        Returns:
            Population: The optimized population with greedy TSP routes.
        """
        logger.info("Applying GreedyTSPOptimization strategy")
        G, _ = population_to_graph(population)
        optimized_routes = [greedy_tsp(G)[:-1]] * population.size
        logger.info("Greedy TSP optimization complete")
        return routes_to_population(optimized_routes, population.distance_matrix)


class MixedOptimization(OptimizationStrategy):
    """Optimization strategy that combines base optimization with original population.

    Attributes:
        base_optimization (OptimizationStrategy): The base optimization strategy to be applied.
        mix_ratio (float): The ratio of optimized routes to retain in the population.
    """

    def __init__(self, base_optimization: OptimizationStrategy, mix_ratio: float):
        """Initializes the MixedOptimization strategy.

        Args:
            base_optimization (OptimizationStrategy): The base optimization strategy.
            mix_ratio (float): The ratio of optimized routes to retain.
        """
        self.base_optimization = base_optimization
        self.mix_ratio = mix_ratio
        logger.info(f"Initialized MixedOptimization with mix_ratio {mix_ratio}")

    def optimize(self, population: Population) -> Population:
        """Applies the base optimization strategy and mixes the results with the original population.

        Args:
            population (Population): The population to be optimized.

        Returns:
            Population: The mixed population containing both optimized and non-optimized routes.
        """
        logger.info("Applying MixedOptimization strategy")
        optimized_population = self.base_optimization.optimize(population)

        num_optimized = int(self.mix_ratio * population.size)
        num_non_optimized = population.size - num_optimized

        logger.debug(
            f"Mixing {num_optimized} optimized and {num_non_optimized} non-optimized routes"
        )

        optimized_routes = optimized_population.routes[:num_optimized]
        non_optimized_routes = population.routes[:num_non_optimized]

        mixed_routes = optimized_routes + non_optimized_routes
        logger.info("Mixed optimization complete")
        return Population(
            size=population.size,
            distance_matrix=population.distance_matrix,
            routes=mixed_routes,
        )
