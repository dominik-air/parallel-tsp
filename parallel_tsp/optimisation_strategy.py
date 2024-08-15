from abc import ABC, abstractmethod

from networkx.algorithms.approximation import christofides, greedy_tsp

from .population import Population
from .route import Route


def two_opt(route: Route) -> None:
    return


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
        raise NotImplementedError("todo")


class GreedyTSPOptimization(OptimizationStrategy):
    def optimize(self, population: Population) -> Population:
        raise NotImplementedError("todo")
