from abc import ABC, abstractmethod

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


class TwoOptOptimization(OptimizationStrategy):
    def optimize(self, population: Population) -> Population:
        for route in population.routes:
            two_opt(route)
        return population
