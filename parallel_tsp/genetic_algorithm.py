from functools import partial
from typing import Iterable, Optional

from .population import Population
from .route import Route


class GeneticAlgorithm:
    def __init__(
        self,
        population: Population,
        mutation_rate: float,
        tournament_size: int,
    ) -> None:
        self.population = population
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.initial_best_route: Route = select_best(self.population.routes)
        self.best_route: Optional[Route] = None

    def run_iteration(self) -> None:
        """Runs a single iteration of the genetic algorithm."""
        self.population.evolve(self.mutation_rate, self.tournament_size)
        current_best = select_best(self.population.routes)
        if self.best_route is None or current_best.length() < self.best_route.length():
            self.best_route = current_best

    def run_iterations(self, num_iterations: int) -> None:
        """Runs the genetic algorithm for a given number of iterations."""
        for _ in range(num_iterations):
            self.run_iteration()


def select_best(routes: Iterable[Route]) -> Route:
    return min(routes, key=lambda route: route.length())


ParametrisedGeneticAlgorithm = partial[GeneticAlgorithm]


def parametrise_genetic_algorithm(
    mutation_rate: float, tournament_size: int
) -> ParametrisedGeneticAlgorithm:

    if mutation_rate > 1.0 or mutation_rate < 0:
        raise ValueError("'mutation_rate' should be between 0 and 1.")

    return partial(
        GeneticAlgorithm,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
    )
