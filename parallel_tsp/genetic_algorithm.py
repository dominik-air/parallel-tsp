from functools import partial
from typing import Iterable, Optional

from .population import Population
from .route import Route
from .stop_condition import StopCondition


class GeneticAlgorithm:
    def __init__(
        self,
        population: Population,
        mutation_rate: float,
        tournament_size: int,
        stop_condition: StopCondition,
    ) -> None:
        self.population = population
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.initial_best_route: Route = select_best(self.population.routes)
        self.best_route: Route = self.initial_best_route
        self.generations_run = 0
        self.stop_condition = stop_condition
        self.stop_condition.update_initial_best_length(self.initial_best_route.length())

    def run_iteration(self) -> None:
        """Runs a single iteration of the genetic algorithm."""
        self.population.evolve(self.mutation_rate, self.tournament_size)
        current_best = select_best(self.population.routes)
        if current_best.length() < self.best_route.length():
            self.best_route = current_best
        self.generations_run += 1

    def run(self) -> None:
        """Runs the genetic algorithm until the stop condition is met."""
        while not self.stop_condition.should_stop(
            generations_run=self.generations_run,
            current_best_length=self.best_route.length(),
        ):
            self.run_iteration()


def select_best(routes: Iterable[Route]) -> Route:
    return min(routes, key=lambda route: route.length())


ParametrisedGeneticAlgorithm = partial[GeneticAlgorithm]


def parametrise_genetic_algorithm(
    mutation_rate: float, tournament_size: int, stop_condition: StopCondition
) -> ParametrisedGeneticAlgorithm:

    if tournament_size < 0:
        raise ValueError("'tournament_size' should be greater than 0.")

    if mutation_rate > 1.0 or mutation_rate < 0:
        raise ValueError("'mutation_rate' should be between 0 and 1.")

    return partial(
        GeneticAlgorithm,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        stop_condition=stop_condition,
    )
