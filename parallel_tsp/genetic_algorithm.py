import logging
from functools import partial
from typing import Iterable

from .population import Population
from .route import Route
from .stop_condition import StopCondition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """Represents a genetic algorithm for solving the Traveling Salesman Problem (TSP).

    Attributes:
        population (Population): The population of routes to evolve.
        mutation_rate (float): The probability of mutation in each generation.
        tournament_size (int): The number of routes selected for the tournament in each generation.
        stop_condition (StopCondition): The condition that determines when the algorithm should stop.
        initial_best_route (Route): The best route found at the start of the algorithm.
        best_route (Route): The best route found so far.
        generations_run (int): The number of generations that have been run.
    """

    def __init__(
        self,
        population: Population,
        mutation_rate: float,
        tournament_size: int,
        stop_condition: StopCondition,
    ) -> None:
        """Initializes the GeneticAlgorithm with the given parameters.

        Args:
            population (Population): The population of routes to evolve.
            mutation_rate (float): The probability of mutation in each generation.
            tournament_size (int): The number of routes selected for the tournament in each generation.
            stop_condition (StopCondition): The condition that determines when the algorithm should stop.
        """
        self.population = population
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.initial_best_route: Route = select_best(self.population.routes)
        self.best_route: Route = self.initial_best_route
        self.generations_run = 0
        self.stop_condition = stop_condition
        self.stop_condition.update_initial_best_length(self.initial_best_route.length())

        if self.stop_condition.max_time_seconds is not None:
            self.stop_condition.start_timer()

        logger.info(
            f"Initialized GeneticAlgorithm with population size {population.size}, "
            f"mutation rate {mutation_rate}, tournament size {tournament_size}, "
            f"initial best route length {self.initial_best_route.length()}"
        )

    def run_iteration(self) -> None:
        """Runs a single iteration of the genetic algorithm.

        During each iteration, the population evolves by selecting, mutating, and
        recombining routes, and the best route is updated.
        """
        logger.debug(f"Running iteration {self.generations_run + 1}")
        self.population.evolve(self.mutation_rate, self.tournament_size)
        current_best = select_best(self.population.routes)
        if current_best.length() < self.best_route.length():
            logger.info(
                f"New best route found: {current_best.length()} (previous best: {self.best_route.length()})"
            )
            self.best_route = current_best
        self.generations_run += 1

    def run(self) -> None:
        """Runs the genetic algorithm until the stop condition is met.

        The algorithm continuously evolves the population until the stopping condition
        is satisfied, updating the best route found so far.
        """
        logger.info("Starting the genetic algorithm run")
        while not self.stop_condition.should_stop(
            generations_run=self.generations_run,
            current_best_length=self.best_route.length(),
        ):
            self.run_iteration()

        logger.info(
            f"Stopping the genetic algorithm after {self.generations_run} generations. "
            f"Best route length: {self.best_route.length()}"
        )


def select_best(routes: Iterable[Route]) -> Route:
    """Selects the best route from a given set of routes.

    Args:
        routes (Iterable[Route]): An iterable of routes to select from.

    Returns:
        Route: The route with the shortest length.
    """
    return min(routes, key=lambda route: route.length())


ParametrisedGeneticAlgorithm = partial[GeneticAlgorithm]


def parametrise_genetic_algorithm(
    mutation_rate: float, tournament_size: int, stop_condition: StopCondition
) -> ParametrisedGeneticAlgorithm:
    """Creates a partially parametrized GeneticAlgorithm class.

    Args:
        mutation_rate (float): The probability of mutation in each generation.
        tournament_size (int): The number of routes selected for the tournament in each generation.
        stop_condition (StopCondition): The condition that determines when the algorithm should stop.

    Returns:
        ParametrisedGeneticAlgorithm: A partial object of GeneticAlgorithm with the specified parameters.

    Raises:
        ValueError: If the tournament_size is less than 0.
        ValueError: If the mutation_rate is not between 0 and 1.
    """
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
