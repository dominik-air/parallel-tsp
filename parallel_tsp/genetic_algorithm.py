from .population import Population
from .route import Route


class GeneticAlgorithm:
    def __init__(
        self,
        population: Population,
        generations: int,
        mutation_rate: float,
        tournament_size: int,
    ) -> None:
        self.population = population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def run(self) -> Route:
        initial_best = self.select_best(self.population)
        for _ in range(self.generations):
            self.population.evolve(self.mutation_rate, self.tournament_size)
        return self.select_best(self.population), initial_best.length()

    @staticmethod
    def select_best(population: Population) -> Route:
        return min(population.routes, key=lambda route: route.length())
