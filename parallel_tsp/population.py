import random
from typing import List

from .distance_matrix import DistanceMatrix
from .route import Route, crossover


class Population:
    def __init__(self, size: int, distance_matrix: DistanceMatrix) -> None:
        self.size = size
        self.routes = [
            Route(self.random_route(len(distance_matrix.matrix)), distance_matrix)
            for _ in range(size)
        ]

    def random_route(self, num_cities: int) -> List[int]:
        route = list(range(num_cities))
        random.shuffle(route)
        return route

    def evolve(self, mutation_rate: float, tournament_size: int) -> None:
        new_routes = []
        for _ in range(self.size):
            parent1 = self.select_parent(tournament_size)
            parent2 = self.select_parent(tournament_size)
            child1, child2 = crossover(parent1, parent2)
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            new_routes.extend([child1, child2])
        self.routes = sorted(new_routes, key=lambda route: route.length())[: self.size]

    def select_parent(self, tournament_size: int) -> Route:
        tournament = random.sample(self.routes, tournament_size)
        return min(tournament, key=lambda route: route.length())

    def get_subset(self, subset_size: int) -> "Population":
        if subset_size > self.size:
            raise ValueError("Subset size cannot be larger than the population size.")
        subset_routes = random.sample(self.routes, subset_size)
        new_population = Population(subset_size, self.routes[0].distance_matrix)
        new_population.routes = subset_routes
        return new_population


def generate_populations(
    num_populations: int, population_size: int, distance_matrix: DistanceMatrix
) -> List[Population]:
    populations = []
    for _ in range(num_populations):
        population = Population(population_size, distance_matrix)
        populations.append(population)
    return populations


def combine_populations(population1: Population, population2: Population) -> Population:
    combined_routes = population1.routes + population2.routes
    combined_routes = sorted(combined_routes, key=lambda route: route.length())[
        : population1.size
    ]
    return Population(population1.size, population1.routes[0].distance_matrix)
