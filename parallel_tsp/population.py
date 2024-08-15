from typing import List

import numpy as np

from parallel_tsp.distance_matrix import DistanceMatrix

from .route import Route, crossover


class Population:
    def __init__(
        self, size: int, distance_matrix: np.ndarray, routes: list[Route] | None = None
    ) -> None:
        self.size = size
        self.distance_matrix = distance_matrix
        self.routes = routes
        if self.routes is None:
            self.routes = [
                Route(self.random_route(len(distance_matrix)), distance_matrix)
                for _ in range(size)
            ]

    def random_route(self, num_cities: int) -> List[int]:
        route = list(range(num_cities))
        np.random.shuffle(route)
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
        tournament = np.random.choice(self.routes, tournament_size)
        return min(tournament, key=lambda route: route.length())

    def get_subset(self, subset_size: int) -> "Population":
        subset_routes = np.random.choice(self.routes, subset_size, replace=False)
        subset_population = Population(subset_size, self.distance_matrix)
        subset_population.routes = list(subset_routes)
        return subset_population

    def serialize(self) -> np.ndarray:
        serialized_data = []
        for route in self.routes:
            serialized_data.extend(route.city_indices)
            serialized_data.append(route.length())
        return np.array(serialized_data, dtype=np.float64)

    @staticmethod
    def deserialize(data: np.ndarray, distance_matrix: np.ndarray) -> "Population":
        route_length = len(distance_matrix)
        population_size = len(data) // (route_length + 1)
        population = Population(population_size, distance_matrix)

        for i in range(population_size):
            start = i * (route_length + 1)
            end = start + route_length
            indices = data[start:end].astype(int).tolist()

            population.routes[i] = Route(indices, distance_matrix)

        return population


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
