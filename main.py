import time
import random
import multiprocessing
from typing import List, Tuple
import numpy as np


class DistanceMatrix:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def distance(self, city1: int, city2: int) -> float:
        return self.matrix[city1][city2]


class Route:
    def __init__(
        self, city_indices: List[int], distance_matrix: DistanceMatrix
    ) -> None:
        self.city_indices = city_indices
        self.distance_matrix = distance_matrix

    def length(self) -> float:
        total_distance = 0
        for i in range(len(self.city_indices)):
            total_distance += self.distance_matrix.distance(
                self.city_indices[i - 1], self.city_indices[i]
            )
        return total_distance

    def mutate(self, mutation_rate: float) -> None:
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(self.city_indices)), 2)
            self.city_indices[idx1], self.city_indices[idx2] = (
                self.city_indices[idx2],
                self.city_indices[idx1],
            )

    @staticmethod
    def crossover(parent1: "Route", parent2: "Route") -> Tuple["Route", "Route"]:
        child1, child2 = Route.crossover_helper(
            parent1, parent2
        ), Route.crossover_helper(parent2, parent1)
        return child1, child2

    @staticmethod
    def crossover_helper(parent1: "Route", parent2: "Route") -> "Route":
        start, end = sorted(random.sample(range(len(parent1.city_indices)), 2))
        child = [None] * len(parent1.city_indices)
        child[start:end] = parent1.city_indices[start:end]
        for city in parent2.city_indices:
            if city not in child:
                for i in range(len(child)):
                    if child[i] is None:
                        child[i] = city
                        break
        return Route(child, parent1.distance_matrix)


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
            child1, child2 = Route.crossover(parent1, parent2)
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            new_routes.extend([child1, child2])
        self.routes = sorted(new_routes, key=lambda route: route.length())[: self.size]

    def select_parent(self, tournament_size: int) -> Route:
        tournament = random.sample(self.routes, tournament_size)
        return min(tournament, key=lambda route: route.length())


class GeneticAlgorithm:
    def __init__(
        self,
        distance_matrix: DistanceMatrix,
        population_size: int,
        generations: int,
        mutation_rate: float,
        tournament_size: int,
    ) -> None:
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def run(self) -> Route:
        population = Population(self.population_size, self.distance_matrix)
        initial_best = self.select_best(population)
        print("Cost:", initial_best.length())
        for _ in range(self.generations):
            population.evolve(self.mutation_rate, self.tournament_size)
        return self.select_best(population)

    @staticmethod
    def select_best(population: Population) -> Route:
        return min(population.routes, key=lambda route: route.length())


class IslandGA:
    def __init__(
        self,
        distance_matrix: DistanceMatrix,
        population_size: int,
        total_generations: int,
        mutation_rate: float,
        tournament_size: int,
    ) -> None:
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.total_generations = total_generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def run_island(self) -> Route:
        island_generations = self.total_generations // multiprocessing.cpu_count()
        population = Population(self.population_size, self.distance_matrix)
        initial_best = self.select_best(population)
        print("Cost:", initial_best.length())
        for _ in range(island_generations):
            population.evolve(self.mutation_rate, self.tournament_size)
        return self.select_best(population)

    @staticmethod
    def select_best(population: Population) -> Route:
        return min(population.routes, key=lambda route: route.length())


def run_sequential_GA():
    num_cities = 500
    matrix = np.random.rand(num_cities, num_cities)
    distance_matrix = DistanceMatrix(matrix)

    population_size = 100
    generations = 200
    mutation_rate = 0.05
    tournament_size = 10

    start = time.perf_counter()
    ga = GeneticAlgorithm(
        distance_matrix, population_size, generations, mutation_rate, tournament_size
    )
    best_route = ga.run()
    end = time.perf_counter()

    print("Cost:", best_route.length())
    print("Time elapsed:", end - start)

def run_island_process(distance_matrix, population_size, total_generations, mutation_rate, tournament_size):
    ga = IslandGA(distance_matrix, population_size, total_generations, mutation_rate, tournament_size)
    return ga.run_island()

def run_island_GA():
    num_cities = 500
    matrix = np.random.rand(num_cities, num_cities)
    distance_matrix = DistanceMatrix(matrix)

    population_size = 100
    total_generations = 200
    mutation_rate = 0.05 
    tournament_size = 10

    start = time.perf_counter()

    with multiprocessing.Pool() as pool:
        args = (distance_matrix, population_size, total_generations, mutation_rate, tournament_size)
        results = pool.starmap(run_island_process, [args for _ in range(multiprocessing.cpu_count())])
        best_route = min(results, key=lambda route: route.length())

    end = time.perf_counter()

    print("Cost:", best_route.length())
    print("Time elapsed:", end-start)


if __name__ == "__main__":
    run_sequential_GA()
