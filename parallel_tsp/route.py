import random
from typing import List, Tuple

from .distance_matrix import DistanceMatrix


class Route:
    def __init__(
        self, city_indices: List[int], distance_matrix: DistanceMatrix
    ) -> None:
        self.city_indices = city_indices
        self.distance_matrix = distance_matrix

    def length(self) -> float:
        total_distance = 0.0
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


def crossover(parent1: Route, parent2: Route) -> Tuple[Route, Route]:
    def crossover_helper(parent1: Route, parent2: Route) -> Route:
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

    child1 = crossover_helper(parent1, parent2)
    child2 = crossover_helper(parent2, parent1)
    return child1, child2
