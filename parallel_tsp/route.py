import random

from .distance_matrix import DistanceMatrix


class Route:
    """Represents a route through a set of cities, defined by a sequence of city indices.

    Attributes:
        city_indices (list[int]): The sequence of city indices that define the route.
        distance_matrix (DistanceMatrix): The matrix containing distances between all pairs of cities.
    """

    def __init__(
        self, city_indices: list[int], distance_matrix: DistanceMatrix
    ) -> None:
        """Initializes a route with a specific sequence of city indices and a distance matrix.

        Args:
            city_indices (list[int]): The sequence of city indices that define the route.
            distance_matrix (DistanceMatrix): The matrix containing distances between all pairs of cities.
        """
        self.city_indices = city_indices
        self.distance_matrix = distance_matrix

    def length(self) -> float:
        """Calculates the total length of the route.

        The length is calculated by summing the distances between consecutive cities in the route.

        Returns:
            float: The total distance of the route.
        """
        total_distance = 0.0
        for i in range(len(self.city_indices)):
            total_distance += self.distance_matrix.distance(
                self.city_indices[i - 1], self.city_indices[i]
            )
        return total_distance

    def mutate(self, mutation_rate: float) -> None:
        """Mutates the route by swapping two cities with a given mutation probability.

        Args:
            mutation_rate (float): The probability of performing a mutation.
        """
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(self.city_indices)), 2)
            self.city_indices[idx1], self.city_indices[idx2] = (
                self.city_indices[idx2],
                self.city_indices[idx1],
            )


def crossover(parent1: Route, parent2: Route) -> tuple[Route, Route]:
    """Performs crossover between two parent routes to produce two child routes.

    The crossover combines segments from each parent route to create new routes.

    Args:
        parent1 (Route): The first parent route.
        parent2 (Route): The second parent route.

    Returns:
        tuple[Route, Route]: A tuple containing the two child routes produced by the crossover.
    """

    def crossover_helper(parent1: Route, parent2: Route) -> Route:
        """Helper function to perform crossover on two parent routes.

        This function selects a random segment from the first parent and fills the remaining cities
        with those from the second parent, maintaining the order of appearance.

        Args:
            parent1 (Route): The first parent route.
            parent2 (Route): The second parent route.

        Returns:
            Route: The resulting child route after crossover.
        """
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
