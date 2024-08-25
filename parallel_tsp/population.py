import numpy as np

from parallel_tsp.distance_matrix import DistanceMatrix

from .route import Route, crossover


class Population:
    """Represents a population of routes in a genetic algorithm.

    Attributes:
        size (int): The number of routes in the population.
        distance_matrix (np.ndarray): The distance matrix representing distances between cities.
        routes (list[Route]): A list of routes that make up the population.
    """

    def __init__(
        self, size: int, distance_matrix: np.ndarray, routes: list[Route] | None = None
    ) -> None:
        """Initializes a population with a given size and distance matrix.

        If no routes are provided, a random population is generated.

        Args:
            size (int): The size of the population.
            distance_matrix (np.ndarray): The distance matrix for the cities.
            routes (list[Route] | None): A list of routes. If None, a random population is generated.
        """
        self.size = size
        self.distance_matrix = distance_matrix
        self.routes = routes
        if self.routes is None:
            self.routes = [
                Route(self.random_route(len(distance_matrix)), distance_matrix)
                for _ in range(size)
            ]

    def random_route(self, num_cities: int) -> list[int]:
        """Generates a random route by shuffling the order of cities.

        Args:
            num_cities (int): The number of cities in the route.

        Returns:
            list[int]: A randomly ordered list of city indices.
        """
        route = list(range(num_cities))
        np.random.shuffle(route)
        return route

    def evolve(self, mutation_rate: float, tournament_size: int) -> None:
        """Evolves the population by selecting parents, performing crossover, and mutating offspring.

        The new population replaces the old one, keeping only the best routes.

        Args:
            mutation_rate (float): The probability of mutating a route.
            tournament_size (int): The number of routes competing in the selection tournament.
        """
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
        """Selects a parent route using tournament selection.

        Args:
            tournament_size (int): The number of routes competing in the selection tournament.

        Returns:
            Route: The selected parent route.
        """
        tournament = np.random.choice(self.routes, tournament_size)
        return min(tournament, key=lambda route: route.length())

    def get_subset(self, subset_size: int) -> "Population":
        """Generates a subset of the population.

        Args:
            subset_size (int): The size of the subset population.

        Returns:
            Population: A new population consisting of the subset of routes.
        """
        subset_routes = np.random.choice(self.routes, subset_size, replace=False)
        subset_population = Population(subset_size, self.distance_matrix)
        subset_population.routes = list(subset_routes)
        return subset_population

    def serialize(self) -> np.ndarray:
        """Serializes the population into a numpy array.

        The array contains the city indices of each route followed by its length.

        Returns:
            np.ndarray: The serialized population.
        """
        serialized_data = []
        for route in self.routes:
            serialized_data.extend(route.city_indices)
            serialized_data.append(route.length())
        return np.array(serialized_data, dtype=np.float64)

    @staticmethod
    def deserialize(data: np.ndarray, distance_matrix: np.ndarray) -> "Population":
        """Deserializes a numpy array into a Population object.

        Args:
            data (np.ndarray): The serialized population data.
            distance_matrix (np.ndarray): The distance matrix for the cities.

        Returns:
            Population: The deserialized population.
        """
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
) -> list[Population]:
    """Generates multiple populations.

    Args:
        num_populations (int): The number of populations to generate.
        population_size (int): The size of each population.
        distance_matrix (DistanceMatrix): The distance matrix for the cities.

    Returns:
        list[Population]: A list of generated populations.
    """
    populations = []
    for _ in range(num_populations):
        population = Population(population_size, distance_matrix)
        populations.append(population)
    return populations


def combine_populations(population1: Population, population2: Population) -> Population:
    """Combines two populations into a single population.

    The combined population retains the best routes from both populations.

    Args:
        population1 (Population): The first population.
        population2 (Population): The second population.

    Returns:
        Population: The combined population with the best routes from both.
    """
    combined_routes = population1.routes + population2.routes
    combined_routes = sorted(combined_routes, key=lambda route: route.length())[
        : population1.size
    ]
    return Population(population1.size, population1.routes[0].distance_matrix)
