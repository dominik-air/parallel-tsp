import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parallel_tsp.distance_matrix import generate_random_distance_matrix
from parallel_tsp.genetic_algorithm import GeneticAlgorithm
from parallel_tsp.population import combine_populations, generate_populations


@pytest.mark.functional
def test_genetic_algorithm_functionality():
    num_cities = 100
    population_size = 100
    generations = 20
    mutation_rate = 0.05
    tournament_size = 10

    distance_matrix = generate_random_distance_matrix(num_cities)

    populations = generate_populations(2, population_size, distance_matrix)
    combined_population = combine_populations(populations[0], populations[1])

    ga = GeneticAlgorithm(
        combined_population,
        generations,
        mutation_rate,
        tournament_size,
    )

    initial_best_route_length = ga.initial_best_route.length()
    assert (
        initial_best_route_length > 0
    ), "Initial route length should be greater than 0"

    ga.run_iterations(generations)

    final_best_route_length = ga.best_route.length()
    assert final_best_route_length > 0, "Final route length should be greater than 0"
    assert final_best_route_length <= initial_best_route_length, (
        f"Final route length ({final_best_route_length}) should be less than or equal "
        f"to initial route length ({initial_best_route_length})"
    )
