import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parallel_tsp.distance_matrix import generate_random_distance_matrix
from parallel_tsp.genetic_algorithm import parametrise_genetic_algorithm, select_best
from parallel_tsp.optimisation_strategy import (
    ChristofidesOptimization,
    GreedyTSPOptimization,
    NoOptimization,
)
from parallel_tsp.population import Population
from parallel_tsp.stop_condition import StopCondition


@pytest.mark.functional
def test_genetic_algorithm_with_christofides_optimization():
    num_cities = 100
    population_size = 20

    distance_matrix = generate_random_distance_matrix(num_cities)

    initial_population = Population(
        size=population_size, distance_matrix=distance_matrix
    )

    initial_best_route = select_best(initial_population.routes)
    initial_best_length = initial_best_route.length()

    assert initial_best_length > 0, "Initial best route length should be greater than 0"

    optimization_strategy = ChristofidesOptimization()
    optimized_population = optimization_strategy.optimize(initial_population)

    optimized_best_route = select_best(optimized_population.routes)
    optimized_best_length = optimized_best_route.length()

    assert (
        optimized_best_length > 0
    ), "Optimized best route length should be greater than 0"
    assert (
        optimized_best_length <= initial_best_length
    ), f"Optimized route length ({optimized_best_length}) should be less than or equal to initial length ({initial_best_length})"

    stop_condition = StopCondition(max_generations=20)

    ga_parameters = parametrise_genetic_algorithm(
        mutation_rate=0.05, tournament_size=5, stop_condition=stop_condition
    )

    ga = ga_parameters(population=optimized_population)
    ga.run()

    final_best_route = ga.best_route
    final_best_length = final_best_route.length()

    assert final_best_length > 0, "Final best route length should be greater than 0"
    assert (
        final_best_length <= optimized_best_length
    ), f"Final route length ({final_best_length}) should be less than or equal to optimized length ({optimized_best_length})"


@pytest.mark.functional
def test_genetic_algorithm_with_greedy_optimization():
    num_cities = 100
    population_size = 20

    distance_matrix = generate_random_distance_matrix(num_cities)

    initial_population = Population(
        size=population_size, distance_matrix=distance_matrix
    )

    initial_best_route = select_best(initial_population.routes)
    initial_best_length = initial_best_route.length()

    assert initial_best_length > 0, "Initial best route length should be greater than 0"

    optimization_strategy = GreedyTSPOptimization()
    optimized_population = optimization_strategy.optimize(initial_population)

    optimized_best_route = select_best(optimized_population.routes)
    optimized_best_length = optimized_best_route.length()

    assert (
        optimized_best_length > 0
    ), "Optimized best route length should be greater than 0"
    assert (
        optimized_best_length <= initial_best_length
    ), f"Optimized route length ({optimized_best_length}) should be less than or equal to initial length ({initial_best_length})"

    stop_condition = StopCondition(max_generations=20)

    ga_parameters = parametrise_genetic_algorithm(
        mutation_rate=0.05, tournament_size=5, stop_condition=stop_condition
    )

    ga = ga_parameters(population=optimized_population)
    ga.run()

    final_best_route = ga.best_route
    final_best_length = final_best_route.length()

    assert final_best_length > 0, "Final best route length should be greater than 0"
    assert (
        final_best_length <= optimized_best_length
    ), f"Final route length ({final_best_length}) should be less than or equal to optimized length ({optimized_best_length})"


@pytest.mark.functional
def test_genetic_algorithm_with_no_optimization():
    num_cities = 100
    population_size = 20

    distance_matrix = generate_random_distance_matrix(num_cities)

    initial_population = Population(
        size=population_size, distance_matrix=distance_matrix
    )

    initial_best_route = select_best(initial_population.routes)
    initial_best_length = initial_best_route.length()

    assert initial_best_length > 0, "Initial best route length should be greater than 0"

    optimization_strategy = NoOptimization()
    optimized_population = optimization_strategy.optimize(initial_population)

    optimized_best_route = select_best(optimized_population.routes)
    optimized_best_length = optimized_best_route.length()

    assert (
        optimized_best_length > 0
    ), "Optimized best route length should be greater than 0"
    assert (
        optimized_best_length == initial_best_length
    ), f"Optimized route length ({optimized_best_length}) should be equal to initial length ({initial_best_length})"

    stop_condition = StopCondition(max_generations=20)

    ga_parameters = parametrise_genetic_algorithm(
        mutation_rate=0.05, tournament_size=5, stop_condition=stop_condition
    )

    ga = ga_parameters(population=optimized_population)
    ga.run()

    final_best_route = ga.best_route
    final_best_length = final_best_route.length()

    assert final_best_length > 0, "Final best route length should be greater than 0"
    assert (
        final_best_length <= optimized_best_length
    ), f"Final route length ({final_best_length}) should be less than or equal to optimized length ({optimized_best_length})"
