import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parallel_tsp.distance_matrix import generate_random_distance_matrix
from parallel_tsp.genetic_algorithm import (
    GeneticAlgorithm,
    parametrise_genetic_algorithm,
)
from parallel_tsp.population import combine_populations, generate_populations
from parallel_tsp.stop_condition import StopCondition


def main():
    num_cities = 100
    distance_matrix = generate_random_distance_matrix(num_cities)

    population_size = 100
    mutation_rate = 0.05
    tournament_size = 10

    stop_condition = StopCondition(
        max_generations=100, improvement_percentage=50, max_time_seconds=10
    )

    populations = generate_populations(2, population_size, distance_matrix)
    combined_population = combine_populations(populations[0], populations[1])

    ga_parameters = parametrise_genetic_algorithm(
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        stop_condition=stop_condition,
    )

    ga = ga_parameters(population=combined_population)

    print(f"Initial best route length: {round(ga.initial_best_route.length(), 2)}")

    ga.run()
    stop_condition = ga.stop_condition.get_triggered_condition()
    print(f"Algorithm stopped due to: {stop_condition}")

    print(f"Best route length: {round(ga.best_route.length(), 2)}")


if __name__ == "__main__":
    main()
