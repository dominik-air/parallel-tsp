import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parallel_tsp.distance_matrix import generate_random_distance_matrix
from parallel_tsp.genetic_algorithm import GeneticAlgorithm
from parallel_tsp.population import combine_populations, generate_populations


def main():
    num_cities = 500
    distance_matrix = generate_random_distance_matrix(num_cities)

    population_size = 100
    generations = 20
    mutation_rate = 0.05
    tournament_size = 10

    populations = generate_populations(2, population_size, distance_matrix)
    combined_population = combine_populations(populations[0], populations[1])

    ga = GeneticAlgorithm(
        combined_population,
        generations,
        mutation_rate,
        tournament_size,
    )
    best_route, initial_cost = ga.run()

    print("Initial Cost:", initial_cost)
    print("Best route length:", best_route.length())


if __name__ == "__main__":
    main()
