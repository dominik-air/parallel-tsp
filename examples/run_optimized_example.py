import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from parallel_tsp.distance_matrix import generate_random_distance_matrix
from parallel_tsp.genetic_algorithm import parametrise_genetic_algorithm, select_best
from parallel_tsp.optimisation_strategy import ChristofidesOptimization
from parallel_tsp.population import Population


def main():
    num_cities = 100
    distance_matrix = generate_random_distance_matrix(num_cities)

    population_size = 20
    initial_population = Population(
        size=population_size, distance_matrix=distance_matrix
    )
    print("Best initial route:", select_best(initial_population.routes).city_indices)
    print(
        "Length of the best initial route:",
        select_best(initial_population.routes).length(),
    )

    optimization_strategy = ChristofidesOptimization()
    optimized_population = optimization_strategy.optimize(initial_population)

    print(
        "Best route after local optimisation:",
        select_best(optimized_population.routes).city_indices,
    )
    print(
        "Length of the best route after local optimisation:",
        select_best(optimized_population.routes).length(),
    )

    ga_parameters = parametrise_genetic_algorithm(
        generations=100,
        mutation_rate=0.05,
        tournament_size=10,
    )

    ga = ga_parameters(population=optimized_population)
    ga.run_iterations(ga.generations)

    best_route = ga.best_route
    print("Best route found with GA:", best_route.city_indices)
    print("Length of the best route:", best_route.length())


if __name__ == "__main__":
    main()
