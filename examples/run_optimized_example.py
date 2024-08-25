import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from parallel_tsp.distance_matrix import generate_random_distance_matrix
from parallel_tsp.genetic_algorithm import parametrise_genetic_algorithm, select_best
from parallel_tsp.optimisation_strategy import ChristofidesOptimization
from parallel_tsp.population import Population
from parallel_tsp.stop_condition import StopCondition


def main():
    num_cities = 100
    distance_matrix = generate_random_distance_matrix(num_cities)

    population_size = 20
    initial_population = Population(
        size=population_size, distance_matrix=distance_matrix
    )

    stop_condition = StopCondition(
        max_generations=100, improvement_percentage=50, max_time_seconds=10
    )

    print(
        f"Initial best route length: {round(select_best(initial_population.routes).length(), 2)}"
    )

    optimization_strategy = ChristofidesOptimization()
    optimized_population = optimization_strategy.optimize(initial_population)

    print(
        f"Best route length after local optimisation: {round(select_best(optimized_population.routes).length(), 2)}"
    )

    ga_parameters = parametrise_genetic_algorithm(
        mutation_rate=0.05, tournament_size=10, stop_condition=stop_condition
    )

    ga = ga_parameters(population=optimized_population)
    ga.run()

    best_route = ga.best_route
    print(f"Best route length: {round(best_route.length(), 2)}")


if __name__ == "__main__":
    main()
