import os
import sys

from mpi4py import MPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parallel_tsp.distance_matrix import generate_random_distance_matrix
from parallel_tsp.genetic_algorithm import parametrise_genetic_algorithm, select_best
from parallel_tsp.mpi_strategy import MPIAllToAllMigration
from parallel_tsp.optimisation_strategy import ChristofidesOptimization
from parallel_tsp.population import Population
from parallel_tsp.runner import GeneticAlgorithmRunner
from parallel_tsp.stop_condition import StopCondition


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    distance_matrix = generate_random_distance_matrix(100)
    population = Population(size=100, distance_matrix=distance_matrix)

    stop_condition = StopCondition(
        max_generations=100, improvement_percentage=50, max_time_seconds=1
    )

    mpi_strategy = MPIAllToAllMigration(
        genetic_algorithm=parametrise_genetic_algorithm(
            mutation_rate=0.05, tournament_size=10, stop_condition=stop_condition
        ),
        population=population,
        migration_size=25,
        generations_per_migration=5,
    )

    local_optimization_strategy = ChristofidesOptimization()

    runner = GeneticAlgorithmRunner(mpi_strategy, local_optimization_strategy)

    if rank == 0:
        initial_best_length = select_best(population.routes).length()
        stop_condition.update_initial_best_length(initial_best_length)

    best_route = runner.run(comm)

    if rank == 0:
        final_best_length = best_route.length()


if __name__ == "__main__":
    main()
