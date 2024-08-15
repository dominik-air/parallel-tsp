import os
import sys

from mpi4py import MPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parallel_tsp.distance_matrix import generate_random_distance_matrix
from parallel_tsp.genetic_algorithm import parametrise_genetic_algorithm, select_best
from parallel_tsp.mpi_strategy import MPIAllToAllMigration, MPIRingMigration
from parallel_tsp.optimisation_strategy import NoOptimization
from parallel_tsp.population import Population
from parallel_tsp.runner import GeneticAlgorithmRunner


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    distance_matrix = generate_random_distance_matrix(100)
    population = Population(size=100, distance_matrix=distance_matrix)

    mpi_strategy = MPIAllToAllMigration(
        genetic_algorithm=parametrise_genetic_algorithm(
            generations=100, mutation_rate=0.05, tournament_size=10
        ),
        population=population,
        migration_size=25,
        migrations_count=5,
    )

    local_optimization_strategy = NoOptimization()

    runner = GeneticAlgorithmRunner(mpi_strategy, local_optimization_strategy)

    if rank == 0:
        print("Best initial route length:", select_best(population.routes).length())

    best_route = runner.run(comm)

    if rank == 0:
        print("Best route length:", best_route.length())


if __name__ == "__main__":
    main()
