import os
import sys

import pytest
from mpi4py import MPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parallel_tsp.distance_matrix import generate_random_distance_matrix
from parallel_tsp.genetic_algorithm import parametrise_genetic_algorithm, select_best
from parallel_tsp.mpi_strategy import (
    MPIAllToAllMigration,
    MPINoMigration,
    MPIRingMigration,
)
from parallel_tsp.optimisation_strategy import ChristofidesOptimization
from parallel_tsp.population import Population
from parallel_tsp.runner import GeneticAlgorithmRunner
from parallel_tsp.stop_condition import StopCondition


@pytest.mark.mpi(min_size=4)
def test_mpi_all_to_all_migration():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    distance_matrix = generate_random_distance_matrix(100)
    population = Population(size=100, distance_matrix=distance_matrix)

    stop_condition = StopCondition(max_generations=50)

    mpi_strategy = MPIAllToAllMigration(
        genetic_algorithm=parametrise_genetic_algorithm(
            mutation_rate=0.05, tournament_size=10, stop_condition=stop_condition
        ),
        population=population,
        migration_size=25,
        migrations_count=5,
    )

    local_optimization_strategy = ChristofidesOptimization()

    runner = GeneticAlgorithmRunner(mpi_strategy, local_optimization_strategy)

    if rank == 0:
        initial_best_length = select_best(population.routes).length()
        assert (
            initial_best_length > 0
        ), "Initial best route length should be greater than 0"

    best_route = runner.run(comm)

    if rank == 0:
        final_best_length = best_route.length()
        assert final_best_length > 0, "Final best route length should be greater than 0"
        assert (
            final_best_length <= initial_best_length
        ), f"Final route length ({final_best_length}) should be less than or equal to initial length ({initial_best_length})"


@pytest.mark.mpi(min_size=4)
def test_mpi_ring_migration():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    distance_matrix = generate_random_distance_matrix(100)
    population = Population(size=100, distance_matrix=distance_matrix)

    stop_condition = StopCondition(max_generations=50)

    mpi_strategy = MPIRingMigration(
        genetic_algorithm=parametrise_genetic_algorithm(
            mutation_rate=0.05, tournament_size=10, stop_condition=stop_condition
        ),
        population=population,
        migration_size=25,
        migrations_count=5,
    )

    local_optimization_strategy = ChristofidesOptimization()

    runner = GeneticAlgorithmRunner(mpi_strategy, local_optimization_strategy)

    if rank == 0:
        initial_best_length = select_best(population.routes).length()
        assert (
            initial_best_length > 0
        ), "Initial best route length should be greater than 0"

    best_route = runner.run(comm)

    if rank == 0:
        final_best_length = best_route.length()
        assert final_best_length > 0, "Final best route length should be greater than 0"
        assert (
            final_best_length <= initial_best_length
        ), f"Final route length ({final_best_length}) should be less than or equal to initial length ({initial_best_length})"


@pytest.mark.mpi(min_size=4)
def test_mpi_no_migration():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    distance_matrix = generate_random_distance_matrix(100)
    population = Population(size=100, distance_matrix=distance_matrix)

    stop_condition = StopCondition(max_generations=50)

    mpi_strategy = MPINoMigration(
        genetic_algorithm=parametrise_genetic_algorithm(
            mutation_rate=0.05, tournament_size=10, stop_condition=stop_condition
        ),
        population=population,
    )

    local_optimization_strategy = ChristofidesOptimization()

    runner = GeneticAlgorithmRunner(mpi_strategy, local_optimization_strategy)

    if rank == 0:
        initial_best_length = select_best(population.routes).length()
        assert (
            initial_best_length > 0
        ), "Initial best route length should be greater than 0"

    best_route = runner.run(comm)

    if rank == 0:
        final_best_length = best_route.length()
        assert final_best_length > 0, "Final best route length should be greater than 0"
        assert (
            final_best_length <= initial_best_length
        ), f"Final route length ({final_best_length}) should be less than or equal to initial length ({initial_best_length})"
