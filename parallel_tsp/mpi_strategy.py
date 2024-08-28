from abc import ABC, abstractmethod

import numpy as np
from mpi4py import MPI

from .genetic_algorithm import ParametrisedGeneticAlgorithm
from .population import Population, combine_populations


class MPIStrategy(ABC):
    """Abstract base class for MPI strategies in a parallel genetic algorithm.

    Attributes:
        genetic_algorithm_partial (ParametrisedGeneticAlgorithm): A partial object of the genetic algorithm.
        population (Population): The population to be evolved.
    """

    def __init__(
        self, genetic_algorithm: ParametrisedGeneticAlgorithm, population: Population
    ):
        """Initializes the MPIStrategy with the given genetic algorithm and population.

        Args:
            genetic_algorithm (ParametrisedGeneticAlgorithm): The genetic algorithm to be used in the strategy.
            population (Population): The population to be evolved.
        """
        self.genetic_algorithm_partial = genetic_algorithm
        self.population = population

    @abstractmethod
    def run(self, comm: MPI.Comm):
        """Run the evolutionary algorithm using the defined MPI strategy.

        Args:
            comm (MPI.Comm): The MPI communicator to be used for communication between processes.
        """
        pass


class MPINoMigration(MPIStrategy):
    """MPI strategy where no migration occurs between different processes."""

    def run(self, comm: MPI.Comm):
        """Runs the genetic algorithm on each process without any migration.

        Args:
            comm (MPI.Comm): The MPI communicator used for communication between processes.

        Returns:
            Route: The best route found across all processes.
        """
        rank = comm.Get_rank()

        ga = self.genetic_algorithm_partial(population=self.population)

        ga.run()
        best_route = ga.best_route
        all_best_routes = comm.gather(best_route, root=0)

        if rank == 0:
            best_route_overall = min(all_best_routes, key=lambda route: route.length())
            return best_route_overall


class MPIRingMigration(MPIStrategy):
    """MPI strategy where migration occurs in a ring topology between processes.

    Attributes:
        migration_size (int): The number of individuals to migrate between processes.
        generations_per_migration (int): The number of generations to run before performing a migration.
    """

    def __init__(
        self,
        genetic_algorithm: ParametrisedGeneticAlgorithm,
        population: Population,
        migration_size: int,
        generations_per_migration: int,
    ):
        """Initializes the MPIRingMigration strategy with the given parameters.

        Args:
            genetic_algorithm (ParametrisedGeneticAlgorithm): The genetic algorithm to be used in the strategy.
            population (Population): The population to be evolved.
            migration_size (int): The number of individuals to migrate between processes.
            generations_per_migration (int): The number of generations to run before performing a migration.
        """
        super().__init__(genetic_algorithm, population)
        self.migration_size = migration_size
        self.generations_per_migration = generations_per_migration

    def run(self, comm: MPI.Comm):
        """Runs the genetic algorithm with ring migration between processes.

        In each migration, a subset of the population is sent to the next process
        and received from the previous process in the ring topology.

        Args:
            comm (MPI.Comm): The MPI communicator used for communication between processes.

        Returns:
            Route: The best route found across all processes.
        """
        ga = self.genetic_algorithm_partial(population=self.population)
        generations_run = 0

        while not ga.stop_condition.should_stop(
            generations_run, ga.best_route.length()
        ):
            for _ in range(self.generations_per_migration):
                if ga.stop_condition.should_stop(
                    generations_run, ga.best_route.length()
                ):
                    break
                ga.run_iteration()
                generations_run += 1

            if ga.stop_condition.should_stop(generations_run, ga.best_route.length()):
                break

            comm.Barrier()

            next_rank = (comm.Get_rank() + 1) % comm.Get_size()
            prev_rank = (comm.Get_rank() - 1 + comm.Get_size()) % comm.Get_size()

            subset_population = self.population.get_subset(self.migration_size)
            serialized_population = subset_population.serialize()

            send_request = comm.Isend(
                [serialized_population, MPI.DOUBLE], dest=next_rank
            )
            received_data = np.empty_like(serialized_population)
            recv_request = comm.Irecv([received_data, MPI.DOUBLE], source=prev_rank)

            MPI.Request.Waitall([send_request, recv_request])

            received_population = Population.deserialize(
                received_data, self.population.distance_matrix
            )

            self.population = combine_populations(self.population, received_population)

            comm.Barrier()

        best_route = ga.best_route
        all_best_routes = comm.gather(best_route, root=0)

        if comm.Get_rank() == 0:
            best_route_overall = min(all_best_routes, key=lambda route: route.length())
            return best_route_overall


class MPIAllToAllMigration(MPIStrategy):
    """MPI strategy where migration occurs between all processes (all-to-all migration).

    Attributes:
        migration_size (int): The number of individuals to migrate between processes.
        generations_per_migration (int): The number of generations to run before performing a migration.
    """

    def __init__(
        self,
        genetic_algorithm: ParametrisedGeneticAlgorithm,
        population: Population,
        migration_size: int,
        generations_per_migration: int,
    ):
        """Initializes the MPIAllToAllMigration strategy with the given parameters.

        Args:
            genetic_algorithm (ParametrisedGeneticAlgorithm): The genetic algorithm to be used in the strategy.
            population (Population): The population to be evolved.
            migration_size (int): The number of individuals to migrate between processes.
            generations_per_migration (int): The number of generations to run before performing a migration.
        """
        super().__init__(genetic_algorithm, population)
        self.migration_size = migration_size
        self.generations_per_migration = generations_per_migration

    def run(self, comm: MPI.Comm):
        """Runs the genetic algorithm with all-to-all migration between processes.

        In each migration, subsets of the population are exchanged between all pairs of processes.

        Args:
            comm (MPI.Comm): The MPI communicator used for communication between processes.

        Returns:
            Route: The best route found across all processes.
        """
        ga = self.genetic_algorithm_partial(population=self.population)
        generations_run = 0

        while not ga.stop_condition.should_stop(
            generations_run, ga.best_route.length()
        ):
            for _ in range(self.generations_per_migration):
                if ga.stop_condition.should_stop(
                    generations_run, ga.best_route.length()
                ):
                    break
                ga.run_iteration()
                generations_run += 1

            if ga.stop_condition.should_stop(generations_run, ga.best_route.length()):
                break

            comm.Barrier()

            requests = []

            for other_rank in range(comm.Get_size()):
                if other_rank != comm.Get_rank():
                    subset_population = self.population.get_subset(self.migration_size)
                    serialized_population = subset_population.serialize()

                    send_request = comm.Isend(
                        [serialized_population, MPI.DOUBLE], dest=other_rank
                    )
                    received_data = np.empty_like(serialized_population)
                    recv_request = comm.Irecv(
                        [received_data, MPI.DOUBLE], source=other_rank
                    )

                    requests.append((send_request, recv_request, received_data))

            for send_request, recv_request, received_data in requests:
                MPI.Request.Waitall([send_request, recv_request])

                received_population = Population.deserialize(
                    received_data, self.population.distance_matrix
                )

                self.population = combine_populations(
                    self.population, received_population
                )

            comm.Barrier()

        best_route = ga.best_route
        all_best_routes = comm.gather(best_route, root=0)

        if comm.Get_rank() == 0:
            best_route_overall = min(all_best_routes, key=lambda route: route.length())
            return best_route_overall
