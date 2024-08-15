from abc import ABC, abstractmethod

import numpy as np
from mpi4py import MPI

from .genetic_algorithm import ParametrisedGeneticAlgorithm
from .population import Population, combine_populations

# TODO: try to not use the serialistion and deserialisation


class MPIStrategy(ABC):
    def __init__(
        self, genetic_algorithm: ParametrisedGeneticAlgorithm, population: Population
    ):
        self.genetic_algorithm_partial = genetic_algorithm
        self.population = population

    @abstractmethod
    def run(self, comm: MPI.Comm):
        """Run the evolutionary algorithm using the defined MPI strategy."""
        pass


class MPINoMigration(MPIStrategy):
    def run(self, comm: MPI.Comm):
        rank = comm.Get_rank()

        ga = self.genetic_algorithm_partial(population=self.population)

        ga.run_iterations(ga.generations)
        best_route = ga.best_route
        all_best_routes = comm.gather(best_route, root=0)

        if rank == 0:
            best_route_overall = min(all_best_routes, key=lambda route: route.length())
            return best_route_overall


class MPIRingMigration(MPIStrategy):
    def __init__(
        self,
        genetic_algorithm: ParametrisedGeneticAlgorithm,
        population: Population,
        migration_size: int,
        migrations_count: int,
    ):
        super().__init__(genetic_algorithm, population)
        self.migration_size = migration_size
        self.migrations_count = migrations_count

    def run(self, comm: MPI.Comm):
        ga = self.genetic_algorithm_partial(population=self.population)

        for _ in range(self.migrations_count):
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

            iterations = ga.generations // self.migrations_count
            ga.run_iterations(iterations)

        best_route = ga.best_route
        all_best_routes = comm.gather(best_route, root=0)

        if comm.Get_rank() == 0:
            best_route_overall = min(all_best_routes, key=lambda route: route.length())
            return best_route_overall


class MPIAllToAllMigration(MPIStrategy):
    def __init__(
        self,
        genetic_algorithm: ParametrisedGeneticAlgorithm,
        population: Population,
        migration_size: int,
        migrations_count: int,
    ):
        super().__init__(genetic_algorithm, population)
        self.migration_size = migration_size
        self.migrations_count = migrations_count

    def run(self, comm: MPI.Comm):
        ga = self.genetic_algorithm_partial(population=self.population)

        for _ in range(self.migrations_count):
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

            iterations = ga.generations // self.migrations_count
            ga.run_iterations(iterations)

        best_route = ga.best_route
        all_best_routes = comm.gather(best_route, root=0)

        if comm.Get_rank() == 0:
            best_route_overall = min(all_best_routes, key=lambda route: route.length())
            return best_route_overall
