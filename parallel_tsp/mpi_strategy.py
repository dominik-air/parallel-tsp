from abc import ABC, abstractmethod

from mpi4py import MPI

from .genetic_algorithm import ParametrisedGeneticAlgorithm
from .population import Population, combine_populations


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
            next_rank = (comm.Get_rank() + 1) % comm.Get_size()
            prev_rank = (comm.Get_rank() - 1 + comm.Get_size()) % comm.Get_size()

            subset_population = self.population.get_subset(self.migration_size)

            comm.Isend(subset_population, dest=next_rank)
            received_population = Population(
                self.migration_size, self.population.routes[0].distance_matrix
            )
            comm.Irecv(received_population, source=prev_rank)

            self.population = combine_populations(self.population, received_population)

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
            for other_rank in range(comm.Get_size()):
                if other_rank != comm.Get_rank():
                    subset_population = self.population.get_subset(self.migration_size)
                    comm.Isend(subset_population, dest=other_rank)

                    received_population = Population(
                        self.migration_size, self.population.routes[0].distance_matrix
                    )
                    comm.Irecv(received_population, source=other_rank)

                    self.population = combine_populations(
                        self.population, received_population
                    )

            iterations = ga.generations // self.migrations_count
            ga.run_iterations(iterations)

        best_route = ga.best_route
        all_best_routes = comm.gather(best_route, root=0)

        if comm.Get_size() == 0:
            best_route_overall = min(all_best_routes, key=lambda route: route.length())
            return best_route_overall
