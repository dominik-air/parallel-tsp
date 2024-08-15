from mpi4py import MPI

from .mpi_strategy import MPIStrategy
from .optimisation_strategy import OptimizationStrategy


class GeneticAlgorithmRunner:
    def __init__(
        self,
        mpi_strategy: MPIStrategy,
        local_optimization_strategy: OptimizationStrategy,
    ):
        self.mpi_strategy = mpi_strategy
        self.local_optimization_strategy = local_optimization_strategy

    def run(self, comm: MPI.Comm):
        rank = comm.Get_rank()

        if rank == 0:
            optimized_population = self.local_optimization_strategy.optimize(
                self.mpi_strategy.population
            )
        else:
            optimized_population = None

        optimized_population = comm.bcast(optimized_population, root=0)

        self.mpi_strategy.population = optimized_population

        return self.mpi_strategy.run(comm)
