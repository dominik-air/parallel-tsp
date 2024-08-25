from mpi4py import MPI

from .mpi_strategy import MPIStrategy
from .optimisation_strategy import OptimizationStrategy


class GeneticAlgorithmRunner:
    """Handles the execution of a genetic algorithm with optional local optimization and MPI strategy.

    This class manages the optimization and execution of a genetic algorithm across multiple processes
    using MPI for parallelization. The optimization is performed on the master node (rank 0), and the
    optimized population is broadcasted to all other nodes.

    Attributes:
        mpi_strategy (MPIStrategy): The MPI strategy used to parallelize the genetic algorithm.
        local_optimization_strategy (OptimizationStrategy): The local optimization strategy applied
            before running the genetic algorithm.
    """

    def __init__(
        self,
        mpi_strategy: MPIStrategy,
        local_optimization_strategy: OptimizationStrategy,
    ):
        """Initializes the GeneticAlgorithmRunner with the given MPI and optimization strategies.

        Args:
            mpi_strategy (MPIStrategy): The MPI strategy for parallelizing the genetic algorithm.
            local_optimization_strategy (OptimizationStrategy): The optimization strategy applied
                to the population before running the genetic algorithm.
        """
        self.mpi_strategy = mpi_strategy
        self.local_optimization_strategy = local_optimization_strategy

    def run(self, comm: MPI.Comm):
        """Runs the genetic algorithm with the specified MPI and optimization strategies.

        The method performs local optimization on the master node (rank 0), broadcasts the optimized
        population to all nodes, and then runs the genetic algorithm using the MPI strategy.

        Args:
            comm (MPI.Comm): The MPI communicator for process communication.

        Returns:
            The best route found after running the genetic algorithm.
        """
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
