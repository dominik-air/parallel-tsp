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
        local_opt_time (float | None): The time taken to perform the local optimization, in seconds.
        mpi_strategy_time (float): The time taken to execute the MPI strategy, in seconds.
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
        self.local_opt_time = None
        self.mpi_strategy_time = None

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
            start_time_local_opt = MPI.Wtime()
            optimized_population = self.local_optimization_strategy.optimize(
                self.mpi_strategy.population
            )
            end_time_local_opt = MPI.Wtime()
            self.local_opt_time = end_time_local_opt - start_time_local_opt
        else:
            optimized_population = None

        optimized_population = comm.bcast(optimized_population, root=0)

        self.mpi_strategy.population = optimized_population

        comm.Barrier()
        start_time_mpi_strategy = MPI.Wtime()
        best_route = self.mpi_strategy.run(comm)
        end_time_mpi_strategy = MPI.Wtime()
        self.mpi_strategy_time = end_time_mpi_strategy - start_time_mpi_strategy

        all_local_opt_times = comm.gather(self.local_opt_time, root=0)
        all_mpi_strategy_times = comm.gather(self.mpi_strategy_time, root=0)

        if rank == 0:
            avg_local_opt_time = (
                sum(filter(None, all_local_opt_times)) / comm.Get_size()
            )
            avg_mpi_strategy_time = sum(all_mpi_strategy_times) / comm.Get_size()

            print(f"Average Local Optimization Time: {avg_local_opt_time:.4f} seconds")
            print(f"Average MPI Strategy Time: {avg_mpi_strategy_time:.4f} seconds")

        return best_route
