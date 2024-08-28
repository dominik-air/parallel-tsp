from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mpi4py import MPI

from parallel_tsp.mpi_strategy import MPIStrategy
from parallel_tsp.optimisation_strategy import OptimizationStrategy
from parallel_tsp.population import Population
from parallel_tsp.runner import GeneticAlgorithmRunner


@pytest.fixture
def comm():
    """Fixture to provide the MPI communicator."""
    return MPI.COMM_WORLD


@pytest.fixture
def mock_mpi_strategy():
    """Fixture to provide a mock MPI strategy."""
    mock = MagicMock(spec=MPIStrategy)
    distance_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    mock.population = Population(size=3, distance_matrix=distance_matrix)
    mock.run.return_value = "best_route"
    return mock


@pytest.fixture
def mock_optimization_strategy():
    """Fixture to provide a mock optimization strategy."""
    mock = MagicMock(spec=OptimizationStrategy)
    return mock


@patch("parallel_tsp.runner.MPI.Wtime", side_effect=[0.0, 2.0, 3.0, 5.0])
def test_genetic_algorithm_runner_run(
    mock_wtime, mock_mpi_strategy, mock_optimization_strategy, comm
):
    """Test the run method of GeneticAlgorithmRunner."""
    runner = GeneticAlgorithmRunner(mock_mpi_strategy, mock_optimization_strategy)

    mock_optimization_strategy.optimize.return_value = mock_mpi_strategy.population

    best_route = runner.run(comm)

    if comm.Get_rank() == 0:
        assert best_route == "best_route"
        assert runner.local_opt_time == 2.0
        assert runner.mpi_strategy_time == 2.0
        optimized_population = mock_optimization_strategy.optimize.call_args[0][0]
        assert optimized_population.size == mock_mpi_strategy.population.size
        np.testing.assert_array_equal(
            optimized_population.distance_matrix,
            mock_mpi_strategy.population.distance_matrix,
        )
        mock_mpi_strategy.run.assert_called_once_with(comm)

    all_local_opt_times = comm.gather(runner.local_opt_time, root=0)
    all_mpi_strategy_times = comm.gather(runner.mpi_strategy_time, root=0)

    if comm.Get_rank() == 0:
        avg_local_opt_time = sum(filter(None, all_local_opt_times)) / comm.Get_size()
        avg_mpi_strategy_time = sum(all_mpi_strategy_times) / comm.Get_size()

        assert avg_local_opt_time == 2.0
        assert avg_mpi_strategy_time == 2.0

        print(f"Average Local Optimization Time: {avg_local_opt_time:.4f} seconds")
        print(f"Average MPI Strategy Time: {avg_mpi_strategy_time:.4f} seconds")


@patch("parallel_tsp.runner.MPI.Wtime", side_effect=[0.0, 1.0, 2.0, 4.0])
def test_genetic_algorithm_runner_timing(
    mock_wtime, mock_mpi_strategy, mock_optimization_strategy, comm
):
    """Test that timing attributes are correctly calculated and stored."""
    runner = GeneticAlgorithmRunner(mock_mpi_strategy, mock_optimization_strategy)

    mock_optimization_strategy.optimize.return_value = mock_mpi_strategy.population

    _ = runner.run(comm)

    if comm.Get_rank() == 0:
        assert runner.local_opt_time == 1.0
        assert runner.mpi_strategy_time == 2.0
    else:
        assert runner.local_opt_time is None
        assert runner.mpi_strategy_time == 2.0
