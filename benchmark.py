import itertools
import time
from functools import partial
from typing import Any

from mpi4py import MPI

from parallel_tsp.distance_matrix import generate_random_distance_matrix
from parallel_tsp.genetic_algorithm import GeneticAlgorithm
from parallel_tsp.mpi_strategy import (
    MPIAllToAllMigration,
    MPINoMigration,
    MPIRingMigration,
)
from parallel_tsp.optimisation_strategy import (
    ChristofidesOptimization,
    GreedyTSPOptimization,
    NoOptimization,
)
from parallel_tsp.population import Population
from parallel_tsp.runner import GeneticAlgorithmRunner
from parallel_tsp.stop_condition import StopCondition


def run_benchmark(
    classes: dict[str, Any], params: dict[str, dict[str, Any]], num_runs: int = 1
) -> dict[str, Any]:
    """Run a benchmark with a given set of parameters."""
    results = []

    GAClass = classes["ga"]
    MPIStrategyClass = classes["mpi"]
    OptimizationClass = classes["opt"]

    for _ in range(num_runs):
        start_time = time.perf_counter()

        optimization_strategy = OptimizationClass()

        mpi_strategy = MPIStrategyClass(
            genetic_algorithm=partial(
                GAClass, population=params["mpi"]["population"], **params["ga"]
            ),
            population=params["mpi"]["population"],
            **params["mpi"]["strategy_params"]
        )

        runner = GeneticAlgorithmRunner(mpi_strategy, optimization_strategy)
        best_route = runner.run(params["mpi"]["comm"])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        route_length = best_route.length()
        results.append({"time": elapsed_time, "route_length": route_length})

    avg_time = sum(r["time"] for r in results) / num_runs
    avg_route_length = sum(r["route_length"] for r in results) / num_runs

    benchmark_result = {
        "params": params,
        "avg_time": avg_time,
        "avg_route_length": avg_route_length,
        "individual_runs": results,
    }

    return benchmark_result


def grid_search(
    search_space: dict[str, dict[str, Any]], classes: dict[str, Any]
) -> dict[str, Any]:
    """Perform a grid search over the parameter space."""
    param_combinations = list(
        itertools.product(
            *[
                [(outer_key, {inner_key: value}) for value in inner_values]
                for outer_key, inner_dict in search_space.items()
                for inner_key, inner_values in inner_dict.items()
            ]
        )
    )

    best_params = None
    best_score = float("inf")
    all_results = []

    for param_set in param_combinations:
        param_dict = {}
        for outer_key, inner_dict in param_set:
            if outer_key not in param_dict:
                param_dict[outer_key] = {}
            param_dict[outer_key].update(inner_dict)

        result = run_benchmark(classes, param_dict)
        all_results.append(result)
        if result["avg_route_length"] < best_score:
            best_score = result["avg_route_length"]
            best_params = param_dict

    return best_params, best_score, all_results


def main():
    search_space = {
        "ga": {
            "mutation_rate": [0.05, 0.1, 0.2],
            "tournament_size": [5, 10, 15],
            "stop_condition": [
                StopCondition(max_generations=10),
                StopCondition(max_generations=20),
                StopCondition(max_generations=30),
            ],
        },
        "mpi": {
            "mpi_strategy": [MPINoMigration, MPIRingMigration, MPIAllToAllMigration],
            "population": [
                Population(
                    size=50, distance_matrix=generate_random_distance_matrix(20)
                ),
                Population(
                    size=100, distance_matrix=generate_random_distance_matrix(50)
                ),
            ],
            "strategy_params": [
                {"migration_size": 10, "migrations_count": 2},
                {"migration_size": 20, "migrations_count": 10},
            ],
            "comm": [MPI.COMM_WORLD],
        },
        "opt": {
            "optimization_strategy": [
                NoOptimization,
                ChristofidesOptimization,
                GreedyTSPOptimization,
            ],
        },
    }

    classes = {
        "ga": GeneticAlgorithm,
        "mpi": MPIRingMigration,
        "opt": ChristofidesOptimization,
    }

    best_params, best_score, _ = grid_search(search_space, classes)
    print(best_params, best_score)


if __name__ == "__main__":
    main()
