import itertools
import json
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
    mpi_strategy_class: Any,
    optimization_class: Any,
    params: dict[str, dict[str, Any]],
    num_runs: int = 3,
) -> dict[str, Any]:
    """Run a benchmark with a given set of parameters."""
    results = []

    for _ in range(num_runs):
        start_time = time.perf_counter()

        optimization_strategy = optimization_class()

        if mpi_strategy_class in [MPIRingMigration, MPIAllToAllMigration]:
            mpi_strategy = mpi_strategy_class(
                genetic_algorithm=partial(
                    GeneticAlgorithm,
                    population=params["mpi"]["population"],
                    **params["ga"],
                ),
                population=params["mpi"]["population"],
                migration_size=params["mpi"]["strategy_params"]["migration_size"],
                migrations_count=params["mpi"]["strategy_params"]["migrations_count"],
            )
        else:
            mpi_strategy = mpi_strategy_class(
                genetic_algorithm=partial(
                    GeneticAlgorithm,
                    population=params["mpi"]["population"],
                    **params["ga"],
                ),
                population=params["mpi"]["population"],
            )

        runner = GeneticAlgorithmRunner(mpi_strategy, optimization_strategy)
        best_route = runner.run(params["mpi"]["comm"])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        if best_route:
            route_length = best_route.length()
        else:
            route_length = float("inf")

        results.append(
            {
                "time": elapsed_time,
                "route_length": route_length,
                "generations": params["ga"]["stop_condition"].max_generations,
            }
        )

    avg_time = sum(r["time"] for r in results) / num_runs
    avg_route_length = sum(r["route_length"] for r in results) / num_runs

    benchmark_result = {
        "mutation_rate": params["ga"]["mutation_rate"],
        "tournament_size": params["ga"]["tournament_size"],
        "max_generations": params["ga"]["stop_condition"].max_generations,
        "mpi_strategy": mpi_strategy_class.__name__,
        "population_size": params["mpi"]["population"].size,
        "num_cities": len(params["mpi"]["population"].distance_matrix.matrix),
        "migration_size": params["mpi"]["strategy_params"].get("migration_size", None),
        "migrations_count": params["mpi"]["strategy_params"].get(
            "migrations_count", None
        ),
        "optimization_strategy": optimization_class.__name__,
        "avg_time": avg_time,
        "avg_route_length": avg_route_length,
    }

    return benchmark_result


def grid_search(search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Perform a grid search over the parameter space."""
    mpi_classes = search_space["mpi"]["mpi_strategy"]
    opt_classes = search_space["opt"]["optimization_strategy"]

    param_combinations = list(
        itertools.product(
            *[
                (
                    [(outer_key, {inner_key: value}) for value in inner_values]
                    if isinstance(inner_values, list)
                    else [(outer_key, inner_values)]
                )
                for outer_key, inner_dict in search_space.items()
                for inner_key, inner_values in inner_dict.items()
            ]
        )
    )

    all_results = []

    for mpi_class in mpi_classes:
        for opt_class in opt_classes:
            for param_set in param_combinations:
                param_dict = {}
                for outer_key, inner_dict in param_set:
                    if outer_key not in param_dict:
                        param_dict[outer_key] = {}
                    param_dict[outer_key].update(inner_dict)

                result = run_benchmark(mpi_class, opt_class, param_dict)
                all_results.append(result)

    return all_results


def save_results_to_json(file_name: str, data: Any):
    """Save the results data to a JSON file."""
    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    search_space = {
        "ga": {
            "mutation_rate": [0.05, 0.1, 0.15, 0.2],
            "tournament_size": [5, 10, 15, 20],
            "stop_condition": [
                StopCondition(max_generations=20),
                StopCondition(max_generations=50),
                StopCondition(max_generations=100),
            ],
        },
        "mpi": {
            "mpi_strategy": [MPINoMigration, MPIRingMigration, MPIAllToAllMigration],
            "population": [
                Population(
                    size=50, distance_matrix=generate_random_distance_matrix(50)
                ),
                Population(
                    size=50, distance_matrix=generate_random_distance_matrix(100)
                ),
                Population(
                    size=100, distance_matrix=generate_random_distance_matrix(50)
                ),
                Population(
                    size=100, distance_matrix=generate_random_distance_matrix(100)
                ),
            ],
            "strategy_params": [
                {"migration_size": 10, "migrations_count": 2},
                {"migration_size": 20, "migrations_count": 4},
                {"migration_size": 30, "migrations_count": 6},
                {"migration_size": 40, "migrations_count": 8},
            ],
            "comm": [comm],
        },
        "opt": {
            "optimization_strategy": [
                NoOptimization,
                ChristofidesOptimization,
                GreedyTSPOptimization,
            ],
        },
    }

    all_results = grid_search(search_space)

    if rank == 0:
        filename = f"benchmark/results/benchmark_results_{size}_cores.json"
        save_results_to_json(filename, all_results)


if __name__ == "__main__":
    main()
