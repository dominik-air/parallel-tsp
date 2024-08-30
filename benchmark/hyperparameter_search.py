import itertools
import json
from functools import partial
from typing import Any, Dict, List

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


def split_population(population: Population, num_splits: int) -> List[Population]:
    """Splits the population into sub-populations."""
    sub_population_size = len(population.routes) // num_splits
    return [
        Population(
            size=sub_population_size,
            distance_matrix=population.distance_matrix,
            routes=population.routes[
                i * sub_population_size : (i + 1) * sub_population_size
            ],
        )
        for i in range(num_splits)
    ]


def run_benchmark(
    mpi_strategy_class: Any,
    optimization_class: Any,
    params: Dict[str, Dict[str, Any]],
    num_runs: int = 1,
) -> Dict[str, Any]:
    """Run a benchmark with a given set of parameters."""
    results = []

    comm = params["mpi"]["comm"]
    rank = comm.Get_rank()

    for _ in range(num_runs):
        optimization_strategy = optimization_class()

        mpi_strategy_params = {
            "genetic_algorithm": partial(
                GeneticAlgorithm,
                population=params["mpi"]["population"],
                **params["ga"],
            ),
            "population": params["mpi"]["population"],
        }

        if mpi_strategy_class in {MPIRingMigration, MPIAllToAllMigration}:
            mpi_strategy_params.update(
                {
                    "migration_size": params["mpi"]["strategy_params"][
                        "migration_size"
                    ],
                    "generations_per_migration": params["mpi"]["strategy_params"][
                        "generations_per_migration"
                    ],
                }
            )

        mpi_strategy = mpi_strategy_class(**mpi_strategy_params)
        runner = GeneticAlgorithmRunner(mpi_strategy, optimization_strategy)
        best_route = runner.run(comm)

        if rank == 0:
            route_length = best_route.length() if best_route else float("inf")

            results.append(
                {
                    "route_length": route_length,
                    "generations": params["ga"]["stop_condition"].max_generations,
                    "local_opt_time": runner.local_opt_time,
                    "mpi_strategy_time": runner.mpi_strategy_time,
                }
            )

    if rank == 0:
        avg_route_length = sum(r["route_length"] for r in results) / num_runs
        avg_local_opt_time = sum(r["local_opt_time"] for r in results) / num_runs
        avg_mpi_strategy_time = sum(r["mpi_strategy_time"] for r in results) / num_runs

        benchmark_result = {
            "mutation_rate": params["ga"]["mutation_rate"],
            "tournament_size": params["ga"]["tournament_size"],
            "max_generations": params["ga"]["stop_condition"].max_generations,
            "mpi_strategy": mpi_strategy_class.__name__,
            "population_size": params["mpi"]["population_size"],
            "num_cities": len(params["mpi"]["population"].distance_matrix.matrix),
            "migration_size": params["mpi"]["strategy_params"].get("migration_size"),
            "generations_per_migration": params["mpi"]["strategy_params"].get(
                "generations_per_migration"
            ),
            "optimization_strategy": optimization_class.__name__,
            "avg_route_length": avg_route_length,
            "avg_local_opt_time": avg_local_opt_time,
            "avg_mpi_strategy_time": avg_mpi_strategy_time,
        }

        return benchmark_result
    return None


def grid_search(search_space: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Perform a grid search over the parameter space."""
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

    for param_set in param_combinations:
        param_dict = {}
        for outer_key, inner_dict in param_set:
            if outer_key not in param_dict:
                param_dict[outer_key] = {}
            param_dict[outer_key].update(inner_dict)

        mpi_class = param_dict["mpi"]["mpi_strategy"]
        opt_class = param_dict["opt"]["optimization_strategy"]

        param_dict["mpi"].pop("mpi_strategy")
        param_dict["opt"].pop("optimization_strategy")

        result = run_benchmark(mpi_class, opt_class, param_dict)
        if result:
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

    population_sizes = [100, 200, 400]

    all_results = []
    for pop_size in population_sizes:
        if rank == 0:
            full_population = Population(
                size=pop_size, distance_matrix=generate_random_distance_matrix(200)
            )
            sub_populations = split_population(full_population, size)
        else:
            sub_populations = None

        sub_population = comm.scatter(sub_populations, root=0)

        search_space = {
            "ga": {
                "mutation_rate": [0.2],
                "tournament_size": [int(sub_population.size / 5)],
                "stop_condition": [
                    StopCondition(max_generations=1000, max_time_seconds=20),
                ],
            },
            "mpi": {
                "mpi_strategy": [
                    MPINoMigration,
                    MPIRingMigration,
                    MPIAllToAllMigration,
                ],
                "population": [
                    sub_population,
                ],
                "population_size": [pop_size],
                "strategy_params": [
                    {
                        "migration_size": int(sub_population.size / 5),
                        "generations_per_migration": 2,
                    },
                    {
                        "migration_size": int(sub_population.size / 5),
                        "generations_per_migration": 5,
                    },
                    {
                        "migration_size": int(sub_population.size / 5),
                        "generations_per_migration": 10,
                    },
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

        results = grid_search(search_space)
        if rank == 0:
            all_results.extend(results)

    if rank == 0:
        filename = f"benchmark/results_perf/benchmark_results_{size}_cores.json"
        save_results_to_json(filename, all_results)


if __name__ == "__main__":
    main()
