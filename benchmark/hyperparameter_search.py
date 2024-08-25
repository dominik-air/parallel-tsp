import itertools
import json
import os
import sys
import time
from functools import partial
from typing import Any, Dict

from mpi4py import MPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

CLASS_MAP = {
    "GeneticAlgorithm": GeneticAlgorithm,
    "MPINoMigration": MPINoMigration,
    "MPIRingMigration": MPIRingMigration,
    "MPIAllToAllMigration": MPIAllToAllMigration,
    "NoOptimization": NoOptimization,
    "ChristofidesOptimization": ChristofidesOptimization,
    "GreedyTSPOptimization": GreedyTSPOptimization,
    "Population": Population,
    "StopCondition": StopCondition,
    "MPI_COMM_WORLD": MPI.COMM_WORLD,
}


def replace_strings_with_classes(d: Any) -> Any:
    """Recursively replace string keys with class objects using CLASS_MAP."""
    if isinstance(d, dict):
        return {
            k: replace_strings_with_classes(CLASS_MAP.get(v, v)) for k, v in d.items()
        }
    elif isinstance(d, list):
        return [replace_strings_with_classes(i) for i in d]
    return d


def load_search_space(file_name: str) -> Dict[str, Dict[str, Any]]:
    """Load the search space from a JSON file and replace strings with class objects."""
    with open(file_name, "r") as json_file:
        search_space = json.load(json_file)

    return replace_strings_with_classes(search_space)


def run_benchmark(
    classes: Dict[str, Any], params: Dict[str, Dict[str, Any]], num_runs: int = 3
) -> Dict[str, Any]:
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
            **params["mpi"]["strategy_params"],
        )

        runner = GeneticAlgorithmRunner(mpi_strategy, optimization_strategy)
        best_route = runner.run(params["mpi"]["comm"])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        route_length = best_route.length()
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
        "params": params,
        "avg_time": avg_time,
        "avg_route_length": avg_route_length,
        "individual_runs": results,
    }

    return benchmark_result


def grid_search(
    search_space: Dict[str, Dict[str, Any]], classes: Dict[str, Any]
) -> Dict[str, Any]:
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


def save_results_to_json(file_name: str, data: Any):
    """Save the results data to a JSON file."""
    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)


def main():
    search_space = load_search_space("benchmark/search_spaces/initial_search.json")

    classes = {
        "ga": GeneticAlgorithm,
        "mpi": MPIRingMigration,
        "opt": ChristofidesOptimization,
    }

    all_results = grid_search(search_space, classes)

    save_results_to_json("benchmark/results/benchmark_results.json", all_results)


if __name__ == "__main__":
    main()
