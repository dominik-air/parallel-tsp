import subprocess
import time
import os
import itertools

def run_benchmark(cores, population_size, total_generations, mutation_rate, tournament_size, result_file):
    command = [
        "mpirun", "-n", str(cores),
        "python", "main.py",
        str(population_size),
        str(total_generations),
        str(mutation_rate),
        str(tournament_size),
        result_file
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

def main():
    START_TIME = time.perf_counter()
    cores_list = [1, 2, 4, 8]
    population_sizes = [100, 250, 500]
    total_generations_list = [50, 100, 200]
    mutation_rates = [0.01, 0.05, 0.1, 0.2]
    tournament_sizes = [5, 10, 25, 50]

    parameter_sets = list(itertools.product(cores_list, population_sizes, total_generations_list, mutation_rates, tournament_sizes))

    os.makedirs('results', exist_ok=True)

    for i, params in enumerate(parameter_sets):
        cores, population_size, total_generations, mutation_rate, tournament_size = params
        result_file = f"results/result_cores_{cores}_pop_{population_size}_gen_{total_generations}_mut_{mutation_rate}_tour_{tournament_size}.json"
        run_benchmark(cores, population_size, total_generations, mutation_rate, tournament_size, result_file)
    END_TIME = time.perf_counter()
    print(f'All tests executed in: {END_TIME-START_TIME}')
if __name__ == "__main__":
    main()
