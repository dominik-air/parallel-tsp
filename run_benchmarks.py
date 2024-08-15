import os
import subprocess
import time


def run_benchmark(
    cores,
    num_cities,
    population_size,
    total_generations,
    mutation_rate,
    tournament_size,
    result_file,
):
    command = [
        "mpirun",
        "-n",
        str(cores),
        "python",
        "main.py",
        str(num_cities),
        str(population_size),
        str(total_generations),
        str(mutation_rate),
        str(tournament_size),
        result_file,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)


def main():
    START_TIME = time.perf_counter()
    cores_list = [1, 2, 4, 8]
    num_cities_list = [500, 600, 700, 800, 900, 1000]
    population_sizes = [100]
    mutation_rate = 0.2
    tournament_size = 50

    total_generations_dict = {
        1: [200, 300, 400, 500],
        2: [100, 200, 300, 400],
        4: [80, 100, 200, 300],
        8: [50, 80, 100, 200],
    }

    # Generate parameter sets
    parameter_sets = []
    for cores in cores_list:
        for num_cities in num_cities_list:
            for population_size in population_sizes:
                for total_generations in total_generations_dict[cores]:
                    parameter_sets.append(
                        (cores, num_cities, population_size, total_generations)
                    )

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    for i, params in enumerate(parameter_sets):
        cores, num_cities, population_size, total_generations = params
        result_file = f"results/result_cores_{cores}_cities_{num_cities}_pop_{population_size}_gen_{total_generations}.json"
        run_benchmark(
            cores,
            num_cities,
            population_size,
            total_generations,
            mutation_rate,
            tournament_size,
            result_file,
        )

    END_TIME = time.perf_counter()
    print(f"All tests run in: {END_TIME-START_TIME}")


if __name__ == "__main__":
    main()
