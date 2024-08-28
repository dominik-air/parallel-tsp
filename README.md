# parallel-tsp

## Inspirations and references

### [Travelling Salesman Problem: Parallel Implementations & Analysis](https://arxiv.org/pdf/2205.14352.pdf)
The project implements several approaches to parallelize the Brute Force TSP algorithm, including OpenMP, MPI and CUDA programming.

## Project dependencies

### Python
Version 3.12

### Open MPI
Version 5.0.3

## Running examples

### Genetic Algorithm

```bash
python examples/run_ga_example.py
```

### Genetic Algorithm with Local Optimisation

```bash
python examples/run_optimized_example.py
```

### Genetic Algorithm with MPI Strategy

```bash
mpiexec -n 4 python examples/run_ga_mpi_example.py
```

## Running tests

### Functional

```bash
pytest -m functional
```

### Functional MPI

```bash
mpiexec -n 4 python -m pytest -m mpi
```

## Running benchmark

The `8` is an arbitrary value and you can use as many parallel environments as many your MPI implementation detects.

```bash
for cores in {1..8}; do echo "Running with $cores cores"; time mpiexec -n $cores python benchmark/hyperparameter_search.py; done
```
