# parallel-tsp

## Inspirations and references

### [Travelling Salesman Problem: Parallel Implementations & Analysis](https://arxiv.org/pdf/2205.14352.pdf)
The project implements several approaches to parallelize the Brute Force TSP algorithm, including OpenMP, MPI and CUDA programming.

## Project dependencies

### Python
Version 3.12

### Open MPI
Version 5.0.3

## Running tests

### Functional

```bash
pytest -m functional
```

### Functional MPI

```bash
mpirun -n 4 python -m pytest --with-mpi -m mpi
```

## Running benchmark

```bash
for cores in {1..8}; do echo "Running with $cores cores"; time mpiexec -n $cores python benchmark/hyperparameter_search.py; done
```
