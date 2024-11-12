# parallel-tsp

## Project dependencies

### Python
Version 3.12

### Open MPI
Version 5.0.3

## Installation

### Get an MPI implementation

While any MPI implementation should work, here are my recommended options.

For macOS, you can install [Open MPI](https://www.open-mpi.org) using Homebrew:
```bash
brew install openmpi
```

For Linux, you may need to either install Open MPI from source or check your distribution’s repositories. For example, on Ubuntu/Debian, [MPICH](https://www.mpich.org) is readily available:
```
sudo apt install mpich
```

For Windows, Microsoft MPI is required. Follow the [Microsoft MPI installation guide](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) for instructions.

### Install poetry

This project uses Poetry for dependency management. Please refer to [the offical docs](https://python-poetry.org/docs/#installing-with-the-official-installer) for installation details.

### Install dependencies

```bash
# install dependencies
poetry install
# activate the virtual environment 
poetry shell
```

You're all set! You can now run examples, tests, and other code from this repository.

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

## Inspirations and references

### [Travelling Salesman Problem: Parallel Implementations & Analysis](https://arxiv.org/pdf/2205.14352.pdf)

### [Optimal Low-Latency Network Topologies for Cluster Performance Enhancement](https://arxiv.org/pdf/1904.00513)

