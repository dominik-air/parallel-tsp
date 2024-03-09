# parallel-tsp

## Inspirations and references

### [Travelling Salesman Problem: Parallel Implementations & Analysis](https://arxiv.org/pdf/2205.14352.pdf)
The project implements several approaches to parallelize the Brute Force TSP algorithm, including OpenMPI, MPI and CUDA programming.
In my project I want to provide a better idea about relative importance of selecting a good algorithm and parallelizing an algorithm by solving TSP with
a non-exact algorithm such as Branch&Bound or Genetic Algorithms.  

## Project dependencies

### Python
Version 3.11.6

### Microsoft MPI
Version 10.1.3

``` ms-mpi/Bin/mpiexec.exe -n 4 python main.py ```

## Ideas

- [ ] Research articles about genetic algorithms used for TSP to borrow parameters or strategies on how to change the parameters depending on the input size
- [ ] Find a reliable way to evaluate the results. Is comparing the best initial route to the best final route meaningful? Maybe compare the final results to the MST of the cities FCG?
