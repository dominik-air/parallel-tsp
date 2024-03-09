from mpi4py import MPI
import numpy as np
import sys
from math import factorial
import random
import time

def precompute_factorial(n):
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i
    return fact

def assign_edge_weights(n, min_weight=1, max_weight=10):
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            weight = random.randint(min_weight, max_weight)
            matrix[i, j] = matrix[j, i] = weight
    np.fill_diagonal(matrix, 0)
    return matrix

def nth_permutation(arr, n):
    result = []
    arr = sorted(arr)
    while arr:
        f = factorial(len(arr) - 1)
        i = n // f
        result.append(arr.pop(i))
        n %= f
    return result

def find_path_cost(matrix, path):
    cost = sum(matrix[path[i], path[i+1]] for i in range(len(path)-1))
    return cost

def main():
    random.seed(123)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    if rank == 0:
        fact = precompute_factorial(N-1)
        matrix = assign_edge_weights(N)
        start_time = time.time()
    else:
        fact = None
        matrix = None

    fact = comm.bcast(fact, root=0)
    matrix = comm.bcast(matrix, root=0)

    nppe = fact[N-1] // size
    extra = fact[N-1] % size
    start_ind = rank * nppe + min(rank, extra)
    end_ind = start_ind + nppe - 1
    if rank < extra:
        end_ind += 1

    local_optimal_cost = float('inf')

    for i in range(start_ind, end_ind + 1):
        perm = nth_permutation(list(range(1, N)), i)
        path = [0] + perm + [0]
        cost = find_path_cost(matrix, path)
        if cost < local_optimal_cost:
            local_optimal_cost = cost

    global_optimal_cost = comm.reduce(local_optimal_cost, op=MPI.MIN, root=0)

    if rank == 0:
        end_time = time.time()
        print(f"Optimal Path Cost: {global_optimal_cost}")
        print(f"Execution Time: {end_time - start_time}")

if __name__ == "__main__":
    main()
