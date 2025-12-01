import numpy as np
import multiprocessing as mp
import time
import sys
import os

try:
    import psutil
except ImportError:
    psutil = None

def get_memory_usage():
    if psutil:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024) 
    else:
        return 0.0 

def vectorized_multiplication(A, B):
    return A @ B  

def parallel_task(A_shape, B_shape, C_shared, M, N, K, start_row, end_row, A_flat, B_flat):
    A = np.frombuffer(A_flat, dtype=np.float64).reshape(A_shape)
    B = np.frombuffer(B_flat, dtype=np.float64).reshape(B_shape)
    C = np.frombuffer(C_shared.get_obj(), dtype=np.float64).reshape((M, N))
    
    for i in range(start_row, end_row):
        C[i, :] = A[i, :] @ B 

def explicit_parallel_multiplication(A, B, num_processes):
    M, K = A.shape
    _, N = B.shape
    
    C_shared = mp.Array('d', M * N)
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    rows_per_process = M // num_processes
    processes = []
    
    for i in range(num_processes):
        start_row = i * rows_per_process
        end_row = (i + 1) * rows_per_process if i < num_processes - 1 else M
        
        p = mp.Process(
            target=parallel_task, 
            args=(A.shape, B.shape, C_shared, M, N, K, start_row, end_row, A_flat, B_flat)
        )
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

    return np.frombuffer(C_shared.get_obj(), dtype=np.float64).reshape((M, N))

def performance_analysis(matrix_size):
    N = matrix_size
    num_processes = mp.cpu_count() 
    
    print("-" * 50)
    print(f"Matrix Dimension: {N} x {N}")
    print(f"Processes used: {num_processes}")
    
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    
    mem_usage = get_memory_usage()
    if mem_usage > 0:
        print(f"Memory Usage (approx): {mem_usage:.2f} MB")
    else:
        print("Memory Usage: (Install 'psutil' library for exact MB)")

    t_start = time.time()
    vectorized_multiplication(A, B)
    T_Base = time.time() - t_start
    print(f"Time T_Base: {T_Base:.6f} s") 

    t_start = time.time()
    explicit_parallel_multiplication(A, B, num_processes)
    T_Parallel = time.time() - t_start
    print(f"Time T_Parallel: {T_Parallel:.6f} s") 
    
    Speedup = T_Base / T_Parallel
    Efficiency = Speedup / num_processes if T_Parallel > 0 else 0

    print(f"Speedup: {Speedup:.6f}x")
    print(f"Efficiency: {Efficiency:.6f}")

if __name__ == '__main__':
    sizes = [512, 1024, 2048]
    for size in sizes:
        performance_analysis(size)