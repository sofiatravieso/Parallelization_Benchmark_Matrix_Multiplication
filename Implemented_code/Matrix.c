#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
    int get_cpu_count() {
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        return (int)sysinfo.dwNumberOfProcessors;
    }

    double get_memory_usage_mb() {
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
        }
        return 0.0;
    }
#else
    #include <unistd.h>
    #include <sys/resource.h>
    int get_cpu_count() {
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        return (nprocs > 0) ? (int)nprocs : 4;
    }

    double get_memory_usage_mb() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss / 1024.0;
    }
#endif

#define NUM_TESTS 3

typedef struct {
    double **A;
    double **B;
    double **C;
    int M;
    int K;
    int N;
    int startRow;
    int endRow;
} thread_args_t;

double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (matrix == NULL) return NULL;
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) free(matrix[i]);
    free(matrix);
}

double** generate_random_matrix(int N) {
    double** matrix = allocate_matrix(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
    return matrix;
}

double** vectorized_multiplication(double** A, double** B, int M, int K, int N) {
    double** C = allocate_matrix(M, N);
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++)
            C[i][j] = 0.0;

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            double r = A[i][k];
            for (int j = 0; j < N; j++) {
                C[i][j] += r * B[k][j];
            }
        }
    }
    return C;
}

void* matrix_multiplication_task(void* arg) {
    thread_args_t* task = (thread_args_t*)arg;
    int N = task->N;
    int K = task->K;

    for (int i = task->startRow; i < task->endRow; i++) {
        for (int j = 0; j < N; j++) task->C[i][j] = 0.0;

        for (int k = 0; k < K; k++) {
            double r = task->A[i][k];
            for (int j = 0; j < N; j++) {
                task->C[i][j] += r * task->B[k][j];
            }
        }
    }
    return NULL;
}

double** parallel_multiplication(double** A, double** B, int M, int K, int N, int numThreads) {
    double** C = allocate_matrix(M, N);
    pthread_t threads[numThreads];
    thread_args_t args[numThreads];
    int rowsPerThread = M / numThreads;
    int startRow = 0;

    for (int i = 0; i < numThreads; i++) {
        int endRow = (i == numThreads - 1) ? M : startRow + rowsPerThread;
        args[i].A = A; args[i].B = B; args[i].C = C;
        args[i].M = M; args[i].K = K; args[i].N = N;
        args[i].startRow = startRow; args[i].endRow = endRow;
        pthread_create(&threads[i], NULL, matrix_multiplication_task, &args[i]);
        startRow = endRow;
    }
    for (int i = 0; i < numThreads; i++) pthread_join(threads[i], NULL);
    return C;
}

double get_time_in_seconds(struct timeval *t_start, struct timeval *t_end) {
    long long t_start_us = t_start->tv_sec * 1000000 + t_start->tv_usec;
    long long t_end_us = t_end->tv_sec * 1000000 + t_end->tv_usec;
    return (double)(t_end_us - t_start_us) / 1000000.0;
}

void performance_analysis(int matrixSize) {
    int N = matrixSize;
    int numThreads = get_cpu_count();
    struct timeval t_start, t_end;

    printf("--------------------------------------------------\n");
    printf("Matrix Dimension: %d x %d\n", N, N);
    printf("Threads used: %d\n", numThreads);

    double** A = generate_random_matrix(N);
    double** B = generate_random_matrix(N);

    printf("Memory Usage (approx): %.2f MB\n", get_memory_usage_mb());

    double T_Base = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        gettimeofday(&t_start, NULL);
        double** C = vectorized_multiplication(A, B, N, N, N);
        gettimeofday(&t_end, NULL);
        T_Base += get_time_in_seconds(&t_start, &t_end);
        free_matrix(C, N);
    }
    T_Base /= NUM_TESTS;
    printf("Time T_Base: %.6f s\n", T_Base);

    double T_Parallel = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        gettimeofday(&t_start, NULL);
        double** C = parallel_multiplication(A, B, N, N, N, numThreads);
        gettimeofday(&t_end, NULL);
        T_Parallel += get_time_in_seconds(&t_start, &t_end);
        free_matrix(C, N);
    }
    T_Parallel /= NUM_TESTS;
    printf("Time T_Parallel: %.6f s\n", T_Parallel);

    printf("Speedup: %.6fx\n", T_Base / T_Parallel);
    printf("Efficiency: %.6f\n", (T_Base / T_Parallel) / numThreads);

    free_matrix(A, N);
    free_matrix(B, N);
}

int main() {
    int sizes[] = {512, 1024, 2048};
    int num_sizes = 3;

    for (int i = 0; i < num_sizes; i++) {
        performance_analysis(sizes[i]);
    }
    return 0;
}