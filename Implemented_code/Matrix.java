import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Matrix {

    private static class MatrixMultiplicationTask implements Callable<Void> {
        private final double[][] A, B, C;
        private final int startRow, endRow;

        public MatrixMultiplicationTask(double[][] A, double[][] B, double[][] C, int startRow, int endRow) {
            this.A = A; this.B = B; this.C = C;
            this.startRow = startRow; this.endRow = endRow;
        }

        @Override
        public Void call() {
            int N = B[0].length;
            int K = A[0].length;
            for (int i = startRow; i < endRow; i++) {
                double[] rowA = A[i];
                double[] rowC = C[i];
                for (int k = 0; k < K; k++) {
                    double valA = rowA[k];
                    double[] rowB = B[k];
                    for (int j = 0; j < N; j++) {
                        rowC[j] += valA * rowB[j];
                    }
                }
            }
            return null;
        }
    }

    public static double[][] vectorizedMultiplication(double[][] A, double[][] B) {
        int M = A.length;
        int N = B[0].length;
        int K = A[0].length;
        double[][] C = new double[M][N];

        for (int i = 0; i < M; i++) {
            double[] rowA = A[i];
            double[] rowC = C[i];
            for (int k = 0; k < K; k++) {
                double valA = rowA[k];
                double[] rowB = B[k];
                for (int j = 0; j < N; j++) {
                    rowC[j] += valA * rowB[j];
                }
            }
        }
        return C;
    }

    public static double[][] parallelMultiplication(double[][] A, double[][] B, int numThreads) throws Exception {
        int M = A.length;
        int N = B[0].length;
        double[][] C = new double[M][N];

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<MatrixMultiplicationTask> tasks = new ArrayList<>();

        int rowsPerThread = M / numThreads;
        int startRow = 0;

        for (int i = 0; i < numThreads; i++) {
            int endRow = (i == numThreads - 1) ? M : startRow + rowsPerThread;
            tasks.add(new MatrixMultiplicationTask(A, B, C, startRow, endRow));
            startRow = endRow;
        }

        executor.invokeAll(tasks);
        executor.shutdown();
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        return C;
    }

    public static double[][] generateRandomMatrix(int N) {
        Random rand = new Random();
        double[][] matrix = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) matrix[i][j] = rand.nextDouble();
        }
        return matrix;
    }

    public static void printMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        long memory = runtime.totalMemory() - runtime.freeMemory();
        System.out.printf("Memory Usage: %.2f MB\n", memory / (1024.0 * 1024.0));
    }

    public static void performanceAnalysis(int matrixSize) {
        try {
            int N = matrixSize;
            int numThreads = Runtime.getRuntime().availableProcessors();

            System.out.println("--------------------------------------------------");
            System.out.printf("Matrix Dimension: %d x %d\n", N, N);
            System.out.printf("Threads available: %d\n", numThreads);

            double[][] A = generateRandomMatrix(N);
            double[][] B = generateRandomMatrix(N);

            printMemoryUsage();

            long t_start = System.nanoTime();
            vectorizedMultiplication(A, B);
            long t_end = System.nanoTime();
            double T_Base = (t_end - t_start) / 1e9;
            System.out.printf(Locale.US, "Time T_Base: %.6f s\n", T_Base);

            t_start = System.nanoTime();
            parallelMultiplication(A, B, numThreads);
            t_end = System.nanoTime();
            double T_Parallel = (t_end - t_start) / 1e9;
            System.out.printf(Locale.US, "Time T_Parallel: %.6f s\n", T_Parallel);

            double Speedup = T_Base / T_Parallel;
            double Efficiency = Speedup / numThreads;

            System.out.printf(Locale.US, "Speedup: %.6fx\n", Speedup);
            System.out.printf(Locale.US, "Efficiency: %.6f\n", Efficiency);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        int[] sizes = {512, 1024, 2048};
        for (int size : sizes) {
            performanceAnalysis(size);
        }
    }
}