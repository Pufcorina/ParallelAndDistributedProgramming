import sun.awt.Mutex;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Main {

    public static void main(String[] args) {
        int TEST_NR = 20;

        Algorithm NORMAL_ALGORITHM = new NormalAlgorithm();
        Algorithm KARATSUBA_ALGORITHM = new KaratsubAlgorithm();

        List<Polynomial> polynomials1 = new ArrayList<>();
        List<Polynomial> polynomials2 = new ArrayList<>();

        Polynomial a = new Polynomial(2, 0, 0);
        a.setCoeficient(0, -7);
        a.setCoeficient(1, 3);
        a.setCoeficient(2, 2);

        Polynomial b = new Polynomial(3, 0, 0);
        b.setCoeficient(0, 0);
        b.setCoeficient(1, 1);
        b.setCoeficient(2, -2);
        b.setCoeficient(3, 7);

        for (int i = 1; i <= TEST_NR; i++) {
            polynomials1.add(new Polynomial(i * 100, randomNumber("1", i*10), randomNumber("9", i*10)));
            polynomials2.add(new Polynomial(i * 100, randomNumber("1", i*10), randomNumber("9", i*10)));
        }

        System.out.println("a = " + a);
        System.out.println("b = " + b);
        System.out.println();
        System.out.println("a * b = " + multiply(a, b, 1, NORMAL_ALGORITHM));
        System.out.println();

        System.out.println("Normal O(N^2) algorithm - sequential: ");
        batchTesting(polynomials1, polynomials2, 1, NORMAL_ALGORITHM);

        System.out.println("Normal O(N^2) algorithm - parallelized: ");
        batchTesting(polynomials1, polynomials2, 4, NORMAL_ALGORITHM);

        System.out.println("Karatsuba O(N^log3) algorithm - sequential: ");
        batchTesting(polynomials1, polynomials2, 1, KARATSUBA_ALGORITHM);

        System.out.println("Karatsuba O(N^log3) algorithm - parallelized: ");
        batchTesting(polynomials1, polynomials2, 4, KARATSUBA_ALGORITHM);
    }


    static void batchTesting(List<Polynomial> polynomials1, List<Polynomial> polynomials2, int threadCount, Algorithm algorithm){
        for (int i = 0; i < polynomials1.size(); i++){
            test(i + 1, polynomials1.get(i), polynomials2.get(i), threadCount, algorithm);
        }
    }

    static void test(int level, Polynomial a, Polynomial b, int threadCount, Algorithm algorthm) {
        long startTime = System.nanoTime();
        multiply(a, b, threadCount, algorthm);
        long endTime = System.nanoTime();
        long duration = (endTime - startTime) / 1000000;
        if (Arrays.asList(5, 10, 15, 20).contains(level))
            System.out.println("Level " + level + ": " + duration + " ms");
    }


    static Polynomial multiply(Polynomial a, Polynomial b, int threadCount, Algorithm algorthm) {
        int rank = a.getRank() + b.getRank();

        Polynomial result = new Polynomial(rank, 0, 0);

        List<Mutex> mutexes = new ArrayList<>(rank + 1);

        for (int i = 0; i <= rank; i++){
            mutexes.add(new Mutex());
        }

        ExecutorService pool = Executors.newFixedThreadPool(threadCount);

        for (int coeficientA = 0; coeficientA <= a.getRank(); coeficientA++){
            for (int coeficientB = 0; coeficientB <= b.getRank(); coeficientB++){
                if (a.getCoeficient(coeficientA).compareTo(BigInteger.ZERO) != 0 && b.getCoeficient(coeficientB).compareTo(BigInteger.ZERO) != 0){
                    pool.execute(new Multiplication(a, b, result, coeficientA, coeficientB, mutexes, algorthm));
                }
            }
        }

        pool.shutdown();

        try {
            pool.awaitTermination(3, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return result;
    }


    static String randomNumber(String startsWith, int length) {
        StringBuilder result = new StringBuilder().append(startsWith);

        Random random = new Random();

        for (int i = 1; i < length; i++) {
            result.append(random.nextInt(9) + 1);
        }

        return result.toString();
    }
}
