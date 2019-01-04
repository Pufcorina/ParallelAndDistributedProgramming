import sun.awt.Mutex;

import java.util.List;

public class Multiplication implements Runnable {
    private Polynomial a;
    private Polynomial b;
    private Polynomial result;

    private int coeficientA;
    private int coeficientB;
    private List<Mutex> coeficientMutex;
    private Algorithm algorithm;

    public Multiplication(Polynomial a, Polynomial b, Polynomial result, int coeficientA, int coeficientB, List<Mutex> coeficientMutex, Algorithm algorithm) {
        this.a = a;
        this.b = b;
        this.result = result;
        this.coeficientA = coeficientA;
        this.coeficientB = coeficientB;
        this.coeficientMutex = coeficientMutex;
        this.algorithm = algorithm;
    }


    @Override
    public void run() {
        int newCoeficient = coeficientA + coeficientB;
        this.coeficientMutex.get(newCoeficient).lock();
        result.addCoeficient(coeficientA + coeficientB, algorithm.multiply(a.getCoeficient(coeficientA), b.getCoeficient(coeficientB)));
        this.coeficientMutex.get(newCoeficient).unlock();
    }
}
