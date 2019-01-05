import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Polynomial {
    private List<BigInteger> coeficients;
    private int rank;

    Polynomial(int rank, BigInteger min, BigInteger max) {
        this.rank = rank;
        coeficients = new ArrayList<>(rank + 1);

        for (int i = 0; i <= rank; i++) {
            if (min.compareTo(BigInteger.ZERO) == 0 && max.compareTo(BigInteger.ZERO) == 0){
                this.coeficients.add(BigInteger.ZERO);
            } else {
                this.coeficients.add(this.getRandomBigInt(min, max));
            }
        }
    }

    Polynomial(int rank, int min, int max) {
        this(rank, new BigInteger(String.valueOf(min)), new BigInteger(String.valueOf(max)));
    }

    Polynomial(int rank, String min, String max) {
        this(rank, new BigInteger(min), new BigInteger(max));
    }

    public BigInteger getCoeficient(int index) {
        return this.coeficients.get(index);
    }

    public void setCoeficient(int index, int value) {
        this.coeficients.set(index, new BigInteger(String.valueOf(value)));
    }

    public void setCoeficient(int index, BigInteger value) {
        this.coeficients.set(index, value);
    }

    public int getRank() {
        return this.rank;
    }

    public void addCoeficient(int index, BigInteger value) {
        this.coeficients.set(index, this.coeficients.get(index).add(value));
    }

    @Override
    public String toString() {
        StringBuilder ss = new StringBuilder();

        BigInteger ZERO = BigInteger.ZERO;

        for (int i = this.rank; i >= 0; i--){
            BigInteger coef = this.coeficients.get(i);
            if (coef.compareTo(ZERO) > 0) {
                ss.append("+").append(this.coeficients.get(i)).append("X^").append(i).append(" ");
            }else if (coef.compareTo(ZERO) < 0) {
                ss.append(this.coeficients.get(i)).append("X^").append(i).append(" ");
            }
        }
        return ss.toString();
    }

    private BigInteger getRandomBigInt(BigInteger min, BigInteger max) {
        BigInteger result;

        Random random = new Random();

        int numBits = max.bitLength() - 1;
        if (numBits <= 0) {
            numBits = 1;
        }

        do {
            result = new BigInteger(numBits, random);
        } while (result.compareTo(min) >= 0 && result.compareTo(max) <= 0);

        return result;
    }
}
