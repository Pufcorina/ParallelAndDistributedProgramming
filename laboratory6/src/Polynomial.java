import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Contains the list of the coefficients of the polynomial.
 * The first coefficients in the list corresponds to x^0 and the last corresponds to the biggest power .
 */
public class Polynomial {
	private static final int BOUND = 10;
	private List<Integer> coefficients;

	public Polynomial(List<Integer> coefficients) {
		this.coefficients = coefficients;
	}

	/**
	 * Generates a polynomial with random coefficients, with a given degree.
	 *
	 * @param degree - Integer
	 */
	public Polynomial(int degree) {
		coefficients = new ArrayList<>(degree + 1);
		//Generate the rest of the coefficients
		Random randomGenerator = new Random();
		for (int i = 0; i < degree; i++) {
			coefficients.add(randomGenerator.nextInt(BOUND));
		}
		//the coefficient of the biggest power has to be different than 0
		coefficients.add(randomGenerator.nextInt(BOUND) + 1);
	}

	public int getDegree() {
		return this.coefficients.size() - 1;
	}

	public int getLength() {
		return this.coefficients.size();
	}

	public List<Integer> getCoefficients() {
		return coefficients;
	}

	@Override
	public String toString() {
		StringBuilder str = new StringBuilder();
		int power = getDegree();
		for (int i = getDegree(); i >= 0; i--) {
			if ( coefficients.get(i) == 0)
				continue;
			str.append(" ").append(coefficients.get(i)).append("x^").append(power).append(" +");
			power--;
		}
		str.deleteCharAt(str.length() - 1); //delete last +
		return str.toString();
	}
}
