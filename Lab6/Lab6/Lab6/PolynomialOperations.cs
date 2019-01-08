using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Lab6
{
    public class PolynomialOperations
    {
        public static Polynomial SynchronousMultiply(Polynomial polynomial1, Polynomial polynomial2)
        {
            int degreeMax = Math.Max(polynomial1.Degree, polynomial2.Degree);
            Polynomial result = new Polynomial(degreeMax * 2);

            for (int i = 0; i < degreeMax; i++)
            {
                for (int j = 0; j < degreeMax; j++)
                {
                    result.Coefficients[i + j] += polynomial1.Coefficients[i] * polynomial2.Coefficients[j];
                }
            }

            return result;
        }

        public static Polynomial AsynchronousMultiply(Polynomial polynomial1, Polynomial polynomial2)
        {
            int degreeMax = Math.Max(polynomial1.Degree, polynomial2.Degree);
            Polynomial result = new Polynomial(degreeMax * 2);

            Mutex[] m = new Mutex[degreeMax];

            for (int i = 0; i < m.Length; i++)
            {
                m[i] = new Mutex();
            }

            List<Task> tasks = new List<Task>();

            for (int i = 0; i < degreeMax; i++)
            {
                int taski = 0 + i;

                tasks.Add(Task.Run(() =>
                {
                    for (int j = 0; j < degreeMax; j++)
                    {
                        m[taski].WaitOne();
                        result.Coefficients[taski + j] += polynomial1.Coefficients[taski] * polynomial2.Coefficients[j];
                        m[taski].ReleaseMutex();
                    }
                }));

            }

            Task.WaitAll(tasks.ToArray());

            return result;
        }

        public static Polynomial KaratsubaMultiply(Polynomial p1, Polynomial p2)
        {
            int degreeMax = Math.Max(p1.Degree, p2.Degree);
            Polynomial result = new Polynomial(degreeMax * 2);

            result.Coefficients = KaratsubaMultiplyRecursive(p1.Coefficients, p2.Coefficients);

            return result;
        }

        public static Polynomial AsynchronousKaratsubaMultiply(Polynomial p1, Polynomial p2)
        {
            int degreeMax = Math.Max(p1.Degree, p2.Degree);
            Polynomial result = new Polynomial(degreeMax * 2);
            result.Coefficients = AsynchronousKaratsubaMultiplyRecursive(p1.Coefficients, p2.Coefficients);

            return result;
        }

        public static int[] KaratsubaMultiplyRecursive(int[] coefficients1, int[] coefficients2)
        {
            int maxLength = Math.Max(coefficients1.Length, coefficients2.Length);
            int[] product = new int[2 * maxLength];

            //Handle the base case where the polynomial has only one coefficient
            if (maxLength == 1)
            {
                product[0] = coefficients1[0] * coefficients2[0];
                return product;
            }

            int halfArraySize = maxLength / 2;

            //Declare arrays to hold halved factors
            int[] coefficients1Low = new int[halfArraySize];
            int[] coefficients1High = new int[halfArraySize];
            int[] coefficients2Low = new int[halfArraySize];
            int[] coefficients2High = new int[halfArraySize];

            int[] coefficients1LowHigh = new int[halfArraySize];
            int[] coefficients2LowHigh = new int[halfArraySize];

            //Fill in the low and high arrays
            for (int halfSizeIndex = 0; halfSizeIndex < halfArraySize; halfSizeIndex++)
            {

                coefficients1Low[halfSizeIndex] = coefficients1[halfSizeIndex];
                coefficients1High[halfSizeIndex] = coefficients1[halfSizeIndex + halfArraySize];
                coefficients1LowHigh[halfSizeIndex] = coefficients1Low[halfSizeIndex] + coefficients1High[halfSizeIndex];

                coefficients2Low[halfSizeIndex] = coefficients2[halfSizeIndex];
                coefficients2High[halfSizeIndex] = coefficients2[halfSizeIndex + halfArraySize];
                coefficients2LowHigh[halfSizeIndex] = coefficients2Low[halfSizeIndex] + coefficients2High[halfSizeIndex];

            }

            //Recursively call method on smaller arrays and construct the low and high parts of the product
            int[] productLow = KaratsubaMultiplyRecursive(coefficients1Low, coefficients2Low);
            int[] productHigh = KaratsubaMultiplyRecursive(coefficients1High, coefficients2High);

            int[] productLowHigh = KaratsubaMultiplyRecursive(coefficients1LowHigh, coefficients2LowHigh);

            //Construct the middle portion of the product
            int[] productMiddle = new int[maxLength];
            for (int halfSizeIndex = 0; halfSizeIndex < maxLength / 2; halfSizeIndex++)
            {
                productMiddle[halfSizeIndex] = productLowHigh[halfSizeIndex] - productLow[halfSizeIndex] - productHigh[halfSizeIndex];
            }

            //Assemble the product from the low, middle and high parts. Start with the low and high parts of the product.
            for (int halfSizeIndex = 0, middleOffset = maxLength / 2; halfSizeIndex < maxLength / 2; ++halfSizeIndex)
            {
                product[halfSizeIndex] += productLow[halfSizeIndex];
                product[halfSizeIndex + maxLength] += productHigh[halfSizeIndex];
                product[halfSizeIndex + middleOffset] += productMiddle[halfSizeIndex];
            }

            return product;

        }

        public static int[] AsynchronousKaratsubaMultiplyRecursive(int[] coefficients1, int[] coefficients2)
        {

            int maxLength = Math.Max(coefficients1.Length, coefficients2.Length);
            int[] product = new int[2 * maxLength];

            //Handle the base case where the polynomial has only one coefficient
            if (maxLength == 1)
            {
                product[0] = coefficients1[0] * coefficients2[0];
                return product;
            }

            int halfArraySize = maxLength / 2;

            //Declare arrays to hold halved factors
            int[] coefficients1Low = new int[halfArraySize];
            int[] coefficients1High = new int[halfArraySize];
            int[] coefficients2Low = new int[halfArraySize];
            int[] coefficients2High = new int[halfArraySize];

            int[] coefficients1LowHigh = new int[halfArraySize];
            int[] coefficients2LowHigh = new int[halfArraySize];

            //Fill in the low and high arrays
            for (int halfSizeIndex = 0; halfSizeIndex < halfArraySize; halfSizeIndex++)
            {

                coefficients1Low[halfSizeIndex] = coefficients1[halfSizeIndex];
                coefficients1High[halfSizeIndex] = coefficients1[halfSizeIndex + halfArraySize];
                coefficients1LowHigh[halfSizeIndex] = coefficients1Low[halfSizeIndex] + coefficients1High[halfSizeIndex];

                coefficients2Low[halfSizeIndex] = coefficients2[halfSizeIndex];
                coefficients2High[halfSizeIndex] = coefficients2[halfSizeIndex + halfArraySize];
                coefficients2LowHigh[halfSizeIndex] = coefficients2Low[halfSizeIndex] + coefficients2High[halfSizeIndex];

            }

            //Recursively call method on smaller arrays and construct the low and high parts of the product
            Task<int[]> t1 = Task<int[]>.Factory.StartNew(() =>
            {
                return AsynchronousKaratsubaMultiplyRecursive(coefficients1Low, coefficients2Low);
            });

            Task<int[]> t2 = Task<int[]>.Factory.StartNew(() =>
            {
                return AsynchronousKaratsubaMultiplyRecursive(coefficients1High, coefficients2High);
            });

            Task<int[]> t3 = Task<int[]>.Factory.StartNew(() =>
            {
                return AsynchronousKaratsubaMultiplyRecursive(coefficients1LowHigh, coefficients2LowHigh);
            });

            int[] productLow = t1.Result;
            int[] productHigh = t2.Result;
            int[] productLowHigh = t3.Result;

            //Construct the middle portion of the product
            int[] productMiddle = new int[coefficients1.Length];
            for (int halfSizeIndex = 0; halfSizeIndex < maxLength / 2; halfSizeIndex++)
            {
                productMiddle[halfSizeIndex] = productLowHigh[halfSizeIndex] - productLow[halfSizeIndex] - productHigh[halfSizeIndex];
            }

            //Assemble the product from the low, middle and high parts. Start with the low and high parts of the product.
            for (int halfSizeIndex = 0, middleOffset = maxLength / 2; halfSizeIndex < maxLength / 2; ++halfSizeIndex)
            {
                product[halfSizeIndex] += productLow[halfSizeIndex];
                product[halfSizeIndex + maxLength] += productHigh[halfSizeIndex];
                product[halfSizeIndex + middleOffset] += productMiddle[halfSizeIndex];
            }

            return product;

        }
    }
}
