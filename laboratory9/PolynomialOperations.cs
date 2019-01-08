using MPI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PPD_MPI
{
    public class PolynomialOperations
    {

        public static Polynomial MPIMultiply(Polynomial polynomial1, Polynomial polynomial2, int begin, int end)
        {
            Polynomial result;
            int maxDegree = Math.Max(polynomial1.Degree, polynomial2.Degree);
            result = new Polynomial(maxDegree * 2);

            for (int i = begin; i < end; i++)
                for (int j = 0; j < polynomial2.size; j++)
                    result.Coefficients[i + j] += polynomial1.Coefficients[i] * polynomial2.Coefficients[j];

            return result;
        }

        public static Polynomial AsynchronousKaratsubaMultiply(Polynomial p1, Polynomial p2)
        {
            Polynomial result = new Polynomial(p1.Degree + p2.Degree);
            result.Coefficients = AsynchronousKaratsubaMultiplyRecursive(p1.Coefficients, p2.Coefficients);

            return result;
        }
        
        public static int[] AsynchronousKaratsubaMultiplyRecursive(int[] coefficients1, int[] coefficients2)
        {

            int[] product = new int[2 * coefficients1.Length];

            //Handle the base case where the polynomial has only one coefficient
            if (coefficients1.Length == 1)
            {
                product[0] = coefficients1[0] * coefficients2[0];
                return product;
            }

            int halfArraySize = coefficients1.Length / 2;

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
            for (int halfSizeIndex = 0; halfSizeIndex < coefficients1.Length; halfSizeIndex++)
                productMiddle[halfSizeIndex] = productLowHigh[halfSizeIndex] - productLow[halfSizeIndex] - productHigh[halfSizeIndex];

            //Assemble the product from the low, middle and high parts. Start with the low and high parts of the product.
            for (int halfSizeIndex = 0, middleOffset = coefficients1.Length / 2; halfSizeIndex < coefficients1.Length; ++halfSizeIndex)
            {
                product[halfSizeIndex] += productLow[halfSizeIndex];
                product[halfSizeIndex + coefficients1.Length] += productHigh[halfSizeIndex];
                product[halfSizeIndex + middleOffset] += productMiddle[halfSizeIndex];
            }

            return product;

        }

        public static void MPIKaratsubaMultiply()
        {

            int from = Communicator.world.Receive<int>(Communicator.anySource, 0);
            int[] coefficients1 = Communicator.world.Receive<int[]>(Communicator.anySource, 0);
            int[] coefficients2 = Communicator.world.Receive<int[]>(Communicator.anySource, 0);
            int[] sendTo = Communicator.world.Receive<int[]>(Communicator.anySource, 0);

            int[] product = new int[2 * coefficients1.Length];

            //Handle the base case where the polynomial has only one coefficient
            if (coefficients1.Length == 1)
            {
                product[0] = coefficients1[0] * coefficients2[0];

                Communicator.world.Send<int[]>(product, from, 0);
                return;
            }

            int halfArraySize = coefficients1.Length / 2;

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
            int[] productLow, productHigh, productLowHigh;

            if (sendTo.Length == 0)
            {
                productLow = AsynchronousKaratsubaMultiplyRecursive(coefficients1Low, coefficients2Low);
                productHigh = AsynchronousKaratsubaMultiplyRecursive(coefficients1High, coefficients2High);
                productLowHigh = AsynchronousKaratsubaMultiplyRecursive(coefficients1LowHigh, coefficients2LowHigh);
            }
            else if (sendTo.Length == 1)
            {
                Communicator.world.Send<int>(Communicator.world.Rank, sendTo[0], 0);
                Communicator.world.Send<int[]>(coefficients1Low, sendTo[0], 0);
                Communicator.world.Send<int[]>(coefficients2Low, sendTo[0], 0);
                Communicator.world.Send<int[]>(new int[0], sendTo[0], 0);
                
                productHigh = AsynchronousKaratsubaMultiplyRecursive(coefficients1High, coefficients2High);
                productLowHigh = AsynchronousKaratsubaMultiplyRecursive(coefficients1LowHigh, coefficients2LowHigh);

                productLow = Communicator.world.Receive<int[]>(sendTo[0], 0);
            }
            else if (sendTo.Length == 2)
            {
                Communicator.world.Send<int>(Communicator.world.Rank, sendTo[0], 0);
                Communicator.world.Send<int[]>(coefficients1Low, sendTo[0], 0);
                Communicator.world.Send<int[]>(coefficients2Low, sendTo[0], 0);
                Communicator.world.Send<int[]>(new int[0], sendTo[0], 0);

                Communicator.world.Send<int>(Communicator.world.Rank, sendTo[1], 0);
                Communicator.world.Send<int[]>(coefficients1High, sendTo[1], 0);
                Communicator.world.Send<int[]>(coefficients2High, sendTo[1], 0);
                Communicator.world.Send<int[]>(new int[0], sendTo[1], 0);

                productLowHigh = AsynchronousKaratsubaMultiplyRecursive(coefficients1LowHigh, coefficients2LowHigh);

                productLow = Communicator.world.Receive<int[]>(sendTo[0], 0);
                productHigh = Communicator.world.Receive<int[]>(sendTo[1], 0);
            }
            else if(sendTo.Length == 3)
            {
                Communicator.world.Send<int>(Communicator.world.Rank, sendTo[0], 0);
                Communicator.world.Send<int[]>(coefficients1Low, sendTo[0], 0);
                Communicator.world.Send<int[]>(coefficients2Low, sendTo[0], 0);
                Communicator.world.Send<int[]>(new int[0], sendTo[0], 0);
                

                Communicator.world.Send<int>(Communicator.world.Rank, sendTo[1], 0);
                Communicator.world.Send<int[]>(coefficients1High, sendTo[1], 0);
                Communicator.world.Send<int[]>(coefficients2High, sendTo[1], 0);
                Communicator.world.Send<int[]>(new int[0], sendTo[1], 0);

                Communicator.world.Send<int>(Communicator.world.Rank, sendTo[2], 0);
                Communicator.world.Send<int[]>(coefficients1LowHigh, sendTo[2], 0);
                Communicator.world.Send<int[]>(coefficients2LowHigh, sendTo[2], 0);
                Communicator.world.Send<int[]>(new int[0], sendTo[2], 0);

                productLow = Communicator.world.Receive<int[]>(sendTo[0], 0);
                productHigh = Communicator.world.Receive<int[]>(sendTo[1], 0);
                productLowHigh = Communicator.world.Receive<int[]>(sendTo[2], 0);
            }
            else
            {
                int[] auxSendTo = sendTo.Skip(3).ToArray();
                int auxLength = auxSendTo.Length / 3;

                Communicator.world.Send<int>(Communicator.world.Rank, sendTo[0], 0);
                Communicator.world.Send<int[]>(coefficients1Low, sendTo[0], 0);
                Communicator.world.Send<int[]>(coefficients2Low, sendTo[0], 0);
                Communicator.world.Send<int[]>(auxSendTo.Take(auxLength).ToArray(), sendTo[0], 0);

                Communicator.world.Send<int>(Communicator.world.Rank, sendTo[1], 0);
                Communicator.world.Send<int[]>(coefficients1High, sendTo[1], 0);
                Communicator.world.Send<int[]>(coefficients2High, sendTo[1], 0);
                Communicator.world.Send<int[]>(auxSendTo.Skip(auxLength).Take(auxLength).ToArray(), sendTo[1], 0);

                Communicator.world.Send<int>(Communicator.world.Rank, sendTo[2], 0);
                Communicator.world.Send<int[]>(coefficients1LowHigh, sendTo[2], 0);
                Communicator.world.Send<int[]>(coefficients2LowHigh, sendTo[2], 0);
                Communicator.world.Send<int[]>(auxSendTo.Skip(2 * auxLength).ToArray(), sendTo[2], 0);

                productLow = Communicator.world.Receive<int[]>(sendTo[0], 0);
                productHigh = Communicator.world.Receive<int[]>(sendTo[1], 0);
                productLowHigh = Communicator.world.Receive<int[]>(sendTo[2], 0);
            }

            //Construct the middle portion of the product
            int[] productMiddle = new int[coefficients1.Length];
            for (int halfSizeIndex = 0; halfSizeIndex < coefficients1.Length; halfSizeIndex++)
            {
                productMiddle[halfSizeIndex] = productLowHigh[halfSizeIndex] - productLow[halfSizeIndex] - productHigh[halfSizeIndex];
            }

            //Assemble the product from the low, middle and high parts. Start with the low and high parts of the product.
            for (int halfSizeIndex = 0, middleOffset = coefficients1.Length / 2; halfSizeIndex < coefficients1.Length; ++halfSizeIndex)
            {
                product[halfSizeIndex] += productLow[halfSizeIndex];
                product[halfSizeIndex + coefficients1.Length] += productHigh[halfSizeIndex];
                product[halfSizeIndex + middleOffset] += productMiddle[halfSizeIndex];
            }

            Communicator.world.Send<int[]>(product, from, 0);

        }
    }
}
