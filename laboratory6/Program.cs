using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Lab6
{
    class Program
    {
        public static void SynchronousMultiplication(Polynomial p1, Polynomial p2)
        {
            DateTime start = DateTime.Now;
            Polynomial result = PolynomialOperations.SynchronousMultiply(p1, p2);
            double time = (DateTime.Now - start).Milliseconds;
            Console.WriteLine("Synchronous Multiplication: " + result.ToString() + "\n" + time + " milliseconds");
        }

        public static void AsynchronousMultiplication(Polynomial p1, Polynomial p2)
        {
            DateTime start = DateTime.Now;
            Polynomial result = PolynomialOperations.AsynchronousMultiply(p1, p2);
            double time = (DateTime.Now - start).Milliseconds;
            Console.WriteLine("Asynchronous Multiplication: " + result.ToString() + "\n" + time + " milliseconds");
        }

        public static void SynchronousKaratsuba(Polynomial p1, Polynomial p2)
        {
            DateTime start = DateTime.Now;
            Polynomial result = PolynomialOperations.KaratsubaMultiply(p1, p2);
            double time = (DateTime.Now - start).Milliseconds;
            Console.WriteLine("Synchronous Karatsuba: " + result.ToString() + "\n" + time + " milliseconds");
        }

        public static void AsynchronousKaratsuba(Polynomial p1, Polynomial p2)
        {
            DateTime start = DateTime.Now;
            Polynomial result = PolynomialOperations.AsynchronousKaratsubaMultiply(p1, p2);
            double time = (DateTime.Now - start).Milliseconds;
            Console.WriteLine("Asynchronous Karatsuba: " + result.ToString() + "\n" + time + " milliseconds");
        }
        static void Main(string[] args)
        {
            int firstLength = 3;
            int secondLength = 5;
            Polynomial polynomial1 = new Polynomial(firstLength);
            polynomial1.GenerateRandom();
            Thread.Sleep(500);
            Polynomial polynomial2 = new Polynomial(secondLength);
            polynomial2.GenerateRandom();

            if (firstLength > secondLength)
                polynomial2 = polynomial2.AddZerosLeft(firstLength - secondLength);
            else if (secondLength > firstLength)
                polynomial1 = polynomial1.AddZerosLeft(secondLength - firstLength);

            SynchronousMultiplication(polynomial1, polynomial2);
            AsynchronousMultiplication(polynomial1, polynomial2);
            SynchronousKaratsuba(polynomial1, polynomial2);
            AsynchronousKaratsuba(polynomial1, polynomial2);
        }
    }
}
