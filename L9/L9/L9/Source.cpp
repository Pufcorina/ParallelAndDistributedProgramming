#include <iostream>
#include "mpi.h"
#include "KaratsubaPolynomial.h"

using namespace std;

void main(int argc, char **argv) {
	MPI_Init(&argc, &argv);

	Polynomial *p = new KaratsubaPolynomial(5, true);
	Polynomial *p2 = new KaratsubaPolynomial(5, false);

	cout << *p;
	cout << *p2;

	MPI_Finalize();
}