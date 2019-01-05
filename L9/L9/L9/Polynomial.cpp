#include "Polynomial.h"
#include <algorithm>


Polynomial::Polynomial(){}

Polynomial::Polynomial(unsigned int size, bool initialization) {
	this->size = size;
	this->polynomial = new int[size];
	if (initialization)
		randomInit();
	else
		for (int i = 0; i < this->size; ++i)
			this->polynomial[i] = 0;
}

Polynomial::Polynomial(Polynomial & polynomial) {
	this->size = polynomial.size;
	this->polynomial = new int[size];

	memcpy(this->polynomial, polynomial.polynomial, size * sizeof(int));
}


Polynomial::~Polynomial(){}

int Polynomial::get(unsigned int index) {
	return this->polynomial[index];
}

void Polynomial::set(unsigned int index, int value) {
	this->polynomial[index] = value;
}

unsigned int Polynomial::getSize() {
	return this->size;
}

int * Polynomial::getPolynomial() {
	return this->polynomial;
}

void Polynomial::randomInit() {
	for (int i = 0; i < this->size; ++i)
		this->polynomial[i] = rand();
}


 std::ostream & operator<<(std::ostream & str, Polynomial const & p) {
	 for (int i = 0; i < p.size; i++)
		 str << p.polynomial[i] << ' ';
	 str << '\n';
	 return str;
 }
