#pragma once
#include "Polynomial.h"

class KaratsubaPolynomial : public Polynomial
{
public:
	KaratsubaPolynomial(); 
	KaratsubaPolynomial(unsigned int size, bool initialization) : Polynomial(size, initialization) {}
	KaratsubaPolynomial(Polynomial &polynomial): Polynomial(polynomial) {}
	~KaratsubaPolynomial();
};

