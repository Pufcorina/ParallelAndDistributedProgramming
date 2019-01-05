#pragma once
#include <string>
class Polynomial
{
protected:
	unsigned int size;
	int* polynomial;


public:
	Polynomial();
	Polynomial(unsigned int size, bool initialization);
	Polynomial(Polynomial &polynomial);
	~Polynomial();
	int get(unsigned int index);
	void set(unsigned int index, int value);
	unsigned int getSize();
	int* getPolynomial();
	void randomInit();
	friend std::ostream & operator<<(std::ostream & str, Polynomial const & p);
};

