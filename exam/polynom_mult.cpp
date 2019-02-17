#include <iostream>
#include <thread>
#include <string> 
#include <vector>
#include <cmath>
using namespace std;


void mult(vector<int> a, vector<int> b, vector<int>& r, int T) {
	if (T == 1)
	{
		for (int i = 0; i < a.size(); i++)
			for (int j = 0; j < a.size(); j++)
				r[i + j] += a[i] * b[j];
	}
	else {
		thread t([&]() {
			for (int i = 0; i < a.size(); i += 2)
				for (int j = 0; j < a.size(); j++)
					r[i + j] += a[i] * b[j];
		});

		for (int i = 1; i < a.size(); i += 2)
			for (int j = 0; j < a.size(); j++)
				r[i + j] += a[i] * b[j];
		t.join();
	}
	
}

int main() {

	vector<int> a;
	vector<int> b;
	vector<int> r(6, 0);

	mult({ 1,2,3 }, { 1, 2, 3 }, r, 3);

	for (auto it : r)
		cout << it << " ";
	cout << "\n";

	return 0;
}