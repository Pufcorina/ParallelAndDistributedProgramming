#include <iostream>
#include <vector>
#include <math.h>
#include <thread>
#include <numeric>
#include <mutex>  

using namespace std;

std::mutex mtx;           // mutex for critical section

void getPrimesLinearly(vector<int> & primes, int n)
{
	for (int x = 2; x <= n; x++)
	{
		bool isPrime = true;
		for (int y = 2; y < x / 2; y++)
		{
			if (x % y == 0)
			{
				isPrime = false;
				break;
			}
		}
		if (isPrime)
		{
			primes.push_back(x);
		}
	}
}

void crossInvalidNumebrs(int x, vector<int> & primes, int n)
{
	int y = x + x;
	while (y <= n)
	{
		mtx.lock();

		auto it = std::find(primes.begin(), primes.end(), y);
		if (it != primes.end())
			primes.erase(it);

		mtx.unlock();
		y += x;
	}
}

void getPrimesThreads(vector<int> & primes, int n, int T)
{
	primes.resize(n - 1);
	std::iota(std::begin(primes), std::end(primes), 2);
	int sqN = sqrt(n);
	vector<int> firstPrimes;
	getPrimesLinearly(firstPrimes, sqN);

	vector<thread> threads(T);
	int i;
	for (i = 0; i < firstPrimes.size() && i < T; i++)
	{
		threads[i] = thread([&]() {
			crossInvalidNumebrs(firstPrimes[i], primes, n);
		});
	}
	while (i < firstPrimes.size())
	{
		crossInvalidNumebrs(firstPrimes[i], primes, n);
		i++;
	}

	for (i = 0; i < T; i++)
	{
		threads[i].join();
	}

}


int main()
{
	vector<int> primes;
	getPrimesThreads(primes, 100, 4);


	for (auto it : primes)
		cout << it << " ";
	cout << "\n";
	return 0;
}