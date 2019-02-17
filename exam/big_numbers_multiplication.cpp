#include <iostream>
#include <fstream>
#include <vector>
#include <thread>

using namespace std;

inline vector <int> solve(vector <int> a, vector <int> b, int T) {
	vector <int> res;
	int n = a.size();
	int m = 2 * n - 1;
	res.resize(m, 0);
	vector <thread> thr;
	for (int idx = 0; idx < T; ++idx) {
		thr.push_back(thread([&, idx]() {
			for (int x = idx; x < m; x += T) {
				for (int i = 0; i < n; ++i) {
					if (x - i >= n || x - i < 0) {
						continue;
					}
					res[x] += a[i] * b[x - i];
				}
			}
		}));
	}
	for (int i = 0; i < thr.size(); ++i) {
		thr[i].join();
	}
	vector<int> result;
	int carry = 0;
	for (int i = res.size() - 1; i >= 0; i--)
	{
		res[i] += carry;
		result.push_back(res[i] % 10);
		if (res[i] > 9)
			carry = res[i] / 10;
		else
			carry = 0;
	}
	while (carry > 0)
	{
		result.push_back(carry % 10);
		carry /= 10;
	}
	reverse(result.begin(), result.end());
	return result;
}

int main() {

	for (auto it : solve({ 1, 2, 3, 4 }, { 1, 2, 3, 4 }, 3))
		cout << it << " ";
	cout << "\n";
}