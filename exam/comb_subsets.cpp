#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>

using namespace std;

atomic <int> cnt;

bool predicate(vector<int> configuration)
{
	if (configuration[0] % 2 == 0)
		return true;

	return true;
}

inline void backtracking(vector <int> configuration, int n, int k, int pos, int T) {

	if (pos == k) {
		if (predicate(configuration)) {
			for (auto it : configuration)
				cout << it << " ";
			cout << "\n";
			cnt++;
		}
			
		return;
	}

	int last = -1;
	if (configuration.size() > 0)
		last = configuration.back();

	if (T == 1) {
		for (int i = last + 1; i < n; ++i) {
			configuration.push_back(i);
			backtracking(configuration, n, k, pos + 1, T);
			configuration.pop_back();
		}
	}
	else {
		thread t([&]() {
			vector <int> newPath(configuration);
			for (int i = last + 1; i < n; i += 2) {
				newPath.push_back(i);
				backtracking(newPath, n, k, pos + 1, T / 2);
				newPath.pop_back();
			}
		});
		vector <int> aux(configuration);
		for (int i = last + 2; i < n; i += 2) {
			aux.push_back(i);
			backtracking(aux, n, k, pos + 1, T - T / 2);
			aux.pop_back();
		}
		t.join();
	}
}

int main() {
	backtracking(vector<int>(), 5, 3, 0, 2);
	cout << cnt << "\n";
}