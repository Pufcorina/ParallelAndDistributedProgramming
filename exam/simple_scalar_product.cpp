#include <iostream>
#include <thread>
#include <vector>

using namespace std;

ostream& operator<<(ostream& stream, vector<pair<int, int> > v) {
	for (int i = 0; i < v.size(); i++) {
		stream << i << ": " << v[i].first << " -> " << v[i].second << "\n";
	}

	return stream;
}

vector< pair<int, int> > splitWorkload(int n, int t) {
	vector< pair<int, int> > intervals;

	int index = 0;
	int step = n / t;
	int mod = n % t;

	while (index < n) {
		intervals.push_back(pair<int, int>(index, index + step + (mod > 0)));
		index += step + (mod > 0);
		mod--;
	}

	return intervals;
}

int scalarProduct(vector<int> a, vector<int> b, int T) {

	vector<int> sums(a.size(), 0);
	vector<thread> threads;
	threads.resize(T);
	int final_sum = 0;

	vector< pair<int, int> > intervals = splitWorkload(a.size(), T);
	cout << intervals;
	for (int i = 0; i < T; i++) {
		threads[i] = thread([&, i]() {
			for (int k = intervals[i].first; k < intervals[i].second; k++) {
				sums[i] += a[k] * b[k];
			}
		});
	}

	for (int i = 0; i < T; i++)
	{
		final_sum += sums[i];
		threads[i].join();
	}
	return final_sum;
}

int main() {

	cout << scalarProduct({ 1, 2, 3, 4 }, { 2, 3, 4, 5 }, 4);
	return 0;
}