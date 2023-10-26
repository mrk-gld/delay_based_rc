
#include <armadillo>
#include <cmath>

using namespace arma;
using namespace std;

#define MAX_RANGE_AFTER_PEAK 8
std::vector<int> MAX_RANGE_AFTER_PEAK_ARRAY{3, 3, 3, 3};

vec generateRandomInputSequenceUni(const int input_length) {

    vec input(input_length);
    mt19937 generator_global(time(0));
    uniform_real_distribution<double> uni_dist_global(-1.0, 1.0);
    for (int i = 0; i < input_length; i++) {
        input[i] = uni_dist_global(generator_global);
    }
    return input;
}

double calculateMemoryCapacityFast(arma::Mat<double>& xT, arma::Mat<double>& ssti, vec& y)
    {
        y -= mean(y);
        y /= stddev(y);
        const auto ny = arma::norm(y);
        const double recNrows = 1.f / double(y.n_elem);

        vec P = xT.t() * y * recNrows;
        mat ghs = P.t() * ssti * P / (ny * ny * recNrows);

        return ghs(0);
    }

// Print all members of ar[] that have given
void findNumbers(vector<int> &ar, int sum, vector<vector<int>> &res, vector<int> &r, int i) {
    // If  current sum becomes negative
    if (sum < 0)
        return;

    // if we get exact answer
    if (sum == 0) {
        res.push_back(r);
        return;
    }

    // Recur for all remaining elements that
    // have value smaller than sum.
    while (i < ar.size() && sum - ar[i] >= 0) {
        // Till every element in the array starting
        // from i which can contribute to the sum
        r.push_back(ar[i]); // add them to list

        // recur for next numbers
        findNumbers(ar, sum - ar[i], res, r, i);
        i++;

        // remove number from list (backtracking)
        r.pop_back();
    }
}

// Returns all combinations of ar[] that have given
// sum.
vector<vector<int>> combinationSum(vector<int> &ar, int sum) {
    // sort input array
    sort(ar.begin(), ar.end());

    // remove duplicates
    ar.erase(unique(ar.begin(), ar.end()), ar.end());

    vector<int> r;
    vector<vector<int>> res;
    findNumbers(ar, sum, res, r, 0);

    return res;
}

// Function to find the permutations
void findPermutations(vector<int> a, vector<vector<int>> &allPerm) {
    // Sort the given array
    sort(a.begin(), a.end());

    // Find all possible permutations
    do {
        allPerm.push_back(a);
    } while (next_permutation(a.begin(), a.end()));
}

vector<vector<int>> getPowerListsByTotalDegree(const int totalDegree, const int numVar) {
    vector<int> ar;

    for (int i = 1; i <= totalDegree; i++) {
        ar.push_back(i);
    }
    int n = ar.size();
    vector<vector<int>> comb = combinationSum(ar, totalDegree);

    vector<int> stepsFalse;
    for (int i = 0; i < comb.size(); i++) {
        if (comb[i].size() == numVar) {
            stepsFalse.push_back(i);
        }
    }

    vector<vector<int>> res;
    for (int i = 0; i < stepsFalse.size(); i++) {
        res.push_back(comb[stepsFalse[i]]);
    }

    vector<vector<int>> resPerm;
    for (int i = 0; i < res.size(); i++) {
        findPermutations(res[i], resPerm);
    }

    return resPerm;
}

vector<pair<int, int>> getInitIdx(vector<int> &powerlist, const int delay) {
    vector<pair<int, int>> stepses;

    for (int i = 0; i < powerlist.size(); i++) {
        pair<int, int> steps(delay + i, powerlist[i]);
        stepses.push_back(steps);
    }
    return stepses;
}

double legendreFunction(int n, double x) {
    if (n == 0)
        return 1;
    else if (n == 1)
        return x;
    else
        return ((2.0 * double(n) - 1.0) * x * legendreFunction(n - 1, x) -
                (double(n) - 1.0) * legendreFunction(n - 2, x)) / double(n);
}

vec setNonlinearMemoryTask(vec input,vector<pair<int, int>> steps, const int train_start) {
    
    vec targets = vec(input.size() - train_start, fill::ones);

    for (auto it = steps.begin(); it != steps.end(); it++) {
        double *dst = targets.memptr();
        const int pos = it->first;
        const int deg = it->second;
        double *mv_inputPtr = &input.memptr()[train_start - pos];
        for (int i = train_start; i < input.size(); i++) {
            dst[0] *= legendreFunction(deg, *mv_inputPtr);
            mv_inputPtr++;
            dst++;
        }
    }

    return targets;
} // setNonlinearMemoryTask

bool getNext(vector<pair<int, int>> &steps,
            vector<int> NUMBER_LOW_MEM_ACC_LAST_PEAKS,
            vector<pair<int, int>> &startSteps, 
            const int pastMax,
            const int pos,
            bool force) {

    const int laststeps = steps.size() - 1;
    const int laststepsPast = steps[pos].first;

    bool checkEnd = true;

    if (steps.size() == 1 || pos == 0) {
        return false;
    }

    for (int i = 1; i < steps.size(); i++) {
        const int stepsPast = steps[i].first;
        const int stepsPastMax = pastMax - steps.size() + i;
        if (stepsPast != stepsPastMax) {
            checkEnd = false;
        }
    }
    if (!checkEnd) {

        if (force) {
            if (pos >= 1) {
                if (steps[pos - 1].first > startSteps[pos - 1].first +
                                                NUMBER_LOW_MEM_ACC_LAST_PEAKS[pos - 1] +
                                                MAX_RANGE_AFTER_PEAK_ARRAY.at(min(3, pos))) {
                    NUMBER_LOW_MEM_ACC_LAST_PEAKS[pos] = 0;
                    return getNext(steps, NUMBER_LOW_MEM_ACC_LAST_PEAKS, startSteps, pastMax,
                                    pos - 1, true);
                }
                NUMBER_LOW_MEM_ACC_LAST_PEAKS[pos] = 0;
                return getNext(steps, NUMBER_LOW_MEM_ACC_LAST_PEAKS, startSteps, pastMax, pos - 1,
                                false);
            }
        }

        const int stepsPastMax = pastMax - steps.size() + pos;
        if (laststepsPast != stepsPastMax) {
            steps[pos].first = steps[pos].first + 1;
            if (pos != laststeps) {
                int p = 0;
                for (int i = pos; i < steps.size(); i++, p++) {
                    steps[i].first = steps[pos].first + p;
                }
            }

            return true;

        } else {
            return false;
        }
    }
    return !checkEnd;
} // getNext
