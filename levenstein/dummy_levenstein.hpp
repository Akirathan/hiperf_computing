#ifndef LEVENSTEIN_TRIVIAL_LEVENSTEIN_HPP
#define LEVENSTEIN_TRIVIAL_LEVENSTEIN_HPP

#include <vector>
#include <algorithm>

class dummy_levenstein {
public:
    using data_element = int;

    template< typename I1, typename I2>
    dummy_levenstein(I1 i1b, I1 i1e, I2 i2b, I2 i2e)
        : array_1(i1b, i1e),
        array_2(i2b, i2e),
        results_rows{array_1.size() + 1},
        results_cols{array_2.size() + 1}
    {
        results.resize(results_rows);
        for (size_t i = 0; i < results_rows; ++i) {
            results[i].resize(results_cols);
        }
    }

    data_element compute()
    {
        fill_boundaries();

        for (size_t i = 1; i < results_rows; ++i) {
            for (size_t j = 1; j < results_cols; ++j) {
                results[i][j] = compute_at_index(i, j);
            }
        }

        return results[results_rows - 1][results_cols - 1];
    }

private:
    std::vector<data_element> array_1;
    std::vector<data_element> array_2;
    std::vector<std::vector<data_element>> results;
    size_t results_rows;
    size_t results_cols;

    data_element compute_at_index(size_t i, size_t j) const
    {
        data_element first = results[i-1][j] + 1;
        data_element second = results[i-1][j-1] + (array_1[i-1] == array_2[j-1] ? 0 : 1);
        data_element third = results[i][j-1] + 1;

        return std::min({first, second, third});
    }

    void fill_boundaries()
    {
        results[0][0] = 0;
        for (size_t i = 0; i < results_rows; ++i) {
            results[i][0] = static_cast<int>(i);
        }

        for (size_t j = 0; j < results_cols; ++j) {
            results[0][j] = static_cast<int>(j);
        }
    }
};

#endif //LEVENSTEIN_TRIVIAL_LEVENSTEIN_HPP
