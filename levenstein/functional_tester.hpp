#ifndef MY_MAIN_FUNCTIONAL_TESTER_HPP
#define MY_MAIN_FUNCTIONAL_TESTER_HPP

#include <initializer_list>
#include <cassert>
#include <iostream>
#include "du4levenstein.hpp"
#include "dummy_levenstein.hpp"

template <typename policy>
class FunctionalTester {
public:
    void run_all_tests()
    {
        rectangle_tests();
        std::cout << "Rectangle tests passed" << std::endl;
        functional_tests();
        bigger_functional_tests();
        std::cout << "All functional tests passed" << std::endl;
    }

private:
    static constexpr size_t max_vector_size = 1024;

    void rectangle_tests()
    {
        auto tmp_arr = {1,2};
        levenstein<policy_sse> lev{tmp_arr.begin(), tmp_arr.end(), tmp_arr.begin(), tmp_arr.end()};

        levenstein<policy_sse>::Rectangle rectangle{5, 5};
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                rectangle.set(i, j, j);
            }
        }

        auto diag = rectangle.get_diagonal(4, 0, 4);
        assert(diag[0] == 0);
        assert(diag[1] == 1);
        assert(diag[2] == 2);
        assert(diag[3] == 3);

        diag = rectangle.get_diagonal(3, 0, 4);
        assert(diag[0] == 0);
        assert(diag[1] == 1);
        assert(diag[2] == 2);
        assert(diag[3] == 3);

        diag = rectangle.get_diagonal(2, 1, 3);
        assert(diag[0] == 1);
        assert(diag[1] == 2);
        assert(diag[2] == 3);
    }

    void functional_tests()
    {
        // One same element in input arrays.
        auto a1 = {1, 2, 3, 4, 5};
        auto a2 = {9, 8, 3, 7, 1};
        compare_both(a1.begin(), a1.end(), a2.begin(), a2.end(), 4);


        a1 = {1, 3, 5, 7, 9};
        a2 = {2, 4, 6, 8, 10};
        compare_both(a1.begin(), a1.end(), a2.begin(), a2.end(), 5);

        // Zero distance.
        a1 = {1, 2, 3};
        a2 = {1, 2, 3};
        compare_both(a1.begin(), a1.end(), a2.begin(), a2.end(), 0);

        // Different sizes.
        a1 = {1, 2, 3};
        a2 = {4, 2, 1, 5};
        compare_both(a1.begin(), a1.end(), a2.begin(), a2.end());
    }

    void bigger_functional_tests()
    {
        for (size_t i = 0; i < 5; ++i) {
            std::vector<int> vec1 = random_vector();
            std::vector<int> vec2 = random_vector();

            compare_both(vec1.begin(), vec1.end(), vec2.begin(), vec2.end());
        }
    }

    template <typename It1, typename It2>
    void compare_both(It1 i1b, It1 i1e, It2 i2b, It2 i2e)
    {
        int std_impl_res = run_levenstein(i1b, i1e, i2b, i2e);
        int dummy_impl_res = run_dummy(i1b, i1e, i2b, i2e);
        assert(std_impl_res == dummy_impl_res);
    }

    template <typename It1, typename It2>
    void compare_both(It1 i1b, It1 i1e, It2 i2b, It2 i2e, int expected)
    {
        int std_impl_res = run_levenstein(i1b, i1e, i2b, i2e);
        int dummy_impl_res = run_dummy(i1b, i1e, i2b, i2e);
        assert(std_impl_res == dummy_impl_res);
        assert(std_impl_res == expected);
    }

    template <typename It1, typename It2>
    int run_levenstein(It1 i1b, It1 i1e, It2 i2b, It2 i2e)
    {
        levenstein<policy> std_impl{i1b, i1e, i2b, i2e};
        return std_impl.compute();
    }

    template <typename It1, typename It2>
    int run_dummy(It1 i1b, It1 i1e, It2 i2b, It2 i2e)
    {
        dummy_levenstein dummy_impl{i1b, i1e, i2b, i2e};
        return dummy_impl.compute();
    }

    std::vector<int> random_vector() const
    {
        size_t size = 0;
        while (size < 10) {
            size = std::rand() % max_vector_size;
        }
        std::vector<int> vec;
        for (size_t i = 0; i < size; ++i) {
            int rand_int = std::rand();
            vec.push_back(rand_int);
        }
        return vec;
    }
};

#endif //MY_MAIN_FUNCTIONAL_TESTER_HPP
