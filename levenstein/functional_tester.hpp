#ifndef MY_MAIN_FUNCTIONAL_TESTER_HPP
#define MY_MAIN_FUNCTIONAL_TESTER_HPP

#include <initializer_list>
#include <cassert>
#include <iostream>
#include <random>
#include <algorithm>
#include <sstream>
#include "exception.hpp"
#include "du4levenstein.hpp"
#include "dummy_levenstein.hpp"

template <typename policy>
class FunctionalTester {
public:
    void run_all_tests()
    {
        rectangle_tests();
        std::cout << "\tRectangle tests passed" << std::endl;
        functional_tests();
        arrays_swap_functional_tests();
        bednarek_random_tests();
        detailed_functional_tests_sse();
#ifdef USE_AVX512
        detailed_functional_tests_avx512();
#endif
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

        // Input string with length 4
        a1 = {1, 2, 3, 4};
        a2 = {5, 2, 5, 1};
        compare_both(a1.begin(), a1.end(), a2.begin(), a2.end());
    }

    /**
     * Checks whether compute(I1,I2) is same as compute(I2,I1).
     * TODO: This is a temporary function.
     */
    void arrays_swap_functional_tests()
    {
        std::cout << "\tArrays swap functional tests:" << std::endl;
        for (size_t i = 0; i < 10; ++i) {
            std::vector<int> a1 = random_sized_random_vector();
            std::vector<int> a2 = random_sized_random_vector();

            dummy_levenstein dummy_normal{a1.begin(), a1.end(), a2.begin(), a2.end()};
            int res_normal = dummy_normal.compute();

            dummy_levenstein dummy_swapped{a2.begin(), a2.end(), a1.begin(), a1.end()};
            int res_swapped = dummy_swapped.compute();

            assert(res_normal == res_swapped);
        }
        std::cout << "\tArrays swap functional tests passed" << std::endl;
    }

    void bednarek_random_tests()
    {
        std::cout << "\tBednarek's random tests started" << std::endl;

        const std::vector<std::pair<size_t, size_t>> sizes = {
                {512, 64}, {4096, 64}, {4096, 512}, {32768, 64}, {32768, 512}, {32768, 4096}
        };

        /*const std::vector<std::pair<size_t, size_t>> sizes = {
                {512, 64}
        };*/
        const size_t iters = 2;

        std::uniform_int_distribution<int> ui{0, 255};
        std::mt19937 engine;

        for (auto &&size : sizes) {
            // Reset engine.
            engine.seed();

            std::vector<int> vec_a(size.first);
            std::vector<int> vec_b(size.second);
            std::generate(vec_a.begin(), vec_a.end(), [&ui, &engine] {
                return ui(engine);
            });
            std::generate(vec_b.begin(), vec_b.end(), [&ui, &engine] {
                return ui(engine);
            });

            std::cout << "\t\tCHKSUM A[" << vec_a.size() << "] = " << bednarek_chksum(vec_a.begin(), vec_a.end()) << std::endl;
            std::cout << "\t\tCHKSUM B[" << vec_b.size() << "] = " << bednarek_chksum(vec_b.begin(), vec_b.end()) << std::endl;

            levenstein<policy> std_impl{vec_a.begin(), vec_a.end(), vec_b.begin(), vec_b.end()};
            dummy_levenstein dummy_impl{vec_a.begin(), vec_a.end(), vec_b.begin(), vec_b.end()};
            for (size_t i = 0; i < iters; ++i) {
                int std_res = std_impl.compute();
                int dummy_res = dummy_impl.compute();
                if (std_res != dummy_res) {
                    std::stringstream ss;
                    ss << "Iteration=" << i <<", std_result=" << std_res << ", dummy_result=" << dummy_res;
                    throw Exception{ss.str()};
                }
            }
        }

        std::cout << "\tBednarek's random tests passed" << std::endl;
    }

    template<typename IT>
    static std::uint64_t bednarek_chksum(IT b, IT e)
    {
        std::uint64_t s = 0;

        for (; b != e; ++b) {
            auto x = *b;
            s = s * 3 + (std::uint64_t) x;
        }

        return s;
    }

    void detailed_functional_tests_sse()
    {
        auto a1 = {1, 2, 3, 4, 5};
        auto a2 = {9, 8, 3, 7, 1};
        levenstein<policy_sse> lev{a1.begin(), a1.end(), a2.begin(), a2.end()};

        lev.compute_first_part_of_stripe(4);
        assert(lev.row[0] == 4);
        assert_vector_equals(lev.vector_y, {4, 3, 2, 3});
        assert_vector_equals(lev.vector_w, {3, 2, 3, 4});
        assert_vector_equals(lev.vector_z, {3, 2, 2, 3});

        lev.compute_one_stripe_diagonal(4, 1);
        assert(lev.row[1] == 4);
        assert_vector_equals(lev.vector_y, {4, 3, 3, 4});
        assert_vector_equals(lev.vector_w, {3, 3, 4, 5});
        assert_vector_equals(lev.vector_z, {3, 2, 3, 4});

        lev.compute_one_stripe_diagonal(4, 2);
        assert(lev.row[2] == 4);
        assert_vector_equals(lev.vector_y, {4, 2, 4, 5});
        assert_vector_equals(lev.vector_z, {3, 3, 4, 5});

        lev.compute_last_part_of_stripe(4);
        assert(lev.row[3] == 3);
        assert(lev.row[4] == 3);
        assert(lev.row[5] == 4);

        std::cout << "\tFunctionalTester: detailed_functional_tests_sse passed" << std::endl;
    }

#ifdef USE_AVX512
    void detailed_functional_tests_avx512()
    {
        auto a1 = {23,25,13,15,18,20,23,25,13,18,25,11,12,17,20,23,24};
        auto a2 = {30,30,30,15,18,20,30,30,30,30,30,30,30,30,20,23};
        assert(a2.size() == 16);
        assert(a1.size() == 17);

        levenstein<policy_avx512> lev{a1.begin(), a1.end(), a2.begin(), a2.end()};
        lev.compute_first_part_of_stripe(16);
        assert(lev.row[0] == 16);
        assert_vector512_equals(lev.vector_y, {16,15,14,13,11,9,7,6,5,6,7,9,11,13,14,15});
        assert_vector512_equals(lev.vector_w, {15,14,13,11,9,7,6,5,6,7,9,11,13,14,15,16});
        assert_vector512_equals(lev.vector_z, {15,14,13,12,10,8,6,5,5,6,8,10,12,13,14,15});

        lev.compute_one_stripe_diagonal(16, 1);
        assert(lev.row[1] == 15);
        assert_vector512_equals(lev.vector_y, {15,15,14,12,10,8,7,6,6,7,8,10,12,14,15,16});
        assert_vector512_equals(lev.vector_w, {15,14,12,10,8,7,6,6,7,8,10,12,14,15,16,17});

        lev.compute_one_stripe_diagonal(16, 2);
        assert(lev.row[2] == 16);
        assert_vector512_equals(lev.vector_y, {16,15,13,11,9,8,7,6,7,8,9,11,13,15,16,17});

        lev.compute_last_part_of_stripe(16);
        assert(lev.row[3] == 16);
        assert(lev.row[4] == 15);
        assert(lev.row[11] == 13);
        assert(lev.row[16] == 11);
        assert(lev.row[lev.row.size() - 1] == 12);

        std::cout << "\tFunctionalTester: detailed_functional_tests_avx512 passed" << std::endl;
    }
#endif


    void assert_vector_equals(const __m128i &vector, const std::array<int, 4> &array) const
    {
        auto arr_from_vec = policy_sse::copy_to_array(vector);
        for (size_t i = 0; i < array.size(); ++i) {
            assert(array[i] == arr_from_vec[i]);
        }
    }

#ifdef USE_AVX512
    void assert_vector512_equals(const __m512i &vector, const std::array<int, 16> &array) const
    {
        auto arr_from_vec = policy_avx512::copy_to_array(vector);
        for (size_t i = 0; i < array.size(); ++i) {
            assert(array[i] == arr_from_vec[i]);
        }
    }
#endif

    void bigger_functional_tests()
    {
        for (size_t i = 0; i < 5; ++i) {
            std::vector<int> vec1 = random_sized_random_vector();
            std::vector<int> vec2 = random_sized_random_vector();

            std::cout << "\tComparing vector of sizes " << vec1.size() << ", " << vec2.size() << std::endl;
            compare_both(vec1.begin(), vec1.end(), vec2.begin(), vec2.end());
        }

        std::cout << "\tFunctionalTester: bigger_functional_tests passed" << std::endl;
    }

    template <typename It1, typename It2>
    void compare_both(It1 i1b, It1 i1e, It2 i2b, It2 i2e, bool verbose = false)
    {
        int std_impl_res = run_levenstein(i1b, i1e, i2b, i2e);
        int dummy_impl_res = run_dummy(i1b, i1e, i2b, i2e);
        if (verbose)
            std::cout << "\t\tMy distance=" << std_impl_res << ", dummy distance=" << dummy_impl_res << std::endl;
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

    std::vector<int> random_sized_random_vector() const
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
