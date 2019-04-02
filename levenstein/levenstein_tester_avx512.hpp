#ifndef MY_MAIN_LEVENSTEIN_TESTER_AVX512_HPP
#define MY_MAIN_LEVENSTEIN_TESTER_AVX512_HPP

#include "du4levenstein.hpp"
#include <cassert>

#ifdef USE_AVX512

template <>
class LevensteinTester<policy_avx512> {
public:
    LevensteinTester()
        : levenstein_{tmp_array1.begin(), tmp_array1.end(), tmp_array2.begin(), tmp_array2.end()}
    {}

    void run_all_tests()
    {
        test_compute_vector();
    }

private:
    using array_type = typename policy_avx512::array_type;
    using vector_type = typename policy_avx512::vector_type;
    using data_element = typename policy_avx512::data_element;
    static constexpr auto tmp_array1 = {1, 2};
    static constexpr auto tmp_array2 = {3, 4};
    levenstein<policy_avx512> levenstein_;

    void test_compute_vector()
    {
        array_type y;
        y.fill(1);
        array_type w;
        w.fill(1);
        array_type z;
        z.fill(1);

        array_type a;
        a.fill(23);
        array_type b;
        b.fill(42);

        array_type expected;
        expected.fill(2);

        array_type scalar_array = compute_scalar(y, w, z, a, b);
        array_type vector_array = compute_vector(y, w, z, a, b);

        assert_same(scalar_array, vector_array);
        assert_same(scalar_array, expected);
    }

    array_type compute_vector(array_type y, array_type w, array_type z, array_type a, array_type b) const
    {
        vector_type vec_y = policy_avx512::copy_to_vector(y);
        vector_type vec_w = policy_avx512::copy_to_vector(w);
        vector_type vec_z = policy_avx512::copy_to_vector(z);
        vector_type vec_a = policy_avx512::copy_to_vector(a);
        vector_type vec_b = policy_avx512::copy_to_vector(b);

        vector_type vec_res = levenstein_.compute_levenstein_distance(vec_y, vec_w, vec_z, vec_a, vec_b);
        return policy_avx512::copy_into_array(vec_res);
    }

    array_type compute_scalar(array_type y, array_type w, array_type z, array_type a, array_type b) const
    {
        array_type res_array = {};
        const size_t max_index = policy_avx512::elements_count_per_register - 1;

        for (size_t i = 0; i < policy_avx512::elements_count_per_register; ++i) {
            data_element upper = w[i];
            data_element left_upper = z[i];
            data_element left = y[i];
            data_element elem_from_a = a[i];
            data_element elem_from_b = b[max_index - i];
            res_array[i] = levenstein_.compute_levenstein_distance(upper, left_upper, left, elem_from_a, elem_from_b);
        }

        return res_array;
    }

    void assert_same(array_type array_1, array_type array_2) const
    {
        for (size_t i = 0; i < array_1.size(); ++i) {
            assert(array_1[i] == array_2[i]);
        }
    }
};

#endif // USE_AVX512

#endif //MY_MAIN_LEVENSTEIN_TESTER_AVX512_HPP
