//
// Created by pal on 2.4.19.
//

#ifndef MY_MAIN_LEVENSTEIN_TESTER_SSE_HPP
#define MY_MAIN_LEVENSTEIN_TESTER_SSE_HPP


#ifdef USE_SSE
template <>
class LevensteinTester <policy_sse> {
public:
    LevensteinTester()
        : levenstein_{tmp_array1.begin(), tmp_array1.end(), tmp_array2.begin(), tmp_array2.end()}
    {

    }

    void run_all_tests()
    {
        test_compute_vector();
        test_random_vectors();
        std::cout << "Tests for SSE passed" << std::endl;
    }

private:
    using vector_type = policy_sse::vector_type;
    using array_t = std::array<int, 4>;
    static constexpr auto tmp_array1 = {1, 2};
    static constexpr auto tmp_array2 = {3, 4};
    levenstein<policy_sse> levenstein_;

    struct arrays_t {
        array_t y;
        array_t w;
        array_t z;
        array_t a;
        array_t b;

        arrays_t(array_t y, array_t w, array_t z, array_t a, array_t b)
            : y{y},
            w{w},
            z{z},
            a{a},
            b{b}
        {
            assert(y[1] == w[0]);
            assert(y[2] == w[1]);
            assert(y[3] == w[2]);
        }
    };

    void test_compute_vector()
    {
        array_t y = {4, 3, 2, 3};
        array_t w = {3, 2, 3, 4};
        array_t z = {3, 2, 2, 3};
        array_t a = {2, 4, 6, 8};
        array_t b = {1, 3, 5, 7};
        array_t expected = {4, 3, 3, 4};

        auto scalar_array = compute_scalar(y, w, z, a, b);
        auto vector_array = compute_vector(y, w, z, a, b);
        assert_same(scalar_array, vector_array);
        assert_same(scalar_array, expected);
    }

    void test_random_vectors()
    {
        for (size_t i = 0; i < 3; ++i) {
            auto arrays = generate_all_arrays_random();
            auto scalar_array = compute_scalar(arrays.y, arrays.w, arrays.z, arrays.a, arrays.b);
            auto vector_array = compute_vector(arrays.y, arrays.w, arrays.z, arrays.a, arrays.b);
            assert_same(scalar_array, vector_array);
        }
    }

    array_t compute_scalar(array_t y, array_t w, array_t z, array_t a, array_t b) const
    {
        array_t res_array = {0, 0, 0, 0};

        res_array[0] = levenstein_.compute_levenstein_distance(y[1], z[0], y[0], a[0], b[3]);
        res_array[1] = levenstein_.compute_levenstein_distance(y[2], z[1], y[1], a[1], b[2]);
        res_array[2] = levenstein_.compute_levenstein_distance(y[3], z[2], y[2], a[2], b[1]);
        res_array[3] = levenstein_.compute_levenstein_distance(w[3], z[3], y[3], a[3], b[0]);

        return res_array;
    }

    array_t compute_vector(array_t y, array_t w, array_t z, array_t a, array_t b) const
    {
        vector_type vec_y = _mm_setr_epi32(y[0], y[1], y[2], y[3]);
        vector_type vec_w = _mm_setr_epi32(w[0], w[1], w[2], w[3]);
        vector_type vec_z = _mm_setr_epi32(z[0], z[1], z[2], z[3]);
        vector_type vec_a = _mm_setr_epi32(a[0], a[1], a[2], a[3]);
        vector_type vec_b = _mm_set_epi32(b[0], b[1], b[2], b[3]); // Reversed.

        vector_type vec_res = levenstein_.compute_levenstein_distance(vec_y, vec_w, vec_z, vec_a, vec_b);

        array_t res_array = {0, 0, 0, 0};
        _mm_storeu_si128(reinterpret_cast<__m128i *>(res_array.data()), vec_res);
        return res_array;
    }

    arrays_t generate_all_arrays_random() const
    {
        array_t a = random_array();
        array_t b = random_array();
        array_t z = random_array();
        array_t w = random_array();

        array_t y;
        y[1] = w[0];
        y[2] = w[1];
        y[3] = w[2];
        y[0] = rand();

        return arrays_t{y, w, z, a, b};
    }

    array_t random_array() const
    {
        array_t array;
        for (int &i : array) {
            i = std::rand();
        }
        return array;
    }

    void assert_same(array_t array_1, array_t array_2) const
    {
        for (size_t i = 0; i < array_1.size(); ++i) {
            assert(array_1[i] == array_2[i]);
        }
    }
};

#endif // USE_SSE

#endif //MY_MAIN_LEVENSTEIN_TESTER_SSE_HPP
