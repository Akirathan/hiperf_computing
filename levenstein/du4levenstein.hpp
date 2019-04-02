#ifndef du4levenstein_hpp_
#define du4levenstein_hpp_

#include <vector>
#include <array>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#include "dummy_levenstein.hpp"

#define USE_AVX512

#if !defined(USE_AVX512) && !defined(USE_AVX)
#define USE_SSE
#endif

#if defined(USE_AVX512) && defined(USE_AVX) || \
	defined(USE_AVX) && defined(USE_SSE) || \
	defined(USE_SSE) && defined(USE_AVX512)
#error Should define just one of USE_AVX512, USE_AVX or USE_SSE.
#endif

template <typename policy>
class LevensteinTester;

static constexpr bool use_dummy_impl = false;

template< typename policy>
class levenstein {
public:
	using data_element = int;

	template< typename I1, typename I2>
	levenstein(I1 i1b, I1 i1e, I2 i2b, I2 i2e)
        : array_1(i1b, i1e),
          array_2(i2b, i2e),
          results_rows{array_2.size() + 1},
          results_cols{array_1.size() + 1},
          vector_stripe_left_boundary{1},
          vector_stripe_right_boundary{results_cols - stripe_size},
          cols_left_boundary{1}
	{
		results.resize(results_rows);
		for (size_t i = 0; i < results_rows; ++i) {
			results[i].resize(results_cols);
		}
		fill_boundaries();
	}

	data_element compute()
	{
	    if (use_dummy_impl) {
	        dummy_levenstein dummy_lev{array_1.begin(), array_1.end(), array_2.begin(), array_2.end()};
	        return dummy_lev.compute();
	    }

	    size_t i = stripe_size;
		for (; i < results_rows; i += stripe_size) {
			compute_stripe(i);
		}

		i = i - stripe_size + 1;
		compute_rest(i);
		return results[results_rows - 1][results_cols - 1];
	}

private:
	using vector_type = typename policy::vector_type;
	using array_type = typename policy::array_type;
	constexpr static size_t stripe_size = policy::register_size / (sizeof(data_element) * 8);
	const std::vector<data_element> array_1;
	const std::vector<data_element> array_2;
	const size_t results_rows;
	const size_t results_cols;
	const size_t vector_stripe_left_boundary;
	const size_t vector_stripe_right_boundary;
	const size_t cols_left_boundary;
	std::vector<std::vector<data_element>> results;

	/// @param i lower index of stripe's row.
	void compute_stripe(size_t i)
	{
		assert(is_stripe_vector_computable(i));

		compute_first_part_of_stripe(i);

		// Start vector computation.
		for (size_t j = vector_stripe_left_boundary; j <= vector_stripe_right_boundary; ++j) {
			compute_one_stripe_diagonal(i, j);
		}

		compute_last_part_of_stripe(i);
	}

	/// Left upper triangle
	void compute_first_part_of_stripe(size_t row)
	{
		size_t k = stripe_size - 1;
		for (size_t i = row - stripe_size + 1; i < row; i++) {
			for (size_t j = 1; j <= k; j++) {
				compute_scalar_at_index(i, j);
			}

			k--;
			if (k <= 0) {
				break;
			}
		}
	}

	void compute_last_part_of_stripe(size_t row)
    {
	    const size_t row_from = row - stripe_size + 2;
	    const size_t row_until = row;
	    const size_t operations_num_from = 1;
	    const size_t operations_num_until = stripe_size - 1;
	    size_t operations_num = operations_num_from;

        for (size_t i = row_from; i <= row_until; ++i) {
            for (size_t j = results_cols - operations_num; j < results_cols; ++j) {
                compute_scalar_at_index(i, j);
            }

            operations_num++;
            if (operations_num > operations_num_until) {
                break;
            }
        }
    }

    void compute_rest(size_t row)
    {
	    assert(results_rows - row < stripe_size);
        for (size_t i = row; i < results_rows; ++i) {
            for (size_t j = cols_left_boundary; j < results_cols; ++j) {
                compute_scalar_at_index(i, j);
            }
        }
    }

	void compute_scalar_at_index(size_t i, size_t j)
	{
		assert(i >= 1 && i < results_rows && j >= 0 && j < results_cols);
		results[i][j] = compute_levenstein_distance(results[i-1][j],   // upper
				                                    results[i-1][j-1], // left_upper
				                                    results[i][j-1],   // left
				                                    array_1[j-1],      // a
				                                    array_2[i-1]);     // b
	}

	/// Compute with vector instructions.
	void compute_one_stripe_diagonal(size_t i, size_t j)
	{
		assert(is_stripe_diagonal_vector_computable(i, j));

		vector_type vector_y = get_vector_from_diagonal(i, j - 1);
		vector_type vector_w = get_vector_from_diagonal(i - 1, j);
		vector_type vector_z = get_vector_from_diagonal(i - 1, j - 1);
		vector_type vector_a = get_vector_from_array(array_1, j - 1);
		vector_type vector_b = get_vector_from_array_reversed(array_2, i - stripe_size);

		// debug
		auto arr_y = policy::copy_into_array(vector_y);
        auto arr_w = policy::copy_into_array(vector_w);
        auto arr_z = policy::copy_into_array(vector_z);
        auto arr_a = policy::copy_into_array(vector_a);
        auto arr_b = policy::copy_into_array(vector_b);

		vector_type result = compute_levenstein_distance(vector_y, vector_w, vector_z, vector_a, vector_b);
		store_vector_to_diagonal(i, j, result);
	}

	vector_type compute_levenstein_distance(vector_type y, vector_type w, vector_type z, vector_type a, vector_type b) const
	{
        vector_type vector_1 = policy::get_vector_1();
#if defined(USE_SSE)
	    vector_type res = _mm_min_epi32(_mm_add_epi32(y, vector_1),
	                                    _mm_add_epi32(w, vector_1));
	    res = _mm_min_epi32(res,
	                        _mm_add_epi32(z, _mm_andnot_si128(_mm_cmpeq_epi32(a, b), vector_1)));
		return res;
#elif defined(USE_AVX)

#elif defined(USE_AVX512)
		vector_type res = _mm512_min_epi32(_mm512_add_epi32(y, vector_1),
		                                   _mm512_add_epi32(w, vector_1));

		__mmask16 cmpeq_mask = _mm512_cmpeq_epi32_mask(a, b);
		__mmask16 not_cmpeq_mask = _mm512_knot(cmpeq_mask);
		z = _mm512_mask_add_epi32(z, not_cmpeq_mask, z, vector_1);

		res = _mm512_min_epi32(res, z);
		return res;
#endif
	}

	data_element compute_levenstein_distance(data_element upper, data_element left_upper, data_element left,
											 data_element a, data_element b) const
	{
		data_element first = upper + 1;
		data_element second = left_upper + (a == b ? 0 : 1);
		data_element third = left + 1;
		return std::min({first, second, third});
	}

	vector_type get_vector_from_diagonal(size_t bottom_left_row, size_t bottom_left_col)
	{
		size_t i = bottom_left_row;
		size_t j = bottom_left_col;
#if defined(USE_SSE)
        return policy::copy_to_vector(results[i][j], results[i-1][j+1], results[i-2][j+2], results[i-3][j+3]);
#elif defined(USE_AVX512)
        array_type arr;
        size_t arr_idx = 0;
        for(; i > bottom_left_row - stripe_size && j < bottom_left_col + stripe_size; i++, j++) {
            arr[arr_idx] = results[i][j];
            arr_idx++;
        }
        return policy::copy_to_vector(arr);
#elif defined(USE_AVX)

#endif
	}

	vector_type get_vector_from_array(const std::vector<data_element> &array, size_t idx) const
	{
        assert(idx + stripe_size <= array.size());
#if defined(USE_SSE)
        return policy::copy_to_vector(array[idx], array[idx+1], array[idx+2], array[idx+3]);
#elif defined(USE_AVX512)
        array_type arr;
        for (size_t i = 0; i < policy::elements_count_per_register; ++i) {
            arr[i] = array[i];
        }
        return policy::copy_to_vector(arr);
#elif defined(USE_AVX)

#endif
	}

    vector_type get_vector_from_array_reversed(const std::vector<data_element> &array, size_t idx) const
    {
        assert(idx + stripe_size <= array.size());
#if defined(USE_SSE)
        return policy::copy_to_vector(array[idx+3], array[idx+2], array[idx+1], array[idx]);
#elif defined(USE_AVX512)
        array_type arr;
        size_t arr_idx = 0;
        for (size_t i = policy::elements_count_per_register; i > 0; i--) {
            arr[arr_idx] = array[idx + i];
            arr_idx++;
        }

        return policy::copy_to_vector(arr);
#elif defined(USE_AVX)

#endif
    }

	void store_vector_to_diagonal(size_t bottom_left_row, size_t bottom_left_col, vector_type src_vec)
	{
        array_type tmp_array;
#if defined(USE_SSE)
		_mm_storeu_si128(reinterpret_cast<__m128i *>(tmp_array.data()), src_vec);
#elif defined(USE_AVX512)
        _mm512_storeu_si512(reinterpret_cast<void *>(tmp_array.data()), src_vec);
#elif defined(USE_AVX)

#endif
        size_t array_idx = 0;
        for(size_t i = bottom_left_row, j = bottom_left_col;
            i > bottom_left_row - stripe_size && j < bottom_left_col + stripe_size;
            i--, j++)
        {
            results[i][j] = tmp_array[array_idx];
            array_idx++;
        }
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

	bool is_index_in_boundaries(size_t i, size_t j) const
	{
		return (i >= 0 && i < results_rows) && (j >= 0 && j < results_cols);
	}

	bool is_stripe_vector_computable(size_t i) const
	{
		return i >= stripe_size;
	}

	bool is_stripe_diagonal_vector_computable(size_t i, size_t j) const
	{
		return i >= stripe_size && j >= vector_stripe_left_boundary && j <= vector_stripe_right_boundary;
	}

	friend class LevensteinTester<policy>;
};


struct policy_sse {
    using data_element = int;
	using vector_type = __m128i;
    constexpr static size_t register_size = 128;
    constexpr static size_t elements_count_per_register = register_size / (sizeof(data_element) * 8);
	using array_type = std::array<data_element, elements_count_per_register>;
    constexpr static size_t alignment = 16;
    static __attribute__ ((aligned(alignment))) array_type aligned_array;
    static bool vector_1_initialized;
    static vector_type vector_1;

    static array_type copy_into_array(vector_type vec)
    {
        array_type arr = {0, 0, 0, 0};
        _mm_storeu_si128(reinterpret_cast<__m128i *>(arr.data()), vec);
        return arr;
    }

    static vector_type get_vector_1()
    {
        if (!vector_1_initialized) {
            array_type arr = {1, 1, 1, 1};
            vector_1 = _mm_load_si128(reinterpret_cast<__m128i *>(arr.data()));
            vector_1_initialized = true;
        }
        return vector_1;
    }

    static vector_type copy_to_vector(data_element elem1, data_element elem2, data_element elem3, data_element elem4)
    {
        aligned_array[0] = elem1;
        aligned_array[1] = elem2;
        aligned_array[2] = elem3;
        aligned_array[3] = elem4;
        return _mm_load_si128(reinterpret_cast<__m128i *>(aligned_array.data()));
    }
};


// TODO
struct policy_avx {
	constexpr static size_t register_size = 0; // TODO
};

struct policy_avx512 {
    using data_element = int;
    using vector_type = __m512i;
    constexpr static size_t register_size = 512;
    constexpr static size_t elements_count_per_register = register_size / (sizeof(data_element) * 8); //16
    constexpr static size_t alignment = 64;
    using array_type = std::array<data_element, elements_count_per_register>;
    static __attribute__ ((aligned(alignment))) array_type aligned_array;
    static __attribute__ ((aligned(alignment))) array_type array_1;
    static vector_type vector_1;
    static bool vector_1_initialized;

    static vector_type get_vector_1()
    {
        if (!vector_1_initialized) {
            vector_1 = _mm512_load_si512(reinterpret_cast<void *>(array_1.data()));
            vector_1_initialized = true;
        }
        return vector_1;
    }

    static vector_type copy_to_vector(const array_type &array)
    {
        aligned_array = array;
        return _mm512_load_si512(reinterpret_cast<void *>(aligned_array.data()));
    }

    static array_type copy_into_array(vector_type vec)
    {
        array_type tmp_array = {};
        _mm512_storeu_si512(reinterpret_cast<void *>(tmp_array.data()), vec);
        return tmp_array;
    }
};

#endif
