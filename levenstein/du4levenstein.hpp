#ifndef du4levenstein_hpp_
#define du4levenstein_hpp_

#include <vector>
#include <array>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <emmintrin.h>
#include <smmintrin.h>

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

template< typename policy>
class levenstein {
public:
	using data_element = int;

	template< typename I1, typename I2>
	levenstein(I1 i1b, I1 i1e, I2 i2b, I2 i2e)
        : array_1(i1b, i1e),
          array_2(i2b, i2e),
          results_rows{array_1.size() + 1},
          results_cols{array_2.size() + 1},
          vector_stripe_left_boundary{1},
          vector_stripe_right_boundary{results_cols - stripe_size}
	{
		results.resize(results_rows);
		for (size_t i = 0; i < results_rows; ++i) {
			results[i].resize(results_cols);
		}
		fill_boundaries();
	}

	data_element compute()
	{
		for (size_t i = stripe_size; i < results_rows; i += stripe_size) {
			compute_stripe(i);
		}
		// Compute rest ...
		return 0;
	}

private:
	using vector_type = typename policy::vector_type;
	constexpr static size_t stripe_size = policy::register_size / (sizeof(data_element) * 8);
	const std::vector<data_element> array_1;
	const std::vector<data_element> array_2;
	const size_t results_rows;
	const size_t results_cols;
	const size_t vector_stripe_left_boundary;
	const size_t vector_stripe_right_boundary;
	std::vector<std::vector<data_element>> results;

	/// @param i lower index of stripe's row.
	void compute_stripe(size_t i)
	{
		assert(is_stripe_vector_computable(i));

		compute_first_part_of_stripe(i);

		// Start vector computation.
		for (size_t j = stripe_size; j < vector_stripe_right_boundary; ++j) {
			compute_one_stripe_diagonal(i, j);
		}

		// TODO: Compute last part of stripe scalar.
	}

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

	void compute_scalar_at_index(size_t i, size_t j)
	{
		assert(i >= 1 && i < results_rows && j >= 0 && j < results_cols);
		results[i][j] = compute_levenstein_distance(results[i-1][j],   // upper
				                                    results[i-1][j-1], // left_upper
				                                    results[i][j-1],   // left
				                                    array_1[i-1],      // a
				                                    array_2[j-1]);     // b
	}

	/// Compute with vector instructions.
	void compute_one_stripe_diagonal(size_t i, size_t j)
	{
		assert(is_stripe_diagonal_vector_computable(i, j));

		vector_type vector_y = get_vector_from_diagonal(i, j - 1);
		vector_type vector_w = get_vector_from_diagonal(i - 1, j);
		vector_type vector_z = get_vector_from_diagonal(i - 1, j - 1);
		// TODO: jeden zvektoru a, b obratit.
		vector_type vector_a = get_vector_from_array_reversed(array_1, j);
		vector_type vector_b = get_vector_from_array(array_2, i-3);

		vector_type result = compute_levenstein_distance(vector_y, vector_w, vector_z, vector_a, vector_b);
		store_vector_to_diagonal(i, j, result);
	}

	vector_type compute_levenstein_distance(vector_type y, vector_type w, vector_type z, vector_type a, vector_type b) const
	{
#if defined(USE_SSE)
		vector_type vector_1 = _mm_setr_epi32(1, 1, 1, 1);
		y = _mm_add_epi32(y, vector_1);
		w = _mm_add_epi32(w, vector_1);
		z = _mm_add_epi32(z, _mm_andnot_si128(_mm_cmpeq_epi32(a, b), vector_1));
		vector_type res = _mm_min_epi32(y, w);
		res = _mm_min_epi32(res, z);
		return res;
#elif defined(USE_AVX)

#elif defined(USE_AVX512)

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

	vector_type get_vector_from_diagonal(size_t bottom_left_row, size_t bottom_left_col) const
	{
		size_t i = bottom_left_row;
		size_t j = bottom_left_col;
#if defined(USE_SSE)
		return _mm_setr_epi32(results[i][j],
                              results[i-1][j+1],
                              results[i-2][j+2],
                              results[i-3][j+3]);
#elif defined(USE_AVX512)

#elif defined(USE_AVX)

#endif
	}

	vector_type get_vector_from_array(const std::vector<data_element> &array, size_t idx) const
	{
        assert(idx + stripe_size <= array.size());
#if defined(USE_SSE)
		return _mm_setr_epi32(array[idx], array[idx+1], array[idx+2], array[idx+3]);
#elif defined(USE_AVX512)

#elif defined(USE_AVX)

#endif
	}

    vector_type get_vector_from_array_reversed(const std::vector<data_element> &array, size_t idx) const
    {
        assert(idx + stripe_size <= array.size());
#if defined(USE_SSE)
        return _mm_set_epi32(array[idx], array[idx+1], array[idx+2], array[idx+3]);
#elif defined(USE_AVX512)

#elif defined(USE_AVX)

#endif
    }

	void store_vector_to_diagonal(size_t bottom_left_row, size_t bottom_left_col, vector_type src_vec)
	{
#if defined(USE_SSE)
		std::array<data_element, stripe_size> tmp_array;
		_mm_storeu_si128(reinterpret_cast<__m128i *>(tmp_array.data()), src_vec);

		size_t array_idx = 0;
		for(size_t i = bottom_left_row, j = bottom_left_col;
			i < bottom_left_row + stripe_size && j < bottom_left_col + stripe_size;
			i++, j++)
		{
			results[i][j] = tmp_array[array_idx];
			array_idx++;
		}
#elif defined(USE_AVX512)

#elif defined(USE_AVX)

#endif
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
	using vector_type = __m128i;
	using array_type = std::array<int, 4>;
    constexpr static size_t register_size = 128;

    static array_type copy_into_array(vector_type vec)
    {
        array_type arr = {0, 0, 0, 0};
        _mm_storeu_si128(reinterpret_cast<__m128i *>(arr.data()), vec);
        return arr;
    }
};

// TODO
struct policy_avx {
	constexpr static size_t register_size = 0; // TODO
};

// TODO
struct policy_avx512 {
	constexpr static size_t register_size = 512;
};

#endif
