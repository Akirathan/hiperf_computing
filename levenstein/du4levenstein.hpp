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

#if defined(USE_AVX512) && defined(USE_AVX) || \
	defined(USE_AVX) && defined(USE_SSE) || \
	defined(USE_SSE) && defined(USE_AVX512)
#error Should define just one of USE_AVX512, USE_AVX or USE_SSE.
#endif

template <typename policy>
class LevensteinTester;

template <typename policy>
class FunctionalTester;

static constexpr bool use_dummy_impl = false;

template< typename policy>
class levenstein {
public:
	using data_element = int;
	using matrix_type = std::vector<std::vector<data_element>>;

	template< typename I1, typename I2>
	levenstein(I1 i1b, I1 i1e, I2 i2b, I2 i2e)
        : input_array_1(i1b, i1e),
          input_array_2(i2b, i2e),
          total_rows_count{input_array_2.size() + 1},
          total_cols_count{input_array_1.size() + 1},
          vector_stripe_left_boundary{1},
          vector_stripe_right_boundary{total_cols_count - stripe_size},
          cols_left_boundary{1},
          vector_y{},
          vector_w{},
          vector_z{}
	{
        row.resize(total_cols_count);
        for (size_t i = 0; i < row.size(); ++i) {
            row[i] = i;
        }
	}

	data_element compute()
	{
	    if (use_dummy_impl) {
	        dummy_levenstein dummy_lev{input_array_1.begin(), input_array_1.end(), input_array_2.begin(), input_array_2.end()};
	        return dummy_lev.compute();
	    }

	    size_t bottom_row_idx = stripe_size;
		for (; bottom_row_idx < total_rows_count; bottom_row_idx += stripe_size) {
			compute_stripe(bottom_row_idx);
		}

		const size_t last_row_idx = total_rows_count - 1;
		const size_t upper_row_idx = bottom_row_idx - stripe_size + 1;
		if (upper_row_idx > last_row_idx) {
		    // We computed everything in stripes.
		    return row[row.size() - 1];
		}
		else {
            return compute_rest(upper_row_idx);
		}
	}

private:
	using vector_type = typename policy::vector_type;
	using array_type = typename policy::array_type;
	constexpr static size_t stripe_size = policy::register_size / (sizeof(data_element) * 8);
	const std::vector<data_element> input_array_1;
	const std::vector<data_element> input_array_2;
	const size_t total_rows_count;
	const size_t total_cols_count;
	const size_t vector_stripe_left_boundary;
	const size_t vector_stripe_right_boundary;
	const size_t cols_left_boundary;
	vector_type vector_y;
	vector_type vector_w;
	vector_type vector_z;
	std::vector<data_element> row;

    class Rectangle {
    public:
        const size_t rows_count;
        const size_t cols_count;

        Rectangle(size_t rows_count, size_t cols_count)
                : rows_count{rows_count},
                  cols_count{cols_count},
                  rectangle(rows_count)
        {
            for (std::vector<data_element> &row_vec : rectangle) {
                row_vec.resize(cols_count);
            }
        }

        void set(size_t i, size_t j, data_element value)
        {
            rectangle[i][j] = value;
        }

        data_element get(size_t i, size_t j) const
        {
            return rectangle[i][j];
        }

        void copy_to_first_row(const std::vector<data_element> &src_row)
        {
            for (size_t i = 0; i < cols_count; ++i) {
                rectangle[0][i] = src_row[i];
            }
        }

        void fill_first_column_with_value_incrementing(const size_t value)
        {
            for (size_t i = 0; i < rows_count; ++i) {
                rectangle[i][0] = value + i;
            }
        }

        array_type get_diagonal(int row, int col, int count) const
        {
            array_type array{};
            size_t array_idx = 0;
            for (int i = row, j = col; i > row-count && j < col+count; i--, j++) {
                array[array_idx] = rectangle[i][j];
                array_idx++;
            }
            return array;
        }

        void set_diagonal(int row, int col, const array_type &array, int count)
        {
            size_t array_idx = 0;
            for (int i = row, j = col; i > row-count && j < col+count; i--, j++) {
                rectangle[i][j] = array[array_idx];
                array_idx++;
            }
        }

    private:
        std::vector<std::vector<data_element>> rectangle;
    };

	/// @param bottom_row_idx lower index of stripe's row.
	void compute_stripe(size_t bottom_row_idx)
	{
		assert(is_stripe_vector_computable(bottom_row_idx));

		compute_first_part_of_stripe(bottom_row_idx);

		// Start vector computation.
		for (size_t j = vector_stripe_left_boundary; j <= vector_stripe_right_boundary; ++j) {
			compute_one_stripe_diagonal(bottom_row_idx, j);
		}

		compute_last_part_of_stripe(bottom_row_idx);
	}

	/// Left upper triangle
	void compute_first_part_of_stripe(size_t bottom_row_idx)
	{
	    const size_t rectangle_row_count = stripe_size + 1;
	    const size_t rectangle_col_count = stripe_size;
	    Rectangle rectangle{rectangle_row_count, rectangle_col_count};

        rectangle.copy_to_first_row(row);
        rectangle.fill_first_column_with_value_incrementing(bottom_row_idx - stripe_size);

        const size_t upper_row_idx = bottom_row_idx - rectangle_row_count + 1;
        const size_t operations_num_from = stripe_size - 1;
        const size_t operations_num_until = 0;
        size_t operations_num = operations_num_from;
        for (size_t rectangle_i = 1; rectangle_i < rectangle_row_count; rectangle_i++) {
            for (size_t rectangle_j = 1; rectangle_j <= operations_num ; ++rectangle_j) {
                compute_distance_in_rectangle(rectangle, rectangle_i, rectangle_j,
                        upper_row_idx + rectangle_i, rectangle_j);
            }

            operations_num--;
            if (operations_num == operations_num_until) {
                break;
            }
        }

        store_vectors_from_rectangle(rectangle);
        row[0] = rectangle.get(rectangle.rows_count - 1, 0);
	}

	void store_vectors_from_rectangle(const Rectangle &rectangle)
    {
        auto array_y = rectangle.get_diagonal(rectangle.rows_count - 1, 0, stripe_size);
        vector_y = policy::copy_to_vector(array_y);

        vector_w = vector_y;
        policy::shift_right(vector_w);
        policy::set_last_idx(vector_w, row[stripe_size]);

        auto array_z = rectangle.get_diagonal(rectangle.rows_count - 2, 0, stripe_size);
        vector_z = policy::copy_to_vector(array_z);
    }

	void compute_distance_in_rectangle(Rectangle &rectangle, size_t rectangle_i, size_t rectangle_j,
	        size_t total_i, size_t total_j) const
    {
	    data_element upper = rectangle.get(rectangle_i-1, rectangle_j);
	    data_element left_upper = rectangle.get(rectangle_i-1, rectangle_j-1);
	    data_element left = rectangle.get(rectangle_i, rectangle_j-1);
	    data_element a = input_array_1[total_j-1];
	    data_element b = input_array_2[total_i-1];
	    rectangle.set(rectangle_i, rectangle_j, compute_levenstein_distance(upper, left_upper, left, a, b));
    }

	void compute_last_part_of_stripe(size_t bottom_row_idx)
    {
	    const size_t rectangle_row_count = stripe_size;
	    const size_t rectangle_col_count = stripe_size;
	    Rectangle rectangle{rectangle_row_count, rectangle_col_count};

	    copy_arrays_to_rectangle_in_last_part(rectangle);

	    const size_t upper_row_idx = bottom_row_idx - stripe_size + 1;
	    const size_t left_col_idx = total_cols_count - stripe_size;
	    const size_t operations_num_from = 1;
	    const size_t operations_num_until = stripe_size - 1;
	    size_t operations_num = operations_num_from;
        for (size_t rectangle_i = 1; rectangle_i < rectangle_row_count; ++rectangle_i) {
            for (size_t rectangle_j = rectangle_col_count - operations_num;
                 rectangle_j < rectangle_col_count;
                 ++rectangle_j)
            {
                compute_distance_in_rectangle(rectangle, rectangle_i, rectangle_j,
                        upper_row_idx + rectangle_i, left_col_idx + rectangle_j);
            }

            operations_num++;
            if (operations_num > operations_num_until) {
                break;
            }
        }

        store_row_from_rectangle_in_last_part(rectangle);
    }

    void copy_arrays_to_rectangle_in_last_part(Rectangle &rectangle) const
    {
        const size_t z_from_row = rectangle.rows_count - 2;
        const size_t z_from_col = 0;
        auto array_z = policy::copy_to_array(vector_z);
        rectangle.set_diagonal(z_from_row, z_from_col, array_z, array_z.size() - 1);

        const size_t y_from_row = rectangle.rows_count - 1;
        const size_t y_from_col = 0;
        auto array_y = policy::copy_to_array(vector_y);
        rectangle.set_diagonal(y_from_row, y_from_col, array_y, array_y.size());
    }

    void store_row_from_rectangle_in_last_part(const Rectangle &rectangle)
    {
        const size_t last_row_idx = rectangle.rows_count - 1;

        size_t col_idx = total_cols_count - stripe_size + 1;
        for (size_t j = 1; j < rectangle.cols_count; ++j) {
            row[col_idx] = rectangle.get(last_row_idx, j);
            col_idx++;
        }
    }

    data_element compute_rest(size_t upper_row_idx)
    {
        assert(upper_row_idx < total_rows_count);
	    assert(total_cols_count == row.size());
	    // We need one row more so we can copy current global variable row inside
	    // the first row of rectangle.
	    size_t rectangle_rows_count = total_rows_count - upper_row_idx + 1;
	    size_t rectangle_cols_count = total_cols_count;
        assert(rectangle_rows_count > 1);
	    Rectangle rectangle{rectangle_rows_count, rectangle_cols_count};

	    // Rectangle starts actually one row above upper_row_idx.
	    upper_row_idx--;

        rectangle.copy_to_first_row(row);
        rectangle.fill_first_column_with_value_incrementing(upper_row_idx);

        for (size_t rectangle_i = 1; rectangle_i < rectangle_rows_count; ++rectangle_i) {
            for (size_t rectangle_j = 1; rectangle_j < rectangle_cols_count; ++rectangle_j) {
                compute_distance_in_rectangle(rectangle, rectangle_i, rectangle_j,
                        upper_row_idx + rectangle_i, rectangle_j);
            }
        }

        return rectangle.get(rectangle_rows_count - 1, rectangle_cols_count - 1);
    }

	/// Compute with vector instructions.
	void compute_one_stripe_diagonal(size_t i, size_t j)
	{
		assert(is_stripe_diagonal_vector_computable(i, j));

		vector_type vector_a = get_vector_from_array(input_array_1, j - 1);
		vector_type vector_b = get_vector_from_array_reversed(input_array_2, i - stripe_size);

		vector_type result = compute_levenstein_distance(vector_y, vector_w, vector_z, vector_a, vector_b);

        replace_arrays_with_result(result, j);
        data_element first_element = policy::get_first(result);
        row[j] = first_element;
	}

	/// Updates internal buffers with new results of vector computation.
	void replace_arrays_with_result(const vector_type &result_vector, const size_t col_idx)
    {
	    vector_z = vector_w;
	    vector_y = result_vector;

        vector_w = vector_y;
        policy::shift_right(vector_w);

        size_t row_element_idx = col_idx + stripe_size;

        bool last_iteration = (col_idx + stripe_size == total_cols_count);
        if (!last_iteration) {
            policy::set_last_idx(vector_w, row[row_element_idx]);
        }
    }

	vector_type compute_levenstein_distance(vector_type y, vector_type w, vector_type z, vector_type a, vector_type b) const
	{
	    return policy::compute_levenstein_distance(y, w, z, a, b);
	}

	data_element compute_levenstein_distance(data_element upper, data_element left_upper, data_element left,
											 data_element a, data_element b) const
	{
		data_element first = upper + 1;
		data_element second = left_upper + (a == b ? 0 : 1);
		data_element third = left + 1;
		return std::min({first, second, third});
	}

	vector_type get_vector_from_array(const std::vector<data_element> &array, size_t idx) const
	{
	    const size_t from_idx = idx;
	    const size_t to_idx = idx + stripe_size - 1;
        array_type arr;
        size_t arr_idx = 0;
        for (size_t i = from_idx; i <= to_idx; ++i) {
            arr[arr_idx] = array[i];
            arr_idx++;
        }
        return policy::copy_to_vector(arr);
	}

    vector_type get_vector_from_array_reversed(const std::vector<data_element> &array, size_t idx) const
    {
	    const int from_idx = idx + stripe_size - 1;
	    const int to_idx = idx;
        array_type arr{};
        size_t arr_idx = 0;
        for (int i = from_idx; i >= to_idx; i--) {
            arr[arr_idx] = array[i];
            arr_idx++;
        }
        return policy::copy_to_vector(arr);
    }

	bool is_index_in_boundaries(size_t i, size_t j) const
	{
		return (i >= 0 && i < total_rows_count) && (j >= 0 && j < total_cols_count);
	}

	bool is_stripe_vector_computable(size_t i) const
	{
		return i >= stripe_size;
	}

	bool is_stripe_diagonal_vector_computable(size_t i, size_t j) const
	{
		return i >= stripe_size && j >= vector_stripe_left_boundary && j <= vector_stripe_right_boundary;
	}

	template <typename T> friend class LevensteinTester;
	template <typename T> friend class FunctionalTester;
};

using matrix_type = std::vector<std::vector<int>>;

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

    static array_type copy_to_array(vector_type vec)
    {
        array_type arr = {0, 0, 0, 0};
        _mm_storeu_si128(reinterpret_cast<__m128i *>(arr.data()), vec);
        return arr;
    }

    static vector_type get_vector_1()
    {
        if (!vector_1_initialized) {
            vector_1 = _mm_insert_epi32(vector_1, 1, 0);
            vector_1 = _mm_insert_epi32(vector_1, 1, 1);
            vector_1 = _mm_insert_epi32(vector_1, 1, 2);
            vector_1 = _mm_insert_epi32(vector_1, 1, 3);
            vector_1_initialized = true;
        }
        return vector_1;
    }

    static vector_type copy_to_vector(const array_type &array)
    {
        vector_type vec{};
        vec = _mm_insert_epi32(vec, array[0], 0);
        vec = _mm_insert_epi32(vec, array[1], 1);
        vec = _mm_insert_epi32(vec, array[2], 2);
        vec = _mm_insert_epi32(vec, array[3], 3);
        return vec;
    }

    static vector_type compute_levenstein_distance(vector_type y, vector_type w, vector_type z,
            vector_type a, vector_type b)
    {
        vector_type vector_1 = get_vector_1();
        vector_type res = _mm_min_epi32(_mm_add_epi32(y, vector_1),
	                                    _mm_add_epi32(w, vector_1));
	    res = _mm_min_epi32(res,
	                        _mm_add_epi32(z, _mm_andnot_si128(_mm_cmpeq_epi32(a, b), vector_1)));
		return res;
    }

    /// Set last 4 bytes of vector.
    static void set_last_idx(vector_type &vector, data_element value)
    {
        vector = _mm_insert_epi32(vector, value, 3);
    }

    /// Get first 4 bytes of vector.
    static data_element get_first(vector_type &vector)
    {
        return _mm_extract_epi32(vector, 0);
    }

    /// Shifts vector left by 4B - what was on index 1 will be on 0. (Shift toward lower indexes).
    static void shift_right(vector_type &vector)
    {
        vector = _mm_srli_si128(vector, 4);
    }
};


// TODO
struct policy_avx {
	constexpr static size_t register_size = 0;
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
            vector_1 = _mm512_load_epi32(reinterpret_cast<void *>(array_1.data()));
            vector_1_initialized = true;
        }
        return vector_1;
    }

    static vector_type copy_to_vector(const array_type &array)
    {
        __m128i first{};
        first = _mm_insert_epi32(first, array[0], 0);
        first = _mm_insert_epi32(first, array[1], 1);
        first = _mm_insert_epi32(first, array[2], 2);
        first = _mm_insert_epi32(first, array[3], 3);

        __m128i second{};
        second = _mm_insert_epi32(second, array[4], 0);
        second = _mm_insert_epi32(second, array[5], 1);
        second = _mm_insert_epi32(second, array[6], 2);
        second = _mm_insert_epi32(second, array[7], 3);

        __m128i third{};
        third = _mm_insert_epi32(third, array[8], 0);
        third = _mm_insert_epi32(third, array[9], 1);
        third = _mm_insert_epi32(third, array[10], 2);
        third = _mm_insert_epi32(third, array[11], 3);

        __m128i fourth{};
        fourth = _mm_insert_epi32(fourth, array[12], 0);
        fourth = _mm_insert_epi32(fourth, array[13], 1);
        fourth = _mm_insert_epi32(fourth, array[14], 2);
        fourth = _mm_insert_epi32(fourth, array[15], 3);

        vector_type vec{};
        vec = _mm512_inserti32x4(vec, first, 0);
        vec = _mm512_inserti32x4(vec, second, 1);
        vec = _mm512_inserti32x4(vec, third, 2);
        vec = _mm512_inserti32x4(vec, fourth, 3);
        return vec;
    }

    static array_type copy_to_array(vector_type vec)
    {
        array_type tmp_array = {};
        _mm512_storeu_si512(reinterpret_cast<void *>(tmp_array.data()), vec);
        return tmp_array;
    }

    static vector_type compute_levenstein_distance(vector_type y, vector_type w, vector_type z,
                                                   vector_type a, vector_type b)
    {
        vector_type vector_1 = get_vector_1();
        vector_type res = _mm512_min_epi32(_mm512_add_epi32(y, vector_1),
                                           _mm512_add_epi32(w, vector_1));

        __mmask16 cmpeq_mask = _mm512_cmpeq_epi32_mask(a, b);
        __mmask16 not_cmpeq_mask = _mm512_knot(cmpeq_mask);
        z = _mm512_mask_add_epi32(z, not_cmpeq_mask, z, vector_1);

        res = _mm512_min_epi32(res, z);
        return res;
    }

    static void set_last_idx(vector_type &vector, data_element value)
    {
        __m128i last_vec = _mm512_extracti32x4_epi32(vector, 3);
        // Set last element in last subvector.
        last_vec = _mm_insert_epi32(last_vec, value, 3);

        vector = _mm512_inserti32x4(vector, last_vec, 3);
    }

    static data_element get_first(vector_type &vector)
    {
        __m128i first_vec = _mm512_extracti32x4_epi32(vector, 0);
        return _mm_extract_epi32(first_vec, 0);
    }

    /// Shifts vector left by 4B - what was on index 1 will be on 0. (Shift toward lower indexes).
    static void shift_right(vector_type &vector)
    {
        __m128i vec1 = _mm512_extracti32x4_epi32(vector, 0);
        __m128i vec2 = _mm512_extracti32x4_epi32(vector, 1);
        __m128i vec3 = _mm512_extracti32x4_epi32(vector, 2);
        __m128i vec4 = _mm512_extracti32x4_epi32(vector, 3);

        data_element tmp = shift_right(vec4, 0);
        tmp = shift_right(vec3, tmp);
        tmp = shift_right(vec2, tmp);
        shift_right(vec1, tmp);

        vector = _mm512_inserti32x4(vector, vec1, 0);
        vector = _mm512_inserti32x4(vector, vec2, 1);
        vector = _mm512_inserti32x4(vector, vec3, 2);
        vector = _mm512_inserti32x4(vector, vec4, 3);
    }

private:
    static data_element shift_right(__m128i &vec, data_element insert_value)
    {
        data_element tmp = _mm_extract_epi32(vec, 0);
        vec = _mm_srli_si128(vec, 4);
        vec = _mm_insert_epi32(vec, insert_value, 3);
        return tmp;
    }
};

#endif
