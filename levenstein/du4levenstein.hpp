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
          results_rows{input_array_2.size() + 1},
          results_cols{input_array_1.size() + 1},
          vector_stripe_left_boundary{1},
          vector_stripe_right_boundary{results_cols - stripe_size},
          cols_left_boundary{1},
          array_y{},
          array_w{},
          array_z{},
          row_idx{0}
	{
        row.resize(results_cols);
	}

	data_element compute()
	{
	    if (use_dummy_impl) {
	        dummy_levenstein dummy_lev{input_array_1.begin(), input_array_1.end(), input_array_2.begin(), input_array_2.end()};
	        return dummy_lev.compute();
	    }

	    size_t i = stripe_size;
		for (; i < results_rows; i += stripe_size) {
			compute_stripe(i);
		}

		i = i - stripe_size + 1;
		return compute_rest(i);
	}

private:
	using vector_type = typename policy::vector_type;
	using array_type = typename policy::array_type;
	constexpr static size_t stripe_size = policy::register_size / (sizeof(data_element) * 8);
	const std::vector<data_element> input_array_1;
	const std::vector<data_element> input_array_2;
	const size_t results_rows;
	const size_t results_cols;
	const size_t vector_stripe_left_boundary;
	const size_t vector_stripe_right_boundary;
	const size_t cols_left_boundary;
	matrix_type results;
	array_type array_y;
	array_type array_w;
	array_type array_z;
	std::vector<data_element> row;
	size_t row_idx;

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
            for (size_t i = 0; i < src_row.size(); ++i) {
                rectangle[0][i] = src_row[i];
            }
        }

        void fill_first_column(const size_t row_idx)
        {
            for (size_t i = 0; i < rows_count; ++i) {
                rectangle[i][0] = row_idx + i;
            }
        }

        array_type get_diagonal(size_t row, size_t col, size_t count) const
        {
            array_type array;
            size_t array_idx = 0;
            for (size_t i = row; i > row - count; i--) {
                for (size_t j = col; j < col + count; j++) {
                    array[array_idx] = rectangle[i][j];
                    array_idx++;
                }
            }
        }

        void set_diagonal(size_t row, size_t col, const array_type &array, size_t count)
        {
            size_t array_idx = 0;
            for (size_t i = row; i > row - count; i--) {
                for (size_t j = col; j < col + count; ++j) {
                    rectangle[i][j] = array[array_idx];
                    array_idx++;
                }
            }
        }

    private:
        std::vector<std::vector<data_element>> rectangle;
    };

	/// @param i lower index of stripe's row.
	void compute_stripe(size_t i)
	{
		assert(is_stripe_vector_computable(i));

		compute_first_part_of_stripe();

		// Start vector computation.
		for (size_t j = vector_stripe_left_boundary; j <= vector_stripe_right_boundary; ++j) {
			compute_one_stripe_diagonal(i, j);
		}

		compute_last_part_of_stripe();
	}

	/// Left upper triangle
	void compute_first_part_of_stripe()
	{
	    const size_t rectangle_row_count = stripe_size + 1;
	    const size_t rectangle_col_count = stripe_size;
	    Rectangle rectangle{rectangle_row_count, rectangle_col_count};

        rectangle.copy_to_first_row(row);
        rectangle.fill_first_column(row_idx);

        const size_t operations_num_from = stripe_size - 1;
        const size_t operations_num_until = 0;
        size_t operations_num = operations_num_from;
        for (size_t i = 1; i < rectangle_row_count; i++) {
            for (size_t j = 1; j <= operations_num ; ++j) {
                compute_distance_in_rectangle(rectangle, i, j);
            }

            operations_num--;
            if (operations_num == operations_num_until) {
                break;
            }
        }

        store_arrays_from_rectangle(rectangle);
	}

	void store_arrays_from_rectangle(const Rectangle &rectangle)
    {
        array_y = rectangle.get_diagonal(rectangle.rows_count - 1, 0, stripe_size);

        for (size_t i = 0; i < array_y.size() - 1; ++i) {
            array_w[i] = array_y[i];
        }
        array_w[array_w.size() - 1] = row[stripe_size];

        array_z = rectangle.get_diagonal(rectangle.rows_count - 2, 0, stripe_size);
    }

	void compute_distance_in_rectangle(Rectangle &rectangle, size_t i, size_t j) const
    {
	    data_element upper = rectangle.get(i-1, j);
	    data_element left_upper = rectangle.get(i-1, j-1);
	    data_element left = rectangle.get(i, j-1);
	    data_element a = input_array_1[j-1];
	    data_element b = input_array_2[i-1];
	    rectangle.set(i, j, compute_levenstein_distance(upper, left_upper, left, a, b));
    }

    void copy_arrays_to_rectangle_in_last_part(Rectangle &rectangle) const
    {
        const size_t z_from_row = rectangle.rows_count - 2;
        const size_t z_from_col = 0;
        rectangle.set_diagonal(z_from_row, z_from_col, array_z, array_z.size() - 1);

        const size_t y_from_row = rectangle.rows_count - 1;
        const size_t y_from_col = 0;
        rectangle.set_diagonal(y_from_row, y_from_col, array_y, array_y.size());
    }

	void compute_last_part_of_stripe()
    {
	    const size_t rectangle_row_count = stripe_size;
	    const size_t rectangle_col_count = stripe_size;
	    Rectangle rectangle{rectangle_row_count, rectangle_col_count};

	    copy_arrays_to_rectangle_in_last_part(rectangle);

	    const size_t operations_num_from = 1;
	    const size_t operations_num_until = stripe_size - 1;
	    size_t operations_num = operations_num_from;
        for (size_t i = 1; i < rectangle_row_count; ++i) {
            for (size_t j = rectangle_col_count - operations_num; j < rectangle_col_count; ++j) {
                compute_distance_in_rectangle(rectangle, i, j);
            }

            operations_num++;
            if (operations_num > operations_num_until) {
                break;
            }
        }

        store_row_from_rectangle_in_last_part(rectangle);
    }

    void store_row_from_rectangle_in_last_part(const Rectangle &rectangle)
    {
        const size_t last_row_idx = rectangle.rows_count - 1;

        size_t col_idx = results_cols - stripe_size + 1;
        for (size_t j = 1; j < rectangle.cols_count; ++j) {
            row[col_idx] = rectangle.get(last_row_idx, j);
            col_idx++;
        }
    }


    data_element compute_rest(const size_t row_idx)
    {
	    assert(results_cols == row.size());
	    const size_t rectangle_rows_count = results_rows - row_idx;
	    const size_t rectangle_cols_count = results_cols;
	    Rectangle rectangle{rectangle_rows_count, rectangle_cols_count};

        rectangle.copy_to_first_row(row);
        rectangle.fill_first_column(row_idx);

        for (size_t i = 1; i < rectangle_rows_count; ++i) {
            for (size_t j = 1; j < rectangle_cols_count; ++j) {
                compute_distance_in_rectangle(rectangle, i, j);
            }
        }

        return rectangle.get(rectangle_rows_count - 1, rectangle_cols_count - 1);
    }

	/// Compute with vector instructions.
	void compute_one_stripe_diagonal(size_t i, size_t j)
	{
		assert(is_stripe_diagonal_vector_computable(i, j));

		vector_type vector_y = policy::copy_to_vector(array_y);
		vector_type vector_w = policy::copy_to_vector(array_w);
		vector_type vector_z = policy::copy_to_vector(array_z);
		vector_type vector_a = get_vector_from_array(input_array_1, j - 1);
		vector_type vector_b = get_vector_from_array_reversed(input_array_2, i - stripe_size);

		// debug
        auto arr_a = policy::copy_to_array(vector_a);
        auto arr_b = policy::copy_to_array(vector_b);

		vector_type result = compute_levenstein_distance(vector_y, vector_w, vector_z, vector_a, vector_b);

        replace_arrays_with_result(result, j);
	}

	/// Updates internal buffers with new results of vector computation.
	void replace_arrays_with_result(vector_type new_result, const size_t col_idx)
    {
        auto new_result_array = policy::copy_to_array(new_result);

	    array_z = array_w;
	    array_y = new_result_array;

        for (size_t i = 0; i < array_w.size() - 1; ++i) {
            array_w[i] = array_y[i + 1];
        }
        size_t row_element_idx = col_idx + stripe_size;

        bool last_iteration = (col_idx + stripe_size == results_cols);
        if (!last_iteration) {
            array_w[array_w.size() - 1] = row[row_element_idx];
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
        array_type arr;
        size_t arr_idx = 0;
        for (int i = from_idx; i >= to_idx; i--) {
            arr[arr_idx] = array[i];
            arr_idx++;
        }
        return policy::copy_to_vector(arr);
    }

	void store_vector_to_diagonal(size_t bottom_left_row, size_t bottom_left_col, vector_type src_vec)
	{
	    const size_t row_from = bottom_left_row;
	    const size_t row_until = bottom_left_row - stripe_size + 1;
	    const size_t col_from = bottom_left_col;
	    const size_t col_until = bottom_left_col + stripe_size - 1;
        array_type tmp_array = policy::copy_to_array(src_vec);
        size_t array_idx = 0;
        for (size_t i = row_from, j = col_from; i >= row_until && j <= col_until; i--, j++)
        {
            results[i][j] = tmp_array[array_idx];
            array_idx++;
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
            array_type arr = {1, 1, 1, 1};
            vector_1 = _mm_load_si128(reinterpret_cast<__m128i *>(arr.data()));
            vector_1_initialized = true;
        }
        return vector_1;
    }

    static vector_type copy_to_vector(const array_type &array)
    {
        aligned_array = array;
        return _mm_load_si128(reinterpret_cast<__m128i *>(aligned_array.data()));
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
};

#endif
