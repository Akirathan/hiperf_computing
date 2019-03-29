
#include <cstdint>
#include <emmintrin.h>
#include <iostream>
#include <vector>
#include "du4levenstein.hpp"

constexpr std::size_t alignment = 16;

static bool is_aligned(const void *ptr)
{
    return (reinterpret_cast<intptr_t>(ptr) % alignment) == 0;
}

static void sum()
{
    constexpr std::size_t array_size = 4;
    int32_t array1[array_size] = {23, 42, 43, 44};
    int32_t array2[array_size] = {23, 42, 43, 44};


    __m128i vec1 = _mm_setr_epi32(array1[0], array1[1], array1[2], array1[3]);
    __m128i vec2 = _mm_setr_epi32(array2[0], array2[1], array2[2], array2[3]);

    __m128i result = _mm_add_epi32(vec1, vec2);

    int32_t result_array[array_size] = {0, 0, 0, 0};
    _mm_storeu_si128(reinterpret_cast<__m128i *>(result_array), result);

    std::cout << "Finished" << std::endl;
}

template <>
class LevensteinTester <policy_sse> {
public:
    void test_compute_vector()
    {
        //vector_type vec = _mm_setr_epi32()
    }
private:
    using vector_type = policy_sse::vector_type;
};

int main()
{
    auto array_1 = {1, 3, 5, 7, 9};
    auto array_2 = {2, 4, 6, 8, 10};
    levenstein<policy_sse> levenstein{array_1.begin(), array_1.end(), array_2.begin(), array_2.end()};

    int result = levenstein.compute();
}
