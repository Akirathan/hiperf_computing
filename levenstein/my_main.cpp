
#include <cstdint>
#include <cstdlib>
#include <emmintrin.h>
#include <iostream>
#include <vector>
#include <tuple>
#include "du4levenstein.hpp"
#include "dummy_levenstein.hpp"
#include "levenstein_tester_avx512.hpp"
#include "levenstein_tester_sse.hpp"

#define USE_AVX512

constexpr std::size_t alignment = 16;

static bool is_aligned(const void *ptr)
{
    return (reinterpret_cast<intptr_t>(ptr) % alignment) == 0;
}

#ifdef USE_AVX512
static void run_avx512_levenstein()
{
    auto arr1 = {1,2,3,4};
    auto arr2 = {1,2,3,4};
    levenstein<policy_avx512> levenstein{arr1.begin(), arr1.end(),
                                         arr2.begin(), arr2.end()};
}

static void run_avx512_vector_tests()
{
    LevensteinTester<policy_avx512> tester;
    tester.run_all_tests();
}

#endif // USE_AVX512

int main()
{
    run_avx512_vector_tests();
}
