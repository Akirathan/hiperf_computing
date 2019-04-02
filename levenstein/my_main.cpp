
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
#include "functional_tester.hpp"

constexpr std::size_t alignment = 16;

static bool is_aligned(const void *ptr)
{
    return (reinterpret_cast<intptr_t>(ptr) % alignment) == 0;
}

#ifdef USE_AVX512
static void run_levenstein()
{
    auto arr1 = {1,2,3,4};
    auto arr2 = {1,2,3,4};
    levenstein<policy_avx512> levenstein{arr1.begin(), arr1.end(),
                                         arr2.begin(), arr2.end()};
}

static void run_all_avx512_tests()
{
    LevensteinTester<policy_avx512> tester;
    FunctionalTester<policy_avx512> functional_tester;
    tester.run_all_tests();
    functional_tester.run_all_tests();
}
#endif // USE_AVX512


static void run_all_sse_tests()
{
    LevensteinTester<policy_sse> tester;
    FunctionalTester<policy_sse> functional_tester;
    tester.run_all_tests();
    functional_tester.run_all_tests();
}

int main()
{
    run_all_sse_tests();
#ifdef USE_AVX512
    run_all_avx512_tests();
#endif
}
