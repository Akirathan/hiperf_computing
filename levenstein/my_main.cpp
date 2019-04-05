
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
    std::cout << "=================================" << std::endl;
    std::cout << "Running tests specific for AVX512" << std::endl;
    std::cout << "=================================" << std::endl;

    LevensteinTester<policy_avx512> tester;
    tester.run_all_tests();
}
#endif // USE_AVX512


static void run_all_sse_tests()
{
    std::cout << "==============================" << std::endl;
    std::cout << "Running tests specific for SSE" << std::endl;
    std::cout << "==============================" << std::endl;

    LevensteinTester<policy_sse> tester;
    tester.run_all_tests();
}

/// For both SSE and AVX512
static void run_functional_tests()
{
    std::cout << "===========================================" << std::endl;
    std::cout << "Running functional tests for SSE and AVX512" << std::endl;
    std::cout << "===========================================" << std::endl;

    FunctionalTester<policy_sse> functional_tester;
    functional_tester.run_all_tests();
}

int main()
{
    auto a1 = {23,25,13,15,18,20,23,25,13,18,25,11,12,17,20,23,24};
    auto a2 = {30,30,30,15,18,20,30,30,30,30,30,30,30,30,20,23};
    assert(a2.size() == 16);
    assert(a1.size() == 17);
    dummy_levenstein dummy{a1.begin(), a1.end(), a2.begin(), a2.end()};
    dummy.compute();

    run_all_sse_tests();
#ifdef USE_AVX512
    run_all_avx512_tests();
#endif
    run_functional_tests();
}
