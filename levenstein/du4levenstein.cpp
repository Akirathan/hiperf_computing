#include "du4levenstein.hpp"

policy_sse::array_type policy_sse::aligned_array = {};
bool policy_sse::vector_1_initialized = false;
policy_sse::vector_type policy_sse::vector_1;


policy_avx512::array_type policy_avx512::aligned_array = {};
policy_avx512::array_type policy_avx512::array_1 = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
bool policy_avx512::vector_1_initialized = false;
policy_avx512::vector_type  policy_avx512::vector_1;

