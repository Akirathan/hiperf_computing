#include "du4levenstein.hpp"

bool policy_sse::vector_1_initialized = false;
policy_sse::vector_type policy_sse::vector_1;


#ifdef USE_AVX512
policy_avx512::array_type policy_avx512::array_1 = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
bool policy_avx512::vector_1_initialized = false;
policy_avx512::vector_type  policy_avx512::vector_1;
#endif // USE_AVX512

