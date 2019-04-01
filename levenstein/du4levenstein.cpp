#include "du4levenstein.hpp"

std::array<policy_sse::data_element, policy_sse::elems_count_per_register> policy_sse::aligned_array =
        std::array<policy_sse::data_element, policy_sse::elems_count_per_register>();
bool policy_sse::vector_1_initialized = false;
policy_sse::vector_type policy_sse::vector_1;

