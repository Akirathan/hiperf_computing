#include "task5.hpp"

#include "testbed.hpp"

#include "du5matrix.hpp"

#include <iostream>
#include <sstream>
#include <fstream>

///////////////////////////////

///////////////////////////////

///////////////////////////////

///////////////////////////////

struct use_sse {
	using policy = policy_sse;
	static std::string name() { return "SSE"; }
};

#ifdef USE_AVX
struct use_avx {
	using policy = policy_avx;
	static std::string name() { return "AVX"; }
};
#endif

#ifdef USE_AVX512
struct use_avx512 {
	using policy = policy_avx512;
	static std::string name() { return "AVX512"; }
};
#endif

template< typename U>
void use( generator_list< std::size_t, time_complexity> & gl)
{
	using data = data_3< matrix< typename U::policy>>;

	gl.push_back(make_generic_generator_task< generator_3< data, policy_zero>, task_5, std::size_t>());
	gl.push_back(make_generic_generator_task< generator_3< data, policy_random>, task_5, std::size_t>());
//	gl.push_back(make_generic_generator_task< generator_3< data, policy_linear>, task_5, std::size_t>());
	gl.push_back(make_generic_generator_task< generator_3< data, policy_one>, task_5, std::size_t>());
}

///////////////////////////////

int main( int argc, char * * argv)
{
	std::vector< std::string> arg{ argv + 1, argv + argc};
	
	generator_list< std::size_t, time_complexity> gl;

#ifdef _DEBUG
	std::size_t min_elements = 64ULL;	
	std::size_t max_elements = 64ULL;
	time_complexity target_complexity = 200ULL;
#else
	std::size_t min_elements = 64ULL;
	std::size_t max_elements = 2048ULL;
	time_complexity target_complexity = 1000000000ULL; // 10000000000ULL
#endif

	bool u_avx = false, u_avx512 = false;

	if (arg.size() >= 1)
	{
		if (arg[0] == "avx")
			u_avx = true;
		if (arg[0] == "avx512")
		{
			u_avx = true;
			u_avx512 = true;
		}
	}

	//use<use_64>(gl);
#ifdef USE_AVX512
	if (u_avx512)
	{
		use<use_avx512>(gl);
	}
	else
#endif
#ifdef USE_AVX
	if (u_avx)
	{
		use<use_avx>(gl);
	}
	else
#endif
	{
		use<use_sse>(gl);
	}

	for ( std::size_t elements = min_elements; elements <= max_elements; elements <<= 1)
	{
		gl.push_back_size( elements, target_complexity);
	}

	std::ostringstream serr;

	logger log( std::cout, serr);

#ifdef _DEBUG
	gl.run< true>( log);
#else
	gl.run< false>( log);
#endif

	std::cerr << serr.str();

	return 0;
}

///////////////////////////////
