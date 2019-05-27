#include "task4.hpp"
#include "du4levenstein.hpp"

#include "testbed.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

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
void use( generator_list< param_type_4, time_complexity> & gl)
{
	using data = data_4< levenstein< typename U::policy>>;

	gl.push_back( make_generic_generator_task< generator_4< data, policy_random>, task_4< U>, param_type_4>());
}

///////////////////////////////
///////////////////////////////
///////////////////////////////

int main( int argc, char * * argv)
{
	std::vector< std::string> arg{ argv + 1, argv + argc };

	generator_list< param_type_4, time_complexity> gl;

#ifdef _DEBUG
	std::size_t min_inner = 64UL;	
	std::size_t step_inner = 4;	
	std::size_t max_inner = 2UL * 1024UL;
	std::size_t min_outer = 64UL;	
	std::size_t step_outer = 4;	
	std::size_t max_outer = 2UL * 1024UL;
	time_complexity target_complexity = 250000UL;
#else
	std::size_t min_inner = 64UL;	
	std::size_t step_inner = 8;	
	std::size_t max_inner = 32UL * 1024UL;
	std::size_t min_outer = 64UL;	
	std::size_t step_outer = 8;	
	std::size_t max_outer = 32UL * 1024UL;
	time_complexity target_complexity = 1000000000UL;
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

	for ( std::size_t outer = min_outer; outer <= max_outer; outer *= step_outer)
	{
		for ( std::size_t inner = min_inner; inner <= max_inner; inner *= step_inner)
		{
			gl.push_back_size( param_type_4( outer, inner), target_complexity);
		}
	}

	std::ostringstream serr;

	logger log(std::cout, serr);

#ifdef _DEBUG
	gl.run< true>(log);
#else
	gl.run< false>(log);
#endif

	std::cerr << serr.str();

	return 0;
}

///////////////////////////////
