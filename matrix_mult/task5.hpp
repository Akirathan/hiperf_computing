#ifndef _TASK5_HPP
#define _TASK5_HPP

#include <cstddef>
#include <memory>
#include <vector>
#include <cassert>
#include <random>
#include <algorithm>

#include "testbed.hpp"

#include <emmintrin.h>

///////////////////////////////
//#define COMPUTE_SGEMM
///////////////////////////////

#ifndef COMPUTE_SGEMM

typedef std::uint16_t matrix_element;

inline matrix_element matrix_element_zero()
{
	return 0xFFFF;
}

inline matrix_element matrix_element_add_mul(matrix_element c, matrix_element a, matrix_element b)
{
	using std::min;
	return min( (int)c, a + b);	// '+' done in int32_t; min() causes saturation
}

#else

typedef float matrix_element;

inline float matrix_element_zero()
{
	return 0.0F;
}

inline matrix_element matrix_element_add_mul(matrix_element c, matrix_element a, matrix_element b)
{
	return c + a * b;
}

#endif

/*
template< typename G>
inline G grouped_matrix_element_add_mul( G c, G a, G b)
{
	using std::min;
	return min(c, a + b);
}

inline __m128i grouped_matrix_element_add_mul( __m128i c, __m128i a, __m128i b)
{
	return _mm_min_epu16( c, _mm_adds_epu16( a, b));	// saturated '+' in uint16_t
}
*/

template< typename M>
inline unsigned long long chksum( const M & m)
{
	unsigned long long chksum = 0;

	for ( std::size_t i = 0; i < m.vsize(); ++ i)
		for ( std::size_t j = 0; j < m.hsize(); ++j)
		{
			matrix_element x = m.get( i, j);

			chksum = chksum * 3 + (unsigned long long)x;
		}

	return chksum;
}


template< typename P, typename N>
inline void fill_matrix( P & gen, N & s)
{
	for ( std::size_t i = 0; i < s.vsize(); ++ i)
	{
		for ( std::size_t j = 0; j < s.hsize(); ++ j)
		{
			s.set( i, j, gen.get());
		}
	}
}

template< typename D, typename P>
struct generator_3 {

	typedef D data_type;
	typedef matrix_element check_type;	// a kind of checksum
		
	static std::string name() { return D::name() + "_" + P::name(); }

	/*
	std::string param() const 
	{ 
		return ulibpp::lexical_cast< std::string>( opsize_);
	}
	*/

	time_complexity complexity() const 
	{ 
		return opsize_ * opsize_ * opsize_;
	}

	check_type check() const
	{
		return 0;
	}

	const D & data() const
	{
		return d_;
	}

	template< typename GP>
	generator_3( const GP &, std::size_t opsize) 
		: opsize_( opsize),
		d_( opsize, opsize, opsize)
	{
		P gen;
		gen.reset();
		fill_matrix( gen, d_.a);
		fill_matrix( gen, d_.b);
	}

private:
	const std::size_t opsize_;
	D d_;
};

struct policy_zero {

	static std::string name() { return "zero"; }

	void reset()
	{
	}

	matrix_element get()
	{
		return 0;
	}
};

const int one = 1000;

struct policy_one {

	static std::string name() { return "one"; }

	void reset()
	{
	}

	matrix_element get()
	{
		return matrix_element(one);
	}
};

struct policy_random {

	static std::string name() { return "random"; }

	policy_random()
//		: distribution(0, one)
	{
	}

	void reset()
	{
		engine.seed(729);
//		distribution.reset();
	}

	matrix_element get()
	{
//		return (matrix_element)distribution(engine);
		return matrix_element(engine() & 0x7FFF);	// avoid overflow for 16-bit add
	}
private:
	std::mt19937 engine;
//	std::uniform_int_distribution<> distribution;
};

struct policy_linear {

	static std::string name() { return "linear"; }

	policy_linear()
		: counter(0)
	{
	}

	void reset()
	{
		counter = 0;
	}

	matrix_element get()
	{
		return matrix_element( (counter ++) & 0x7FFF);	// avoid overflow in '+'
	}
private:
	unsigned counter;
};

template< typename N>
struct data_3 {
public:

	static std::string name() { return "matrix"; }

	std::size_t byte_size() const
	{
//		return a.byte_size() + b.byte_size() + c.byte_size();
		return 0;
	}

	data_3( std::size_t isize, std::size_t jsize, std::size_t ksize)
		: a( isize, jsize), b( jsize, ksize), c( isize, ksize)
	{
	}

	N a, b;
	mutable N c;
};

struct task_5 {

	static std::string name() { return "mul"; }

	template< bool cold, bool debug, typename D, typename C>
	static void run( const D & data, const C & check)
	{
		data.c.assign_mul( data.a, data.b);

		// assert( s == check);
	}


	template< bool debug, typename D, typename C>
	static void initial_check( logger & log, const D & data, const C & check)
	{
		log.ss() << "CHKSUM A[" << data.a.vsize() << "," << data.a.hsize() << "] = " << chksum( data.a) << std::endl;
		log.ss() << "CHKSUM B[" << data.b.vsize() << "," << data.b.hsize() << "] = " << chksum( data.b) << std::endl;
	}

	template< bool debug, typename D, typename C>
	static void final_check( logger & log, const D & data, const C & check)
	{
		log.ss() << "CHKSUM C[" << data.c.vsize() << "," << data.c.hsize() << "] = " << chksum( data.c) << std::endl;
		if (debug)
		{
			verify(log, data.c, data.a, data.b);
		}
	}

private:
	template< typename CT, typename AT, typename BT>
	static void verify(logger & log, const CT & c, const AT & a, const BT & b)
	{
		for (std::size_t i = 0; i < c.vsize(); ++i)
			for (std::size_t j = 0; j < c.hsize(); ++j)
			{
				auto cc = matrix_element_zero();

				for (std::size_t k = 0; k < a.hsize(); ++k)
				{
					cc = matrix_element_add_mul(cc, a.get(i, k), b.get(k, j));
				}

				if (cc != c.get(i, j))
				{
					log.ss() << "VERIFY FAILED: C[" << i << "," << j << "] = " << c.get(i, j) << " EXPECTED " << cc << std::endl;
					for (std::size_t k = 0; k < a.hsize(); ++k)
					{
						auto ccc = matrix_element_add_mul(matrix_element_zero(), a.get(i, k), b.get(k, j));
						if (ccc == cc)
						{
							log.ss() << "A[" << i << "," << k << "] = " << a.get(i, k) << " B[" << k << "," << j << "] = " << b.get(k, j) 
								<< " C = " << ccc << std::endl;
						}
					}
					return;
				}
			}
		log.ss() << "VERIFIED." << std::endl;
	}
};

#endif

