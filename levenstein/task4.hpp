#ifndef _TASK4_HPP
#define _TASK4_HPP

#include <cstddef>
#include <memory>
#include <vector>
#include <cassert>
#include <random>

#include "testbed.hpp"

///////////////////////////////

typedef int data_element;

typedef std::pair< std::size_t, std::size_t> param_type_4;	// string sizes

inline std::ostream & operator<<( std::ostream & os, const param_type_4 & v)
{
	return os << v.first << "*" << v.second;
}

template< typename D, typename P>
struct generator_4 {

	typedef D data_type;
	struct check_type {	// a kind of checksum
		check_type(std::size_t a, std::size_t b, std::uint64_t chka, std::uint64_t chkb)
			: asize(a), bsize(b), chka(chka), chkb(chkb)
		{}

		std::size_t asize, bsize;
		std::uint64_t chka, chkb;
	};
		
	static std::string name() { return D::name() + "_" + P::name(); }

	/*	
	std::string param() const 
	{ 
		return ulibpp::lexical_cast< std::string>( opsize_);
	}
	*/

	time_complexity complexity() const 
	{ 
		return s1_ * s2_;
	}

	check_type check() const
	{
		return check_type( s1_, s2_, chksum(v1_.begin(), v1_.end()), chksum(v2_.begin(), v2_.end()));
	}

	template< typename IT>
	static std::uint64_t chksum(IT b, IT e)
	{
		std::uint64_t s = 0;

		for (;b != e; ++ b)
		{
			auto x = * b;

			s = s * 3 + (std::uint64_t)x;
		}

		return s;
	}

	const D & data() const
	{
		return d_;
	}

	template< typename GP>
	generator_4( const GP &, param_type_4 opsize) 
		: p_(),
		s2_( opsize.second),
		s1_( opsize.first),
		v1_( opsize.first),
		v2_( opsize.second),
		randgen1_( v1_.begin(), v1_.end(), p_),
		randgen2_(v2_.begin(), v2_.end(), p_),
		d_( v1_.cbegin(), v1_.cend(), v2_.cbegin(), v2_.cend())
	{
	}

private:

	class randgen {
	public:
		template< typename IT>
		randgen(IT b, IT e, P & p)
		{
			std::generate(b, e, [& p]() -> data_element {
				return p.get();
			});
		}
	};

	P p_;
	std::size_t s1_, s2_;
	std::vector< data_element> v1_, v2_;
	randgen randgen1_, randgen2_;
	D d_;
};

struct policy_random {

	static std::string name() { return "random"; }

	policy_random()
		: ui_(0, 255)
	{
	}

	void reset()
	{
	}

	data_element get()
	{
		return ui_(engine_);
	}

private:
	typedef std::mt19937 M;
	M engine_;
	typedef std::uniform_int_distribution< data_element> D;
	D ui_;
};

template< typename L>
struct data_4 {
public:

	static std::string name() { return "du4"; }
	/*
	std::size_t byte_size() const
	{
		return inner.byte_size();
	}
	*/
	template< typename I1, typename I2>
	data_4(I1 i1b, I1 i1e, I2 i2b, I2 i2e)
		: loew_(i1b, i1e, i2b, i2e)
	{
	}

	mutable L loew_;
	mutable data_element result_;
};

template< typename P>
struct task_4 {

	static std::string name() { return "levenstein<" + P::name() + ">"; }

	template< bool cold, bool debug, typename D, typename C>
	static void run( const D & data, const C & check)
	{
		data.result_ = data.loew_.compute();
	}

	template< bool debug, typename D, typename C>
	static void initial_check( logger & log, const D & data, const C & check)
	{
		log.ss() << "CHKSUM A[" << check.asize << "] = " << check.chka << std::endl;
		log.ss() << "CHKSUM B[" << check.bsize << "] = " << check.chkb << std::endl;
	}

	template< bool debug, typename D, typename C>
	static void final_check( logger & log, const D & data, const C & check)
	{
		log.ss() << "DISTANCE = " << data.result_ << std::endl;
	}
};

#endif

