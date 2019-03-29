#ifndef _TESTBED_HPP
#define _TESTBED_HPP

#include <ostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

//#include "upp_mthr.hpp"
//#include "upp_lexical_cast.hpp"

namespace ulibpp {
	typedef unsigned long long uUINT64;
	struct system_info {
		static uUINT64 get_relative_time();
	};
};

#ifdef _MSC_VER

#include <windows.h>
#undef max

namespace ulibpp {
	inline uUINT64 system_info::get_relative_time()
	{
		::SYSTEMTIME stm;
		::FILETIME ftm;
		uUINT64 utm;

		::GetSystemTime(&stm);
		::SystemTimeToFileTime(&stm,&ftm);

		utm=*(uUINT64 *)&ftm;

		return utm/10000;
	}
};

#else

//#include <sys/fcntl.h>
#include <sys/time.h>
//#include <sys/mman.h>
//#include <cctype>
#undef max

namespace ulibpp {
	inline uUINT64 system_info::get_relative_time()
	{
		::timeval tv;
		//struct timezone tz;

		if(gettimeofday(&tv, 0))
			return 0;

		return ((uUINT64)tv.tv_sec*1000)+tv.tv_usec/1000;
	}
};

#endif

class logger {
public:
	logger( std::ostream & o, std::ostream & s)
		: os_( o), ss_( s)
	{
	}

	std::ostream & os() { return os_; }
	std::ostream & ss() { return ss_; }
private:
	std::ostream & os_;
	std::ostream & ss_;
};

template< typename SP, typename SQ>
class abstract_generator {
public:
	virtual ~abstract_generator() {}
	void label( logger & log) const { label_( log); }
	template< bool debug>
	void run( logger & log, const SP & sp, const SQ & sq) const 
	{ 
		if ( debug )
			run_debug_( log, sp, sq); 
		else
			run_( log, sp, sq); 
	}
private:
	virtual void label_( logger & log) const = 0;
	virtual void run_( logger & log, const SP & sp, const SQ & sq) const = 0;
	virtual void run_debug_( logger & log, const SP & sp, const SQ & sq) const = 0;
};

template< typename SP, typename SQ>
class generator_list {
public:
	typedef std::shared_ptr< abstract_generator< SP, SQ> > generator_ptr;

	void push_back( generator_ptr p)
	{
		tv_.push_back( p);
	}

	void push_back_size( const SP & sp, const SQ & sq)
	{
		spv_.push_back( size_param( sp, sq));
	}

	template< bool debug>
	void run_row( logger & log, const SP & sp, const SQ & sq) const
	{
		log.os() << sp;
		std::for_each( tv_.begin(), tv_.end(), [&]( const generator_ptr & p){ p->template run< debug>( log, sp, sq); });
		log.os() << std::endl;
	}

	template< bool debug>
	void run( logger & log) const
	{
		log.os() << "size";
		std::for_each( tv_.begin(), tv_.end(), [&]( const generator_ptr & p){ p->label( log); });
		log.os() << std::endl;
		std::for_each( spv_.begin(), spv_.end(), [&]( const size_param & p){ run_row< debug>( log, p.sp_, p.sq_); });
	}
private:
	typedef std::vector< generator_ptr> generator_vector;

	struct size_param {
		size_param( const SP & sp, const SQ & sq)
			: sp_( sp), sq_( sq)
		{
		}
		SP sp_;
		SQ sq_;
	};

	typedef std::vector< size_param> sp_vector;

	generator_vector tv_;
	sp_vector spv_;
};

template< typename D, typename C, typename SQ>
class abstract_task {
public:
	virtual ~abstract_task() {}
	std::string name() const { return name_(); }
	template< bool debug>
	void run( logger & log, const D & data, const C & check, const SQ & sq) const 
	{ 
		if ( debug )
			run_debug_( log, data, check, sq); 
		else
			run_( log, data, check, sq); 
	}
private:
	virtual void run_( logger & log, const D & data, const C & check, const SQ & sq) const = 0;
	virtual void run_debug_( logger & log, const D & data, const C & check, const SQ & sq) const = 0;
	virtual std::string name_() const = 0;
};

template< typename D, typename C, typename SQ>
class task_list {
public:
	typedef std::shared_ptr< abstract_task< D, C, SQ> > task_ptr;

	void push_back( task_ptr p)
	{
		tv_.push_back( p);
	}

	void label( logger & log) const
	{
		std::for_each( tv_.begin(), tv_.end(), [&]( const task_ptr & p){ log.os() << "\t" << p->name(); });
	}

	template< bool debug>
	void run( logger & log, const D & data, const C & check, const SQ & sq) const
	{
		std::for_each( tv_.begin(), tv_.end(), [&]( const task_ptr & p){ p->template run< debug>( log, data, check, sq); });
	}
private:
	typedef std::vector< task_ptr> task_vector;

	task_vector tv_;
};

///////////////////////////////

typedef unsigned long long time_complexity;

struct generic_SQ {
	time_complexity iterations;
	time_complexity complexity;
};

template< typename G, typename GP, typename SP>
class generic_generator : public abstract_generator< SP, time_complexity> {
public:
	typedef typename G::data_type data_type;
	typedef typename G::check_type check_type;

	generic_generator( const task_list< data_type, check_type, generic_SQ> & ml, const GP & p)
		: ml_( ml), p_( p)
	{
	}
private:
	virtual void label_( logger & log) const
	{
		//log.os() << "\t";
		ml_.label(log);
		log.os() << "/";
		log.os() << G::name();
	}
	virtual void run_( logger & log, const SP & sp, const time_complexity & target_complexity) const
	{
		run_impl< false>( log, sp, target_complexity);
	}
	virtual void run_debug_( logger & log, const SP & sp, const time_complexity & target_complexity) const
	{
		run_impl< true>( log, sp, target_complexity);
	}
	template< bool debug>
	void run_impl( logger & log, const SP & sp, const time_complexity & target_complexity) const
	{
		G generator( p_, sp);

		generic_SQ sq;
		sq.complexity = generator.complexity();
		sq.iterations = std::max( time_complexity( 1), target_complexity / sq.complexity);

		ml_.template run< debug>( log, generator.data(), generator.check(), sq);
	}

	task_list< data_type, check_type, generic_SQ> ml_;
	GP p_;
};
/*
template< typename G, typename GP, typename SP, typename SQ>
generic_generator< G, GP, SP, SQ> * make_generic_generator( const task_list< G, SQ> & ml, const GP & p)
{
	return new generic_generator< G, GP, SP, SQ>( ml, p);
}
*/

template< typename D, typename C, typename M>
class generic_task : public abstract_task< D, C, generic_SQ> {
public:
	generic_task()
	{
	}
private:
	virtual std::string name_() const
	{
		return M::name();
	}
private:
	virtual void run_( logger & log, const D & data, const C & check, const generic_SQ & sq) const
	{
		run_impl< false>( log, data, check, sq);
	}
	virtual void run_debug_( logger & log, const D & data, const C & check, const generic_SQ & sq) const
	{
		run_impl< true>( log, data, check, sq);
	}
	template< bool debug>
	void run_impl( logger & log, const D & data, const C & check, const generic_SQ & sq) const
	{
		/*
		log.ss() << "START " << D::name() << " " << data.byte_size() << " " << M::name() << " " << sq.complexity << " " << sq.iterations << std::endl;
		*/

		M::template initial_check< debug>( log, data, check);

		// cold run
		M::template run< true, debug>( data, check);

		ulibpp::uUINT64 pre_mach = ulibpp::system_info::get_relative_time();

		for ( std::size_t i = 0; i < sq.iterations; ++ i)
		{
			M::template run< false, false>( data, check);
		}

		ulibpp::uUINT64 post_mach = ulibpp::system_info::get_relative_time();

		double ns = (1000000.0 * (double)(post_mach - pre_mach) / ((double)sq.iterations * (double)sq.complexity));

		log.os() 
			<< "\t"
			<< ns
			;

		M::template final_check< debug>( log, data, check);
		/*
		log.ss() 
			<< "STOP " 
			<< ((double)(post_mach - pre_mach))
			<< " ms total," 
			<< " "
			<< (1000000.0 * (double)(post_mach - pre_mach) / sq.iterations)
			<< " ns" 
			<< " per iter,"
			<< " "
			<< ns
			<< " ns" 
			<< " per element"
			<< std::endl
			;
		*/
	}
};
/*
template< typename G, typename T, typename GP, typename Q, typename SP, typename SQ>
generic_generator< G, GP, SP, SQ> * make_generic_generator_task( const GP & p, const Q & q)
{
	task_list< G> ml;
	ml.push_back( new generic_task< G, T>( q));
	return new generic_generator< G, GP>( ml, p);
}

template< typename G, typename T, typename GP, typename SP, typename SQ>
generic_generator< G, GP, SP, SQ> * make_generic_generator_task( const GP & p)
{
	task_list< G> ml;
	ml.push_back( new generic_task< G, T>());
	return new generic_generator< G, GP, SP, SQ>( ml, p);
}
*/
struct empty_GP {};

template< typename G, typename T, typename SP>
typename generator_list< SP, time_complexity>::generator_ptr make_generic_generator_task()
{
	typedef typename G::data_type data_type;
	typedef typename G::check_type check_type;

	task_list< data_type, check_type, generic_SQ> ml;
	ml.push_back( typename task_list< data_type, check_type, generic_SQ>::task_ptr( new generic_task< data_type, check_type, T>()));
	return typename generator_list< SP, time_complexity>::generator_ptr( new generic_generator< G, empty_GP, SP>( ml, empty_GP()));
}

#endif
