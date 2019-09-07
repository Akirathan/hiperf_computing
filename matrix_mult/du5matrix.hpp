#ifndef matrix_h_
#define matrix_h_

#include <cstddef>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>

#undef min

#ifdef _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT 
#endif

template< typename policy>
class matrix {
public:
	static std::string name() { return "matrix"; }

	typedef std::uint16_t matrix_element;

	std::size_t byte_size() const
	{
		return 0;
	}

	matrix( std::size_t m, std::size_t n)
		: m_(m), n_(n), v_(m * n, matrix_element{})
	{
	}

	std::size_t vsize() const
	{
		return m_;
	}

	std::size_t hsize() const
	{
		return n_;
	}

	void set( std::size_t i, std::size_t j, matrix_element e)
	{
		v_[ i * n_ + j] = e;
	}

	matrix_element get( std::size_t i, std::size_t j) const
	{
		return v_[ i * n_ + j];
	}

	void assign_mul_dummy( const matrix & a, const matrix & b)
	{
		std::size_t L = m_;
		assert( L == a.m_);
		std::size_t M = n_;
		assert( M == b.n_);
		std::size_t N = a.n_;
		assert( N == b.m_);

		matrix_element * RESTRICT cv = v_.data();
		const matrix_element * RESTRICT av = a.v_.data();
		const matrix_element * RESTRICT bv = b.v_.data();

		for( std::size_t i = 0; i < L; ++ i)
			for (std::size_t j = 0; j < M; ++j)
			{
				cv[i * M + j] = 0xFFFF;
			}
		for (std::size_t i = 0; i < L; ++i)
			for (std::size_t k = 0; k < N; ++k)
			{
				auto ax = av[i * N + k];
				for (std::size_t j = 0; j < M; ++j)
					cv[i * M + j] = std::min(cv[i * M + j],
						matrix_element(ax + bv[k * M + j]));
			}
	}

	void assign_mul( const matrix & a, const matrix & b)
    {
        std::size_t L = m_;
        assert( L == a.m_);
        std::size_t M = n_;
        assert( M == b.n_);
        std::size_t N = a.n_;
        assert( N == b.m_);

        matrix_element * RESTRICT cv = v_.data();
        const matrix_element * RESTRICT av = a.v_.data();
        const matrix_element * RESTRICT bv = b.v_.data();

        for (std::size_t i = 0; i < L; ++i)
            for (std::size_t j = 0; j < M; ++j)
                cv[i * M + j] = 0xFFFF;

        for (std::size_t i = 0; i < L; ++i) {
            for (std::size_t k = 0; k < N; ++k) {
                matrix_element ax = av[i * N + k];
                for (std::size_t j = 0; j < M; ++j) {
                    cv[i*M + j] = std::min(cv[i*M + j],
                            matrix_element{static_cast<matrix_element>(ax + bv[k*M + j])}
                            );
                }
            }
        }
    }

private:
	std::vector< matrix_element> v_;
	std::size_t m_, n_;
};

struct policy_sse {
};

struct policy_avx {
};

struct policy_avx512 {
};

#endif
