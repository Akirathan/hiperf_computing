#ifndef matrix_h_
#define matrix_h_

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>

#undef min

#ifdef _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
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
		: m_(m), n_(n)
	{
	    std::size_t size = m * n;
	    if (!is_aligned(size)) {
	        std::size_t new_size = align(size);
	        size = new_size;
	    }
	    v_ = static_cast<matrix_element *>(std::aligned_alloc(policy::alignment, size * sizeof(matrix_element)));
	    assert(v_ != nullptr);
	    assert(is_aligned(reinterpret_cast<uintptr_t>(v_)));
	}

	~matrix()
    {
	    std::free(v_);
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

	void assign_mul( const matrix & a, const matrix & b)
    {
        const std::size_t L = m_;
        assert( L == a.m_);
        const std::size_t M = n_;
        assert( M == b.n_);
        const std::size_t N = a.n_;
        assert( N == b.m_);

        __attribute__ ((aligned(policy::alignment))) matrix_element * RESTRICT cv = v_;
        __attribute__ ((aligned(policy::alignment))) const matrix_element * RESTRICT av = a.v_;
        __attribute__ ((aligned(policy::alignment))) const matrix_element * RESTRICT bv = b.v_;

        // TODO: Move to constructor?
        for (std::size_t i = 0; i < L; ++i)
            for (std::size_t j = 0; j < M; ++j)
                cv[i * M + j] = 0xFFFF;

        for (std::size_t i = 0; i < L; i++) {
            for (std::size_t k = 0; k < N; k++) {
                for (std::size_t j = 0; j < M; j++) {
                    std::size_t a_idx = i * N + k; // A[i,k]
                    std::size_t b_idx = k * M + j; // B[k,j]
                    std::size_t c_idx = i * M + j; // C[i,j]
                    matrix_element ax = av[a_idx];
                    matrix_element bx = bv[b_idx];
                    // C[i,j] = min(C[i,j], A[i,k] + B[k,j])
                    cv[c_idx] = cv[c_idx] < ax + bx ? cv[c_idx] : ax + bx;
                }
            }
        }
    }

private:
	matrix_element *v_;
	std::size_t m_, n_;

	bool is_aligned(uintptr_t addr) const
    {
	    return (addr % policy::alignment) == 0;
    }

    uintptr_t align(uintptr_t addr) const
    {
	    while (!is_aligned(addr))
	        ++addr;
	    return addr;
    }
};

struct policy_sse {
    static constexpr std::size_t alignment = 8;
};

struct policy_avx {
    static constexpr std::size_t alignment = 16;
};

struct policy_avx512 {
    static constexpr std::size_t alignment = 32;
};

#endif
