#pragma once
#include <cmath>
#include <iostream>
#include <sstream>

namespace util {

template <typename T>
inline bool almost_equal(const T& v1, const T& v2, const T& tol) 
{
    return std::abs(v1-v2) < tol;
}


template <typename T>
inline T round_to(const T value, int precision) 
{
    double mul = std::pow(10, precision);
    return (T)((int)(value*mul + 0.5) ) / mul;
}

template<typename T>
void print_array(const T &v, size_t size, bool to_unsigned=false)
{
    std::stringstream s;
    for (size_t i=0; i<size; i++) {
        s << (s.tellp()==0? "{": ", ");
	if (to_unsigned) {
	    s << (unsigned)v[i];
	} else {
	    s << v[i];
	}
    }
    s << "}";
    std::cout << s.str() << std::endl;
}

template<typename T>
void print_array(const T* v, size_t size, bool to_unsigned=false)
{
    std::stringstream s;
    for (size_t i=0; i<size; i++) {
        s << (s.tellp()==0? "{": ", ");
	if (to_unsigned) {
	    s << (unsigned)v[i];
	} else {
	    s << v[i];
	}
    }
    s << "}";
    std::cout << s.str() << std::endl;
}


template<size_t M, size_t N, size_t K>
void matmul_rowmajor(const float* A, const float* B, float* C) {
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            *(C + m*N + n) = 0.0f;
            for (size_t k = 0; k < K; k++) {
                *(C + m*N + n) += (*(A + m*K + k)) * (*(B + k*N + n));
            }
        }
    }
}

} // util
