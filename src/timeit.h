#pragma once
#include <iostream>
#include <string>
#include <chrono>


namespace util::timeit {


inline void 
timeit(char what, char unit='u', std::string msg="") {
    static std::chrono::steady_clock::time_point timeit_begin;
    if (!what) {
        timeit_begin = std::chrono::steady_clock::now();
    } else {
        std::chrono::steady_clock::time_point timeit_end = std::chrono::steady_clock::now();
        
        switch (unit) {
            case 's':
                std::cout << msg << ": " << std::chrono::duration_cast<std::chrono::seconds>(timeit_end - timeit_begin).count() <<
                " [s]" << std::endl;
                break;
            case 'm':
                std::cout << msg << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(timeit_end - timeit_begin).count() <<
                " [ms]" << std::endl;
                break;
            case 'u':
                std::cout << msg << ": " << std::chrono::duration_cast<std::chrono::microseconds>(timeit_end - timeit_begin).count() <<
                " [us]" << std::endl;
                break;
            case 'n':
                std::cout << msg << ": " << std::chrono::duration_cast<std::chrono::nanoseconds>(timeit_end - timeit_begin).count() <<
                " [ns]" << std::endl;
                break;
            default:
                std::cout << "pass s, m, u or n" << std::endl;
        }
    }
}

} // util
