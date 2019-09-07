#include <memory>
#include <vector>
#include <iostream>
#include "du5matrix.hpp"


int main()
{
    matrix<policy_sse> a{2, 2};
    matrix<policy_sse> b{2, 2};
    a.set(0, 0, 1);
    a.set(0, 1, 2);
    a.set(1, 0, 3);
    a.set(1, 1, 4);

    b.set(0, 0, 5);
    b.set(0, 1, 6);
    b.set(1, 0, 7);
    b.set(1, 1, 8);
    matrix<policy_sse> c{2, 2};
    c.assign_mul(a, b);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++)
            std::cout << c.get(i, j) << " ";
        std::cout << std::endl;
    }
}
