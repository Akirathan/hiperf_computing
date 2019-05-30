
#ifndef MY_MAIN_EXCEPTION_HPP
#define MY_MAIN_EXCEPTION_HPP

#include <exception>
#include <string>

struct Exception : public std::exception {
    std::string msg;

    explicit Exception(const std::string &ss)
            : msg(ss)
    {}

    const char *what() const noexcept override
    {
        return msg.c_str();
    }
};

#endif //MY_MAIN_EXCEPTION_HPP
