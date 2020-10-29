#ifndef HYBRID_LOG_UTILS_H_
#define HYBRID_LOG_UTILS_H_

#include <fmt/core.h>
#include <fmt/format.h>
#include <regex>
#include <string>

#include "mpi_utils.h"


namespace logutils {


template <typename... Args>
std::string format(const char* format, const Args &... args) {

    auto prefix = fmt::format("[RANK {: ^4}] ", mpiutils::getCommRank());
    auto s      = fmt::format("{}{}", prefix, fmt::format(format, args...));

    std::string line_end = "";
    std::regex re_endline("^([\\s\\S]*.*[\\s\\S]*)\\n$");
    std::smatch re_match;
    if (std::regex_match(s, re_match, re_endline)) {
        s = re_match[1].str();
        if (re_match.size() == 2)
            line_end = "\n";
    }

    std::regex re_newline("\\n");
    s = std::regex_replace(s, re_newline, "\n" + prefix);

    return s + line_end;
}


template <typename... Args>
void print(const char* format, const Args &... args) {
    fmt::print("{}", logutils::format(format, args...));
}


template <typename Container>
std::string join(const Container &c, const std::string &delimiter) {
    return fmt::format("{}", fmt::join(c, delimiter));
}


}


#endif  // HYBRID_LOG_UTILS_H_
