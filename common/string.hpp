/**
 * @file string.hpp
 * @brief
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-25 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef COMMON_STRING_HPP_
#define COMMON_STRING_HPP_

#include <string>
#include <utility>
#include <vector>

namespace common {

std::vector<std::string> split(std::string s, std::string delimiters = " ") {
    std::vector<std::string> tokens;
    std::string::size_type lastPos = s.find_first_not_of(delimiters, 0);
    std::string::size_type pos = s.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
        tokens.emplace_back(s.substr(lastPos, pos - lastPos));
        lastPos = s.find_first_not_of(delimiters, pos);
        pos = s.find_first_of(delimiters, lastPos);
    }
    return std::move(tokens);
}

}  // namespace common

#endif  // COMMON_STRING_HPP_
