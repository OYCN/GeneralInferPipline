/**
 * @file file.hpp
 * @brief Func about File
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

#ifndef COMMON_FILE_HPP_
#define COMMON_FILE_HPP_

#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace common {

std::vector<char> read(std::string name) {
    std::vector<char> buff;
    std::ifstream fin(name);
    if (fin.good()) {
        fin.seekg(0, fin.end);
        size_t size = fin.tellg();
        buff.resize(size);
        fin.seekg(0);
        fin.read(buff.data(), size);
    }
    return std::move(buff);
}

}  // namespace common

#endif  // COMMON_FILE_HPP_
