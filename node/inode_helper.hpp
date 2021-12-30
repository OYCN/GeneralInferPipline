/**
 * @file inode_macro.hpp
 * @brief
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-29 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef NODE_INODE_HELPER_HPP_
#define NODE_INODE_HELPER_HPP_

#include <glog/logging.h>

#include <tuple>
#include <vector>

namespace node {

#define NODE_ASSERT(cond, msg) \
    if (!(cond)) {             \
        LOG(ERROR) << msg;     \
        return false;          \
    }

#define IF_HAS_ATTR(class, cfg, attr, type) \
    NODE_ASSERT(cfg[attr],                  \
                "[" << class << "] Need attr: " << attr << " " << type);

#define GET_FROM_MAP(class, map, key, v)                       \
    NODE_ASSERT(map.find(key) != map.end(),                    \
                "[" << class << "] i/o not fetched: " << key); \
    v = map.at(key);

using BCHW = std::tuple<size_t, size_t, size_t, size_t>;

inline BCHW parseShape(std::vector<size_t> shape) {
    CHECK(shape.size() > 2) << "Size of shape is illegal: " << shape.size();
    auto iter = shape.cend() - 1;
    size_t W = *(iter--);
    size_t H = *(iter--);
    size_t C = iter != shape.cbegin() ? *(iter--) : 1;
    size_t B = 1;
    for (; iter != shape.cbegin(); iter--) {
        B *= *iter;
    }
    return std::make_tuple(B, C, H, W);
}

}  // namespace node

#endif  // NODE_INODE_HELPER_HPP_
