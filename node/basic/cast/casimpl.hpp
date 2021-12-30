/**
 * @file casimpl.hpp
 * @brief
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-27 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef NODE_BASIC_CAST_CASIMPL_HPP_
#define NODE_BASIC_CAST_CASIMPL_HPP_

#include <glog/logging.h>

#include <type_traits>

#include "common/datatype.hpp"

namespace node {
namespace castimpl {
using CastImpl = void(void* in, void* out, size_t len);

template <bool _Cond, typename _Then, typename _Else>
struct If;

template <typename _Then, typename _Else>
struct If<true, _Then, _Else> {
    static CastImpl* run() { return _Then::run(); }
};

template <typename _Then, typename _Else>
struct If<false, _Then, _Else> {
    static CastImpl* run() { return _Else::run(); }
};

using done_t = void;

template <typename _From, typename _To>
struct Impl {
    static void fun(void* in, void* out, size_t len) {
        _From* i = reinterpret_cast<_From*>(in);
        _To* o = reinterpret_cast<_To*>(out);
        for (size_t j = 0; j < len; j++) {
            o[j] = static_cast<_To>(i[j]);
        }
    }
    static CastImpl* run() {
        VLOG(0) << "Cast impl with " << common::getStr<_From>() << " to "
                << common::getStr<_To>();
        return fun;
    }
};

struct Empty {
    static void fun(void* in, void* out, size_t len) { return; }
    static CastImpl* run() {
        VLOG(0) << "Cast impl empty func";
        return fun;
    }
};

template <typename _From, typename _To>
struct ImplIf {
    static CastImpl* run() {
        return If < std::is_same<_From, void>::value ||
                   std::is_same<_To, void>::value ||
                   std::is_same<_From, _To>::value,
               Empty, Impl<_From, _To> > ::run();
    }
};

template <typename _From = done_t, typename _To = done_t>
CastImpl* getCastImpl(common::DataType from, common::DataType to) {
    common::DataType type = common::DataType::kUNKNOW;
    if (std::is_same<_From, done_t>::value) {
        type = to;
    } else if (std::is_same<_To, done_t>::value) {
        type = from;
    } else {
        return ImplIf<_From, _To>::run();
    }
    switch (type) {
        case common::DataType::kBOOL:
            return getCastImpl<bool, _From>(from, to);
        case common::DataType::kINT8:
            return getCastImpl<int8_t, _From>(from, to);
        case common::DataType::kUINT8:
            return getCastImpl<uint8_t, _From>(from, to);
        case common::DataType::kINT16:
            return getCastImpl<int16_t, _From>(from, to);
        case common::DataType::kUINT16:
            return getCastImpl<uint16_t, _From>(from, to);
        case common::DataType::kINT32:
            return getCastImpl<int32_t, _From>(from, to);
        case common::DataType::kUINT32:
            return getCastImpl<uint32_t, _From>(from, to);
        case common::DataType::kINT64:
            return getCastImpl<int64_t, _From>(from, to);
        case common::DataType::kUINT64:
            return getCastImpl<uint64_t, _From>(from, to);
            // case common::DataType::kFLOAT16:
            //         return getCastImpl<half, _From>(from, to);
        case common::DataType::kFLOAT32:
            return getCastImpl<float, _From>(from, to);
        case common::DataType::kFLOAT64:
            return getCastImpl<double, _From>(from, to);
        default:
            LOG(ERROR) << "Unknow Impl of " << common::getString(type);
            return nullptr;
    }
}

}  // namespace castimpl
}  // namespace node

#endif  // NODE_BASIC_CAST_CASIMPL_HPP_
