/**
 * @file factory.hpp
 * @brief Define the Factory
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-24 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef CORE_FACTORY_HPP_
#define CORE_FACTORY_HPP_

#include <map>
#include <utility>

namespace core {

enum class FactoryType : int {
    kNODE  // for node
};

template <FactoryType _Type, typename _Key, typename _Val, typename... _Args>
class Factory {
 public:
    using Creator = _Val*(_Args... args);
    static constexpr FactoryType ft = _Type;

 public:
    static Factory& getFactory() {
        static Factory f;
        return f;
    }

    bool registerCreator(_Key k, Creator* c) {
        if (mMap.find(k) == mMap.end()) {
            mMap[k] = c;
            return true;
        } else {
            return false;
        }
    }

    bool isHas(_Key k) {
        // If has the key
        return mMap.find(k) != mMap.end();
    }

    Creator* getCreator(_Key k) {
        if (mMap.find(k) == mMap.end()) {
            return nullptr;
        } else {
            return mMap[k];
        }
    }

    // TODO(oPluss): Using right value
    _Val* getInstance(_Key k, _Args... args) {
        if (mMap.find(k) == mMap.end()) {
            return nullptr;
        } else {
            return mMap[k](std::forward<_Args>(args)...);
        }
    }

 private:
    Factory() {}

 private:
    std::map<_Key, Creator*> mMap;
};

}  // namespace core

#endif  // CORE_FACTORY_HPP_
