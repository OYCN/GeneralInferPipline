/**
 * @file blobmgr.hpp
 * @brief Manager of blob
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-21 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef CORE_BLOBMGR_HPP_
#define CORE_BLOBMGR_HPP_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "blob.hpp"

namespace core {

class BlobManager {
 public:
    Blob* get(std::string name) {
        if (mMap.find(name) != mMap.end()) {
            return mMap[name].get();
        } else {
            return nullptr;
        }
    }

    bool add(std::string name, std::unique_ptr<Blob> blob) {
        if (mMap.find(name) != mMap.end()) {
            return false;
        } else {
            mMap.emplace(name, std::move(blob));
            return true;
        }
    }

    bool add(std::string name, Context* ctx, Blob::BlobArgs args) {
        if (mMap.find(name) != mMap.end()) {
            return false;
        } else {
            mMap.emplace(name, new Blob(ctx, args));
            return true;
        }
    }

 private:
    std::map<std::string, std::unique_ptr<Blob>> mMap;
};

}  // namespace core

#endif  // CORE_BLOBMGR_HPP_
