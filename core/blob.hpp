/**
 * @file blob.hpp
 * @brief Define the Blob class
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-19 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef CORE_BLOB_HPP_
#define CORE_BLOB_HPP_

#include "common/datatype.hpp"

namespace core {

class Context;

/**
 * @brief Manage the memory
 */
class Blob {
 public:
    enum Target : uint8_t { kHOST = 1, kDEVICE = 1 << 1, kALL = (1 << 1) | 1 };
    enum Mode : uint8_t { kNORMAL, kPAGELOCK, kUNIFIED, kZEROCOPY };
    struct BlobArgs {
        size_t size;
        Target target;
        Mode mode;
    };

 public:
    Blob(core::Context* ctx, BlobArgs args);
    Blob(core::Context* ctx, size_t size, Target target, Mode mode);
    ~Blob();

 public:
    /**
     * @brief Get the size of the blob
     * @return size_t
     */
    size_t getSize() { return mSize; }

    /**
     * @brief Resize the blob
     * @param  size             target size
     */
    void resize(size_t size);

    /**
     * @brief Copy the mem to Host
     * @param  size             Size of mem will copy
     * @param  hoff             Offset of the Host start addr
     * @param  doff             Offset of the Device start addr
     */
    void toHost(size_t size, size_t hoff = 0, size_t doff = 0);

    /**
     * @brief Copy the mem to Device
     * @param  size             Size of mem will copy
     * @param  hoff             Offset of the Host start addr
     * @param  doff             Offset of the Device start addr
     */
    void toDevice(size_t size, size_t hoff = 0, size_t doff = 0);

 public:
    /**
     * @brief Get the Host Ptr
     * @tparam _T Type of the ptr
     * @return _T
     */
    template <typename _T>
    _T* getHostPtr() {
        return reinterpret_cast<_T*>(mHostPtr);
    }

    /**
     * @brieUTILS_BLOB_HPP_ Get the Device Ptr
     * @tparam _T Type of the ptr
     * @return _T
     */
    template <typename _T>
    _T* getDevicePtr() {
        return reinterpret_cast<_T*>(mDevicePtr);
    }

 private:
    /**
     * @brief If the Blob::Target has Host flag
     * @return true
     * @return false
     */
    bool targetHasHost() { return (mTarget & kHOST) == kHOST; }

    /**
     * @brief If the Blob::Target has Device flag
     * @return true
     * @return false
     */
    bool targetHasDevice() { return (mTarget & kDEVICE) == kDEVICE; }

    /**
     * @brief alloc memory by object attribute
     */
    void memAlloc();

    /**
     * @brief free memory by object attribute
     */
    void memFree();

 private:
    core::Context* mCtx;
    Mode mMode;
    Target mTarget;
    void* mHostPtr = nullptr;
    void* mDevicePtr = nullptr;
    size_t mSize = 0;
};

}  // namespace core

#endif  // CORE_BLOB_HPP_
