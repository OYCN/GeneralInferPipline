/**
 * @file blob.cpp
 * @brief Impl the Blob class
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

#include "blob.hpp"

#include "common/config.hpp"
#include "common/cuda.hpp"
#include "context.hpp"

namespace core {

Blob::Blob(core::Context* ctx, BlobArgs args)
    : Blob(ctx, args.size, args.target, args.mode) {
    // Pass
}

Blob::Blob(core::Context* ctx, size_t size, Target target, Mode mode)
    : mCtx(ctx), mMode(mode), mTarget(target), mSize(size) {
    CHECK(mCtx != nullptr);
    memAlloc();
}

Blob::~Blob() {
    memFree();
    mSize = 0;
}

void Blob::resize(size_t size) {
    if (size != mSize) {
        memFree();
        mSize = size;
        if (size != 0) {
            memAlloc();
        }
    }
}

void Blob::toHost(size_t size, size_t hoff, size_t doff) {
    if (mMode != Mode::kZEROCOPY && mMode != Mode::kUNIFIED) {
        if (mTarget != Target::kALL) {
            CHECK(!BLOB_STRICT_TARGET);
            mTarget = Target::kALL;
            memAlloc();
        }
        uint8_t* h = reinterpret_cast<uint8_t*>(mHostPtr);
        uint8_t* d = reinterpret_cast<uint8_t*>(mDevicePtr);
        CUDA_CHECK(cudaMemcpyAsync(h + hoff, d + doff, size,
                                   cudaMemcpyDeviceToHost,
                                   mCtx->getCudaStream()));
    }
}

void Blob::toDevice(size_t size, size_t hoff, size_t doff) {
    if (mMode != Mode::kZEROCOPY && mMode != Mode::kUNIFIED) {
        if (mTarget != Target::kALL) {
            CHECK(!BLOB_STRICT_TARGET);
            mTarget = Target::kALL;
            memAlloc();
        }
        uint8_t* h = reinterpret_cast<uint8_t*>(mHostPtr);
        uint8_t* d = reinterpret_cast<uint8_t*>(mDevicePtr);
        CUDA_CHECK(cudaMemcpyAsync(d + doff, h + hoff, size,
                                   cudaMemcpyHostToDevice,
                                   mCtx->getCudaStream()));
    }
}

void Blob::memAlloc() {
    CHECK_GE(mSize, 0);
    if (mMode == Mode::kNORMAL) {
        if (targetHasHost() && mHostPtr == nullptr) {
            mHostPtr = malloc(mSize);
            VLOG(0) << "Malloc cpu " << mSize << "byte";
        }
        if (targetHasDevice() && mDevicePtr == nullptr) {
            CUDA_CHECK(cudaMalloc(&mDevicePtr, mSize));
            VLOG(0) << "Malloc cpu " << mSize << "byte";
        }
    } else if (mMode == Mode::kPAGELOCK) {
        if (targetHasHost() && mHostPtr == nullptr) {
            CUDA_CHECK(cudaMallocHost(&mHostPtr, mSize));
            VLOG(0) << "Malloc cpu-page-lock " << mSize << "byte";
        }
        if (targetHasDevice() && mDevicePtr == nullptr) {
            CUDA_CHECK(cudaMalloc(&mDevicePtr, mSize));
            VLOG(0) << "Malloc gpu " << mSize << "byte";
        }
    } else if (mMode == Mode::kZEROCOPY) {
        if (mHostPtr == nullptr) {
            CUDA_CHECK(cudaHostAlloc(&mHostPtr, mSize, cudaHostAllocMapped));
            VLOG(0) << "Malloc cpu-page-lock " << mSize << "byte";
        }
        if (mDevicePtr == nullptr) {
            CUDA_CHECK(cudaHostGetDevicePointer(&mDevicePtr, mHostPtr,
                                                cudaHostRegisterDefault));
            VLOG(0) << "Map gpu addr from host " << mSize << "byte";
        }
    } else if (mMode == Mode::kUNIFIED) {
        if (mHostPtr == nullptr) {
            CUDA_CHECK(cudaMallocManaged(&mHostPtr, mSize));
            VLOG(0) << "Malloc gpu-managed " << mSize << "byte";
        }
        if (mDevicePtr == nullptr) {
            mDevicePtr = mHostPtr;
        }
    }
}

void Blob::memFree() {
    if (mMode == Mode::kNORMAL) {
        if (mHostPtr != nullptr) {
            free(mHostPtr);
            VLOG(0) << "Free cpu " << mSize << "byte";
        }
        if (mDevicePtr != nullptr) {
            CUDA_CHECK(cudaFree(mDevicePtr));
            VLOG(0) << "Free gpu " << mSize << "byte";
        }
    } else if (mMode == Mode::kPAGELOCK) {
        if (mHostPtr != nullptr) {
            CUDA_CHECK(cudaFreeHost(mHostPtr));
            VLOG(0) << "Free cpu-page-lock " << mSize << "byte";
        }
        if (mDevicePtr != nullptr) {
            CUDA_CHECK(cudaFree(mDevicePtr));
            VLOG(0) << "Free gpu " << mSize << "byte";
        }
    } else if (mMode == Mode::kZEROCOPY) {
        CUDA_CHECK(cudaFreeHost(mHostPtr));
        VLOG(0) << "Free cpu-page-lock " << mSize << "byte";
    } else if (mMode == Mode::kUNIFIED) {
        CUDA_CHECK(cudaFree(mHostPtr));
        VLOG(0) << "Free gpu(managed) " << mSize << "byte";
    }
    mHostPtr = nullptr;
    mDevicePtr = nullptr;
}

}  // namespace core
