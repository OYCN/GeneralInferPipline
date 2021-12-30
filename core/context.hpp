/**
 * @file context.hpp
 * @brief Context of the project
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

#ifndef CORE_CONTEXT_HPP_
#define CORE_CONTEXT_HPP_

#include <cuda_runtime_api.h>

#include "blobmgr.hpp"

namespace core {

class Context {
 public:
    Context();
    cudaStream_t& getCudaStream() { return mCudaStream; }
    BlobManager& getBlobManager() { return mBmgr; }

 private:
    cudaStream_t mCudaStream;
    BlobManager mBmgr;
};

}  // namespace core

#endif  // CORE_CONTEXT_HPP_
