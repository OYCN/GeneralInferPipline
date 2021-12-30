/**
 * @file cuda.hpp
 * @brief Common func or macro about cuda
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

#ifndef COMMON_CUDA_HPP_
#define COMMON_CUDA_HPP_

#include <cuda_runtime_api.h>
#include <glog/logging.h>

#define CUDA_CHECK(call)                                                       \
    if ((call) != cudaSuccess) {                                               \
        cudaError_t err = cudaGetLastError();                                  \
        LOG(FATAL) << "CUDA error calling \"" #call "\", "                     \
                   << cudaGetErrorName(err) << " " << cudaGetErrorString(err); \
    }

#endif  // COMMON_CUDA_HPP_
