/**
 * @file context.cpp
 * @brief
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
#include "context.hpp"

#include "common/cuda.hpp"

namespace core {

Context::Context() { CUDA_CHECK(cudaStreamCreate(&mCudaStream)); }

}  // namespace core
