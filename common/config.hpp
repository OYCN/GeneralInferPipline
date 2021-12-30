/**
 * @file config.hpp
 * @brief Config for this proj
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

#ifndef COMMON_CONFIG_HPP_
#define COMMON_CONFIG_HPP_

// #define BLOB_STRICT_TARGET true
#define BLOB_STRICT_TARGET false
// #define BLOB_GLOBAL_MODE core::Blob::Mode::kNORMAL
#define BLOB_GLOBAL_MODE core::Blob::Mode::kPAGELOCK
// #define BLOB_GLOBAL_MODE core::Blob::Mode::kUNIFIED
// #define BLOB_GLOBAL_MODE core::Blob::Mode::kZEROCOPY

#endif  // COMMON_CONFIG_HPP_
