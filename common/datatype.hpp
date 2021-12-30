/**
 * @file datatype.hpp
 * @brief Define the common func in this project
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

#ifndef COMMON_DATATYPE_HPP_
#define COMMON_DATATYPE_HPP_

#include <NvInfer.h>
#include <cuda_fp16.h>  // NOLINT(build/include_subdir)
#include <glog/logging.h>

#include <sstream>
#include <string>

namespace common {

#define MACRO_DATATYPE_TYPE uint32_t

template <typename _T>
inline const char* getStr() {
    return "unknow type";
}

#define DEFINE_TYPE2STR(type)           \
    template <>                         \
    inline const char* getStr<type>() { \
        return #type;                   \
    }

DEFINE_TYPE2STR(bool);
DEFINE_TYPE2STR(int8_t);
DEFINE_TYPE2STR(uint8_t);
DEFINE_TYPE2STR(int16_t);
DEFINE_TYPE2STR(uint16_t);
DEFINE_TYPE2STR(int32_t);
DEFINE_TYPE2STR(uint32_t);
DEFINE_TYPE2STR(int64_t);
DEFINE_TYPE2STR(uint64_t);
DEFINE_TYPE2STR(half);
DEFINE_TYPE2STR(float);
DEFINE_TYPE2STR(double);

/**
 * @brief Definition of dataType in this project
 */
enum class DataType : MACRO_DATATYPE_TYPE {
    kUNKNOW = 0,
    kBOOL,
    kINT8,
    kINT16,
    kINT32,
    kINT64,
    kUINT8,
    kUINT16,
    kUINT32,
    kUINT64,
    kFLOAT16,
    kFLOAT32,
    kFLOAT64
};

/**
 * @brief Convert core::DataType to nvinfer1::DataType
 * @param  dt               one of the core::DataType
 * @return nvinfer1::DataType
 */
nvinfer1::DataType toNvDataType(DataType dt);

/**
 * @brief Convert nvinfer1::DataType to core::DataType
 * @param  dt               one of the nvinfer1::DataType
 * @return DataType
 */
DataType fromNvDataType(nvinfer1::DataType dt);

/**
 * @brief Convert core::DataType to string
 * @param  dt               core::DataType
 * @return const char*
 */
const char* getString(DataType dt);

/**
 * @brief The memory size of input
 * @param  dt               core::DataType
 * @return size_t
 */
size_t getSize(DataType dt);

/**
 * @brief Get the String Stream of the ptr by datatype
 * @param  dt               core::DataType
 * @param  ptr              a ptr
 * @return std::stringstream
 */
std::stringstream toStringStream(DataType dt, void* ptr);

/**
 * @brief Convert string to type
 * @param  str              str of the type
 * @param  tolower          lower the str
 * @param  simplest         del the space
 * @return DataType
 */
DataType str2Type(std::string str, bool tolower = true, bool simplest = true);

}  // namespace common

#endif  // COMMON_DATATYPE_HPP_
