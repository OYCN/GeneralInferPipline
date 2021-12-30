/**
 * @file datatype.cpp
 * @brief Impl of common.hpp
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

#include "datatype.hpp"

#include <algorithm>
#include <utility>

#define MACRO_UNKNOW_NVDATATYPE(dt) \
    LOG(FATAL) << "Unknow nvinfer1::DataType with " << static_cast<int32_t>(dt);
#define MACRO_UNKNOW_DATATYPE(dt)               \
    LOG(FATAL) << "Unknow core::DataType with " \
               << static_cast<MACRO_DATATYPE_TYPE>(dt);

namespace common {

nvinfer1::DataType toNvDataType(DataType dt) {
    switch (dt) {
        case DataType::kBOOL:
            return nvinfer1::DataType::kBOOL;
        case DataType::kFLOAT32:
            return nvinfer1::DataType::kFLOAT;
        case DataType::kFLOAT16:
            return nvinfer1::DataType::kHALF;
        case DataType::kINT32:
            return nvinfer1::DataType::kINT32;
        case DataType::kINT8:
            return nvinfer1::DataType::kINT8;
        default:
            MACRO_UNKNOW_DATATYPE(dt);
    }
}

DataType fromNvDataType(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kBOOL:
            return DataType::kBOOL;
        case nvinfer1::DataType::kFLOAT:
            return DataType::kFLOAT32;
        case nvinfer1::DataType::kHALF:
            return DataType::kFLOAT16;
        case nvinfer1::DataType::kINT32:
            return DataType::kINT32;
        case nvinfer1::DataType::kINT8:
            return DataType::kINT8;
        default:
            MACRO_UNKNOW_NVDATATYPE(dt);
    }
}

const char* getString(DataType dt) {
    switch (dt) {
        case DataType::kUNKNOW:
            return "Unknow";
        case DataType::kBOOL:
            return "Bool";
        case DataType::kINT8:
            return "Int8";
        case DataType::kINT16:
            return "Int16";
        case DataType::kINT32:
            return "Int32";
        case DataType::kINT64:
            return "Int64";
        case DataType::kUINT8:
            return "UInt8";
        case DataType::kUINT16:
            return "UInt16";
        case DataType::kUINT32:
            return "UInt32";
        case DataType::kUINT64:
            return "UInt64";
        case DataType::kFLOAT16:
            return "Float16";
        case DataType::kFLOAT32:
            return "Float32";
        case DataType::kFLOAT64:
            return "Float64";
        default:
            MACRO_UNKNOW_DATATYPE(dt);
    }
}

size_t getSize(DataType dt) {
    switch (dt) {
        case DataType::kBOOL:
            return 1;
        case DataType::kINT8:
            return 1;
        case DataType::kINT16:
            return 2;
        case DataType::kINT32:
            return 4;
        case DataType::kINT64:
            return 8;
        case DataType::kUINT8:
            return 1;
        case DataType::kUINT16:
            return 2;
        case DataType::kUINT32:
            return 4;
        case DataType::kUINT64:
            return 8;
        case DataType::kFLOAT16:
            return 2;
        case DataType::kFLOAT32:
            return 4;
        case DataType::kFLOAT64:
            return 8;
        default:
            MACRO_UNKNOW_DATATYPE(dt);
    }
}

std::stringstream toStringStream(DataType dt, void* ptr) {
    std::stringstream ss;
    switch (dt) {
        case DataType::kBOOL:
            ss << *reinterpret_cast<bool*>(ptr);
        case DataType::kINT8:
            ss << *reinterpret_cast<int8_t*>(ptr);
        case DataType::kINT16:
            ss << *reinterpret_cast<int16_t*>(ptr);
        case DataType::kINT32:
            ss << *reinterpret_cast<int32_t*>(ptr);
        case DataType::kINT64:
            ss << *reinterpret_cast<int64_t*>(ptr);
        case DataType::kUINT8:
            ss << *reinterpret_cast<uint8_t*>(ptr);
        case DataType::kUINT16:
            ss << *reinterpret_cast<uint16_t*>(ptr);
        case DataType::kUINT32:
            ss << *reinterpret_cast<uint32_t*>(ptr);
        case DataType::kUINT64:
            ss << *reinterpret_cast<uint64_t*>(ptr);
        case DataType::kFLOAT16:
            ss << *reinterpret_cast<half*>(ptr);
        case DataType::kFLOAT32:
            ss << *reinterpret_cast<float*>(ptr);
        case DataType::kFLOAT64:
            ss << *reinterpret_cast<double*>(ptr);
        default:
            MACRO_UNKNOW_DATATYPE(dt);
    }
    return std::move(ss);
}

// C++ 14
static constexpr inline size_t string_hash(const char* s) {
    size_t hash{}, c{};
    for (auto p = s; *p; ++p, ++c) {
        hash += *p << c;
    }
    return hash;
}

static inline size_t string_hash(const std::string s) {
    return string_hash(s.c_str());
}

static constexpr inline size_t operator"" _sh(const char* s, size_t) {
    return string_hash(s);
}

DataType str2Type(std::string str, bool tolower, bool simplest) {
    if (tolower) {
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    }
    if (simplest) {
        str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
    }
    switch (string_hash(str)) {
        case "bool"_sh:
            return DataType::kBOOL;
        case "uchar"_sh:
        case "uint8"_sh:
            return DataType::kUINT8;
        case "char"_sh:
        case "int8"_sh:
            return DataType::kINT8;
        case "uchar2"_sh:
        case "uint16"_sh:
            return DataType::kUINT16;
        case "char2"_sh:
        case "int16"_sh:
            return DataType::kINT16;
        case "uchar4"_sh:
        case "uint32"_sh:
            return DataType::kUINT32;
        case "char4"_sh:
        case "int32"_sh:
            return DataType::kINT32;
        case "uchar8"_sh:
        case "uint64"_sh:
            return DataType::kUINT64;
        case "char8"_sh:
        case "int64"_sh:
            return DataType::kINT64;
        case "float16"_sh:
        case "fp16"_sh:
        case "half"_sh:
            return DataType::kFLOAT16;
        case "float32"_sh:
        case "fp32"_sh:
        case "float"_sh:
            return DataType::kFLOAT32;
        case "float64"_sh:
        case "fp64"_sh:
        case "double"_sh:
            return DataType::kFLOAT64;
        default:
            return DataType::kUNKNOW;
    }
}

}  // namespace common
