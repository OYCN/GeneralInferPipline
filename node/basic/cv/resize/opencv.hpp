/**
 * @file opencv.hpp
 * @brief
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-29 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef NODE_BASIC_CV_RESIZE_OPENCV_HPP_
#define NODE_BASIC_CV_RESIZE_OPENCV_HPP_

#include <map>
#include <string>
#include <vector>

#include "common/datatype.hpp"
#include "node/inode.hpp"

namespace node {

class ResizeOpenCV : public INode {
 private:
    inline static const char input_name[] = "img";   // C++ 17
    inline static const char output_name[] = "out";  // C++ 17

 public:
    explicit ResizeOpenCV(core::Context* ctx, const char* name);
    ~ResizeOpenCV();

 public:
    bool init(YAML::Node cfg) override;
    std::vector<BlobInfo> registerBlob() override;
    bool fetchBlob(const std::map<std::string, core::Blob*>& m) override;
    bool verification() override;
    bool exec(bool debug = false) override;
    const char* getName() override { return mName.c_str(); }

 private:
    std::string mName;

    core::Blob* mInputBlob;
    core::Blob* mOutputBlob;

    size_t mInputW = 0;
    size_t mInputH = 0;
    size_t mInputC = 0;
    size_t mInputB = 0;
    size_t mOutputW = 0;
    size_t mOutputH = 0;
    size_t mOutputC = 0;
    size_t mOutputB = 0;
    size_t mOutputSize = 0;
    bool mPadding = false;
    bool mSplitC = false;
};

}  // namespace node

#endif  // NODE_BASIC_CV_RESIZE_OPENCV_HPP_
