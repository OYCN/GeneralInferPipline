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
 * <tr><td>2021-12-30 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef NODE_BASIC_INPUT_IMAGE_OPENCV_HPP_
#define NODE_BASIC_INPUT_IMAGE_OPENCV_HPP_

#include <map>
#include <string>
#include <vector>

#include "common/datatype.hpp"
#include "node/inode.hpp"

namespace node {

class ImageOpenCV : public INode {
 private:
    inline static const char output_name[] = "out";  // C++ 17

 public:
    explicit ImageOpenCV(core::Context* ctx, const char* name);
    ~ImageOpenCV();

 public:
    bool init(YAML::Node cfg) override;
    std::vector<BlobInfo> registerBlob() override;
    bool fetchBlob(const std::map<std::string, core::Blob*>& m) override;
    bool verification() override;
    bool exec(bool debug = false) override;
    const char* getName() override { return mName.c_str(); }

 private:
    std::string mName;

    core::Blob* mOutput = nullptr;
    std::vector<std::string> mFiles;

    size_t mImgW = 0;
    size_t mImgH = 0;
    size_t mImgC = 0;
    size_t mImgB = 0;
    size_t mLoopTime = 0;
};

}  // namespace node

#endif  // NODE_BASIC_INPUT_IMAGE_OPENCV_HPP_
