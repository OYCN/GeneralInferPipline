/**
 * @file detection.hpp
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

#ifndef NODE_SPECIFIC_DET_DETECTION_HPP_
#define NODE_SPECIFIC_DET_DETECTION_HPP_

#include <map>
#include <string>
#include <vector>

#include "common/datatype.hpp"
#include "node/inode.hpp"

namespace node {

class DetectionImgGen : public INode {
 private:
    inline static const char input_box[] = "box";      // C++ 17
    inline static const char input_label[] = "label";  // C++ 17
    inline static const char input_prob[] = "prob";    // C++ 17
    inline static const char input_num[] = "num";      // C++ 17
    inline static const char input_img[] = "img";      // C++ 17
    inline static const char output_img[] = "out";     // C++ 17

 public:
    explicit DetectionImgGen(core::Context* ctx, const char* name);
    ~DetectionImgGen();

 public:
    bool init(YAML::Node cfg) override;
    std::vector<BlobInfo> registerBlob() override;
    bool fetchBlob(const std::map<std::string, core::Blob*>& m) override;
    bool verification() override;
    bool exec(bool debug = false) override;
    const char* getName() override { return mName.c_str(); }

 private:
    std::string mName;
    core::Context* mCtx = nullptr;

    core::Blob* mInputBox = nullptr;
    core::Blob* mInputLabel = nullptr;
    core::Blob* mInputProb = nullptr;
    core::Blob* mInputNum = nullptr;
    core::Blob* mInputImg = nullptr;
    core::Blob* mOutImg = nullptr;

    size_t mImgH = 0;
    size_t mImgW = 0;
    size_t mImgC = 0;
    size_t mImgB = 0;
    const float* (*mGetColor)(int i);
    const char* (*mGetClass)(int i);
};

}  // namespace node

#endif  // NODE_SPECIFIC_DET_DETECTION_HPP_
