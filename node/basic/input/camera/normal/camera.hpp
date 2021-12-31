/**
 * @file camera.hpp
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

#ifndef NODE_BASIC_INPUT_CAMERA_NORMAL_CAMERA_HPP_
#define NODE_BASIC_INPUT_CAMERA_NORMAL_CAMERA_HPP_

#include <map>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

#include "common/datatype.hpp"
#include "node/inode.hpp"

namespace node {

class CameraNormal : public INode {
 private:
    inline static const char output_name[] = "out";  // C++ 17

 public:
    explicit CameraNormal(core::Context* ctx, const char* name);
    ~CameraNormal();

 public:
    bool init(YAML::Node* c) override;
    std::vector<BlobInfo> registerBlob() override;
    bool fetchBlob(const std::map<std::string, core::Blob*>& m) override;
    bool verification() override;
    bool exec(bool debug = false) override;
    const char* getName() override { return mName.c_str(); }

 private:
    std::string mName;

    core::Blob* mBlob = nullptr;

    cv::VideoCapture mCapture;
    size_t mOutputW = 0;
    size_t mOutputH = 0;
    size_t mOutputC = 0;
};

}  // namespace node

#endif  // NODE_BASIC_INPUT_CAMERA_NORMAL_CAMERA_HPP_
