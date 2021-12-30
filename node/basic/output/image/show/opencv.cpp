/**
 * @file opencv.cpp
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

#include "opencv.hpp"

#include <opencv2/opencv.hpp>

#include "common/config.hpp"

namespace node {

ShowOpenCV::ShowOpenCV(core::Context* ctx, const char* name) {
    // Pass
    mName = name;
}

ShowOpenCV::~ShowOpenCV() {
    // Pass
}

bool ShowOpenCV::init(YAML::Node cfg) {
    // wait_num
    IF_HAS_ATTR("ShowOpenCV", cfg, "wait_num", "int");
    mWaitNum = cfg["wait_num"].as<int>();
    // shape
    IF_HAS_ATTR("ShowOpenCV", cfg, "shape", "list(int)");
    std::vector<size_t> shape = cfg["shape"].as<std::vector<size_t>>();
    NODE_ASSERT(shape.size() >= 2,
                "[ShowOpenCV] shape must at least 2D: " << cfg["shape"]);
    auto bchw = parseShape(shape);
    mImgW = std::get<3>(bchw);
    mImgH = std::get<2>(bchw);
    mImgC = std::get<1>(bchw);
    mImgB = std::get<0>(bchw);
    // type
    IF_HAS_ATTR("ShowOpenCV", cfg, "data_type", "str");
    std::string type = cfg["data_type"].as<std::string>();
    auto t = common::str2Type(type);
    NODE_ASSERT(t == common::DataType::kUINT8 || t == common::DataType::kINT8,
                "Unsupported type: " << type);

    return true;
}

std::vector<BlobInfo> ShowOpenCV::registerBlob() {
    std::vector<BlobInfo> ret;
    BlobInfo info;
    info.name = input_name;
    info.type = BlobInfo::IOType::kINPUT;
    ret.emplace_back(info);
    return ret;
}

bool ShowOpenCV::fetchBlob(const std::map<std::string, core::Blob*>& m) {
    if (m.find(input_name) == m.end()) {
        return false;
    }
    mBlob = m.at(input_name);
    return true;
}

bool ShowOpenCV::verification() {
    if (mBlob->getSize() != mImgW * mImgH * mImgC * mImgB) {
        return false;
    } else {
        return true;
    }
}

bool ShowOpenCV::exec(bool debug) {
    for (size_t i = 0; i < mImgB; i++) {
        cv::Mat img(mImgH, mImgW, CV_MAKETYPE(CV_8U, mImgC),
                    mBlob->getHostPtr<char>() + mImgH * mImgW * mImgC * i);
        cv::imshow(std::to_string(i), img);
    }
    cv::waitKey(mWaitNum);
    return true;
}

REGISTER_NODE(ShowOpenCV);

}  // namespace node
