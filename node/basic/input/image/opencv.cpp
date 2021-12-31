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
 * <tr><td>2021-12-30 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#include "opencv.hpp"

#include <opencv2/opencv.hpp>

#include "common/config.hpp"

namespace node {

ImageOpenCV::ImageOpenCV(core::Context* ctx, const char* name) {
    // Pass
    mName = name;
}

ImageOpenCV::~ImageOpenCV() {
    // Pass
}

bool ImageOpenCV::init(YAML::Node* c) {
    YAML::Node& cfg = *c;
    IF_HAS_ATTR("ImageOpenCV", cfg, "files", "list(str)");
    mFiles = cfg["files"].as<std::vector<std::string>>();
    IF_HAS_ATTR("ImageOpenCV", cfg, "shape", "list(int)");
    {
        std::vector<size_t> shape = cfg["shape"].as<std::vector<size_t>>();
        NODE_ASSERT(shape.size() >= 2, "[ImageOpenCV]");
        auto bchw = parseShape(shape);
        mImgW = std::get<3>(bchw);
        mImgH = std::get<2>(bchw);
        mImgC = std::get<1>(bchw);
        mImgB = std::get<0>(bchw);
    }
    std::string type = cfg["data_type"].as<std::string>();
    auto t = common::str2Type(type);
    CHECK(t == common::DataType::kUINT8 || t == common::DataType::kINT8)
        << "Unsupported type: " << type;
    return true;
}

std::vector<BlobInfo> ImageOpenCV::registerBlob() {
    BlobInfo info;
    info.name = output_name;
    info.type = BlobInfo::kOUTPUT;
    info.args.mode = BLOB_GLOBAL_MODE;
    info.args.target = core::Blob::Target::kHOST;
    info.args.size = mImgW * mImgH * mImgC * mImgB;
    return {info};
}

bool ImageOpenCV::fetchBlob(const std::map<std::string, core::Blob*>& m) {
    GET_FROM_MAP("ImageOpenCV", m, output_name, mOutput);
    return true;
}

bool ImageOpenCV::verification() {
    // Pass
    return true;
}

bool ImageOpenCV::exec(bool debug) {
    cv::Mat fram;
    auto iter = mFiles.cbegin();
    for (size_t i = 0; i < mImgB; i++) {
        if (iter == mFiles.cend()) {
            break;
        }
        fram = cv::imread(*iter);
        cv::Mat out(mImgH, mImgW, CV_MAKETYPE(CV_8U, mImgC),
                    mOutput->getHostPtr<char>() + i * mImgW * mImgH * mImgC);
        cv::resize(fram, out,
                   {static_cast<int>(mImgW), static_cast<int>(mImgH)});
    }
    return true;
}

REGISTER_NODE(ImageOpenCV);

}  // namespace node
