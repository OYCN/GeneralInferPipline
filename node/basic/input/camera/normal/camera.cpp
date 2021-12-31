/**
 * @file camera.cpp
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

#include "camera.hpp"

#include "common/config.hpp"

namespace node {

CameraNormal::CameraNormal(core::Context* ctx, const char* name) {
    // Pass
    mName = name;
}

CameraNormal::~CameraNormal() {
    // Pass
}

bool CameraNormal::init(YAML::Node* c) {
    YAML::Node& cfg = *c;
    IF_HAS_ATTR("CameraNormal", cfg, "data_type", "str");
    std::string type = cfg["data_type"].as<std::string>();
    auto t = common::str2Type(type);
    CHECK(t == common::DataType::kUINT8 || t == common::DataType::kINT8)
        << "Unsupported type: " << type;

    IF_HAS_ATTR("CameraNormal", cfg, "shape", "list(int)");
    auto shape = cfg["shape"].as<std::vector<size_t>>();
    NODE_ASSERT(shape.size() >= 2,
                "[CameraNormal] shape must at least 2D: " << cfg["shape"]);
    auto bchw = parseShape(shape);
    mOutputW = std::get<3>(bchw);
    mOutputH = std::get<2>(bchw);
    mOutputC = std::get<1>(bchw);
    mOutputB = std::get<0>(bchw);
    CHECK_EQ(mOutputB, 1) << "Only supported 1 batchsize in camera";

    mSize = common::getSize(t) * mOutputW * mOutputH * mOutputC * mOutputB;
    NODE_ASSERT(mSize > 0, "Size is illegal: " << mSize);

    IF_HAS_ATTR("CameraNormal", cfg, "device", "int");
    int device = cfg["device"].as<int>();
    NODE_ASSERT(mCapture.open(device), "Device: " << device << " open failed");

    return true;
}

std::vector<BlobInfo> CameraNormal::registerBlob() {
    BlobInfo info;
    info.name = output_name;
    info.type = BlobInfo::kOUTPUT;
    info.args.mode = BLOB_GLOBAL_MODE;
    info.args.target = core::Blob::Target::kALL;
    info.args.size = mSize;
    return {info};
}

bool CameraNormal::fetchBlob(const std::map<std::string, core::Blob*>& m) {
    if (m.find(output_name) != m.end()) {
        mBlob = m.at(output_name);
        return true;
    } else {
        return false;
    }
}

bool CameraNormal::verification() {
    // Pass
    return true;
}

bool CameraNormal::exec(bool debug) {
    cv::Mat frame;
    mCapture >> frame;
    cv::Mat output(mOutputH, mOutputW, CV_MAKETYPE(CV_8U, mOutputC),
                   mBlob->getHostPtr<char>());
    cv::resize(frame, output,
               {static_cast<int>(mOutputH), static_cast<int>(mOutputW)});
    return true;
}

REGISTER_NODE(CameraNormal);

}  // namespace node