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
 * <tr><td>2021-12-31 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#include "camera.hpp"

#include "common/config.hpp"

namespace node {

CameraCSI::CameraCSI(core::Context* ctx, const char* name) {
    // Pass
    mName = name;
}

CameraCSI::~CameraCSI() {
    // Pass
}

bool CameraCSI::init(YAML::Node* c) {
    YAML::Node& cfg = *c;

    IF_HAS_ATTR("CameraCSI", cfg, "capture_width", "int");
    int capture_width = cfg["capture_width"].as<int>();
    IF_HAS_ATTR("CameraCSI", cfg, "capture_height", "int");
    int capture_height = cfg["capture_height"].as<int>();
    IF_HAS_ATTR("CameraCSI", cfg, "display_width", "int");
    int display_width = cfg["display_width"].as<int>();
    mOutputW = display_width;
    IF_HAS_ATTR("CameraCSI", cfg, "display_height", "int");
    int display_height = cfg["display_height"].as<int>();
    mOutputH = display_height;
    IF_HAS_ATTR("CameraCSI", cfg, "framerate", "int");
    int framerate = cfg["framerate"].as<int>();
    IF_HAS_ATTR("CameraCSI", cfg, "flip_method", "int");
    int flip_method = cfg["flip_method"].as<int>();
    std::string d =
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" +
        std::to_string(capture_width) + ", height=(int)" +
        std::to_string(capture_height) +
        ", format=(string)NV12, framerate=(fraction)" +
        std::to_string(framerate) +
        "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) +
        " ! video/x-raw, width=(int)" + std::to_string(display_width) +
        ", height=(int)" + std::to_string(display_height) +
        ", format=(string)BGRx ! videoconvert ! video/x-raw, "
        "format=(string)BGR ! appsink";
    NODE_ASSERT(mCapture.open(d, cv::CAP_GSTREAMER),
                "CSI device open failed: " << d);

    // update cfg
    cfg["data_type"] = common::getString(common::DataType::kINT8);
    cfg["shape"] = std::vector<size_t>({1, 3, mOutputH, mOutputW});

    return true;
}

std::vector<BlobInfo> CameraCSI::registerBlob() {
    BlobInfo info;
    info.name = output_name;
    info.type = BlobInfo::kOUTPUT;
    info.args.mode = BLOB_GLOBAL_MODE;
    info.args.target = core::Blob::Target::kALL;
    info.args.size = mOutputW * mOutputH * 3;
    return {info};
}

bool CameraCSI::fetchBlob(const std::map<std::string, core::Blob*>& m) {
    if (m.find(output_name) != m.end()) {
        mBlob = m.at(output_name);
        return true;
    } else {
        return false;
    }
}

bool CameraCSI::verification() {
    // Pass
    return true;
}

bool CameraCSI::exec(bool debug) {
    cv::Mat output(mOutputH, mOutputW, CV_MAKETYPE(CV_8U, 3),
                   mBlob->getHostPtr<char>());
    NODE_ASSERT(mCapture.read(output), "[" << mName << "]"
                                           << " Camera read failed");
    return true;
}

REGISTER_NODE(CameraCSI);

}  // namespace node
