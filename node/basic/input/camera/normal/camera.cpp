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

    IF_HAS_ATTR("CameraNormal", cfg, "device", "int");
    int device = cfg["device"].as<int>();
    NODE_ASSERT(mCapture.open(device), "Device: " << device << " open failed");

    mOutputH = mCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    mOutputW = mCapture.get(cv::CAP_PROP_FRAME_WIDTH);

    cv::Mat t;
    mCapture.read(t);
    mOutputC = t.channels();

    NODE_ASSERT(mOutputH > 0 && mOutputW > 0 && mOutputC > 0,
                mOutputH << ", " << mOutputW << ", " << mOutputC);

    cfg["data_type"] = common::getString(common::DataType::kINT8);
    cfg["shape"] = std::vector<size_t>({1, mOutputC, mOutputH, mOutputW});

    return true;
}

std::vector<BlobInfo> CameraNormal::registerBlob() {
    BlobInfo info;
    info.name = output_name;
    info.type = BlobInfo::kOUTPUT;
    info.args.mode = BLOB_GLOBAL_MODE;
    info.args.target = core::Blob::Target::kALL;
    info.args.size = mOutputH * mOutputW * mOutputC;
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
    cv::Mat output(mOutputH, mOutputW, CV_MAKETYPE(CV_8U, mOutputC),
                   mBlob->getHostPtr<char>());
    mCapture.read(output);
    return true;
}

REGISTER_NODE(CameraNormal);

}  // namespace node
