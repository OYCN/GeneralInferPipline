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

#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "common/config.hpp"

namespace node {

ResizeOpenCV::ResizeOpenCV(core::Context* ctx, const char* name) {
    // Pass
    mName = name;
}

ResizeOpenCV::~ResizeOpenCV() {
    // Pass
}

bool ResizeOpenCV::init(YAML::Node cfg) {
    IF_HAS_ATTR("Camera", cfg, "data_type", "str");
    std::string type = cfg["data_type"].as<std::string>();
    auto t = common::str2Type(type);
    CHECK(t == common::DataType::kUINT8 || t == common::DataType::kINT8)
        << "Unsupported type: " << type;

    IF_HAS_ATTR("ResizeOpenCV", cfg, "from", "list(int)")
    IF_HAS_ATTR("ResizeOpenCV", cfg, "to", "list(int)")
    auto from = cfg["from"].as<std::vector<size_t>>();
    auto to = cfg["to"].as<std::vector<size_t>>();
    NODE_ASSERT(from.size() == to.size(),
                "len of \"from\" and \"to\" must be equal");
    {
        auto s = parseShape(from);
        mInputW = std::get<3>(s);
        mInputH = std::get<2>(s);
        mInputC = std::get<1>(s);
        mInputB = std::get<0>(s);
    }
    {
        auto s = parseShape(to);
        mOutputW = std::get<3>(s);
        mOutputH = std::get<2>(s);
        mOutputC = std::get<1>(s);
        mOutputB = std::get<0>(s);
    }
    NODE_ASSERT(mInputC == mOutputC,
                "Not supported resize C: " << mInputC << " to " << mOutputC);
    NODE_ASSERT(mInputB == mOutputB,
                "Not supported different B: " << mInputB << " to " << mOutputB);

    mOutputSize =
        common::getSize(t) * mOutputW * mOutputH * mOutputC * mOutputB;
    NODE_ASSERT(mOutputSize > 0, " size of output error: " << mOutputSize);
    IF_HAS_ATTR("ResizeOpenCV", cfg, "padding", "bool")
    mPadding = cfg["padding"].as<bool>();
    IF_HAS_ATTR("ResizeOpenCV", cfg, "splitC", "bool")
    mSplitC = cfg["splitC"].as<bool>();
    return true;
}

std::vector<BlobInfo> ResizeOpenCV::registerBlob() {
    std::vector<BlobInfo> ret;
    // Input
    {
        BlobInfo info;
        info.name = input_name;
        info.type = BlobInfo::kINPUT;
        ret.push_back(info);
    }
    // Output
    {
        BlobInfo info;
        info.name = output_name;
        info.type = BlobInfo::kOUTPUT;
        info.args.target = core::Blob::Target::kALL;
        info.args.mode = BLOB_GLOBAL_MODE;
        info.args.size = mOutputSize;
        ret.push_back(info);
    }
    return ret;
}

bool ResizeOpenCV::fetchBlob(const std::map<std::string, core::Blob*>& m) {
    if (m.find(input_name) == m.end()) {
        return false;
    }
    mInputBlob = m.at(input_name);
    if (mInputBlob == nullptr) {
        return false;
    }
    if (m.find(output_name) == m.end()) {
        return false;
    }
    mOutputBlob = m.at(output_name);
    if (mOutputBlob == nullptr) {
        return false;
    }
    return true;
}

bool ResizeOpenCV::verification() {
    // Pass
    return true;
}

bool ResizeOpenCV::exec(bool debug) {
    for (size_t i = 0; i < mOutputB; i++) {
        cv::Mat input(
            mInputH, mInputW, CV_MAKETYPE(CV_8U, mInputC),
            mInputBlob->getHostPtr<char>() + i * mInputW * mInputH * mInputC);
        cv::Mat output;
        if (mSplitC) {
            output = cv::Mat(mOutputH, mOutputW, CV_MAKETYPE(CV_8U, mOutputC));
        } else {
            output = cv::Mat(mOutputH, mOutputW, CV_MAKETYPE(CV_8U, mOutputC),
                             mOutputBlob->getHostPtr<char>() +
                                 i * mOutputW * mOutputH * mOutputC);
        }
        if (mPadding) {
            float r = std::min(mOutputW / (mInputW * 1.0),
                               mOutputH / (mInputH * 1.0));
            int unpad_w = r * mInputW;
            int unpad_h = r * mInputH;
            cv::Mat re(unpad_h, unpad_w, CV_MAKETYPE(CV_8U, mInputC));
            cv::resize(input, re, re.size());
            output.setTo(cv::Scalar(114, 114, 114));
            re.copyTo(output(cv::Rect(0, 0, re.cols, re.rows)));
        } else {
            cv::resize(input, output, output.size());
        }
        if (mSplitC) {
            std::vector<cv::Mat> v;
            for (int c = 0; c < mInputC; c++) {
                v.emplace_back(mOutputH, mOutputW, CV_MAKETYPE(CV_8U, 1),
                               mOutputBlob->getHostPtr<char>() +
                                   i * mOutputW * mOutputH * mOutputC +
                                   mOutputW * mOutputH * c);
            }
            cv::split(output, v);
        }
    }
    return true;
}

REGISTER_NODE(ResizeOpenCV);

}  // namespace node
