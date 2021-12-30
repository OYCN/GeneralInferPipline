/**
 * @file detection.cpp
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

#include "detection.hpp"

#include <opencv2/opencv.hpp>

#include "common/config.hpp"
#include "common/cuda.hpp"
#include "core/context.hpp"
#include "yolox80.hpp"

namespace node {

DetectionImgGen::DetectionImgGen(core::Context* ctx, const char* name) {
    // Pass
    mName = name;
    mCtx = ctx;
}

DetectionImgGen::~DetectionImgGen() {
    // Pass
}

bool DetectionImgGen::init(YAML::Node cfg) {
    IF_HAS_ATTR("DetectionImgGen", cfg, "data_type", "str");
    std::string type = cfg["data_type"].as<std::string>();
    auto t = common::str2Type(type);
    CHECK(t == common::DataType::kUINT8 || t == common::DataType::kINT8)
        << "Unsupported type: " << type;

    IF_HAS_ATTR("DetectionImgGen", cfg, "shape", "list(int)");
    std::vector<size_t> shape = cfg["shape"].as<std::vector<size_t>>();
    auto s = parseShape(shape);
    mImgW = std::get<3>(s);
    mImgH = std::get<2>(s);
    mImgC = std::get<1>(s);
    mImgB = std::get<0>(s);
    NODE_ASSERT(mImgB == 1, "Only supported 1 Batchsize");

    IF_HAS_ATTR("DetectionImgGen", cfg, "rule", "str");
    std::string rule = cfg["rule"].as<std::string>();
    if (rule.compare("yolox80") == 0) {
        mGetColor = yolox80::getColor;
        mGetClass = yolox80::getClass;
    } else {
        LOG(ERROR) << "[DetectionImgGen] Rule(" << rule << ") is not supported";
        return false;
    }
    NODE_ASSERT(mGetColor != nullptr && mGetClass != nullptr,
                "[DetectionImgGen] Need valid \"rule\"");
    return true;
}

std::vector<BlobInfo> DetectionImgGen::registerBlob() {
    std::vector<BlobInfo> ret;
    // Input
    {
        BlobInfo info;
        info.type = BlobInfo::kINPUT;
        info.name = input_box;
        ret.push_back(info);
        info.name = input_label;
        ret.push_back(info);
        info.name = input_prob;
        ret.push_back(info);
        info.name = input_num;
        ret.push_back(info);
        info.name = input_img;
        ret.push_back(info);
    }
    // Output
    {
        BlobInfo info;
        info.type = BlobInfo::kOUTPUT;
        info.name = output_img;
        info.args.mode = BLOB_GLOBAL_MODE;
        info.args.target = core::Blob::Target::kALL;
        info.args.size = mImgW * mImgH * mImgC * mImgB;
        ret.push_back(info);
    }
    return ret;
}

bool DetectionImgGen::fetchBlob(const std::map<std::string, core::Blob*>& m) {
    GET_FROM_MAP("DetectionImgGen", m, input_box, mInputBox);
    GET_FROM_MAP("DetectionImgGen", m, input_label, mInputLabel);
    GET_FROM_MAP("DetectionImgGen", m, input_prob, mInputProb);
    GET_FROM_MAP("DetectionImgGen", m, input_num, mInputNum);
    GET_FROM_MAP("DetectionImgGen", m, input_img, mInputImg);
    GET_FROM_MAP("DetectionImgGen", m, output_img, mOutImg);
    return true;
}

bool DetectionImgGen::verification() {
    NODE_ASSERT(mInputNum->getSize() == sizeof(size_t) * mImgB,
                "[DetectionImgGen] Byte of Num is: " << mInputNum->getSize()
                                                     << " expect size_t("
                                                     << sizeof(size_t) << ")");
    NODE_ASSERT(mOutImg->getSize() == mInputImg->getSize(),
                "IO img blob need equal");
    return true;
}

bool DetectionImgGen::exec(bool debug) {
    size_t* num = mInputNum->getHostPtr<size_t>();
    float* box = mInputBox->getHostPtr<float>();
    float* prob = mInputProb->getHostPtr<float>();
    size_t* label = mInputLabel->getHostPtr<size_t>();
    CUDA_CHECK(cudaStreamSynchronize(mCtx->getCudaStream()));

    memcpy(mOutImg->getHostPtr<char>(), mInputImg->getHostPtr<char>(),
           mInputImg->getSize());
    cv::Mat img(mImgH, mImgW, CV_MAKETYPE(CV_8U, mImgC),
                mOutImg->getHostPtr<char>());

    for (size_t i = 0; i < *num; i++) {
        const float* c = mGetColor(label[i]);
        cv::Scalar color = cv::Scalar(c[0], c[1], c[2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5) {
            txt_color = cv::Scalar(0, 0, 0);
        } else {
            txt_color = cv::Scalar(255, 255, 255);
        }
        float* this_box = box + i * 4;
        cv::rectangle(
            img,
            {static_cast<int>(this_box[0]), static_cast<int>(this_box[1]),
             static_cast<int>(this_box[2]), static_cast<int>(this_box[3])},
            color * 255, 2);
        std::string label_name = mGetClass(label[i]);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(
            label_name, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        cv::Scalar txt_bk_color = color * 0.7 * 255;
        int x = this_box[0];
        int y = this_box[1] + 1;
        if (y > mImgH) y = mImgH;
        cv::rectangle(
            img,
            cv::Rect(cv::Point(x, y),
                     cv::Size(label_size.width, label_size.height + baseLine)),
            txt_bk_color, -1);
        cv::putText(img, label_name, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }
    return true;
}

REGISTER_NODE(DetectionImgGen);

}  // namespace node
