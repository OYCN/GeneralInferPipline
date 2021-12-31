/**
 * @file post.cpp
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

#include "post.hpp"

#include <algorithm>
#include <utility>

#include "common/config.hpp"
#include "common/cuda.hpp"
#include "core/context.hpp"

namespace node {

YoloPost::YoloPost(core::Context* ctx, const char* name) {
    // Pass
    mName = name;
    mCtx = ctx;
}

YoloPost::~YoloPost() {
    // Pass
}

bool YoloPost::init(YAML::Node* c) {
    YAML::Node& cfg = *c;
    std::vector<size_t> shape;
    shape = cfg["org_shape"].as<std::vector<size_t>>();
    mOrgImgW = shape[shape.size() - 1];
    mOrgImgH = shape[shape.size() - 2];
    shape = cfg["net_shape"].as<std::vector<size_t>>();
    mNetImgW = shape[shape.size() - 1];
    mNetImgH = shape[shape.size() - 2];
    mNumClasses = cfg["num_classes"].as<size_t>();
    mNmsThresh = cfg["nms_thresh"].as<float>();
    mBboxConfThresh = cfg["bbox_conf_thresh"].as<float>();
    mStrides = cfg["strides"].as<std::vector<size_t>>();
    mScale_x = static_cast<float>(mNetImgW) / mOrgImgW;
    mScale_y = static_cast<float>(mNetImgH) / mOrgImgH;
    mScale_x = std::min(mScale_x, mScale_y);
    mScale_y = mScale_x;
    return generateGridsAndStride();
}

std::vector<BlobInfo> YoloPost::registerBlob() {
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
        info.type = BlobInfo::kOUTPUT;
        info.args.target = core::Blob::Target::kHOST;
        info.args.mode = BLOB_GLOBAL_MODE;
        info.args.size = 0;
        info.name = output_box;
        ret.push_back(info);
        info.name = output_prob;
        ret.push_back(info);
        info.name = output_label;
        ret.push_back(info);
        info.name = output_num;
        info.args.size = sizeof(size_t);
        ret.push_back(info);
    }
    return ret;
}

bool YoloPost::fetchBlob(const std::map<std::string, core::Blob*>& m) {
    if (m.find(input_name) != m.end()) {
        mInputBlob = m.at(input_name);
    } else {
        return false;
    }
    if (m.find(output_box) != m.end()) {
        mOutputBox = m.at(output_box);
    } else {
        return false;
    }
    if (m.find(output_prob) != m.end()) {
        mOutputProb = m.at(output_prob);
    } else {
        return false;
    }
    if (m.find(output_label) != m.end()) {
        mOutputLabel = m.at(output_label);
    } else {
        return false;
    }
    if (m.find(output_num) != m.end()) {
        mOutputNum = m.at(output_num);
    } else {
        return false;
    }
    return true;
}

bool YoloPost::verification() {
    // Pass
    return true;
}

bool YoloPost::exec(bool debug) {
    generateProposals();
    LOG(INFO) << "Gen proposals: " << mProposals.size();
    qsortDescentInplace();
    nmsSortedBboxes();
    LOG(INFO) << "Picked proposals: " << mPicked.size();
    setOutputBlob();
    return true;
}

bool YoloPost::generateGridsAndStride() {
    if (mStrides.size() <= 0) {
        return false;
    }

    mGridAndStrides.clear();
    for (auto stride : mStrides) {
        int num_grid_y = mNetImgH / stride;
        int num_grid_x = mNetImgW / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++) {
            for (int g0 = 0; g0 < num_grid_x; g0++) {
                mGridAndStrides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
    return true;
}

void YoloPost::generateProposals() {
    mProposals.clear();
    mInputBlob->toHost(mInputBlob->getSize());
    float* feat_blob = mInputBlob->getHostPtr<float>();
    const int num_anchors = mGridAndStrides.size();
    CUDA_CHECK(cudaStreamSynchronize(mCtx->getCudaStream()));
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
        const int grid0 = mGridAndStrides[anchor_idx].grid0;
        const int grid1 = mGridAndStrides[anchor_idx].grid1;
        const int stride = mGridAndStrides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (mNumClasses + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos + 2]) * stride;
        float h = exp(feat_blob[basic_pos + 3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos + 4];
        for (size_t class_idx = 0; class_idx < mNumClasses; class_idx++) {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > mBboxConfThresh) {
                mProposals.push_back({
                    x0,         // x
                    y0,         // y
                    w,          // w
                    h,          // h
                    class_idx,  // label
                    box_prob    // prob
                });
            }
        }  // class loop
    }      // point anchor loop
}

void YoloPost::qsortDescentInplace(int left, int right) {
    int i = left;
    int j = right;
    float p = mProposals[(left + right) / 2].prob;

    while (i <= j) {
        while (mProposals[i].prob > p) i++;
        while (mProposals[j].prob < p) j--;
        if (i <= j) {
            // swap
            std::swap(mProposals[i], mProposals[j]);
            i++;
            j--;
        }
    }
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsortDescentInplace(left, j);
        }
#pragma omp section
        {
            if (i < right) qsortDescentInplace(i, right);
        }
    }
}

void YoloPost::qsortDescentInplace() {
    if (mProposals.size() == 0) {
        return;
    }
    qsortDescentInplace(0, mProposals.size() - 1);
}

float YoloPost::getIouValue(Proposal a, Proposal b) {
    int xx1, yy1, xx2, yy2;

    xx1 = std::max(a.x, b.x);
    yy1 = std::max(a.y, b.y);
    xx2 = std::min(a.x + a.w - 1, b.x + b.w - 1);
    yy2 = std::min(a.y + a.h - 1, b.y + b.h - 1);

    int insection_width, insection_height;
    insection_width = std::max(0, xx2 - xx1 + 1);
    insection_height = std::max(0, yy2 - yy1 + 1);

    float insection_area, union_area, iou;
    insection_area = static_cast<float>(insection_width) * insection_height;
    union_area = static_cast<float>(a.w * a.h + b.w * b.h - insection_area);
    iou = insection_area / union_area;
    return iou;
}

void YoloPost::nmsSortedBboxes() {
    mPicked.clear();
    const int n = mProposals.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = mProposals[i].w * mProposals[i].h;
    }
    for (int i = 0; i < n; i++) {
        const auto& a = mProposals[i];
        int keep = 1;
        for (int j = 0; j < static_cast<int>(mPicked.size()); j++) {
            const auto& b = mProposals[mPicked[j]];
            if (getIouValue(a, b) > mNmsThresh) keep = 0;
        }
        if (keep) mPicked.push_back(i);
    }
}

void YoloPost::setOutputBlob() {
    size_t* num = mOutputNum->getHostPtr<size_t>();
    *num = mPicked.size();
    if (mOutputBox->getSize() < *num * 4 * sizeof(float))
        mOutputBox->resize(*num * 4 * sizeof(float));
    if (mOutputProb->getSize() < *num * sizeof(float))
        mOutputProb->resize(*num * sizeof(float));
    if (mOutputLabel->getSize() < *num * sizeof(size_t))
        mOutputLabel->resize(*num * sizeof(size_t));
    float* box = mOutputBox->getHostPtr<float>();
    float* prob = mOutputProb->getHostPtr<float>();
    size_t* label = mOutputLabel->getHostPtr<size_t>();
    for (size_t i = 0; i < *num; i++) {
        const auto& p = mProposals[mPicked[i]];

        float x0 = (p.x) / mScale_x;
        float y0 = (p.y) / mScale_y;
        float x1 = (p.x + p.w) / mScale_x;
        float y1 = (p.y + p.h) / mScale_y;

        x0 = std::max(std::min(x0, static_cast<float>(mOrgImgW - 1)), 0.f);
        y0 = std::max(std::min(y0, static_cast<float>(mOrgImgH - 1)), 0.f);
        x1 = std::max(std::min(x1, static_cast<float>(mOrgImgW - 1)), 0.f);
        y1 = std::max(std::min(y1, static_cast<float>(mOrgImgH - 1)), 0.f);

        box[i * 4 + 0] = x0;
        box[i * 4 + 1] = y0;
        box[i * 4 + 2] = x1 - x0;
        box[i * 4 + 3] = y1 - y0;

        prob[i] = p.prob;
        label[i] = p.label;
    }
}

REGISTER_NODE(YoloPost);

}  // namespace node
