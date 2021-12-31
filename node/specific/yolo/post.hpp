/**
 * @file post.hpp
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

#ifndef NODE_SPECIFIC_YOLO_POST_HPP_
#define NODE_SPECIFIC_YOLO_POST_HPP_

#include <map>
#include <string>
#include <vector>

#include "common/datatype.hpp"
#include "node/inode.hpp"

namespace node {

class YoloPost : public INode {
 private:
    inline static const char input_name[] = "in";       // C++ 17
    inline static const char output_box[] = "box";      // C++ 17
    inline static const char output_prob[] = "prob";    // C++ 17
    inline static const char output_label[] = "label";  // C++ 17
    inline static const char output_num[] = "num";      // C++ 17

 private:
    struct GridAndStride {
        int grid0;
        int grid1;
        size_t stride;
    };
    struct Proposal {
        float x;
        float y;
        float w;
        float h;
        size_t label;
        float prob;
    };

 public:
    explicit YoloPost(core::Context* ctx, const char* name);
    ~YoloPost();

 public:
    bool init(YAML::Node* c) override;
    std::vector<BlobInfo> registerBlob() override;
    bool fetchBlob(const std::map<std::string, core::Blob*>& m) override;
    bool verification() override;
    bool exec(bool debug = false) override;
    const char* getName() override { return mName.c_str(); }

 private:
    // once
    bool generateGridsAndStride();
    // every time
    void generateProposals();
    void qsortDescentInplace(int left, int right);
    void qsortDescentInplace();
    float getIouValue(Proposal a, Proposal b);
    void nmsSortedBboxes();
    void setOutputBlob();

 private:
    std::string mName;
    core::Context* mCtx;

    core::Blob* mInputBlob = nullptr;
    core::Blob* mOutputBox = nullptr;
    core::Blob* mOutputProb = nullptr;
    core::Blob* mOutputLabel = nullptr;
    core::Blob* mOutputNum = nullptr;

    size_t mOrgImgW = 0;
    size_t mOrgImgH = 0;
    size_t mNetImgW = 0;
    size_t mNetImgH = 0;
    float mScale_x = 0;
    float mScale_y = 0;
    size_t mNumClasses = 0;
    float mNmsThresh = 0.45;
    float mBboxConfThresh = 0.3;
    std::vector<size_t> mStrides;
    std::vector<GridAndStride> mGridAndStrides;
    std::vector<Proposal> mProposals;
    std::vector<int> mPicked;
};

}  // namespace node

#endif  // NODE_SPECIFIC_YOLO_POST_HPP_
