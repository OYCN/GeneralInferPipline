/**
 * @file fakedata.hpp
 * @brief
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-25 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef NODE_BASIC_INPUT_FAKE_FAKEDATA_HPP_
#define NODE_BASIC_INPUT_FAKE_FAKEDATA_HPP_

#include <map>
#include <string>
#include <vector>

#include "common/datatype.hpp"
#include "node/inode.hpp"

namespace node {

class FakeData : public INode {
 private:
    inline static const char output_name[] = "out";  // C++ 17

 public:
    explicit FakeData(core::Context* ctx, const char* name);
    ~FakeData();

 public:
    bool init(YAML::Node cfg) override;
    std::vector<BlobInfo> registerBlob() override;
    bool fetchBlob(const std::map<std::string, core::Blob*>& m) override;
    bool verification() override;
    bool exec(bool debug = false) override;
    const char* getName() override { return mName.c_str(); }

 private:
    std::string mName;

    core::Blob* mBlob = nullptr;

    common::DataType mDataType = common::DataType::kUNKNOW;
    std::vector<size_t> mSahpe;
    size_t mSize;
};

}  // namespace node

#endif  // NODE_BASIC_INPUT_FAKE_FAKEDATA_HPP_
