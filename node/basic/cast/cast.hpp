/**
 * @file cast.hpp
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

#ifndef NODE_BASIC_CAST_CAST_HPP_
#define NODE_BASIC_CAST_CAST_HPP_

#include <map>
#include <string>
#include <vector>

#include "casimpl.hpp"
#include "common/datatype.hpp"
#include "node/inode.hpp"

namespace node {

class Cast : public INode {
 private:
    inline static const char input_name[] = "in";    // C++ 17
    inline static const char output_name[] = "out";  // C++ 17

 public:
    explicit Cast(core::Context* ctx, const char* name);
    ~Cast();

 public:
    bool init(YAML::Node* c) override;
    std::vector<BlobInfo> registerBlob() override;
    bool fetchBlob(const std::map<std::string, core::Blob*>& m) override;
    bool verification() override;
    bool exec(bool debug = false) override;
    const char* getName() override { return mName.c_str(); }

 private:
    std::string mName;

    core::Blob* mInput = nullptr;
    core::Blob* mOutput = nullptr;

    size_t mSize = 0;
    common::DataType mFrom = common::DataType::kUNKNOW;
    common::DataType mTo = common::DataType::kUNKNOW;

    castimpl::CastImpl* mCastImplFunc = nullptr;
};

}  // namespace node

#endif  // NODE_BASIC_CAST_CAST_HPP_
