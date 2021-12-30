/**
 * @file template.hpp
 * @brief
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-26 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef NODE_TEMPLATE_TEMPLATE_HPP_
#define NODE_TEMPLATE_TEMPLATE_HPP_

#include <map>
#include <string>
#include <vector>

#include "common/datatype.hpp"
#include "node/inode.hpp"

namespace node {

class Template : public INode {
 public:
    explicit Template(core::Context* ctx, const char* name);
    ~Template();

 public:
    bool init(YAML::Node cfg) override;
    std::vector<BlobInfo> registerBlob() override;
    bool fetchBlob(const std::map<std::string, core::Blob*>& m) override;
    bool verification() override;
    bool exec(bool debug = false) override;
    const char* getName() override { return mName.c_str(); }

 private:
    std::string mName;
};

}  // namespace node

#endif  // NODE_TEMPLATE_TEMPLATE_HPP_
