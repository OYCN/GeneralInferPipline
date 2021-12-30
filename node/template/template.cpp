/**
 * @file template.cpp
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

#include "template.hpp"

#include "common/config.hpp"

namespace node {

Template::Template(core::Context* ctx, const char* name) {
    // Pass
    mName = name;
}

Template::~Template() {
    // Pass
}

bool Template::init(YAML::Node cfg) {
    // Pass
    return false;
}

std::vector<BlobInfo> Template::registerBlob() {
    // Pass
    return {};
}

bool Template::fetchBlob(const std::map<std::string, core::Blob*>& m) {
    // Pass
    return false;
}

bool Template::verification() {
    // Pass
    return false;
}

bool Template::exec(bool debug) {
    // Pass
    return false;
}

// REGISTER_NODE(Template);

}  // namespace node
