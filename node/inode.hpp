/**
 * @file inode.hpp
 * @brief Interface of node class
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-21 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef NODE_INODE_HPP_
#define NODE_INODE_HPP_

#include <yaml-cpp/yaml.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/blobmgr.hpp"
#include "core/factory.hpp"
#include "inode_helper.hpp"

namespace node {

struct BlobInfo {
    enum IOType {
        kINPUT,  // is input, will fetch from last node
        kOUTPUT  // is output, will alloc and share to next node
    };
    std::string name;
    IOType type;
    core::Blob::BlobArgs args;
};

class INode {
 public:
    /**
     * @brief Init the Node by config, invoked after instance
     * @param  cfg              The config of node
     * @return true
     * @return false
     */
    virtual bool init(YAML::Node* c) = 0;

    /**
     * @brief Register the blob to memory manager
     * @return std::vector<BlobInfo>
     */
    virtual std::vector<BlobInfo> registerBlob() = 0;

    /**
     * @brief Fetch the Blob pointer which the node need
     * @param  bmgr             Blob Manager
     * @return true
     * @return false
     */
    virtual bool fetchBlob(const std::map<std::string, core::Blob*>& m) = 0;

    /**
     * @brief Simulate exec, verify the env
     * @return true
     * @return false
     */
    virtual bool verification() = 0;

    /**
     * @brief Execute the Node
     * @param  debug            if enable debug mode
     * @return true
     * @return false
     */
    virtual bool exec(bool debug = false) = 0;

    /**
     * @brief Get the Name
     * @return const char*
     */
    virtual const char* getName() = 0;
};

using NodeFactory = core::Factory<core::FactoryType::kNODE, std::string,
                                  node::INode, core::Context*, const char*>;

#define REGISTER_NODE(cls)                                                   \
    static INode* cls##Creator(core::Context* ctx, const char* name) {       \
        return new cls(ctx, name);                                           \
    }                                                                        \
    static bool register##cls = []() {                                       \
        if (NodeFactory::getFactory().registerCreator(#cls, cls##Creator)) { \
            VLOG(0) << "Register " << #cls << " Success";                    \
            return true;                                                     \
        } else {                                                             \
            LOG(WARNING) << "Register " << #cls << " Failed";                \
            return false;                                                    \
        }                                                                    \
    }();

}  // namespace node

#endif  // NODE_INODE_HPP_
