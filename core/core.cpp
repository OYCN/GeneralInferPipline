/**
 * @file core.cpp
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

#include "core.hpp"

#include <glog/logging.h>

#include <map>
#include <utility>

#include "common/config.hpp"
#include "common/string.hpp"
#include "context.hpp"
#include "graph.hpp"

namespace core {

Core::Core(Context* ctx) : mCtx(ctx) {
    if (mCtx == nullptr) {
        mSelfCtx.reset(new Context);
        mCtx = mSelfCtx.get();
    }
}

Core::~Core() {
    // Pass
}

bool Core::readCfg(std::string file) {
    mCfg = YAML::LoadFile(file);
    Graph g;
    // Add node
    for (const auto& item : mCfg) {
        std::string name = item.first.as<std::string>();
        g.addNode(name);
        VLOG(0) << "Add Node into Graph: " << name;
        CHECK(item.second.IsMap()) << name << " is not a Map";
        CHECK(item.second["node_type"] && item.second["node_type"].IsScalar())
            << "[" << name << "]";
        std::string node_type = item.second["node_type"].as<std::string>();
        CHECK(node::NodeFactory::getFactory().isHas(node_type))
            << "[" << name << "]"
            << " Unsupported Node Type: " << node_type;
    }
    // Add edge
    for (const auto& item : mCfg) {
        std::string name = item.first.as<std::string>();
        CHECK(item.second.IsMap()) << name << " is not a Map";
        for (const auto& subitem : item.second) {
            if (subitem.second.IsScalar()) {
                std::string attr_value = subitem.second.as<std::string>();
                if (attr_value[0] == '@' || attr_value[0] == '$') {
                    // it's a data dependency
                    attr_value = attr_value.substr(1);
                    auto res = common::split(attr_value, ".");
                    g.addEdge(res[0], name);
                }
            }
        }
    }
    LOG(INFO) << "Edge add done";
    // topo sort
    mTopoOrder = g.topoSort();
    if (mTopoOrder.size() == 0) {
        LOG(ERROR) << "Can't Impl topo-sort";
        return false;
    }
    LOG(INFO) << "Topo-Sort done";
    return true;
}

bool Core::genPipline() {
    if (mTopoOrder.size() == 0) {
        LOG(ERROR) << "Don't have mTopoOrder";
        return false;
    }
    for (const auto& name : mTopoOrder) {
        // TODO(oPluss): Replace value which start with '$'
        std::vector<std::pair<std::string, YAML::Node>> replace_list;
        // I don't know if it's safe when I modify the value on iteration
        for (const auto& item : mCfg[name]) {
            if (item.second.IsScalar()) {
                std::string value = item.second.as<std::string>();
                if (value[0] == '$') {
                    value = value.substr(1);
                    auto res = common::split(value, ".");
                    CHECK_EQ(res.size(), 2) << "[" << name << "]";
                    CHECK(mCfg[res[0]][res[1]])
                        << "[" << name << "]" << value << " Not Found";
                    replace_list.emplace_back(item.first.as<std::string>(),
                                              mCfg[res[0]][res[1]]);
                }
            }
        }
        LOG(INFO) << "[" << name << "]"
                  << "Replace list of '$' collect done";
        // Apply the replace
        for (auto& item : replace_list) {
            VLOG(0) << "[" << name << "]"
                    << "Replace " << name << "." << item.first << " to "
                    << item.second;
            mCfg[name][item.first] = item.second;
        }
        LOG(INFO) << "[" << name << "]"
                  << "Replace of '$' done";

        // Create and init Node
        std::string node_type = mCfg[name]["node_type"].as<std::string>();
        node::INode* node = node::NodeFactory::getFactory().getInstance(
            node_type, mCtx, name.c_str());
        mPipline.emplace_back(node);
        CHECK(node != nullptr) << "[" << name << "]"
                               << "Unsupported Node Type: " << node_type;
        LOG(INFO) << "[" << name << "]"
                  << "Creat node done";
        CHECK(node->init(mCfg[name])) << "[" << name << "]";
        LOG(INFO) << "[" << name << "]"
                  << "Init node done";

        // Register Blob
        std::map<std::string, core::Blob*> distributes;
        std::vector<std::string> input_names;
        auto info = node->registerBlob();
        for (auto& item : info) {
            if (item.type == node::BlobInfo::kOUTPUT) {
                auto key = name + "." + item.name;
                CHECK(mCtx->getBlobManager().add(key, mCtx, item.args))
                    << "[" << name << "]" << key;
                core::Blob* b = mCtx->getBlobManager().get(key);
                CHECK(b != nullptr) << "[" << name << "]";
                if (distributes.find(item.name) != distributes.end()) {
                    LOG(WARNING)
                        << "Will distribute a repeated blob: " << item.name;
                }
                distributes.emplace(item.name, b);
                VLOG(0) << "[" << name << "]"
                        << "Create Blob(collected): " << item.name << "("
                        << b->getSize() << ")";
            } else {
                input_names.emplace_back(item.name);
            }
        }
        LOG(INFO) << "[" << name << "]"
                  << "Register blob done";

        // Distribute Blob
        for (auto& item : input_names) {
            CHECK(mCfg[name][item]) << "[" << name << "]"
                                    << "Not found " << name << "." << item;
            std::string val = mCfg[name][item].as<std::string>();
            CHECK(val[0] == '@') << "[" << name << "]"
                                 << "As input with " << name << "." << item
                                 << ", The val must start with '@'";
            val = val.substr(1);
            core::Blob* b = mCtx->getBlobManager().get(val);
            CHECK(b != nullptr)
                << "[" << name << "] Not found i/o named " << val;
            if (distributes.find(item) != distributes.end()) {
                LOG(WARNING) << "Will distribute a repeated blob: " << item;
            }
            distributes.emplace(item, b);
            VLOG(0) << "[" << name << "]"
                    << "Collected Blob: " << val << "(" << b->getSize() << ")";
        }
        CHECK(node->fetchBlob(distributes)) << "[" << name << "]";
        LOG(INFO) << "[" << name << "]"
                  << "Distribute node done";
    }
    return true;
}

bool Core::initPipline() {
    if (mPipline.size() == 0) {
        LOG(ERROR) << "Pipline is Empty";
        return false;
    }
    for (auto& item : mPipline) {
        if (!item->verification()) {
            LOG(ERROR) << "Get error when verification the node: "
                       << item->getName();
            return false;
        }
        LOG(INFO) << item->getName() << " init done";
    }
    return true;
}

bool Core::exec(bool debug) {
    for (auto& item : mPipline) {
        if (!item->exec(debug)) {
            LOG(ERROR) << "Get error when exec the node: " << item->getName();
            return false;
        }
    }
    return true;
}

}  // namespace core
