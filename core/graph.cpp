/**
 * @file graph.cpp
 * @brief Impl the Graph
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

#include "graph.hpp"

#include <algorithm>

namespace core {
Graph::Graph() {
    // Pass
}

Graph::~Graph() {
    // Pass
}

bool Graph::addNode(std::string name) {
    if (mAdjTable.find(name) == mAdjTable.end()) {
        mAdjTable[name] = {};
        mTopoOrder.clear();
        return true;
    } else {
        return false;
    }
}

bool Graph::addEdge(std::string from, std::string to) {
    if (mAdjTable.find(from) != mAdjTable.end()) {
        mAdjTable[from].push_back(to);
        mTopoOrder.clear();
        return true;
    } else {
        return false;
    }
}

const std::vector<std::string>& Graph::topoSort() {
    if (mTopoOrder.size() == 0) {
        // Init the Status Map
        mStatusMap.clear();
        for (auto& item : mAdjTable) {
            mStatusMap[item.first] = Status::kUNVISIT;
        }

        for (auto& item : mStatusMap) {
            if (item.second != Status::kVISITED) {
                if (!dfs(item.first)) {
                    // Have Loop
                    mTopoOrder.clear();
                    return mTopoOrder;
                }
            }
        }
        std::reverse(mTopoOrder.begin(), mTopoOrder.end());
    }
    return mTopoOrder;
}

bool Graph::dfs(std::string name) {
    mStatusMap[name] = Status::kVISITING;
    for (auto& item : mAdjTable[name]) {
        if (mStatusMap[item] == Status::kUNVISIT) {
            if (!dfs(item)) {
                return false;
            }
        } else if (mStatusMap[item] == Status::kVISITING) {
            return false;
        }
    }
    mStatusMap[name] = Status::kVISITED;
    mTopoOrder.push_back(name);
    return true;
}

}  // namespace core
