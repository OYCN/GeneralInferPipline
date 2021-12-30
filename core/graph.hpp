/**
 * @file graph.hpp
 * @brief Define the Graph
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

#ifndef CORE_GRAPH_HPP_
#define CORE_GRAPH_HPP_

#include <map>
#include <string>
#include <vector>

namespace core {

/**
 * @brief Graph Class for Topo Sort
 */
class Graph {
 private:
    enum Status : uint8_t {
        kUNVISIT = 0,  // Not be visited
        kVISITING,     // has be visiting
        kVISITED       // has been pushed into topoOrder list
    };

 public:
    Graph();
    ~Graph();

 public:
    /**
     * @brief Add a node into graph
     * @param  name             Name of the node
     * @return true
     * @return false
     */
    bool addNode(std::string name);

    /**
     * @brief Add the edge between the double nodes
     * @param  from             The first node
     * @param  to               The second node
     * @return true
     * @return false
     */
    bool addEdge(std::string from, std::string to);

    /**
     * @brief Get the order with topo-sort
     * @return const std::vector<std::string>&
     */
    const std::vector<std::string>& topoSort();

 private:
    /**
     * @brief dfs for a node
     * @param  name             Start node
     * @return true
     * @return false
     */
    bool dfs(std::string name);

 private:
    std::map<std::string, std::vector<std::string>> mAdjTable;
    std::map<std::string, Status> mStatusMap;
    std::vector<std::string> mTopoOrder;
};

}  // namespace core

#endif  // CORE_GRAPH_HPP_
