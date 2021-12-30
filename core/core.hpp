/**
 * @file core.hpp
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

#ifndef CORE_CORE_HPP_
#define CORE_CORE_HPP_

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <vector>

#include "node/inode.hpp"

namespace core {

class Context;

class Core {
 public:
    explicit Core(Context* ctx = nullptr);
    ~Core();

 public:
    bool readCfg(std::string file);
    bool genPipline();
    bool initPipline();
    bool exec(bool debug = false);

 private:
    std::unique_ptr<Context> mSelfCtx;
    Context* mCtx = nullptr;
    YAML::Node mCfg;
    std::vector<std::string> mTopoOrder;
    std::vector<std::unique_ptr<node::INode>> mPipline;
};

}  // namespace core

#endif  // CORE_CORE_HPP_
