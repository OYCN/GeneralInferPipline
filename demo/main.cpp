/**
 * @file main.cpp
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

#include <glog/logging.h>

#include "core/core.hpp"

int main(int argc, char* argv[]) {
    FLAGS_logtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_v = 0;
    google::InitGoogleLogging(argv[0]);
    CHECK_EQ(argc, 2) << argv[0] << " cfg.yaml";
    core::Core c;
    CHECK(c.readCfg(argv[1]));
    CHECK(c.genPipline());
    CHECK(c.initPipline());
    while (1) {
        CHECK(c.exec());
    }

    return 0;
}
