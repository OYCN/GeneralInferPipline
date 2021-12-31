/**
 * @file tensorrt.hpp
 * @brief Inferrer network by config using TensorRT
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-24 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#ifndef NODE_BASIC_INFERRER_TENSORRT_HPP_
#define NODE_BASIC_INFERRER_TENSORRT_HPP_

#include <NvInfer.h>

#include <map>
#include <string>
#include <vector>

#include "node/inode.hpp"

namespace node {

class Logger : public nvinfer1::ILogger {
 public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kVERBOSE:
                VLOG(0) << "[TRT]" << msg;
                break;
            case Severity::kINFO:
                LOG(INFO) << "[TRT]" << msg;
                break;
            case Severity::kWARNING:
                LOG(WARNING) << "[TRT]" << msg;
                break;
            case Severity::kERROR:
                LOG(ERROR) << "[TRT]" << msg;
                break;
            case Severity::kINTERNAL_ERROR:
                LOG(FATAL) << "[TRT]" << msg;
                break;
            default:
                break;
        }
    }
};

/**
 * @brief Inferrer by TensorRT \n
 * Yaml Cfg:
 *    file: /path/to/engine_file
 */
class InferrerTRT : public INode {
 public:
    explicit InferrerTRT(core::Context* ctx, const char* name);
    ~InferrerTRT();

 public:
    bool init(YAML::Node* c) override;
    std::vector<BlobInfo> registerBlob() override;
    bool fetchBlob(const std::map<std::string, core::Blob*>& m) override;
    bool verification() override;
    bool exec(bool debug = false) override;
    const char* getName() override { return mName.c_str(); }

 private:
    std::string mName;

    core::Context* mCtx = nullptr;

    nvinfer1::IRuntime* mRuntime = nullptr;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::IExecutionContext* mExec = nullptr;
    Logger mLogger;

    std::vector<core::Blob*> mInputs;
    std::vector<void*> mEngineIO;
};

}  // namespace node

#endif  // NODE_BASIC_INFERRER_TENSORRT_HPP_
