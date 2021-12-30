/**
 * @file tensorrt.cpp
 * @brief
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

#include "tensorrt.hpp"

#include <functional>
#include <numeric>

#include "common/config.hpp"
#include "common/datatype.hpp"
#include "common/file.hpp"
#include "core/context.hpp"

namespace node {

InferrerTRT::InferrerTRT(core::Context* ctx, const char* name) : mCtx(ctx) {
    mName = name;
    CHECK(mCtx != nullptr);
    mRuntime = nvinfer1::createInferRuntime(mLogger);
}

InferrerTRT::~InferrerTRT() {
    if (mRuntime != nullptr) {
        mRuntime->destroy();
        mRuntime = nullptr;
    }
    if (mEngine != nullptr) {
        mEngine->destroy();
        mEngine = nullptr;
    }
}

bool InferrerTRT::init(YAML::Node cfg) {
    if (cfg.IsNull()) {
        return false;
    }
    std::string file = cfg["file"].as<std::string>();
    std::vector<char> buff = common::read(file);
    mEngine = mRuntime->deserializeCudaEngine(buff.data(), buff.size());
    if (mEngine == nullptr) {
        return false;
    }
    if (mEngine->hasImplicitBatchDimension()) {
        LOG(ERROR) << "Only support explicit BatchSize";
        return false;
    }

    // TODO(oPluss): about dynamic shapes

    mExec = mEngine->createExecutionContext();
    return true;
}

std::vector<BlobInfo> InferrerTRT::registerBlob() {
    std::vector<BlobInfo> rets;
    for (size_t i = 0; i < mEngine->getNbBindings(); i++) {
        BlobInfo info;
        if (mEngine->bindingIsInput(i)) {
            info.type = BlobInfo::kINPUT;
        } else {
            info.type = BlobInfo::kOUTPUT;
        }
        info.name = mEngine->getBindingName(i);

        // get size
        size_t vol = 1;
        nvinfer1::Dims dims = mExec->getBindingDimensions(i);
        nvinfer1::DataType type = mEngine->getBindingDataType(i);
        int vecDim = mEngine->getBindingVectorizedDim(i);
        if (-1 != vecDim) {
            int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
            dims.d[vecDim] =
                (dims.d[vecDim] + scalarsPerVec - 1) / scalarsPerVec;
            vol *= scalarsPerVec;
        }
        vol *= std::accumulate(dims.d, dims.d + dims.nbDims, 1,
                               std::multiplies<int64_t>());
        vol *= common::getSize(common::fromNvDataType(type));

        info.args.size = vol;
        info.args.target = core::Blob::Target::kDEVICE;
        info.args.mode = BLOB_GLOBAL_MODE;

        rets.push_back(info);
    }
    return rets;
}

bool InferrerTRT::fetchBlob(const std::map<std::string, core::Blob*>& m) {
    for (size_t i = 0; i < mEngine->getNbBindings(); i++) {
        std::string name = mEngine->getBindingName(i);
        if (m.find(name) == m.end()) {
            LOG(ERROR) << "The correct Blob was not found in InferrerTRT: "
                       << name;
            return false;
        }
        core::Blob* b = m.at(name);
        NODE_ASSERT(b->getDevicePtr<void>() != nullptr, "Need device mem");
        mEngineIO.push_back(b->getDevicePtr<void>());
        if (mEngine->bindingIsInput(i)) {
            mInputs.push_back(b);
        }
    }
    return true;
}

bool InferrerTRT::verification() {
    if (mEngineIO.size() != mEngine->getNbBindings()) {
        return false;
    }
    return true;
}

bool InferrerTRT::exec(bool debug) {
    for (auto& i : mInputs) {
        i->toDevice(i->getSize());
    }
    if (debug) {
        mExec->executeV2(mEngineIO.data());
    } else {
        mExec->enqueueV2(mEngineIO.data(), mCtx->getCudaStream(), nullptr);
    }
    return true;
}

REGISTER_NODE(InferrerTRT);

}  // namespace node
