/**
 * @file cast.cpp
 * @brief
 * @author oPluss (opluss@qq.com)
 *
 * @copyright Copyright (c) 2021  oPluss
 *
 * @par Modify log:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2021-12-29 <td>1.0     <td>lijiaqi     <td>Initial
 * </table>
 */

#include "cast.hpp"

#include "common/config.hpp"

namespace node {

Cast::Cast(core::Context* ctx, const char* name) {
    // Pass
    mName = name;
}

Cast::~Cast() {
    // Pass
}

bool Cast::init(YAML::Node cfg) {
    std::vector<size_t> shape = cfg["shape"].as<std::vector<size_t>>();
    if (shape.size() <= 0) {
        LOG(ERROR) << "Size of shape is 0";
        return false;
    }
    mSize = 1;
    for (const auto& i : shape) {
        mSize *= i;
    }
    mFrom = common::str2Type(cfg["from"].as<std::string>());
    if (mFrom == common::DataType::kUNKNOW) {
        LOG(ERROR) << "Type of \"from\" is unknow";
        return false;
    }
    mTo = common::str2Type(cfg["to"].as<std::string>());
    if (mTo == common::DataType::kUNKNOW) {
        LOG(ERROR) << "Type of \"to\" is unknow";
        return false;
    }
    mCastImplFunc = castimpl::getCastImpl(mFrom, mTo);
    if (mCastImplFunc == nullptr) {
        LOG(ERROR) << "Can't get impl func";
        return false;
    }
    return true;
}

std::vector<BlobInfo> Cast::registerBlob() {
    std::vector<BlobInfo> ret;
    // Input
    {
        BlobInfo info;
        info.name = input_name;
        info.type = BlobInfo::kINPUT;
        ret.push_back(info);
    }
    // Output
    {
        BlobInfo info;
        info.name = output_name;
        info.type = BlobInfo::kOUTPUT;
        info.args.target = core::Blob::Target::kALL;
        info.args.mode = BLOB_GLOBAL_MODE;
        info.args.size = mSize * common::getSize(mTo);
        ret.push_back(info);
    }
    return ret;
}

bool Cast::fetchBlob(const std::map<std::string, core::Blob*>& m) {
    if (m.find(input_name) == m.end()) {
        return false;
    }
    mInput = m.at(input_name);
    if (mInput == nullptr) {
        return false;
    }
    if (m.find(output_name) == m.end()) {
        return false;
    }
    mOutput = m.at(output_name);
    if (mOutput == nullptr) {
        return false;
    }
    return true;
}

bool Cast::verification() {
    size_t input_len = mInput->getSize() / common::getSize(mFrom);
    size_t output_len = mOutput->getSize() / common::getSize(mTo);
    if (input_len != output_len) {
        LOG(ERROR) << "Shape of io is not equal, i:" << input_len
                   << " and o:" << output_len;
        return false;
    } else {
        return true;
    }
}

bool Cast::exec(bool debug) {
    mCastImplFunc(mInput->getHostPtr<void>(), mOutput->getHostPtr<void>(),
                  mSize);
    return true;
}

REGISTER_NODE(Cast);

}  // namespace node
