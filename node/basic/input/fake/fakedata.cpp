/**
 * @file fakedata.cpp
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

#include "fakedata.hpp"

#include "common/config.hpp"

namespace node {

FakeData::FakeData(core::Context* ctx, const char* name) {
    // Pass
    mName = name;
}

FakeData::~FakeData() {
    // Pass
}

bool FakeData::init(YAML::Node cfg) {
    std::string type = cfg["data_type"].as<std::string>();
    mDataType = common::str2Type(type);
    CHECK(mDataType != common::DataType::kUNKNOW)
        << "Unsupported type: " << type;
    CHECK(cfg["shape"].IsSequence()) << cfg["shape"];
    mSize = common::getSize(mDataType);
    for (const auto& item : cfg["shape"]) {
        CHECK(item.IsScalar()) << "It's not a Scalar: " << item;
        size_t t = item.as<size_t>();
        mSahpe.push_back(t);
        mSize *= t;
    }
    CHECK(mSize > 0) << "Size is illegal: " << mSize;
    CHECK(mSahpe.size() > 0) << "Size of shape is illegal: " << mSahpe.size();
    return true;
}

std::vector<BlobInfo> FakeData::registerBlob() {
    BlobInfo info;
    info.name = output_name;
    info.type = BlobInfo::kOUTPUT;
    info.args.mode = BLOB_GLOBAL_MODE;
    info.args.target = core::Blob::Target::kALL;
    info.args.size = mSize;
    return {info};
}

bool FakeData::fetchBlob(const std::map<std::string, core::Blob*>& m) {
    if (m.find(output_name) != m.end()) {
        mBlob = m.at(output_name);
        return true;
    } else {
        return false;
    }
}

bool FakeData::verification() {
    // Pass
    return true;
}

bool FakeData::exec(bool debug) {
    // Pass
    return true;
}

REGISTER_NODE(FakeData);

}  // namespace node
