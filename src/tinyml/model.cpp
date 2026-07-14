#include <tinyml/model.hpp>
#include <cstring>

namespace tinyml {

Error Model::ParseHeader(const uint8_t* data, uint32_t size) {
    if (size < sizeof(ModelHeader)) return Error::kBadSize;
    std::memcpy(&header_, data, sizeof(ModelHeader));
    if (std::memcmp(header_.magic, MODEL_MAGIC, 4) != 0) return Error::kBadMagic;
    if (header_.version_major != 0) return Error::kBadVersion;
    data_ = data;
    data_size_ = size;
    return Error::kNone;
}

} // namespace tinyml
