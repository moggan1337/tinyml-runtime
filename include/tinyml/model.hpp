#pragma once
#include <tinyml/model_types.hpp>
#include <cstdint>
#include <cstdio>

namespace tinyml {

enum class Error : int {
    kNone = 0,
    kBadMagic,
    kBadVersion,
    kBadSize,
    kBadSectionOffset,
    kBadCRC,
};

class Model {
public:
    Error ParseHeader(const uint8_t* data, uint32_t size);

    uint16_t num_nodes()    const { return header_.num_nodes; }
    uint32_t num_tensors()  const { return header_.num_tensors; }
    uint32_t arena_size()   const { return header_.arena_size; }

    const ModelHeader& header() const { return header_; }
    const SectionHeader* graph_sec()   const { return graph_sec_; }
    const SectionHeader* weights_sec() const { return weights_sec_; }
    const SectionHeader* act_sec()     const { return act_sec_; }

private:
    ModelHeader    header_{};
    const uint8_t* data_ = nullptr;
    uint32_t       data_size_ = 0;
    const SectionHeader* graph_sec_   = nullptr;
    const SectionHeader* weights_sec_ = nullptr;
    const SectionHeader* act_sec_     = nullptr;
};

} // namespace tinyml
