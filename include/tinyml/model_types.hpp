#pragma once
#include <cstdint>
#include <cstring>

namespace tinyml {

// Model binary magic: "TML\0"
static constexpr char MODEL_MAGIC[4] = {'T', 'M', 'L', 0};

enum class SectionType : uint8_t {
    kPadding   = 0,
    kGraph     = 1,
    kWeights   = 2,
    kActivations = 3,
};

// All structs are packed — no padding
#pragma pack(push, 1)
struct ModelHeader {
    char     magic[4];
    uint8_t  version_major;
    uint8_t  version_minor;
    uint16_t num_nodes;
    uint32_t num_tensors;
    uint32_t weights_size;       // bytes
    uint32_t activations_size;   // bytes
    uint32_t arena_size;         // required scratch (bytes)
    uint32_t checksum;          // CRC32 of entire file
};

struct SectionHeader {
    uint8_t  type;       // SectionType
    uint8_t  flags;
    uint16_t reserved;
    uint32_t offset;     // from start of file
    uint32_t size;      // bytes
};

struct TensorDescriptor {
    char    name[16];    // null-terminated
    uint8_t dtype;       // 0=int8, 1=int16, 2=float32
    uint8_t per_channel; // 0=per-tensor, 1=per-channel
    uint8_t num_dims;
    uint16_t dims[4];   // max 4D
    float    scale;      // for quantized: scale; for float: 1.0f
    int32_t  zero_point; // for quantized
    float    channel_scales[8];
    int32_t  channel_zero_points[8];
};

enum class OpType : uint8_t {
    kConv2D=1, kDepthwiseConv2D, kMaxPool, kAvgPool,
    kAdd, kMul, kReLU, kReLU6,
    kFullyConnected, kReshape, kSoftmax, kConcat,
    kIdentity,
};

struct NodeDef {
    uint8_t  op_type;       // OpType
    uint8_t  padding[3];
    uint32_t inputs[4];    // tensor indices
    uint32_t outputs[2];
    uint8_t  padding_params[64];
};
#pragma pack(pop)

static_assert(sizeof(ModelHeader)      == 28, "ModelHeader must be 28 bytes");
static_assert(sizeof(SectionHeader)    == 12, "SectionHeader must be 12 bytes");
static_assert(sizeof(TensorDescriptor) == 99, "TensorDescriptor must be 99 bytes");
static_assert(sizeof(NodeDef)          == 92, "NodeDef must be 92 bytes");

} // namespace tinyml
