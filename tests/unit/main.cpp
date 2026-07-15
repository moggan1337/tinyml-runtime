// Minimal unit test framework — each test_xxx returns 0 on pass, 1 on fail.

#include <cstdio>
#include <cstring>
#include <string>
#include <array>

#include <tinyml/version.hpp>
#include <tinyml/model.hpp>
#include <tinyml/model_types.hpp>
#include <tinyml/tensor.hpp>
#include <tinyml/memory_arena.hpp>
#include <tinyml/quantize.hpp>
#include <climits>

// --- test framework ---
static int g_total = 0;
static int g_failed = 0;

#define CHECK(cond) do { \
    ++g_total; \
    if (!(cond)) { \
        ++g_failed; \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

// --- version tests ---
static int test_version_is_non_empty() {
    CHECK(tinyml_version() != nullptr);
    CHECK(std::string(tinyml_version()) == "0.1.0");
    return 0;
}

static int test_version_macro_matches_function() {
    CHECK(std::string(TINYML_VERSION) == tinyml_version());
    return 0;
}

// --- model header tests ---
static int test_model_header_magic() {
    tinyml::ModelHeader hdr{};
    std::memcpy(hdr.magic, tinyml::MODEL_MAGIC, 4);
    CHECK(std::memcmp(hdr.magic, tinyml::MODEL_MAGIC, 4) == 0);
    return 0;
}

static int test_section_type_enum() {
    CHECK(static_cast<uint8_t>(tinyml::SectionType::kGraph)       == 1);
    CHECK(static_cast<uint8_t>(tinyml::SectionType::kWeights)     == 2);
    CHECK(static_cast<uint8_t>(tinyml::SectionType::kActivations) == 3);
    return 0;
}

static int test_struct_sizes() {
    CHECK(sizeof(tinyml::TensorDescriptor) == 99);
    CHECK(sizeof(tinyml::ModelHeader) == 28);
    CHECK(sizeof(tinyml::SectionHeader) == 12);
    CHECK(sizeof(tinyml::NodeDef) == 92);
    return 0;
}

static int test_parse_valid_header() {
    uint8_t buf[256] = {0};
    std::memcpy(buf, tinyml::MODEL_MAGIC, 4);
    buf[4] = 0;
    buf[5] = 1;
    buf[6] = 1; buf[7] = 0;
    buf[8] = 2; buf[9] = 0; buf[10] = 0; buf[11] = 0;
    tinyml::Model m;
    auto err = m.ParseHeader(buf, sizeof(buf));
    CHECK(err == tinyml::Error::kNone);
    CHECK(m.num_nodes() == 1);
    CHECK(m.num_tensors() == 2);
    return 0;
}

static int test_parse_wrong_magic() {
    uint8_t buf[32] = {0};
    std::memcpy(buf, "FOO", 3);
    tinyml::Model m;
    auto err = m.ParseHeader(buf, sizeof(buf));
    CHECK(err == tinyml::Error::kBadMagic);
    return 0;
}

// --- tensor tests ---
static int test_tensor_zero_init() {
    std::array<uint8_t, 128> buf{};
    tinyml::Tensor t(buf.data(), 0, tinyml::DType::kInt8, 1.0f, 0);
    CHECK(t.dtype() == tinyml::DType::kInt8);
    CHECK(t.scale() == 1.0f);
    CHECK(t.zero_point() == 0);
    return 0;
}

static int test_tensor_flat_index() {
    std::array<int8_t, 16> buf{};
    tinyml::Tensor t(buf.data(), 4, tinyml::DType::kInt8, 1.0f, 0);
    CHECK(t.flat_size() == 4);
    CHECK(t.flat_index(0, 0) == 0);
    CHECK(t.flat_index(1, 0) == 1);
    return 0;
}

// Forward declarations for tests defined after main()
static int test_requantize_positive();
static int test_requantize_saturates_int8_max();
static int test_requantize_saturates_int8_min();
static int test_sat_add_int32_no_overflow();
static int test_sat_add_int32_overflow();
static int test_sat_mul_int8();
static int test_quantize_float_to_int8();
static int test_dequantize_int8_array();
static int test_relu();
static int test_relu6();

// --- arena tests ---
static int test_arena_alloc_reset() {
    std::array<uint8_t, 256> buf{};
    tinyml::MemoryArena arena(buf.data(), 256);
    CHECK(arena.allocated_bytes() == 0);
    auto* p = arena.alloc(64);
    CHECK(p != nullptr);
    CHECK(arena.allocated_bytes() == 64);
    arena.reset();
    CHECK(arena.allocated_bytes() == 0);
    return 0;
}

static int test_arena_fits() {
    std::array<uint8_t, 32> buf{};
    tinyml::MemoryArena arena(buf.data(), 32);
    CHECK(arena.alloc(16) != nullptr);
    CHECK(arena.alloc(16) != nullptr);
    CHECK(arena.alloc(1) == nullptr);
    return 0;
}

static int test_arena_alignment() {
    std::array<uint8_t, 128> buf{};
    tinyml::MemoryArena arena(buf.data(), 128);
    auto* a = arena.alloc(1);
    CHECK(reinterpret_cast<uintptr_t>(a) % 4 == 0);
    return 0;
}

int main() {
    std::fprintf(stderr, "Running tinyml_test...\n");
    test_version_is_non_empty();
    test_version_macro_matches_function();
    test_model_header_magic();
    test_section_type_enum();
    test_struct_sizes();
    test_parse_valid_header();
    test_parse_wrong_magic();
    test_tensor_zero_init();
    test_tensor_flat_index();
    test_arena_alloc_reset();
    test_arena_fits();
    test_arena_alignment();
    test_requantize_positive();
    test_requantize_saturates_int8_max();
    test_requantize_saturates_int8_min();
    test_sat_add_int32_no_overflow();
    test_sat_add_int32_overflow();
    test_sat_mul_int8();
    test_quantize_float_to_int8();
    test_dequantize_int8_array();
    test_relu();
    test_relu6();
    std::fprintf(stderr, "%d/%d tests passed (%d failed)\n",
        g_total - g_failed, g_total, g_failed);
    return g_failed == 0 ? 0 : 1;
}

static int test_requantize_positive() {
    // 0.0 with scale=0.5, zp=-128: q = round(0/0.5) - 128 = -128
    int32_t out = tinyml::requantize(0, {0.5f, -128}, tinyml::DType::kInt8);
    CHECK(out == -128);
    return 0;
}

static int test_requantize_saturates_int8_max() {
    int32_t out = tinyml::requantize(10000, {0.5f, -128}, tinyml::DType::kInt8);
    CHECK(out == 127);
    return 0;
}

static int test_requantize_saturates_int8_min() {
    int32_t out = tinyml::requantize(-10000, {0.5f, -128}, tinyml::DType::kInt8);
    CHECK(out == -128);
    return 0;
}

static int test_sat_add_int32_no_overflow() {
    CHECK(tinyml::sat_add<int32_t>(1000, 2000) == 3000);
    return 0;
}

static int test_sat_add_int32_overflow() {
    CHECK(tinyml::sat_add<int32_t>(INT32_MAX, 1) == INT32_MAX);
    CHECK(tinyml::sat_add<int32_t>(INT32_MIN, -1) == INT32_MIN);
    return 0;
}

static int test_sat_mul_int8() {
    CHECK(tinyml::sat_mul<int8_t>(50, 2) == static_cast<int8_t>(100));
    CHECK(tinyml::sat_mul<int8_t>(100, 100) == 127); // saturate
    CHECK(tinyml::sat_mul<int8_t>(-100, 2) == static_cast<int8_t>(-128)); // saturate
    CHECK(tinyml::sat_mul<int8_t>(-100, -2) == 127); // 200 saturates to 127
    return 0;
}

static int test_quantize_float_to_int8() {
    float input[] = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};
    int8_t output[5] = {0};
    tinyml::quantize_float_array(input, output, 5, 0.01f, -128);
    CHECK(output[0] == -128);  // 0.0
    CHECK(output[1] == static_cast<int8_t>(-78));    // 0.5/0.01 - 128 = 50 - 128 = -78
    // -0.5f / 0.01f rounds to -50; + (-128) = -178 -- but int8 cannot represent -178!
    // Saturation to int8 should clamp to -128.
    CHECK(output[2] == static_cast<int8_t>(-128));   // saturated (correct post-saturation result)
    return 0;
}

static int test_dequantize_int8_array() {
    int8_t input[] = {-128, 0, 127};
    float output[3] = {0};
    tinyml::dequantize_array(input, output, 3, 0.01f, -128);
    CHECK(output[0] == 0.0f);
    CHECK(output[2] == 2.55f);
    return 0;
}

static int test_relu() {
    CHECK(tinyml::relu(-1.0f) == 0.0f);
    CHECK(tinyml::relu(0.0f)  == 0.0f);
    CHECK(tinyml::relu(2.0f)  == 2.0f);
    return 0;
}

static int test_relu6() {
    CHECK(tinyml::relu6(-1.0f) == 0.0f);
    CHECK(tinyml::relu6(3.0f) == 3.0f);
    CHECK(tinyml::relu6(10.0f) == 6.0f);
    return 0;
}

