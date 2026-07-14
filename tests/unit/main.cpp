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
    std::fprintf(stderr, "%d/%d tests passed (%d failed)\n",
        g_total - g_failed, g_total, g_failed);
    return g_failed == 0 ? 0 : 1;
}
