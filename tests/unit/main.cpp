// Minimal unit test framework — each test_xxx returns 0 on pass, 1 on fail.
// Suitable for embedded/no-std libraries that compile without exceptions.

#include <cstdio>
#include <cstring>
#include <string>

#include <tinyml/version.hpp>

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

// --- tests ---
static int test_version_is_non_empty() {
    CHECK(tinyml_version() != nullptr);
    CHECK(std::string(tinyml_version()) == "0.1.0");
    return 0;
}

static int test_version_macro_matches_function() {
    CHECK(std::string(TINYML_VERSION) == tinyml_version());
    return 0;
}

int main() {
    std::fprintf(stderr, "Running tinyml_test...\n");
    test_version_is_non_empty();
    test_version_macro_matches_function();
    std::fprintf(stderr, "%d/%d tests passed (%d failed)\n",
        g_total - g_failed, g_total, g_failed);
    return g_failed == 0 ? 0 : 1;
}
