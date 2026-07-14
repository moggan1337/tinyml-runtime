#pragma once
#include <cstdint>
#include <cstddef>

namespace tinyml {

class MemoryArena {
public:
    MemoryArena() = default;
    MemoryArena(void* base, uint32_t size) : base_(static_cast<uint8_t*>(base)), size_(size) {}

    // Allocate n bytes, 4-byte aligned. Returns nullptr if out of memory.
    void* alloc(uint32_t n) {
        uint32_t aligned = (n + 3) & ~3u;
        if (offset_ + aligned > size_) return nullptr;
        void* p = base_ + offset_;
        offset_ += aligned;
        return p;
    }

    void reset() { offset_ = 0; }
    uint32_t allocated_bytes() const { return offset_; }
    uint32_t capacity() const { return size_; }
    bool fits(uint32_t n) const {
        uint32_t aligned = (n + 3) & ~3u;
        return (offset_ + aligned) <= size_;
    }
    void* base() const { return base_; }

private:
    uint8_t* base_ = nullptr;
    uint32_t size_ = 0;
    uint32_t offset_ = 0;
};

} // namespace tinyml
