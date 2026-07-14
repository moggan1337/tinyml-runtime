#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace tinyml {

enum class DType : uint8_t { kInt8 = 0, kInt16 = 1, kFloat32 = 2 };

struct alignas(4) QParams {
    float scale;
    int32_t zero_point;
};

class Tensor {
public:
    Tensor() = default;

    Tensor(void* buf, uint32_t flat_elements, DType dtype, float scale, int32_t zero_point)
        : buf_(static_cast<uint8_t*>(buf))
        , flat_elements_(flat_elements)
        , dtype_(dtype)
        , q_{scale, zero_point} {}

    DType dtype() const { return dtype_; }
    float scale() const { return q_.scale; }
    int32_t zero_point() const { return q_.zero_point; }
    const QParams* q() const { return &q_; }
    uint32_t flat_size() const { return flat_elements_; }
    uint8_t* data() { return buf_; }
    const uint8_t* data() const { return buf_; }
    int8_t* data_i8() { return reinterpret_cast<int8_t*>(buf_); }
    int16_t* data_i16() { return reinterpret_cast<int16_t*>(buf_); }
    float* data_f32() { return reinterpret_cast<float*>(buf_); }

    uint32_t flat_index(uint32_t d0, uint32_t d1 = 0, uint32_t d2 = 0, uint32_t d3 = 0) const {
        return ((d0 * dims_[1] + d1) * dims_[2] + d2) * dims_[3] + d3;
    }

    void set_dims(uint8_t n, const uint16_t d[4]) {
        num_dims_ = n;
        for (int i = 0; i < 4; ++i) dims_[i] = (i < n) ? d[i] : 1;
    }

    uint16_t dim(uint8_t i) const { return dims_[i]; }
    uint8_t  num_dims() const { return num_dims_; }

    void fill_zero() { std::memset(buf_, 0, flat_elements_ * element_size()); }

    int32_t read_q(uint32_t idx) const;
    void    write_q(uint32_t idx, int32_t val);

    float   dequantize(int32_t q) const { return static_cast<float>(q - q_.zero_point) * q_.scale; }
    int32_t quantize(float v) const;

private:
    uint8_t* buf_ = nullptr;
    uint32_t flat_elements_ = 0;
    DType dtype_ = DType::kInt8;
    QParams q_{1.0f, 0};
    uint8_t num_dims_ = 1;
    uint16_t dims_[4] = {1, 1, 1, 1};

    uint32_t element_size() const {
        switch (dtype_) {
            case DType::kInt8:  return 1;
            case DType::kInt16: return 2;
            case DType::kFloat32: return 4;
        }
        return 1;
    }
};

inline int32_t Tensor::read_q(uint32_t idx) const {
    if (dtype_ == DType::kInt8) return static_cast<int8_t*>(static_cast<void*>(buf_))[idx];
    if (dtype_ == DType::kInt16) return static_cast<int16_t*>(static_cast<void*>(buf_))[idx];
    return static_cast<int32_t*>(static_cast<void*>(buf_))[idx];
}

inline void Tensor::write_q(uint32_t idx, int32_t val) {
    if (dtype_ == DType::kInt8) {
        if (val > 127)  val = 127;
        if (val < -128) val = -128;
        static_cast<int8_t*>(static_cast<void*>(buf_))[idx] = static_cast<int8_t>(val);
    } else if (dtype_ == DType::kInt16) {
        static_cast<int16_t*>(static_cast<void*>(buf_))[idx] = static_cast<int16_t>(val);
    } else {
        static_cast<float*>(static_cast<void*>(buf_))[idx] = static_cast<float>(val);
    }
}

inline int32_t Tensor::quantize(float v) const {
    if (q_.scale == 0.0f) return q_.zero_point;
    float scale_f = static_cast<float>(q_.scale);
    float rounded = v / scale_f + 0.5f;
    return static_cast<int32_t>(rounded) + q_.zero_point;
}

[[maybe_unused]] static constexpr int _tensor_unused_anchor = 0;

} // namespace tinyml
