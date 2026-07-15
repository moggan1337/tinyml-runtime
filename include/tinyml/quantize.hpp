#pragma once
#include <tinyml/tensor.hpp>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

namespace tinyml {

// Requantize: int32 accumulator → quantized int8/int16 output.
// (multiplier, shift) come from per-channel quant params (computed offline).
inline int32_t requantize(int32_t acc,
                           const QParams& dst,
                           DType dtype,
                           int32_t multiplier = 1,
                           int right_shift = 0) {
    int64_t tmp = static_cast<int64_t>(acc) * multiplier;
    if (right_shift > 0) {
        tmp += (1LL << (right_shift - 1));
        tmp >>= right_shift;
    }
    int32_t q = static_cast<int32_t>(tmp) + dst.zero_point;
    switch (dtype) {
        case DType::kInt8:
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            break;
        case DType::kInt16:
            if (q >  32767) q =  32767;
            if (q < -32768) q = -32768;
            break;
        case DType::kFloat32:
            break;
    }
    return q;
}

// Saturating arithmetic for safe int8 / int16 dot-products
template<typename T>
inline T sat_add(T a, T b) {
    if (a > 0 && b > T(INT_MAX) - a) return T(INT_MAX);
    if (a < 0 && b < T(INT_MIN) - a) return T(INT_MIN);
    return a + b;
}

template<>
inline int32_t sat_add<int32_t>(int32_t a, int32_t b) {
    int64_t sum = static_cast<int64_t>(a) + static_cast<int64_t>(b);
    if (sum > INT32_MAX) return INT32_MAX;
    if (sum < INT32_MIN) return INT32_MIN;
    return static_cast<int32_t>(sum);
}

template<>
inline int8_t sat_add<int8_t>(int8_t a, int8_t b) {
    int16_t sum = static_cast<int16_t>(a) + static_cast<int16_t>(b);
    if (sum >  127) return  127;
    if (sum < -128) return -128;
    return static_cast<int8_t>(sum);
}

template<typename T>
inline T sat_mul(T a, T b) {
    int64_t prod = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    constexpr int64_t MAX = (1LL << (sizeof(T)*8 - 1)) - 1;
    if (prod >  MAX) return static_cast<T>(MAX);
    if (prod < -MAX-1) return static_cast<T>(-MAX-1);
    return static_cast<T>(prod);
}

template<>
inline int8_t sat_mul<int8_t>(int8_t a, int8_t b) {
    int16_t prod = static_cast<int16_t>(a) * static_cast<int16_t>(b);
    if (prod >  127) return  127;
    if (prod < -128) return -128;
    return static_cast<int8_t>(prod);
}

template<>
inline int32_t sat_mul<int32_t>(int32_t a, int32_t b) {
    int64_t prod = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    if (prod > INT32_MAX) return INT32_MAX;
    if (prod < INT32_MIN) return INT32_MIN;
    return static_cast<int32_t>(prod);
}

// Per-channel requantize: per-channel multiplier/shift for symmetric quantization.
inline void requantize_per_channel(
    const int32_t* acc, int32_t* out, int num_elements,
    const float* channel_scales, int32_t zero_point,
    const int32_t* multipliers, const int* shifts, int num_channels) {
    for (int i = 0; i < num_elements; ++i) {
        int ch = i % num_channels;
        QParams qp{channel_scales[ch], zero_point};
        out[i] = requantize(acc[i], qp, DType::kInt8, multipliers[ch], shifts[ch]);
    }
}

// Float → int8 quantization helpers (reference test).
inline void quantize_float_array(const float* in, int8_t* out, int n,
                                float scale, int32_t zero_point) {
    if (scale == 0.0f) { for (int i = 0; i < n; ++i) out[i] = static_cast<int8_t>(zero_point); return; }
    float inv = 1.0f / scale;
    for (int i = 0; i < n; ++i) {
        int32_t q = static_cast<int32_t>(in[i] * inv + 0.5f) + zero_point;
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        out[i] = static_cast<int8_t>(q);
    }
}

inline void dequantize_array(const int8_t* in, float* out, int n,
                              float scale, int32_t zero_point) {
    for (int i = 0; i < n; ++i) {
        out[i] = (static_cast<float>(in[i]) - static_cast<float>(zero_point)) * scale;
    }
}

// Activation functions
inline float relu(float x)        { return x > 0.0f ? x : 0.0f; }
inline float relu6(float x)       { return x < 0.0f ? 0.0f : (x > 6.0f ? 6.0f : x); }

} // namespace tinyml
