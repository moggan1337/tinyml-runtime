#pragma once
#include <cstdint>

namespace tinyml {
namespace kernels {

// Naive Conv2D (NCHW layout)
// input:  [N][C_in][H][W]
// filter: [C_out][C_in][K_h][K_w]
// output: [N][C_out][H_out][W_out]
// acc: int32 output accumulator buffer (must be pre-allocated)
inline void conv2d_naive(
    const int8_t* input,
    int N, int H, int W, int C_in,
    const int8_t* filter,
    int K_h, int K_w, int C_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int32_t* acc
) {
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) return;

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < C_out; ++co) {
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    int32_t sum = 0;
                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            int ih = ho * stride_h + kh - pad_h;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < K_w; ++kw) {
                                int iw = wo * stride_w + kw - pad_w;
                                if (iw < 0 || iw >= W) continue;
                                int in_idx = ((n * C_in + ci) * H + ih) * W + iw;
                                int fi_idx = ((co * C_in + ci) * K_h + kh) * K_w + kw;
                                sum += static_cast<int32_t>(input[in_idx]) *
                                       static_cast<int32_t>(filter[fi_idx]);
                            }
                        }
                    }
                    int out_idx = ((n * C_out + co) * H_out + ho) * W_out + wo;
                    acc[out_idx] = sum;
                }
            }
        }
    }
}

// Naive DepthwiseConv2D (per-channel filter)
inline void depthwise_conv2d_naive(
    const int8_t* input,
    int N, int C, int H, int W,
    const int8_t* filter,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int32_t* acc
) {
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) return;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    int32_t sum = 0;
                    for (int kh = 0; kh < K_h; ++kh) {
                        int ih = ho * stride_h + kh - pad_h;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < K_w; ++kw) {
                            int iw = wo * stride_w + kw - pad_w;
                            if (iw < 0 || iw >= W) continue;
                            int in_idx  = ((n * C + c) * H + ih) * W + iw;
                            int fi_idx  = (c * K_h + kh) * K_w + kw;
                            sum += static_cast<int32_t>(input[in_idx]) *
                                   static_cast<int32_t>(filter[fi_idx]);
                        }
                    }
                    int out_idx = ((n * C + c) * H_out + ho) * W_out + wo;
                    acc[out_idx] = sum;
                }
            }
        }
    }
}

// Naive MaxPool
inline void maxpool2d_naive(
    const int8_t* input,
    int N, int C, int H, int W,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int8_t* out
) {
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    int8_t max_val = -128;
                    for (int kh = 0; kh < K_h; ++kh) {
                        int ih = ho * stride_h + kh - pad_h;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < K_w; ++kw) {
                            int iw = wo * stride_w + kw - pad_w;
                            if (iw < 0 || iw >= W) continue;
                            int idx = ((n * C + c) * H + ih) * W + iw;
                            max_val = (input[idx] > max_val) ? input[idx] : max_val;
                        }
                    }
                    int out_idx = ((n * C + c) * H_out + ho) * W_out + wo;
                    out[out_idx] = max_val;
                }
            }
        }
    }
}

// Naive AvgPool (output is int32 accumulator divided by count)
inline void avgpool2d_naive(
    const int8_t* input,
    int N, int C, int H, int W,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int32_t* acc
) {
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    int32_t sum = 0;
                    int count = 0;
                    for (int kh = 0; kh < K_h; ++kh) {
                        int ih = ho * stride_h + kh - pad_h;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < K_w; ++kw) {
                            int iw = wo * stride_w + kw - pad_w;
                            if (iw < 0 || iw >= W) continue;
                            int idx = ((n * C + c) * H + ih) * W + iw;
                            sum += static_cast<int32_t>(input[idx]);
                            ++count;
                        }
                    }
                    int out_idx = ((n * C + c) * H_out + ho) * W_out + wo;
                    acc[out_idx] = (count > 0) ? (sum / count) : 0;
                }
            }
        }
    }
}

} // namespace kernels
} // namespace tinyml
