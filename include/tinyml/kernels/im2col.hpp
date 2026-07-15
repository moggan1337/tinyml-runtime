#pragma once
#include <cstdint>

namespace tinyml {
namespace kernels {

// im2col: convert NHWC input into column matrix for GEMM
// input:  [N][H][W][C_in]  (NHWC layout)
// col:    [K_h*K_w*C_in][H_out*W_out]  (column-major: one column per output pixel)
//
// For a 1x1 kernel, im2col is essentially a reshape: col[n*h*w*c + (kh*kw*c + ci)*h*w + (ho*wo+wo)] = input[n*h*w*c + (ho*sh+kh-pad)*w*c + (wo*sw+kw-pad)*c + ci]
//
// For larger kernels, do explicit nested loops.
inline void im2col(
    const int8_t* input,
    int N, int H, int W, int C_in,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int8_t* col
) {
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) return;

    int KHKWC = K_h * K_w * C_in;
    int HW = H * W;

    for (int n = 0; n < N; ++n) {
        const int8_t* in_n = input + n * HW * C_in;
        for (int ho = 0; ho < H_out; ++ho) {
            for (int wo = 0; wo < W_out; ++wo) {
                int8_t* col_col = col + (ho * W_out + wo) * KHKWC;
                for (int kh = 0; kh < K_h; ++kh) {
                    int ih = ho * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < K_w; ++kw) {
                        int iw = wo * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= W) continue;
                        const int8_t* in_pix = in_n + (ih * W + iw) * C_in;
                        int8_t* col_kw = col_col + (kh * K_w + kw) * C_in;
                        for (int ci = 0; ci < C_in; ++ci) {
                            col_kw[ci] = in_pix[ci];
                        }
                    }
                }
            }
        }
    }
}

} // namespace kernels
} // namespace tinyml
