#pragma once
#include <cstdint>

namespace tinyml {
namespace kernels {

// C[M][N] = A[M][K] * B[K][N]  (overwrite — no accumulate yet)
// A: row-major, B: column-major (B[k,n] is at offset n*K + k), C: row-major
inline void gemm_scalar(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N
) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t sum = 0;
            const int8_t* A_row = A + m * K;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A_row[k]) *
                       static_cast<int32_t>(B[n * K + k]);
            }
            C[m * N + n] = sum;
        }
    }
}

// C[M][N] += A[M][K] * B[K][N] (accumulate)
inline void gemm_scalar_acc(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N
) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t sum = C[m * N + n];
            const int8_t* A_row = A + m * K;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A_row[k]) *
                       static_cast<int32_t>(B[n * K + k]);
            }
            C[m * N + n] = sum;
        }
    }
}

} // namespace kernels
} // namespace tinyml
