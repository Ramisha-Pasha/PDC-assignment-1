#include <stdio.h>
#include <algorithm>
#include <immintrin.h>  
#include "CycleTimer.h"
#include "saxpy_ispc.h"
extern "C" void avx2(int N, float scale, float* X, float* Y, float* result);
extern "C" void avx2(int N, float scale, float* X, float* Y, float* result) {
    __m256 scale_vec = _mm256_set1_ps(scale);

    int blockSize = 256;  // Process in blocks for better cache use
    for (int j = 0; j < N; j += blockSize) {
        for (int i = j; i < j + blockSize; i += 8) {  // Process 8 elements at a time
            _mm_prefetch((const char*)&X[i + 16], _MM_HINT_T0);  // Prefetch next cache line
            _mm_prefetch((const char*)&Y[i + 16], _MM_HINT_T0);

            __m256 x = _mm256_loadu_ps(&X[i]);  
            __m256 y = _mm256_loadu_ps(&Y[i]);  
            
            __m256 res = _mm256_fmadd_ps(scale_vec, x, y);  
            //__m256 x2 = _mm256_loadu_ps(&X[i+8]);  
           //__m256 y2 = _mm256_loadu_ps(&Y[i+8]);  
           //__m256 res2 = _mm256_fmadd_ps(scale_vec, x2, y2);
            _mm256_stream_ps(&result[i], res); // Non-temporal store to reduce cache pollution
            //_mm256_stream_ps(&result[i+8], res2);
            
        }
    }
}