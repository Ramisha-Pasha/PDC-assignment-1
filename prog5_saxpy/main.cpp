#include <stdio.h>
#include <algorithm>
#include <immintrin.h>  
#include <cstdlib>
#include <xmmintrin.h>  // For _mm_malloc()
#include "CycleTimer.h"
#include "saxpy_ispc.h"
#include <stdint.h>
extern void saxpySerial(int N, float a, float* X, float* Y, float* result);

 void avx2(int N, float scale, float* X, float* Y, float* result);






//------------------------------------avx 
 void avx2(int N, float scale, float* X, float* Y, float* result) {
    __m256 scale_vec = _mm256_set1_ps(scale);

    int blockSize = 128;  // Process in blocks for better cache use
    for (int j = 0; j < N; j += blockSize) {
        for (int i = j; i < j + blockSize; i += 8) {  // Process 8 elements at a time
            _mm_prefetch((const char*)&X[i + 16], _MM_HINT_T0);  // Prefetch next cache line
            _mm_prefetch((const char*)&Y[i + 16], _MM_HINT_T0);

            __m256 x = _mm256_load_ps(&X[i]);  
            __m256 y = _mm256_load_ps(&Y[i]);  
            
            __m256 res = _mm256_fmadd_ps(scale_vec, x, y);  
            
            _mm256_stream_ps(&result[i], res); // Non-temporal store to reduce cache pollution
           
            
        }
    }
    for (int i = (N / 8) * 8; i < N; i++) {
        result[i] = scale * X[i] + Y[i];
    }
}

//-------------------------------------------------

// return GB/s
static float
toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

static float
toGFLOPS(int ops, float sec) {
    return static_cast<float>(ops) / 1e9 / sec;
}

static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (result[i] != gold[i]) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

using namespace ispc;


int main() {

    const unsigned int N = 20 * 1000 * 1000; // 20 M element vectors (~80 MB)
    const unsigned int TOTAL_BYTES = 4 * N * sizeof(float);
    const unsigned int TOTAL_FLOPS = 2 * N;

    float scale = 2.f;
    float* arrayX = (float*)aligned_alloc(32, N * sizeof(float));
    float* arrayY = (float*)aligned_alloc(32, N * sizeof(float));
    float* resultSerial = (float*)aligned_alloc(32, N * sizeof(float));
    float* resultTasks = (float*)aligned_alloc(32, N * sizeof(float));
    float* resultAVX2 = (float*)aligned_alloc(32, N * sizeof(float));
    float* resultISPC = (float*)aligned_alloc(32, N * sizeof(float));



    // initialize array values
    for (unsigned int i=0; i<N; i++)
    {
        arrayX[i] = i;
        arrayY[i] = i;
        resultSerial[i] = 0.f;
        resultISPC[i] = 0.f;
        resultTasks[i] = 0.f;
        resultAVX2[i]=0.f;
       
    }

    //
    // Run the serial implementation. Repeat three times for robust
    // timing.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime =CycleTimer::currentSeconds();
        saxpySerial(N, scale, arrayX, arrayY, resultSerial);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

// printf("[saxpy serial]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
    //       minSerial * 1000,
    //       toBW(TOTAL_BYTES, minSerial),
    //       toGFLOPS(TOTAL_FLOPS, minSerial));

    //
    // Run the ISPC (single core) implementation
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        saxpy_ispc(N, scale, arrayX, arrayY, resultISPC);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    verifyResult(N, resultISPC, resultSerial);

    printf("[saxpy ispc]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
           minISPC * 1000,
           toBW(TOTAL_BYTES, minISPC),
           toGFLOPS(TOTAL_FLOPS, minISPC));

    //
    // Run the ISPC (multi-core) implementation
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        saxpy_ispc_withtasks(N, scale, arrayX, arrayY, resultTasks);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    verifyResult(N, resultTasks, resultSerial);


    // Run AVX2 Implementation
    double minAVX2 = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        avx2(N, scale, arrayX, arrayY, resultAVX2);
        double endTime = CycleTimer::currentSeconds();
        minAVX2 = std::min(minAVX2, endTime - startTime);
    }

    // Verify correctness
  
    verifyResult(N, resultAVX2, resultSerial);


    printf("[saxpy task ispc]:\t[%.3f] ms\t[%.3f] GB/s\t[%.3f] GFLOPS\n",
           minTaskISPC * 1000,
           toBW(TOTAL_BYTES, minTaskISPC),
           toGFLOPS(TOTAL_FLOPS, minTaskISPC));

    printf("\t\t\t\t(%.2fx speedup from use of tasks)\n", minISPC/minTaskISPC);
    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);
    printf("[saxpy AVX2]:        [%.3f] ms  [%.3f] GB/s  [%.3f] GFLOPS\n",
        minAVX2 * 1000, toBW(TOTAL_BYTES, minAVX2), toGFLOPS(TOTAL_FLOPS, minAVX2));

 printf("Speedup from AVX2:       %.2fx\n", minSerial / minAVX2);



    free(resultSerial);
    free(resultISPC);
    free(resultAVX2);
    free(arrayX);
    free(arrayY);
    free(resultTasks);



    return 0;
}
