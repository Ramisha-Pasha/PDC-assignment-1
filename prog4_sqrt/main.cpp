#include <stdio.h>
#include <algorithm>
#include <pthread.h>
#include <math.h>
#include "CycleTimer.h"
#include "sqrt_ispc.h"
#include <immintrin.h>  
using namespace ispc;

void avx2(float* values, float* output, int N);





void avx2( float *values, float *output, int N) {
    const int maxIterations = 5;  // Maximum number of Newton-Raphson refinements
    const float threshold = 1e-6f; // Convergence threshold

    for (int i = 0; i < N; i += 8) {
        __m256 x = _mm256_loadu_ps(&values[i]); // Load 8 values
        __m256 guess = _mm256_rsqrt_ps(x); // Initial approximation of 1/sqrt(x)

        // First Newton-Raphson refinement step for better accuracy
        __m256 half_x = _mm256_mul_ps(x, _mm256_set1_ps(0.5f));
        __m256 three_half = _mm256_set1_ps(1.5f);
        __m256 g2 = _mm256_mul_ps(guess, guess);
        guess = _mm256_mul_ps(guess, _mm256_sub_ps(three_half, _mm256_mul_ps(half_x, g2)));

        // Second Newton-Raphson step (for even higher accuracy)
        g2 = _mm256_mul_ps(guess, guess);
        guess = _mm256_mul_ps(guess, _mm256_sub_ps(three_half, _mm256_mul_ps(half_x, g2)));

        // Compute sqrt(x) = x * guess (final accurate result)
        __m256 result = _mm256_mul_ps(x, guess);

        // Store results
        _mm256_storeu_ps(&output[i], result);
    }
}




   
extern void sqrtSerial(int N, float startGuess, float* values, float* output);

static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (fabs(result[i] - gold[i]) > 1e-4) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

int main() {

    const unsigned int N = 20 * 1000 * 1000;
    const float initialGuess = 1.0f;

    float* values = new float[N];
    float* output = new float[N];
    float* gold = new float[N];

    for (unsigned int i=0; i<N; i++)
    {
        // TODO: CS149 students.  Attempt to change the values in the
        // array here to meet the instructions in the handout: we want
        // to you generate best and worse-case speedups
        
        // starter code populates array with random input values
        values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;       //original input
        //-----------------------------part 3-----------------------
        //values[i]= (i & 7) ? 1.0f : 2.99f;              
        //---------------------------- part 2----------------------
        //values[i]=0.001f;   //fastest 
        //values[i] = 1.0;  //this reduces the execution times for all three kinds of codes.
       
    }

    // generate a gold version to check results
    for (unsigned int i=0; i<N; i++)
        gold[i] = sqrt(values[i]);

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtSerial(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[sqrt serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    verifyResult(N, output, gold);

    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[sqrt ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

    verifyResult(N, output, gold);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of the ISPC code
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc_withtasks(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[sqrt task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

    verifyResult(N, output, gold);
     // Clear out the buffer
     for (unsigned int i = 0; i < N; ++i)
     output[i] = 0;
    double minTaskAVX = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        avx2( values, output,N);
        double endTime = CycleTimer::currentSeconds();
        minTaskAVX = std::min(minTaskAVX, endTime - startTime);
    }

    printf("[sqrt AVX]:\t[%.3f] ms\n", minTaskAVX * 1000);

    verifyResult(N, output, gold);
 


   
 
    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);
    printf("\t\t\t\t(%.2fx speedup from avx)\n", minSerial/minTaskAVX);
    



    delete [] values;
    delete [] output;
    delete [] gold;

    return 0;
}
