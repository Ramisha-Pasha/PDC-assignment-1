#include <stdio.h>
#include <thread>
#include<iostream>
#include<stdlib.h>
#include "CycleTimer.h"
#include <atomic>
#include <vector>
#include <cstdio>
#include <algorithm>
static inline int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}
typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int* output;
    int threadId;
    int numThreads;
} WorkerArgs;


extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);


//
// workerThreadStart --
//
// Thread entrypoint.
//----------------------------------------------part 1-------------------------------------------------------
void workerThreadStart(WorkerArgs * const args) {
    int height = args->height;
    int startRow, endRow;

    // If threadId is 0 → compute the top half, else compute the bottom half
    if (args->threadId == 0) {
        startRow = 0;
        endRow = height / 2;
    } else {
        startRow = height / 2;
        endRow = height;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    printf("Thread %d computing rows %d to %d\n", args->threadId, startRow, endRow);

    
    mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                     args->width, args->height,
                     startRow, endRow - startRow,
                     args->maxIterations, args->output);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    printf("Thread %d completed in %.5f seconds\n", args->threadId, elapsed.count());
}


//----------------------------------------part 2(extending to 8 threads)-----------------------------------
void workerThreadStart_8Threads(WorkerArgs * const args) {
    int h=args->height;
    int num_threads=args->numThreads;
    int rowSt,rowEnd;
    //rows per thread
    int rowStep=h/num_threads;
    rowSt=args->threadId*rowStep;
    rowEnd = (args->threadId == num_threads - 1) ? h : rowSt + rowStep;
    auto start_time = std::chrono::high_resolution_clock::now();
    printf("Thread %d computing rows %d to %d\n", args->threadId, rowSt, rowEnd);
    mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
        args->width, args->height,
        rowSt, rowEnd - rowSt,
        args->maxIterations, args->output);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    printf("Thread %d completed in %.5f seconds\n", args->threadId, elapsed.count());


}


//-----------------------------------------part 4 ( dynamic task decomposition with static blocks(8 rows))

void workerThreadStart_dynamic(WorkerArgs * const args) {
    const int blockSize = 8;  // Assign 4 rows at once
    std::atomic<int> nextRow(0);
   
    float stepSize_x = (args->x1 - args->x0) / args->width;
    float stepSize_y = (args->y1 - args->y0) / args->height;


    while (true) {
        int begin_row = nextRow.fetch_add(blockSize);
        if (begin_row >= args->height) break;
        int rowEnd = std::min(static_cast<int>(args->height), begin_row + blockSize);


        for (int r = begin_row; r < rowEnd; r += args->numThreads) {
            float cordinate_y = args->y0 + r * stepSize_y;  // Compute only once per row
            for (int c = 0; c < args->width; c++) {
                int ind = (r * args->width) + c;
                float cordinate_x = args->x0 + c * stepSize_x;
                args->output[ind] = mandel(cordinate_x, cordinate_y, args->maxIterations);
            }
        }
    }
}



    






//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[]) {
    std::cout << " Running mandelbrotThread() with " << numThreads << " threads\n";
//-------------------part 1---------------------------------
    // if (numThreads != 2) {
    //     fprintf(stderr, "Error: This version only supports 2 threads.\n");
    //     exit(1);
    // }

    // std::thread workers[2];
    // WorkerArgs args[2];

    // for (int i = 0; i < 2; i++) {
    //     args[i].x0 = x0;
    //     args[i].y0 = y0;
    //     args[i].x1 = x1;
    //     args[i].y1 = y1;
    //     args[i].width = width;
    //     args[i].height = height;
    //     args[i].maxIterations = maxIterations;
    //     args[i].numThreads = 2;
    //     args[i].output = output;
    //     args[i].threadId = i;
    // }

    // // Spawn one thread
    // workers[1] = std::thread(workerThreadStart, &args[1]);

    // // Run first worker on main thread
    // workerThreadStart(&args[0]);

    // // Wait for second thread to finish
    // workers[1].join();

//-------------------------------part 2--------------------------------
    static constexpr int MAX_THREADS = 16; // Support up to 16 threads
    if (numThreads > MAX_THREADS) {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    // Assign work to threads
    for (int i = 0; i < numThreads; i++) {
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;
        args[i].threadId = i;
    }

    // Spawn worker threads
    for (int i = 1; i < numThreads; i++) {
        workers[i] = std::thread(workerThreadStart_dynamic, &args[i]);
    }

    
    workerThreadStart_dynamic(&args[0]);

    // Join all worker threads
    for (int i = 1; i < numThreads; i++) {
        workers[i].join();
    }
}

