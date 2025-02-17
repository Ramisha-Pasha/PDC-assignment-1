#include <stdio.h>
#include <thread>
#include<iostream>
#include<stdlib.h>
#include "CycleTimer.h"
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

    //  For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.

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

    // Compute Mandelbrot set for assigned region
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


//-----------------------------------------part 3 ( a good decomposition)
#include <atomic>
#include <iostream>

#include <atomic>

#include <atomic>
#include <algorithm>



void workerThreadStartt(WorkerArgs *args) {
    int width = args->width;
    int height = args->height;
    int numThreads = args->numThreads;
    int threadId = args->threadId;
    int maxIterations = args->maxIterations;
    
    const int tileSize = 16; // Fixed tile size
    int numTilesX = (width + tileSize - 1) / tileSize;
    int numTilesY = (height + tileSize - 1) / tileSize;
    int totalTiles = numTilesX * numTilesY;

    // **Block-Cyclic Assignment**: Each thread picks every N-th tile
    for (int tileIndex = threadId; tileIndex < totalTiles; tileIndex += numThreads) {
        int tileX = (tileIndex % numTilesX) * tileSize;
        int tileY = (tileIndex / numTilesX) * tileSize;

        for (int j = tileY; j < std::min(height, tileY + tileSize); j++) {
            for (int i = tileX; i < std::min(width, tileX + tileSize); i++) {
                float x = args->x0 + i * (args->x1 - args->x0) / (width - 1);
                float y = args->y0 + j * (args->y1 - args->y0) / (height - 1);
                int index = j * width + i;
                args->output[index] = mandel(x, y, maxIterations);
            }
        }
    }
}

//-----------------------yet to explore( dynamic threadin, each thread takes on the next available row without having to explicitly assign work)
void workerThreadStart_opt(WorkerArgs * const args) {
    const int chunkSize = 10;  // Assign 4 rows at once
    std::atomic<int> nextRow(0);
    
    float dx = (args->x1 - args->x0) / args->width;
    float dy = (args->y1 - args->y0) / args->height;

    while (true) {
        int rowStart = nextRow.fetch_add(chunkSize);
        if (rowStart >= args->height) break;
        int rowEnd = std::min(static_cast<int>(args->height), rowStart + chunkSize);

        for (int row = rowStart; row < rowEnd; row += args->numThreads) {
            float yCoord = args->y0 + row * dy;  // Compute only once per row
            for (int col = 0; col < args->width; col++) {
                int index = (row * args->width) + col;
                float xCoord = args->x0 + col * dx;
                args->output[index] = mandel(xCoord, yCoord, args->maxIterations);
            }
        }
    }
}



//--------------------------------part(4)optimization---------------------------
// void workerThreadStart_opt(WorkerArgs * const args) {
//     int height = args->height;
//     int width = args->width;
//     int numThreads = args->numThreads;
//     int threadId = args->threadId;

//     // Compute proper row assignment
//     int rowsPerThread = height / numThreads;
//     int startRow = threadId * rowsPerThread;
//     int endRow = (threadId == numThreads - 1) ? height : startRow + rowsPerThread;

//     printf("Thread %d computing rows %d to %d\n", threadId, startRow, endRow);

//     auto start_time = std::chrono::high_resolution_clock::now();

//     for (int y = startRow; y < endRow; y++) {
//         for (int x = 0; x < width; x++) {
//             int index = (y * width) + x;
//             float xCoord = args->x0 + x * ((args->x1 - args->x0) / width);
//             float yCoord = args->y0 + y * ((args->y1 - args->y0) / height);
//             args->output[index] = mandel(xCoord, yCoord, args->maxIterations);
//         }
//     }

//     auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end_time - start_time;
//     printf("Thread %d completed in %.5f seconds\n", threadId, elapsed.count());
// }


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
    std::cout << "[DEBUG] Running mandelbrotThread() with " << numThreads << " threads\n";
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
    static constexpr int MAX_THREADS = 8; // Support up to 8 threads
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
        workers[i] = std::thread(workerThreadStartt, &args[i]);
    }

    // Run worker 0 on the main thread
    workerThreadStartt(&args[0]);

    // Join all worker threads
    for (int i = 1; i < numThreads; i++) {
        workers[i].join();
    }
}

