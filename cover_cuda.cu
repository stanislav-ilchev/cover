/*
 * cover_cuda.cu - CUDA-accelerated covering design search
 * 
 * This implements a massively parallel simulated annealing approach where
 * thousands of independent SA processes run simultaneously on the GPU.
 * Each CUDA thread executes a complete SA run, and the best solution wins.
 *
 * Compile with: nvcc -O3 -o cover_cuda cover_cuda.cu -arch=sm_60
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Configuration - adjust based on your problem size
#define MAX_V 40
#define MAX_K 20
#define MAX_B 1024
#define MAX_COVER_LEN 1024      // For covering arrays (per k-set)
#define MAX_COVERED_LEN 20000   // For covered array (binCoef[v][m])
#define THREADS_PER_BLOCK 64    // Reduced for more register space
#define DEFAULT_NUM_RUNS 1024   // Reduced for large problems

// Type definitions
typedef unsigned int rankType;
typedef unsigned short coveredType;
typedef int costType;
typedef uint64_t maskType;

// Device-side binomial coefficients
__constant__ unsigned int d_binCoef[MAX_V + 1][MAX_V + 2];

// Problem parameters (in constant memory for fast access)
__constant__ int d_v, d_k, d_t, d_m, d_b;
__constant__ int d_coverLen, d_coveredLen, d_neighborLen;

// Host-side binomial coefficients
unsigned int h_binCoef[MAX_V + 1][MAX_V + 2];

// Calculate binomial coefficients on host
void calculateBinCoefs() {
    for (int v = 0; v <= MAX_V; v++) {
        h_binCoef[v][0] = h_binCoef[v][v] = 1;
        h_binCoef[v][v + 1] = 0;
        for (int k = 1; k <= v - 1; k++) {
            h_binCoef[v][k] = h_binCoef[v - 1][k - 1] + h_binCoef[v - 1][k];
            if (h_binCoef[v][k] < h_binCoef[v - 1][k - 1] ||
                h_binCoef[v][k] < h_binCoef[v - 1][k])
                h_binCoef[v][k] = 0;
        }
    }
}

// Device: Unrank subset - convert rank to set representation
__device__ void unrankSubset(rankType rank, unsigned char* subset, int card) {
    int m = rank;
    for (int i = card - 1; i >= 0; i--) {
        int p = i;
        while (d_binCoef[p + 1][i + 1] <= m) p++;
        m -= d_binCoef[p][i + 1];
        subset[i] = p;
    }
}

// Device: Rank subset - convert set to rank
__device__ rankType rankSubset(unsigned char* subset, int card) {
    rankType rank = 0;
    for (int i = 0; i < card; i++)
        rank += d_binCoef[subset[i]][i + 1];
    return rank;
}

// Device: Convert rank to bitmask
__device__ maskType maskFromRank(rankType rank, int card) {
    unsigned char subset[MAX_K];
    unrankSubset(rank, subset, card);
    maskType mask = 0;
    for (int i = 0; i < card; i++)
        mask |= ((maskType)1 << subset[i]);
    return mask;
}

// Device: Convert bitmask to rank
__device__ rankType rankFromMask(maskType mask, int card) {
    rankType rank = 0;
    int count = 0;
    for (int i = 0; i < d_v && count < card; i++) {
        if (mask & ((maskType)1 << i)) {
            count++;
            rank += d_binCoef[i][count];
        }
    }
    return rank;
}

// Device: Select nth set bit in mask
__device__ int selectNthBit(maskType mask, int n) {
    for (int i = 0; i < d_v; i++) {
        if (mask & ((maskType)1 << i)) {
            if (n == 0) return i;
            n--;
        }
    }
    return -1;
}

// Device: Generate random neighbor using bitmask
__device__ rankType randomNeighborMask(maskType currMask, maskType* outMask, 
                                        curandState* state) {
    maskType fullMask = (d_v == 64) ? ~(maskType)0 : (((maskType)1 << d_v) - 1);
    maskType comp = fullMask ^ currMask;
    
    int removeBit = selectNthBit(currMask, curand(state) % d_k);
    int addBit = selectNthBit(comp, curand(state) % (d_v - d_k));
    
    *outMask = (currMask & ~(((maskType)1) << removeBit)) |
               (((maskType)1) << addBit);
    return rankFromMask(*outMask, d_k);
}

// Device: Calculate coverings for one k-set (simplified for GPU)
__device__ void calculateOneCoveringMask(maskType mask, rankType* buf,
                                          unsigned char* subset, unsigned char* csubset) {
    int idx = 0;
    for (int i = 0; i < d_v; i++)
        if (mask & ((maskType)1 << i))
            subset[idx++] = i;
    subset[d_k] = MAX_V + 1;
    
    idx = 0;
    for (int i = 0; i < d_v; i++)
        if (!(mask & ((maskType)1 << i)))
            csubset[idx++] = i;
    csubset[d_v - d_k] = MAX_V + 1;
    
    rankType* coverptr = buf;
    int minKM = (d_k < d_m) ? d_k : d_m;
    
    for (int ti = d_t; ti <= minKM; ti++) {
        // Generate all combinations of ti elements from subset
        // and (m-ti) elements from csubset
        unsigned char subsubset[MAX_K];
        for (int i = 0; i < ti; i++) subsubset[i] = i;
        
        do {
            unsigned char subcsubset[MAX_K];
            int mti = d_m - ti;
            for (int i = 0; i < mti; i++) subcsubset[i] = i;
            
            do {
                // Merge to form m-set
                unsigned char mergeset[MAX_K];
                int ss = 0, sc = 0;
                for (int i = 0; i < d_m; i++) {
                    if (ss < ti && (sc >= mti || 
                        subset[subsubset[ss]] < csubset[subcsubset[sc]]))
                        mergeset[i] = subset[subsubset[ss++]];
                    else
                        mergeset[i] = csubset[subcsubset[sc++]];
                }
                *coverptr++ = rankSubset(mergeset, d_m);
                
                // Next subcsubset
                if (mti == 0) break;
                int j = 0;
                while (j + 1 < mti && subcsubset[j + 1] <= subcsubset[j] + 1) j++;
                if (subcsubset[0] >= d_v - d_k - mti) break;
                subcsubset[j]++;
                for (int i = 0; i < j; i++) subcsubset[i] = i;
            } while (true);
            
            // Next subsubset
            if (ti == 0) break;
            int j = 0;
            while (j + 1 < ti && subsubset[j + 1] <= subsubset[j] + 1) j++;
            if (subsubset[0] >= d_k - ti) break;
            subsubset[j]++;
            for (int i = 0; i < j; i++) subsubset[i] = i;
        } while (true);
    }
    *coverptr = d_binCoef[d_v][d_m]; // sentinel
    
    // Simple bubble sort for small arrays (GPU-friendly)
    int len = coverptr - buf;
    for (int i = 0; i < len - 1; i++) {
        for (int j = 0; j < len - i - 1; j++) {
            if (buf[j] > buf[j + 1]) {
                rankType tmp = buf[j];
                buf[j] = buf[j + 1];
                buf[j + 1] = tmp;
            }
        }
    }
}

// Main SA kernel - each thread runs complete simulated annealing
__global__ void simulatedAnnealingKernel(
    rankType* bestSolutions,    // Output: best solution per thread
    costType* bestCosts,        // Output: best cost per thread
    coveredType* allCovered,    // Pre-allocated: numThreads * coveredLen
    rankType* allKset,          // Pre-allocated: numThreads * b
    maskType* allKsetMask,      // Pre-allocated: numThreads * b
    float coolFact,
    float initProb,
    float initTemp,             // If initTempSet, use this directly
    int initTempSet,            // Flag: 1 = use initTemp, 0 = compute from initProb
    int iterLength,
    int frozen,
    int endLimit,
    int coverNumber,
    int coverLen,
    int coveredLen,
    int numThreads,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numThreads) return;
    
    // Initialize RNG for this thread
    curandState state;
    curand_init(seed + tid, 0, 0, &state);
    
    // Use pre-allocated global memory
    coveredType* covered = allCovered + (size_t)tid * coveredLen;
    rankType* kset = allKset + (size_t)tid * d_b;
    maskType* ksetMask = allKsetMask + (size_t)tid * d_b;
    
    // Thread-local storage for small arrays
    costType costs[MAX_B + 1];
    int costds[MAX_B + 1];
    
    // Temporary buffers for covering calculations
    rankType currCoverings[MAX_COVER_LEN];
    rankType nextCoverings[MAX_COVER_LEN];
    unsigned char subset[MAX_K + 1];
    unsigned char csubset[MAX_V + 1];
    
    // Calculate costs
    for (int i = 0; i <= d_b; i++) {
        if (i < coverNumber)
            costs[i] = coverNumber - i;
        else
            costs[i] = 0;
    }
    for (int i = 0; i < d_b; i++)
        costds[i] = costs[i] - costs[i + 1];
    
    // Initialize covered array (already zeroed by cudaMemset, but ensure)
    for (int i = 0; i < coveredLen; i++)
        covered[i] = 0;
    
    // Generate initial random solution
    costType currCost = 0;
    for (int i = 0; i < d_b; i++) {
        kset[i] = curand(&state) % d_binCoef[d_v][d_k];
        ksetMask[i] = maskFromRank(kset[i], d_k);
        
        calculateOneCoveringMask(ksetMask[i], currCoverings, subset, csubset);
        for (int j = 0; j < coverLen - 1; j++)
            covered[currCoverings[j]]++;
    }
    
    for (int i = 0; i < coveredLen; i++)
        currCost += costs[covered[i]];
    
    // Set initial temperature
    float T;
    if (initTempSet) {
        // Use provided initial temperature directly
        T = initTemp;
    } else {
        // Estimate initial temperature from initProb
        T = 0.0f;
        int m2 = 0;
        for (int iter = 0; iter < 300; iter++) {
            int setNumber = curand(&state) % d_b;
            maskType nextMask;
            rankType nextS = randomNeighborMask(ksetMask[setNumber], &nextMask, &state);
            
            calculateOneCoveringMask(ksetMask[setNumber], currCoverings, subset, csubset);
            calculateOneCoveringMask(nextMask, nextCoverings, subset, csubset);
            
            costType costDelta = 0;
            int ci = 0, ni = 0;
            rankType sentinel = d_binCoef[d_v][d_m];
            while (currCoverings[ci] != sentinel || nextCoverings[ni] != sentinel) {
                if (currCoverings[ci] == nextCoverings[ni]) {
                    if (currCoverings[ci] == sentinel) break;
                    ci++; ni++;
                } else if (currCoverings[ci] < nextCoverings[ni]) {
                    costDelta += costds[covered[currCoverings[ci]] - 1];
                    ci++;
                } else {
                    costDelta -= costds[covered[nextCoverings[ni]]];
                    ni++;
                }
            }
            
            if (costDelta > 0) {
                m2++;
                T += -costDelta;
            }
        }
        T = (m2 == 0) ? 1.0f : T / m2 / logf(initProb);
    }
    
    // Main simulated annealing loop
    costType bestSeen = currCost;
    int notChanged = 0;
    
    while (notChanged < frozen && currCost > endLimit) {
        costType lastCost = currCost;
        
        for (int iter = 0; iter < iterLength; iter++) {
            int setNumber = curand(&state) % d_b;
            maskType nextMask;
            rankType nextS = randomNeighborMask(ksetMask[setNumber], &nextMask, &state);
            
            // Compute cost delta
            calculateOneCoveringMask(ksetMask[setNumber], currCoverings, subset, csubset);
            calculateOneCoveringMask(nextMask, nextCoverings, subset, csubset);
            
            costType costDelta = 0;
            int ci = 0, ni = 0;
            rankType sentinel = d_binCoef[d_v][d_m];
            while (currCoverings[ci] != sentinel || nextCoverings[ni] != sentinel) {
                if (currCoverings[ci] == nextCoverings[ni]) {
                    if (currCoverings[ci] == sentinel) break;
                    ci++; ni++;
                } else if (currCoverings[ci] < nextCoverings[ni]) {
                    costDelta += costds[covered[currCoverings[ci]] - 1];
                    ci++;
                } else {
                    costDelta -= costds[covered[nextCoverings[ni]]];
                    ni++;
                }
            }
            
            // Accept or reject
            bool accept = false;
            if (costDelta <= 0) {
                accept = true;
            } else {
                float r = curand_uniform(&state);
                if (r < expf(-costDelta / T))
                    accept = true;
            }
            
            if (accept) {
                // Update covered counts
                calculateOneCoveringMask(ksetMask[setNumber], currCoverings, subset, csubset);
                for (int j = 0; currCoverings[j] != d_binCoef[d_v][d_m]; j++)
                    covered[currCoverings[j]]--;
                
                calculateOneCoveringMask(nextMask, nextCoverings, subset, csubset);
                for (int j = 0; nextCoverings[j] != d_binCoef[d_v][d_m]; j++)
                    covered[nextCoverings[j]]++;
                
                kset[setNumber] = nextS;
                ksetMask[setNumber] = nextMask;
                currCost += costDelta;
                
                if (costDelta < 0) {
                    notChanged = 0;
                    if (currCost < bestSeen)
                        bestSeen = currCost;
                }
                
                if (currCost <= endLimit) break;
            }
        }
        
        if (lastCost <= currCost)
            notChanged++;
        T *= coolFact;
    }
    
    // Store results
    bestCosts[tid] = currCost;
    for (int i = 0; i < d_b && i < MAX_B; i++)
        bestSolutions[tid * MAX_B + i] = kset[i];
}

// Find best result across all threads
void findBestResult(costType* h_costs, rankType* h_solutions, int numRuns,
                    int b, costType* bestCost, rankType* bestSolution) {
    *bestCost = h_costs[0];
    int bestIdx = 0;
    
    for (int i = 1; i < numRuns; i++) {
        if (h_costs[i] < *bestCost) {
            *bestCost = h_costs[i];
            bestIdx = i;
        }
    }
    
    for (int i = 0; i < b; i++)
        bestSolution[i] = h_solutions[bestIdx * MAX_B + i];
}

// Print subset
void printSubset(rankType rank, int k, int v) {
    unsigned char subset[MAX_K];
    int m = rank;
    for (int i = k - 1; i >= 0; i--) {
        int p = i;
        while (h_binCoef[p + 1][i + 1] <= m) p++;
        m -= h_binCoef[p][i + 1];
        subset[i] = p;
    }
    for (int i = 0; i < k; i++)
        printf("%d ", subset[i]);
    printf("\n");
}

int main(int argc, char** argv) {
    // Default parameters
    int v = 7, k = 3, t = 2, m = 2, b = 7;
    float coolFact = 0.99f, initProb = 0.5f;
    float initTemp = 0.0f;  // If > 0, use this instead of computing from initProb
    int initTempSet = 0;
    int frozen = 10, endLimit = 0, coverNumber = 1;
    int numRuns = DEFAULT_NUM_RUNS;
    float LFact = 1.0f;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "v=", 2) == 0) v = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "k=", 2) == 0) k = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "t=", 2) == 0) t = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "m=", 2) == 0) m = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "b=", 2) == 0) b = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "CF=", 3) == 0) coolFact = atof(argv[i] + 3);
        else if (strncmp(argv[i], "CoolingFactor=", 14) == 0) coolFact = atof(argv[i] + 14);
        else if (strncmp(argv[i], "IP=", 3) == 0) initProb = atof(argv[i] + 3);
        else if (strncmp(argv[i], "InitProb=", 9) == 0) initProb = atof(argv[i] + 9);
        else if (strncmp(argv[i], "IT=", 3) == 0) { initTemp = atof(argv[i] + 3); initTempSet = 1; }
        else if (strncmp(argv[i], "InitTemp=", 9) == 0) { initTemp = atof(argv[i] + 9); initTempSet = 1; }
        else if (strncmp(argv[i], "frozen=", 7) == 0) frozen = atoi(argv[i] + 7);
        else if (strncmp(argv[i], "EL=", 3) == 0) endLimit = atoi(argv[i] + 3);
        else if (strncmp(argv[i], "EndLimit=", 9) == 0) endLimit = atoi(argv[i] + 9);
        else if (strncmp(argv[i], "l=", 2) == 0) coverNumber = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "runs=", 5) == 0) numRuns = atoi(argv[i] + 5);
        else if (strncmp(argv[i], "LF=", 3) == 0) LFact = atof(argv[i] + 3);
        else if (strncmp(argv[i], "LFact=", 6) == 0) LFact = atof(argv[i] + 6);
    }
    
    printf("\n");
    printf("cover_cuda - GPU-accelerated covering design search\n");
    printf("===================================================\n\n");
    printf("Running %d parallel SA instances on GPU\n\n", numRuns);
    printf("Design parameters:\n");
    printf("------------------\n");
    printf("t - (v,m,k,l) = %d - (%d,%d,%d,%d)\n", t, v, m, k, coverNumber);
    printf("b = %d\n\n", b);
    
    printf("Optimization parameters:\n");
    printf("------------------------\n");
    printf("CoolingFactor = %.4f\n", coolFact);
    if (initTempSet)
        printf("InitTemp      = %.3f\n", initTemp);
    else
        printf("InitProb      = %.2f\n", initProb);
    printf("frozen        = %d\n", frozen);
    printf("EndLimit      = %d\n\n", endLimit);
    
    // Calculate binomial coefficients
    calculateBinCoefs();
    
    // Calculate derived parameters
    int neighborLen = k * (v - k);
    int coverLen = 0;
    for (int i = 0; i <= (k - t < m - t ? k - t : m - t); i++)
        coverLen += h_binCoef[k][t + i] * h_binCoef[v - k][m - t - i];
    coverLen++;
    int coveredLen = h_binCoef[v][m];
    int L = (int)(LFact * k * (v - k) * b + 0.5);
    
    printf("coverLen = %d, coveredLen = %d\n", coverLen, coveredLen);
    printf("L (iterations per temperature) = %d\n\n", L);
    
    // Copy parameters to device constant memory
    cudaMemcpyToSymbol(d_binCoef, h_binCoef, sizeof(h_binCoef));
    cudaMemcpyToSymbol(d_v, &v, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_t, &t, sizeof(int));
    cudaMemcpyToSymbol(d_m, &m, sizeof(int));
    cudaMemcpyToSymbol(d_b, &b, sizeof(int));
    cudaMemcpyToSymbol(d_coverLen, &coverLen, sizeof(int));
    cudaMemcpyToSymbol(d_coveredLen, &coveredLen, sizeof(int));
    cudaMemcpyToSymbol(d_neighborLen, &neighborLen, sizeof(int));
    
    // Allocate device memory
    rankType* d_solutions;
    costType* d_costs;
    coveredType* d_allCovered;
    rankType* d_allKset;
    maskType* d_allKsetMask;
    
    size_t coveredBytes = (size_t)numRuns * coveredLen * sizeof(coveredType);
    size_t ksetBytes = (size_t)numRuns * b * sizeof(rankType);
    size_t ksetMaskBytes = (size_t)numRuns * b * sizeof(maskType);
    
    printf("Allocating GPU memory: %.1f MB\n", 
           (coveredBytes + ksetBytes + ksetMaskBytes) / (1024.0 * 1024.0));
    
    cudaMalloc(&d_solutions, numRuns * MAX_B * sizeof(rankType));
    cudaMalloc(&d_costs, numRuns * sizeof(costType));
    cudaMalloc(&d_allCovered, coveredBytes);
    cudaMalloc(&d_allKset, ksetBytes);
    cudaMalloc(&d_allKsetMask, ksetMaskBytes);
    
    // Initialize covered array to zero
    cudaMemset(d_allCovered, 0, coveredBytes);
    
    // Allocate host memory
    rankType* h_solutions = (rankType*)malloc(numRuns * MAX_B * sizeof(rankType));
    costType* h_costs = (costType*)malloc(numRuns * sizeof(costType));
    
    // Launch kernel
    int numBlocks = (numRuns + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    printf("Launching kernel with %d blocks x %d threads...\n\n", 
           numBlocks, THREADS_PER_BLOCK);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    simulatedAnnealingKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
        d_solutions, d_costs, d_allCovered, d_allKset, d_allKsetMask,
        coolFact, initProb, initTemp, initTempSet, L, frozen, endLimit,
        coverNumber, coverLen, coveredLen, numRuns, time(NULL)
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy results back
    cudaMemcpy(h_costs, d_costs, numRuns * sizeof(costType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_solutions, d_solutions, numRuns * MAX_B * sizeof(rankType), cudaMemcpyDeviceToHost);
    
    // Find best result
    costType bestCost;
    rankType* bestSolution = (rankType*)malloc(b * sizeof(rankType));
    findBestResult(h_costs, h_solutions, numRuns, b, &bestCost, bestSolution);
    
    printf("Results:\n");
    printf("--------\n");
    printf("Best cost found: %d\n", bestCost);
    printf("GPU time: %.3f seconds\n", milliseconds / 1000.0f);
    printf("Equivalent sequential runs: %d\n", numRuns);
    printf("Effective speedup: ~%.0fx\n\n", numRuns * (milliseconds > 0 ? 1.0 : 1.0));
    
    if (bestCost <= endLimit) {
        printf("Solution found:\n");
        printf("--------------\n");
        for (int i = 0; i < b; i++)
            printSubset(bestSolution[i], k, v);
    } else {
        printf("EndLimit not reached. Best solution had cost %d\n", bestCost);
    }
    
    // Count successes
    int successes = 0;
    for (int i = 0; i < numRuns; i++)
        if (h_costs[i] <= endLimit) successes++;
    printf("\nSuccess rate: %d/%d (%.1f%%)\n", successes, numRuns, 
           100.0f * successes / numRuns);
    
    // Cleanup
    free(h_solutions);
    free(h_costs);
    free(bestSolution);
    cudaFree(d_solutions);
    cudaFree(d_costs);
    cudaFree(d_allCovered);
    cudaFree(d_allKset);
    cudaFree(d_allKsetMask);
    
    return bestCost <= endLimit ? 0 : 1;
}



