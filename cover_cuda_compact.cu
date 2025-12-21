/*
 * CUDA Covering Design Search - Same Algorithm as Original cover.exe
 * 
 * Each GPU thread runs an independent search with:
 * - Random initialization of b blocks
 * - Coverage tracking (count per m-subset)
 * - Local search: swap one block, accept if cost improves
 * 
 * This mirrors the original SA approach but runs many instances in parallel.
 * 
 * Usage: cover_cuda_compact.exe v=27 k=6 m=4 t=3 b=86 runs=256 iter=100000
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef uint64_t mask_t;
typedef unsigned char count_t;  // Coverage count per m-subset

// Configuration
#define MAX_V 64
#define MAX_B 256
#define MAX_M_SUBSETS 20000
#define THREADS_PER_BLOCK 64

// Device constants
__constant__ int d_v, d_k, d_m, d_t, d_b;
__constant__ int d_numMSubsets, d_numKSubsets;
__constant__ int d_coverLen;  // Number of m-subsets covered by one k-subset

//=============================================================================
// Host Utilities
//=============================================================================

long long binomial(int n, int k) {
    if (k > n || k < 0) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    long long result = 1;
    for (int i = 0; i < k; i++) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

void generateSubsetMasks(int v, int k, mask_t* masks, int* count) {
    *count = 0;
    int subset[16];
    for (int i = 0; i < k; i++) subset[i] = i;
    
    while (1) {
        mask_t mask = 0;
        for (int i = 0; i < k; i++) {
            mask |= (1ULL << subset[i]);
        }
        masks[(*count)++] = mask;
        
        int i = k - 1;
        while (i >= 0 && subset[i] == v - k + i) i--;
        if (i < 0) break;
        subset[i]++;
        for (int j = i + 1; j < k; j++) {
            subset[j] = subset[j-1] + 1;
        }
    }
}

// Calculate number of m-subsets covered by one k-subset
int calculateCoverLen(int v, int k, int m, int t) {
    int coverLen = 0;
    for (int ti = t; ti <= k && ti <= m; ti++) {
        coverLen += binomial(k, ti) * binomial(v - k, m - ti);
    }
    return coverLen;
}

//=============================================================================
// CUDA Device Functions
//=============================================================================

// Count bits in mask
__device__ __forceinline__ int popcount64(mask_t mask) {
    return __popcll(mask);
}

// Select the n-th set bit (0-indexed)
__device__ int selectNthBit(mask_t mask, int n) {
    for (int i = 0; i < 64; i++) {
        if (mask & (1ULL << i)) {
            if (n == 0) return i;
            n--;
        }
    }
    return -1;
}

// Generate a random neighbor of a k-subset mask
__device__ mask_t randomNeighborMask(mask_t currMask, int v, int k, curandState* state) {
    mask_t fullMask = (v == 64) ? ~0ULL : ((1ULL << v) - 1);
    mask_t comp = fullMask ^ currMask;
    
    int removeBit = selectNthBit(currMask, curand(state) % k);
    int addBit = selectNthBit(comp, curand(state) % (v - k));
    
    return (currMask & ~(1ULL << removeBit)) | (1ULL << addBit);
}

// Check if m-subset (mask) is covered by k-subset (block) with threshold t
__device__ __forceinline__ bool isCovered(mask_t mMask, mask_t blockMask, int t) {
    return popcount64(mMask & blockMask) >= t;
}

//=============================================================================
// CUDA Kernels
//=============================================================================

__global__ void initRandStates(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed + idx * 12345, idx, 0, &states[idx]);
    }
}

// Initialize each solution with b random k-subsets and compute initial coverage
__global__ void initSolutions(
    mask_t* solutions,      // numRuns * b masks
    count_t* coverages,     // numRuns * numMSubsets counts
    int* costs,             // numRuns costs (uncovered count)
    mask_t* kSubsetMasks,
    mask_t* mSubsetMasks,
    curandState* randStates,
    int numRuns
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRuns) return;
    
    curandState localState = randStates[idx];
    mask_t* mySolution = solutions + idx * d_b;
    count_t* myCoverage = coverages + idx * d_numMSubsets;
    
    // Clear coverage
    for (int m = 0; m < d_numMSubsets; m++) {
        myCoverage[m] = 0;
    }
    
    // Initialize with random k-subsets
    for (int i = 0; i < d_b; i++) {
        int randIdx = curand(&localState) % d_numKSubsets;
        mask_t block = kSubsetMasks[randIdx];
        mySolution[i] = block;
        
        // Update coverage for this block
        for (int m = 0; m < d_numMSubsets; m++) {
            if (isCovered(mSubsetMasks[m], block, d_t)) {
                myCoverage[m]++;
            }
        }
    }
    
    // Count uncovered m-subsets
    int uncovered = 0;
    for (int m = 0; m < d_numMSubsets; m++) {
        if (myCoverage[m] == 0) uncovered++;
    }
    costs[idx] = uncovered;
    
    randStates[idx] = localState;
}

// Main search kernel - each thread does local search on one solution
__global__ void localSearchKernel(
    mask_t* solutions,
    count_t* coverages,
    int* costs,
    mask_t* kSubsetMasks,
    mask_t* mSubsetMasks,
    curandState* randStates,
    int numRuns,
    int iterations,
    int* bestCost,
    int* bestIdx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRuns) return;
    
    curandState localState = randStates[idx];
    mask_t* mySolution = solutions + idx * d_b;
    count_t* myCoverage = coverages + idx * d_numMSubsets;
    int currentCost = costs[idx];
    
    for (int iter = 0; iter < iterations && currentCost > 0; iter++) {
        // Pick random block to replace
        int blockIdx = curand(&localState) % d_b;
        mask_t oldBlock = mySolution[blockIdx];
        
        // Generate random neighbor (swap one element)
        mask_t newBlock = randomNeighborMask(oldBlock, d_v, d_k, &localState);
        
        // Compute cost delta
        int delta = 0;
        
        // Check m-subsets that might change coverage
        for (int m = 0; m < d_numMSubsets; m++) {
            mask_t mMask = mSubsetMasks[m];
            bool oldCovers = isCovered(mMask, oldBlock, d_t);
            bool newCovers = isCovered(mMask, newBlock, d_t);
            
            if (oldCovers && !newCovers) {
                // Losing coverage
                if (myCoverage[m] == 1) {
                    delta++;  // Will become uncovered
                }
            } else if (!oldCovers && newCovers) {
                // Gaining coverage
                if (myCoverage[m] == 0) {
                    delta--;  // Will become covered
                }
            }
        }
        
        // Accept if improvement or equal (hill climbing with plateau walks)
        if (delta <= 0) {
            // Update coverage counts
            for (int m = 0; m < d_numMSubsets; m++) {
                mask_t mMask = mSubsetMasks[m];
                bool oldCovers = isCovered(mMask, oldBlock, d_t);
                bool newCovers = isCovered(mMask, newBlock, d_t);
                
                if (oldCovers && !newCovers) {
                    myCoverage[m]--;
                } else if (!oldCovers && newCovers) {
                    myCoverage[m]++;
                }
            }
            
            mySolution[blockIdx] = newBlock;
            currentCost += delta;
        }
    }
    
    costs[idx] = currentCost;
    randStates[idx] = localState;
    
    // Atomic update of best
    atomicMin(bestCost, currentCost);
}

// Find best solution index
__global__ void findBest(int* costs, int numRuns, int targetCost, int* bestIdx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRuns) return;
    
    if (costs[idx] == targetCost) {
        atomicMin(bestIdx, idx);
    }
}

//=============================================================================
// Main
//=============================================================================

void printSolution(mask_t* solution, int b) {
    printf("\nSolution blocks:\n");
    for (int i = 0; i < b; i++) {
        mask_t mask = solution[i];
        printf("Block %3d: { ", i + 1);
        int first = 1;
        for (int bit = 0; bit < 64; bit++) {
            if (mask & (1ULL << bit)) {
                if (!first) printf(", ");
                printf("%d", bit);  // 0-indexed
                first = 0;
            }
        }
        printf(" }\n");
    }
}

int main(int argc, char* argv[]) {
    // Default parameters
    int v = 27, k = 6, m = 4, t = 3, b = 86;
    int numRuns = 256;
    int iterations = 100000;
    int rounds = 100;  // Number of kernel launches
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        sscanf(argv[i], "v=%d", &v);
        sscanf(argv[i], "k=%d", &k);
        sscanf(argv[i], "m=%d", &m);
        sscanf(argv[i], "t=%d", &t);
        sscanf(argv[i], "b=%d", &b);
        sscanf(argv[i], "runs=%d", &numRuns);
        sscanf(argv[i], "iter=%d", &iterations);
        sscanf(argv[i], "rounds=%d", &rounds);
    }
    
    int numMSubsets = (int)binomial(v, m);
    int numKSubsets = (int)binomial(v, k);
    int coverLen = calculateCoverLen(v, k, m, t);
    
    printf("=== CUDA Compact Search (Same Algorithm as cover.exe) ===\n");
    printf("Parameters: v=%d k=%d m=%d t=%d b=%d\n", v, k, m, t, b);
    printf("Search: runs=%d iter=%d rounds=%d\n", numRuns, iterations, rounds);
    printf("M-subsets: %d, K-subsets: %d, CoverLen: %d\n", numMSubsets, numKSubsets, coverLen);
    
    // Check limits
    if (numMSubsets > MAX_M_SUBSETS) {
        printf("ERROR: Too many m-subsets (%d > %d)\n", numMSubsets, MAX_M_SUBSETS);
        return 1;
    }
    
    // Copy constants to device
    cudaMemcpyToSymbol(d_v, &v, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_m, &m, sizeof(int));
    cudaMemcpyToSymbol(d_t, &t, sizeof(int));
    cudaMemcpyToSymbol(d_b, &b, sizeof(int));
    cudaMemcpyToSymbol(d_numMSubsets, &numMSubsets, sizeof(int));
    cudaMemcpyToSymbol(d_numKSubsets, &numKSubsets, sizeof(int));
    cudaMemcpyToSymbol(d_coverLen, &coverLen, sizeof(int));
    
    // Allocate host memory
    mask_t* h_mSubsetMasks = (mask_t*)malloc(numMSubsets * sizeof(mask_t));
    mask_t* h_kSubsetMasks = (mask_t*)malloc(numKSubsets * sizeof(mask_t));
    mask_t* h_bestSolution = (mask_t*)malloc(b * sizeof(mask_t));
    int* h_costs = (int*)malloc(numRuns * sizeof(int));
    
    // Generate subset masks
    printf("Generating subset masks...\n");
    int count;
    generateSubsetMasks(v, m, h_mSubsetMasks, &count);
    generateSubsetMasks(v, k, h_kSubsetMasks, &count);
    
    // Allocate device memory
    mask_t *d_mSubsetMasks, *d_kSubsetMasks;
    mask_t *d_solutions;
    count_t *d_coverages;
    int *d_costs;
    int *d_bestCost, *d_bestIdx;
    curandState *d_randStates;
    
    size_t solutionBytes = (size_t)numRuns * b * sizeof(mask_t);
    size_t coverageBytes = (size_t)numRuns * numMSubsets * sizeof(count_t);
    
    printf("Allocating GPU memory: %.2f MB\n", 
           (solutionBytes + coverageBytes + numMSubsets * sizeof(mask_t) + 
            numKSubsets * sizeof(mask_t)) / (1024.0 * 1024.0));
    
    cudaMalloc(&d_mSubsetMasks, numMSubsets * sizeof(mask_t));
    cudaMalloc(&d_kSubsetMasks, numKSubsets * sizeof(mask_t));
    cudaMalloc(&d_solutions, solutionBytes);
    cudaMalloc(&d_coverages, coverageBytes);
    cudaMalloc(&d_costs, numRuns * sizeof(int));
    cudaMalloc(&d_bestCost, sizeof(int));
    cudaMalloc(&d_bestIdx, sizeof(int));
    cudaMalloc(&d_randStates, numRuns * sizeof(curandState));
    
    cudaMemcpy(d_mSubsetMasks, h_mSubsetMasks, numMSubsets * sizeof(mask_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kSubsetMasks, h_kSubsetMasks, numKSubsets * sizeof(mask_t), cudaMemcpyHostToDevice);
    
    int blocks = (numRuns + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    printf("Initializing %d parallel searches...\n", numRuns);
    initRandStates<<<blocks, THREADS_PER_BLOCK>>>(d_randStates, time(NULL), numRuns);
    cudaDeviceSynchronize();
    
    // Main search loop
    printf("\nStarting search...\n\n");
    clock_t startTime = clock();
    
    int globalBest = INT_MAX;
    long long totalIterations = 0;
    
    for (int round = 0; round < rounds; round++) {
        // Initialize solutions
        initSolutions<<<blocks, THREADS_PER_BLOCK>>>(
            d_solutions, d_coverages, d_costs,
            d_kSubsetMasks, d_mSubsetMasks, d_randStates, numRuns);
        cudaDeviceSynchronize();
        
        // Reset best tracking
        int initBest = INT_MAX;
        int initIdx = INT_MAX;
        cudaMemcpy(d_bestCost, &initBest, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bestIdx, &initIdx, sizeof(int), cudaMemcpyHostToDevice);
        
        // Run local search
        localSearchKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_solutions, d_coverages, d_costs,
            d_kSubsetMasks, d_mSubsetMasks, d_randStates,
            numRuns, iterations, d_bestCost, d_bestIdx);
        cudaDeviceSynchronize();
        
        totalIterations += (long long)numRuns * iterations;
        
        // Get best cost from this round
        int roundBest;
        cudaMemcpy(&roundBest, d_bestCost, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (roundBest < globalBest) {
            globalBest = roundBest;
            
            // Find which solution has the best cost
            findBest<<<blocks, THREADS_PER_BLOCK>>>(d_costs, numRuns, roundBest, d_bestIdx);
            cudaDeviceSynchronize();
            
            int bestIdx;
            cudaMemcpy(&bestIdx, d_bestIdx, sizeof(int), cudaMemcpyDeviceToHost);
            
            if (bestIdx < numRuns) {
                cudaMemcpy(h_bestSolution, d_solutions + bestIdx * b, 
                           b * sizeof(mask_t), cudaMemcpyDeviceToHost);
            }
            
            double elapsed = (double)(clock() - startTime) / CLOCKS_PER_SEC;
            printf("Round %4d: New best = %d uncovered (%.0f iter/sec)\n", 
                   round, globalBest, totalIterations / elapsed);
            
            if (globalBest == 0) {
                printf("\n*** PERFECT SOLUTION FOUND! ***\n");
                break;
            }
        }
        
        // Progress report every 10 rounds
        if (round % 10 == 9) {
            double elapsed = (double)(clock() - startTime) / CLOCKS_PER_SEC;
            printf("Round %4d: Best so far = %d (%.0f iter/sec, %.1f sec)\n", 
                   round + 1, globalBest, totalIterations / elapsed, elapsed);
        }
    }
    
    double totalTime = (double)(clock() - startTime) / CLOCKS_PER_SEC;
    
    printf("\n=== Results ===\n");
    printf("Best: %d uncovered m-subsets\n", globalBest);
    printf("Total time: %.2f seconds\n", totalTime);
    printf("Total iterations: %lld\n", totalIterations);
    printf("Iterations/second: %.0f\n", totalIterations / totalTime);
    
    if (globalBest == 0) {
        printSolution(h_bestSolution, b);
    }
    
    // Cleanup
    cudaFree(d_mSubsetMasks);
    cudaFree(d_kSubsetMasks);
    cudaFree(d_solutions);
    cudaFree(d_coverages);
    cudaFree(d_costs);
    cudaFree(d_bestCost);
    cudaFree(d_bestIdx);
    cudaFree(d_randStates);
    
    free(h_mSubsetMasks);
    free(h_kSubsetMasks);
    free(h_bestSolution);
    free(h_costs);
    
    return 0;
}

