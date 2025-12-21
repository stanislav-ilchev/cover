/*
 * CUDA Covering Design Search with Merge-Diff Algorithm
 * 
 * This implements the SAME efficient algorithm as original cover.exe (OF=0):
 * - Precompute all k-subset → m-subset coverings
 * - Use merge-diff to compute cost delta (O(coverLen) not O(numMSubsets))
 * - Run many parallel independent searches
 * 
 * Usage: cover_cuda_mergediff.exe v=27 k=6 m=4 t=3 b=86 runs=256 iter=100000
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef uint64_t mask_t;
typedef uint32_t rank_t;
typedef unsigned char count_t;

// Configuration - adjust based on GPU memory
#define THREADS_PER_BLOCK 64
#define MAX_COVER_LEN 512

// Device constants
__constant__ int d_v, d_k, d_m, d_t, d_b;
__constant__ int d_numMSubsets, d_numKSubsets;
__constant__ int d_coverLen;
__constant__ rank_t d_sentinel;

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

int* h_binCoef = NULL;
int h_maxV = 0;

void initBinCoef(int maxV) {
    h_maxV = maxV + 1;
    h_binCoef = (int*)malloc(h_maxV * h_maxV * sizeof(int));
    for (int n = 0; n < h_maxV; n++) {
        for (int k = 0; k <= n; k++) {
            h_binCoef[n * h_maxV + k] = (int)binomial(n, k);
        }
    }
}

int getBinCoef(int n, int k) {
    if (k > n || k < 0) return 0;
    return h_binCoef[n * h_maxV + k];
}

// Rank a subset (combinatorial number system)
rank_t rankSubset(int* subset, int k) {
    rank_t rank = 0;
    for (int i = 0; i < k; i++) {
        rank += getBinCoef(subset[i], i + 1);
    }
    return rank;
}

// Unrank a subset
void unrankSubset(rank_t rank, int* subset, int k, int v) {
    int x = v - 1;
    for (int i = k - 1; i >= 0; i--) {
        while (getBinCoef(x, i + 1) > rank) x--;
        subset[i] = x;
        rank -= getBinCoef(x, i + 1);
        x--;
    }
}

// Convert mask to rank
rank_t maskToRank(mask_t mask, int k) {
    int subset[16];
    int idx = 0;
    for (int i = 0; i < 64 && idx < k; i++) {
        if (mask & (1ULL << i)) {
            subset[idx++] = i;
        }
    }
    return rankSubset(subset, k);
}

// Convert rank to mask
mask_t rankToMask(rank_t rank, int k, int v) {
    int subset[16];
    unrankSubset(rank, subset, k, v);
    mask_t mask = 0;
    for (int i = 0; i < k; i++) {
        mask |= (1ULL << subset[i]);
    }
    return mask;
}

// Calculate m-subsets covered by a k-subset
// Returns sorted list of m-subset ranks
int calculateCovering(int v, int k, int m, int t, int* kSubset, rank_t* output) {
    int count = 0;
    int mSubset[16];
    int complement[64];
    int compLen = 0;
    
    // Build complement
    int kIdx = 0;
    for (int i = 0; i < v; i++) {
        if (kIdx < k && kSubset[kIdx] == i) {
            kIdx++;
        } else {
            complement[compLen++] = i;
        }
    }
    
    // Generate all m-subsets that intersect kSubset in at least t elements
    for (int ti = t; ti <= k && ti <= m; ti++) {
        // Choose ti elements from kSubset
        int subsubset[16];
        for (int i = 0; i < ti; i++) subsubset[i] = i;
        
        while (1) {
            // Choose m-ti elements from complement
            int csubset[16];
            for (int i = 0; i < m - ti; i++) csubset[i] = i;
            
            while (1) {
                // Build m-subset
                int si = 0, ci = 0, mi = 0;
                while (mi < m) {
                    int fromK = (si < ti) ? kSubset[subsubset[si]] : v + 1;
                    int fromC = (ci < m - ti) ? complement[csubset[ci]] : v + 1;
                    if (fromK < fromC) {
                        mSubset[mi++] = fromK;
                        si++;
                    } else {
                        mSubset[mi++] = fromC;
                        ci++;
                    }
                }
                output[count++] = rankSubset(mSubset, m);
                
                // Next complement subset
                int i = m - ti - 1;
                while (i >= 0 && csubset[i] == compLen - (m - ti) + i) i--;
                if (i < 0) break;
                csubset[i]++;
                for (int j = i + 1; j < m - ti; j++) csubset[j] = csubset[j-1] + 1;
            }
            
            // Next kSubset subset
            int i = ti - 1;
            while (i >= 0 && subsubset[i] == k - ti + i) i--;
            if (i < 0) break;
            subsubset[i]++;
            for (int j = i + 1; j < ti; j++) subsubset[j] = subsubset[j-1] + 1;
        }
    }
    
    // Sort
    std::sort(output, output + count);
    return count;
}

//=============================================================================
// CUDA Device Code
//=============================================================================

__device__ int d_binCoef[64 * 64];  // Device binomial coefficients

__device__ rank_t deviceMaskToRank(mask_t mask, int k) {
    rank_t rank = 0;
    int idx = 0;
    for (int i = 0; i < 64 && idx < k; i++) {
        if (mask & (1ULL << i)) {
            rank += d_binCoef[i * 64 + (idx + 1)];
            idx++;
        }
    }
    return rank;
}

__device__ mask_t deviceRankToMask(rank_t rank, int k, int v) {
    mask_t mask = 0;
    int x = v - 1;
    for (int i = k - 1; i >= 0; i--) {
        while (d_binCoef[x * 64 + (i + 1)] > rank) x--;
        mask |= (1ULL << x);
        rank -= d_binCoef[x * 64 + (i + 1)];
        x--;
    }
    return mask;
}

__device__ int selectNthBit(mask_t mask, int n) {
    for (int i = 0; i < 64; i++) {
        if (mask & (1ULL << i)) {
            if (n == 0) return i;
            n--;
        }
    }
    return -1;
}

__device__ mask_t randomNeighborMask(mask_t currMask, int v, int k, curandState* state) {
    mask_t fullMask = (v == 64) ? ~0ULL : ((1ULL << v) - 1);
    mask_t comp = fullMask ^ currMask;
    int removeBit = selectNthBit(currMask, curand(state) % k);
    int addBit = selectNthBit(comp, curand(state) % (v - k));
    return (currMask & ~(1ULL << removeBit)) | (1ULL << addBit);
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

// Initialize solutions with random blocks
__global__ void initSolutions(
    mask_t* solutions,           // [numRuns][b] block masks
    rank_t* solutionRanks,       // [numRuns][b] block ranks
    rank_t* solutionCoverings,   // [numRuns][b][coverLen] sorted coverings
    count_t* coverages,          // [numRuns][numMSubsets] coverage counts
    int* costs,                  // [numRuns] uncovered counts
    rank_t* allCoverings,        // [numKSubsets][coverLen] precomputed
    curandState* randStates,
    int numRuns
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRuns) return;
    
    curandState localState = randStates[idx];
    
    mask_t* mySolutions = solutions + idx * d_b;
    rank_t* myRanks = solutionRanks + idx * d_b;
    rank_t* myCoverings = solutionCoverings + idx * d_b * d_coverLen;
    count_t* myCoverage = coverages + idx * d_numMSubsets;
    
    // Clear coverage
    for (int i = 0; i < d_numMSubsets; i++) {
        myCoverage[i] = 0;
    }
    
    // Initialize with random blocks
    for (int blk = 0; blk < d_b; blk++) {
        rank_t rank = curand(&localState) % d_numKSubsets;
        mask_t mask = deviceRankToMask(rank, d_k, d_v);
        
        mySolutions[blk] = mask;
        myRanks[blk] = rank;
        
        // Copy precomputed coverings
        rank_t* srcCoverings = allCoverings + rank * d_coverLen;
        rank_t* dstCoverings = myCoverings + blk * d_coverLen;
        for (int i = 0; i < d_coverLen; i++) {
            dstCoverings[i] = srcCoverings[i];
        }
        
        // Update coverage counts
        for (int i = 0; i < d_coverLen - 1; i++) {
            myCoverage[srcCoverings[i]]++;
        }
    }
    
    // Count uncovered
    int uncovered = 0;
    for (int i = 0; i < d_numMSubsets; i++) {
        if (myCoverage[i] == 0) uncovered++;
    }
    costs[idx] = uncovered;
    
    randStates[idx] = localState;
}

// Main search kernel with merge-diff
__global__ void searchKernel(
    mask_t* solutions,
    rank_t* solutionRanks,
    rank_t* solutionCoverings,
    count_t* coverages,
    int* costs,
    rank_t* allCoverings,
    curandState* randStates,
    int numRuns,
    int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRuns) return;
    
    curandState localState = randStates[idx];
    
    mask_t* mySolutions = solutions + idx * d_b;
    rank_t* myRanks = solutionRanks + idx * d_b;
    rank_t* myCoverings = solutionCoverings + idx * d_b * d_coverLen;
    count_t* myCoverage = coverages + idx * d_numMSubsets;
    int currentCost = costs[idx];
    
    for (int iter = 0; iter < iterations && currentCost > 0; iter++) {
        // Pick random block to modify
        int blkIdx = curand(&localState) % d_b;
        mask_t oldMask = mySolutions[blkIdx];
        
        // Generate random neighbor
        mask_t newMask = randomNeighborMask(oldMask, d_v, d_k, &localState);
        rank_t newRank = deviceMaskToRank(newMask, d_k);
        
        // Get covering lists
        rank_t* oldCov = myCoverings + blkIdx * d_coverLen;
        rank_t* newCov = allCoverings + newRank * d_coverLen;
        
        // Merge-diff to compute cost delta
        int costDelta = 0;
        int oldPtr = 0, newPtr = 0;
        
        while (oldCov[oldPtr] != d_sentinel || newCov[newPtr] != d_sentinel) {
            if (oldCov[oldPtr] == newCov[newPtr]) {
                if (oldCov[oldPtr] == d_sentinel) break;
                oldPtr++;
                newPtr++;
            } else if (oldCov[oldPtr] < newCov[newPtr]) {
                // Losing coverage of this m-subset
                if (myCoverage[oldCov[oldPtr]] == 1) {
                    costDelta++;  // Will become uncovered
                }
                oldPtr++;
            } else {
                // Gaining coverage of this m-subset
                if (myCoverage[newCov[newPtr]] == 0) {
                    costDelta--;  // Will become covered
                }
                newPtr++;
            }
        }
        
        // Accept if improvement or equal (hill climbing with plateau walks)
        if (costDelta <= 0) {
            // Update coverage counts using merge-diff
            oldPtr = 0;
            newPtr = 0;
            while (oldCov[oldPtr] != d_sentinel || newCov[newPtr] != d_sentinel) {
                if (oldCov[oldPtr] == newCov[newPtr]) {
                    if (oldCov[oldPtr] == d_sentinel) break;
                    oldPtr++;
                    newPtr++;
                } else if (oldCov[oldPtr] < newCov[newPtr]) {
                    myCoverage[oldCov[oldPtr]]--;
                    oldPtr++;
                } else {
                    myCoverage[newCov[newPtr]]++;
                    newPtr++;
                }
            }
            
            // Update solution
            mySolutions[blkIdx] = newMask;
            myRanks[blkIdx] = newRank;
            
            // Copy new coverings
            for (int i = 0; i < d_coverLen; i++) {
                oldCov[i] = newCov[i];
            }
            
            currentCost += costDelta;
        }
    }
    
    costs[idx] = currentCost;
    randStates[idx] = localState;
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[]) {
    // Default parameters
    int v = 27, k = 6, m = 4, t = 3, b = 86;
    int numRuns = 256;
    int iterations = 100000;
    int rounds = 100;
    
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
    
    initBinCoef(v);
    
    int numMSubsets = (int)binomial(v, m);
    int numKSubsets = (int)binomial(v, k);
    
    // Calculate coverLen
    int kSubset[16];
    for (int i = 0; i < k; i++) kSubset[i] = i;
    rank_t tempCov[MAX_COVER_LEN];
    int coverLen = calculateCovering(v, k, m, t, kSubset, tempCov) + 1;  // +1 for sentinel
    rank_t sentinel = numMSubsets;
    
    printf("=== CUDA Merge-Diff Search (Same Algorithm as cover.exe OF=0) ===\n");
    printf("Parameters: v=%d k=%d m=%d t=%d b=%d\n", v, k, m, t, b);
    printf("Search: runs=%d iter=%d rounds=%d\n", numRuns, iterations, rounds);
    printf("M-subsets: %d, K-subsets: %d, CoverLen: %d\n", numMSubsets, numKSubsets, coverLen - 1);
    
    // Copy constants
    cudaMemcpyToSymbol(d_v, &v, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_m, &m, sizeof(int));
    cudaMemcpyToSymbol(d_t, &t, sizeof(int));
    cudaMemcpyToSymbol(d_b, &b, sizeof(int));
    cudaMemcpyToSymbol(d_numMSubsets, &numMSubsets, sizeof(int));
    cudaMemcpyToSymbol(d_numKSubsets, &numKSubsets, sizeof(int));
    cudaMemcpyToSymbol(d_coverLen, &coverLen, sizeof(int));
    cudaMemcpyToSymbol(d_sentinel, &sentinel, sizeof(rank_t));
    
    // Copy binomial coefficients to device
    int h_binCoefFlat[64 * 64] = {0};
    for (int n = 0; n < 64 && n <= v; n++) {
        for (int kk = 0; kk <= n && kk <= k; kk++) {
            h_binCoefFlat[n * 64 + kk] = getBinCoef(n, kk);
        }
    }
    cudaMemcpyToSymbol(d_binCoef, h_binCoefFlat, 64 * 64 * sizeof(int));
    
    // Precompute all coverings
    printf("Precomputing coverings for %d k-subsets...\n", numKSubsets);
    size_t allCoveringsBytes = (size_t)numKSubsets * coverLen * sizeof(rank_t);
    rank_t* h_allCoverings = (rank_t*)malloc(allCoveringsBytes);
    
    clock_t precompStart = clock();
    for (int r = 0; r < numKSubsets; r++) {
        unrankSubset(r, kSubset, k, v);
        int len = calculateCovering(v, k, m, t, kSubset, h_allCoverings + r * coverLen);
        h_allCoverings[r * coverLen + len] = sentinel;  // Add sentinel
        
        if (r % 50000 == 0 && r > 0) {
            printf("  %d/%d k-subsets processed...\n", r, numKSubsets);
        }
    }
    double precompTime = (double)(clock() - precompStart) / CLOCKS_PER_SEC;
    printf("Precomputation done in %.2f seconds\n", precompTime);
    
    // Memory calculation
    size_t solutionsBytes = (size_t)numRuns * b * sizeof(mask_t);
    size_t ranksBytes = (size_t)numRuns * b * sizeof(rank_t);
    size_t solCoveringsBytes = (size_t)numRuns * b * coverLen * sizeof(rank_t);
    size_t coveragesBytes = (size_t)numRuns * numMSubsets * sizeof(count_t);
    size_t totalBytes = allCoveringsBytes + solutionsBytes + ranksBytes + solCoveringsBytes + coveragesBytes;
    
    printf("GPU Memory: %.2f MB (coverings: %.2f MB)\n", 
           totalBytes / (1024.0 * 1024.0), allCoveringsBytes / (1024.0 * 1024.0));
    
    // Allocate device memory
    rank_t* d_allCoverings;
    mask_t* d_solutions;
    rank_t* d_solutionRanks;
    rank_t* d_solutionCoverings;
    count_t* d_coverages;
    int* d_costs;
    curandState* d_randStates;
    
    cudaMalloc(&d_allCoverings, allCoveringsBytes);
    cudaMalloc(&d_solutions, solutionsBytes);
    cudaMalloc(&d_solutionRanks, ranksBytes);
    cudaMalloc(&d_solutionCoverings, solCoveringsBytes);
    cudaMalloc(&d_coverages, coveragesBytes);
    cudaMalloc(&d_costs, numRuns * sizeof(int));
    cudaMalloc(&d_randStates, numRuns * sizeof(curandState));
    
    cudaMemcpy(d_allCoverings, h_allCoverings, allCoveringsBytes, cudaMemcpyHostToDevice);
    
    int blocks = (numRuns + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    printf("Initializing %d parallel searches...\n", numRuns);
    initRandStates<<<blocks, THREADS_PER_BLOCK>>>(d_randStates, time(NULL), numRuns);
    cudaDeviceSynchronize();
    
    // Allocate host memory for results
    int* h_costs = (int*)malloc(numRuns * sizeof(int));
    mask_t* h_bestSolution = (mask_t*)malloc(b * sizeof(mask_t));
    
    printf("\nStarting search...\n\n");
    clock_t startTime = clock();
    
    int globalBest = INT_MAX;
    long long totalIterations = 0;
    
    for (int round = 0; round < rounds; round++) {
        // Initialize solutions
        initSolutions<<<blocks, THREADS_PER_BLOCK>>>(
            d_solutions, d_solutionRanks, d_solutionCoverings,
            d_coverages, d_costs, d_allCoverings, d_randStates, numRuns);
        cudaDeviceSynchronize();
        
        // Run search
        searchKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_solutions, d_solutionRanks, d_solutionCoverings,
            d_coverages, d_costs, d_allCoverings, d_randStates,
            numRuns, iterations);
        cudaDeviceSynchronize();
        
        totalIterations += (long long)numRuns * iterations;
        
        // Get results
        cudaMemcpy(h_costs, d_costs, numRuns * sizeof(int), cudaMemcpyDeviceToHost);
        
        int roundBest = INT_MAX;
        int bestIdx = 0;
        for (int i = 0; i < numRuns; i++) {
            if (h_costs[i] < roundBest) {
                roundBest = h_costs[i];
                bestIdx = i;
            }
        }
        
        if (roundBest < globalBest) {
            globalBest = roundBest;
            cudaMemcpy(h_bestSolution, d_solutions + bestIdx * b, 
                       b * sizeof(mask_t), cudaMemcpyDeviceToHost);
            
            double elapsed = (double)(clock() - startTime) / CLOCKS_PER_SEC;
            printf("Round %4d: New best = %d uncovered (%.0f iter/sec)\n",
                   round, globalBest, totalIterations / elapsed);
            
            if (globalBest == 0) {
                printf("\n*** PERFECT SOLUTION FOUND! ***\n");
                break;
            }
        }
        
        if (round % 10 == 9) {
            double elapsed = (double)(clock() - startTime) / CLOCKS_PER_SEC;
            printf("Round %4d: Best = %d (%.0f iter/sec)\n",
                   round + 1, globalBest, totalIterations / elapsed);
        }
    }
    
    double totalTime = (double)(clock() - startTime) / CLOCKS_PER_SEC;
    
    printf("\n=== Results ===\n");
    printf("Best: %d uncovered m-subsets\n", globalBest);
    printf("Total time: %.2f seconds (+ %.2f precomp)\n", totalTime, precompTime);
    printf("Total iterations: %lld\n", totalIterations);
    printf("Iterations/second: %.0f\n", totalIterations / totalTime);
    
    if (globalBest == 0) {
        printf("\nSolution blocks (0-indexed):\n");
        for (int i = 0; i < b; i++) {
            printf("Block %3d: { ", i);
            int first = 1;
            for (int bit = 0; bit < 64; bit++) {
                if (h_bestSolution[i] & (1ULL << bit)) {
                    if (!first) printf(", ");
                    printf("%d", bit);
                    first = 0;
                }
            }
            printf(" }\n");
        }
    }
    
    // Cleanup
    cudaFree(d_allCoverings);
    cudaFree(d_solutions);
    cudaFree(d_solutionRanks);
    cudaFree(d_solutionCoverings);
    cudaFree(d_coverages);
    cudaFree(d_costs);
    cudaFree(d_randStates);
    
    free(h_allCoverings);
    free(h_costs);
    free(h_bestSolution);
    free(h_binCoef);
    
    return 0;
}

