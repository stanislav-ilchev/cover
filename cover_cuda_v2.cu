/*
** cover_cuda_v2.cu - GPU-accelerated covering design finder
**
** Key insight: parallelize WITHIN each SA iteration
** - 512 threads cooperate to check all 17,550 m-subsets
** - Each thread checks ~35 m-subsets using popcount
** - Warp reduction computes total cost delta
** - No sorting needed!
**
** Compile: nvcc -O3 -arch=sm_75 -o cover_cuda_v2.exe cover_cuda_v2.cu
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef uint64_t mask_t;
typedef int32_t cost_t;
typedef uint8_t cover_t;

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

/* Error checking macro */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* Warp-level reduction */
__device__ __forceinline__ int warpReduceSum(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/* Block-level reduction */
__device__ int blockReduceSum(int val) {
    __shared__ int shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

/* Select nth set bit from mask */
__device__ int selectNthBit(mask_t mask, int n) {
    for (int i = 0; i < 64; i++) {
        if (mask & ((mask_t)1 << i)) {
            if (n == 0) return i;
            n--;
        }
    }
    return -1;
}

/* Generate random neighbor mask */
__device__ mask_t randomNeighbor(curandState* state, mask_t curr, int k, int v, mask_t fullMask) {
    mask_t comp = fullMask ^ curr;
    int removeBit = selectNthBit(curr, curand(state) % k);
    int addBit = selectNthBit(comp, curand(state) % (v - k));
    return (curr ^ ((mask_t)1 << removeBit)) | ((mask_t)1 << addBit);
}

/* Generate random k-subset */
__device__ mask_t randomKSubset(curandState* state, int k, int v) {
    mask_t mask = 0;
    int remaining = k;
    for (int i = 0; i < v && remaining > 0; i++) {
        if ((curand(state) % (v - i)) < remaining) {
            mask |= ((mask_t)1 << i);
            remaining--;
        }
    }
    return mask;
}

/*
** Main SA kernel - one block per SA process
** All threads in block cooperate on each iteration
*/
__global__ void simulatedAnnealingKernel(
    const mask_t* __restrict__ mSubsetMasks,
    int numMSubsets,
    int v, int k, int t, int b,
    mask_t fullMask,
    double coolFact, double initialT, int frozen, int endLimit, int iterLength,
    mask_t* bestSolution,
    cost_t* bestCost,
    unsigned long long seed
) {
    __shared__ mask_t blocks[128];  // Max 128 blocks per solution
    __shared__ cover_t covered[18000];  // Max ~18000 m-subsets
    __shared__ int sharedDelta;
    __shared__ int sharedCost;
    __shared__ mask_t sharedOldMask, sharedNewMask;
    __shared__ int sharedBlockIdx;
    __shared__ int sharedAccept;
    __shared__ curandState rngState;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Initialize RNG (one per block)
    if (tid == 0) {
        curand_init(seed + bid, 0, 0, &rngState);
    }
    __syncthreads();
    
    // Initialize random solution
    if (tid < b) {
        blocks[tid] = randomKSubset(&rngState, k, v);
    }
    __syncthreads();
    
    // Initialize covered array
    for (int i = tid; i < numMSubsets; i += blockDim.x) {
        cover_t cnt = 0;
        mask_t mSub = mSubsetMasks[i];
        for (int j = 0; j < b; j++) {
            if (__popcll(mSub & blocks[j]) >= t) {
                cnt++;
                break;  // Just need to know if covered, not count
            }
        }
        covered[i] = cnt;
    }
    __syncthreads();
    
    // Compute initial cost
    int localCost = 0;
    for (int i = tid; i < numMSubsets; i += blockDim.x) {
        if (covered[i] == 0) localCost++;
    }
    int totalCost = blockReduceSum(localCost);
    if (tid == 0) {
        sharedCost = totalCost;
    }
    __syncthreads();
    
    // SA loop
    double T = initialT;
    int notChanged = 0;
    cost_t currCost = sharedCost;
    cost_t bestSeenCost = currCost;
    
    while (notChanged < frozen && currCost > endLimit) {
        for (int iter = 0; iter < iterLength; iter++) {
            // Thread 0 generates the move
            if (tid == 0) {
                sharedBlockIdx = curand(&rngState) % b;
                sharedOldMask = blocks[sharedBlockIdx];
                sharedNewMask = randomNeighbor(&rngState, sharedOldMask, k, v, fullMask);
            }
            __syncthreads();
            
            mask_t oldMask = sharedOldMask;
            mask_t newMask = sharedNewMask;
            int blockIdx = sharedBlockIdx;
            
            // All threads compute partial delta in parallel
            int myDelta = 0;
            for (int i = tid; i < numMSubsets; i += blockDim.x) {
                mask_t mSub = mSubsetMasks[i];
                int oldCovers = (__popcll(mSub & oldMask) >= t);
                int newCovers = (__popcll(mSub & newMask) >= t);
                
                if (oldCovers != newCovers) {
                    if (oldCovers && !newCovers) {
                        // Losing coverage - check if still covered by other blocks
                        int stillCovered = 0;
                        for (int j = 0; j < b && !stillCovered; j++) {
                            if (j != blockIdx && __popcll(mSub & blocks[j]) >= t) {
                                stillCovered = 1;
                            }
                        }
                        if (!stillCovered) myDelta++;  // Becomes uncovered
                    } else {
                        // Gaining coverage
                        if (covered[i] == 0) myDelta--;  // Was uncovered, now covered
                    }
                }
            }
            
            // Reduce to get total delta
            int totalDelta = blockReduceSum(myDelta);
            if (tid == 0) {
                sharedDelta = totalDelta;
            }
            __syncthreads();
            
            int delta = sharedDelta;
            
            // Acceptance decision (thread 0)
            if (tid == 0) {
                int accept = 0;
                if (delta <= 0) {
                    accept = 1;
                    if (delta < 0) notChanged = 0;
                } else {
                    float r = curand_uniform(&rngState);
                    if (r < expf(-delta / T)) {
                        accept = 1;
                    }
                }
                sharedAccept = accept;
            }
            __syncthreads();
            
            // Apply move if accepted
            if (sharedAccept) {
                // Update blocks array
                if (tid == 0) {
                    blocks[blockIdx] = newMask;
                    currCost += delta;
                    if (currCost < bestSeenCost) bestSeenCost = currCost;
                }
                
                // Update covered array in parallel
                for (int i = tid; i < numMSubsets; i += blockDim.x) {
                    mask_t mSub = mSubsetMasks[i];
                    int oldCovers = (__popcll(mSub & oldMask) >= t);
                    int newCovers = (__popcll(mSub & newMask) >= t);
                    
                    if (oldCovers && !newCovers) {
                        int stillCovered = 0;
                        for (int j = 0; j < b && !stillCovered; j++) {
                            if (j != blockIdx && __popcll(mSub & blocks[j]) >= t) {
                                stillCovered = 1;
                            }
                        }
                        covered[i] = stillCovered;
                    } else if (!oldCovers && newCovers) {
                        covered[i] = 1;
                    }
                }
                __syncthreads();
                
                // Check if solution found
                if (currCost <= endLimit) {
                    break;
                }
            }
        }
        
        notChanged++;
        T *= coolFact;
        __syncthreads();
    }
    
    // Write best result
    if (tid == 0) {
        // Atomic update of global best
        int oldBest = atomicMin((int*)bestCost, currCost);
        if (currCost < oldBest) {
            // We found a better solution, copy it
            for (int i = 0; i < b; i++) {
                bestSolution[i] = blocks[i];
            }
        }
    }
}

/* Generate all m-subsets using Gosper's hack */
int generateMSubsets(int v, int m, mask_t** outMasks) {
    // Calculate binomial coefficient
    unsigned long long count = 1;
    for (int i = 0; i < m; i++) {
        count = count * (v - i) / (i + 1);
    }
    
    *outMasks = (mask_t*)malloc(count * sizeof(mask_t));
    if (!*outMasks) return 0;
    
    mask_t mask = ((mask_t)1 << m) - 1;
    mask_t limit = (mask_t)1 << v;
    int idx = 0;
    
    while (mask < limit && idx < (int)count) {
        (*outMasks)[idx++] = mask;
        mask_t c = mask & -(long long)mask;
        mask_t r = mask + c;
        mask = (((r ^ mask) >> 2) / c) | r;
    }
    return idx;
}

void printSolution(mask_t* blocks, int b, int v) {
    for (int j = 0; j < b; j++) {
        mask_t mask = blocks[j];
        for (int i = 0; i < v; i++) {
            if (mask & ((mask_t)1 << i)) printf("%d ", i);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int v = 27, k = 6, m = 4, t = 3, b = 86;
    double coolFact = 0.999;
    double initialT = 0.6;
    int frozen = 1000;
    int endLimit = 0;
    int numBlocks = 64;  // Number of parallel SA processes
    int iterLength = 1000;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "v=", 2) == 0) v = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "k=", 2) == 0) k = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "m=", 2) == 0) m = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "t=", 2) == 0) t = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "b=", 2) == 0) b = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "CF=", 3) == 0) coolFact = atof(argv[i] + 3);
        else if (strncmp(argv[i], "IT=", 3) == 0) initialT = atof(argv[i] + 3);
        else if (strncmp(argv[i], "frozen=", 7) == 0) frozen = atoi(argv[i] + 7);
        else if (strncmp(argv[i], "EL=", 3) == 0) endLimit = atoi(argv[i] + 3);
        else if (strncmp(argv[i], "blocks=", 7) == 0) numBlocks = atoi(argv[i] + 7);
        else if (strncmp(argv[i], "L=", 2) == 0) iterLength = atoi(argv[i] + 2);
    }
    
    printf("\ncover_cuda_v2 - GPU-accelerated covering design finder\n");
    printf("======================================================\n\n");
    printf("Parameters: v=%d k=%d m=%d t=%d b=%d\n", v, k, m, t, b);
    printf("SA params: CF=%.4f IT=%.3f frozen=%d EL=%d L=%d\n", coolFact, initialT, frozen, endLimit, iterLength);
    printf("GPU: %d parallel SA blocks, %d threads each\n\n", numBlocks, THREADS_PER_BLOCK);
    
    // Generate m-subset masks
    mask_t* h_mSubsets;
    int numMSubsets = generateMSubsets(v, m, &h_mSubsets);
    printf("Generated %d m-subsets\n", numMSubsets);
    
    // Allocate GPU memory
    mask_t* d_mSubsets;
    mask_t* d_bestSolution;
    cost_t* d_bestCost;
    
    CUDA_CHECK(cudaMalloc(&d_mSubsets, numMSubsets * sizeof(mask_t)));
    CUDA_CHECK(cudaMalloc(&d_bestSolution, b * sizeof(mask_t)));
    CUDA_CHECK(cudaMalloc(&d_bestCost, sizeof(cost_t)));
    
    CUDA_CHECK(cudaMemcpy(d_mSubsets, h_mSubsets, numMSubsets * sizeof(mask_t), cudaMemcpyHostToDevice));
    
    // Initialize best cost to large value
    cost_t initCost = numMSubsets;
    CUDA_CHECK(cudaMemcpy(d_bestCost, &initCost, sizeof(cost_t), cudaMemcpyHostToDevice));
    
    mask_t fullMask = (v == 64) ? ~(mask_t)0 : ((mask_t)1 << v) - 1;
    unsigned long long seed = time(NULL);
    
    printf("Launching kernel...\n\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Launch kernel
    simulatedAnnealingKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
        d_mSubsets, numMSubsets,
        v, k, t, b, fullMask,
        coolFact, initialT, frozen, endLimit, iterLength,
        d_bestSolution, d_bestCost, seed
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Get results
    cost_t finalCost;
    mask_t* h_bestSolution = (mask_t*)malloc(b * sizeof(mask_t));
    CUDA_CHECK(cudaMemcpy(&finalCost, d_bestCost, sizeof(cost_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bestSolution, d_bestSolution, b * sizeof(mask_t), cudaMemcpyDeviceToHost));
    
    printf("Results:\n");
    printf("--------\n");
    printf("Best cost: %d\n", finalCost);
    printf("GPU time: %.2f ms (%.2f sec)\n", milliseconds, milliseconds/1000.0);
    
    if (finalCost <= endLimit) {
        printf("\nSOLUTION FOUND!\n");
        printf("--------------\n");
        printSolution(h_bestSolution, b, v);
        
        FILE* fp = fopen("cover_cuda_v2.res", "w");
        if (fp) {
            for (int j = 0; j < b; j++) {
                mask_t mask = h_bestSolution[j];
                for (int i = 0; i < v; i++) {
                    if (mask & ((mask_t)1 << i)) fprintf(fp, "%d ", i);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    } else {
        printf("\nEndLimit not reached.\n");
    }
    
    // Cleanup
    free(h_mSubsets);
    free(h_bestSolution);
    cudaFree(d_mSubsets);
    cudaFree(d_bestSolution);
    cudaFree(d_bestCost);
    
    return (finalCost <= endLimit) ? 0 : 1;
}


