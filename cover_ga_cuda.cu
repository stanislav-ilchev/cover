/*
 * CUDA Genetic Algorithm for Covering Designs
 * 
 * This implements a population-based search that leverages GPU parallelism.
 * Each thread evaluates one individual's fitness in parallel.
 * 
 * Usage: cover_ga_cuda.exe v=27 k=6 m=4 t=3 b=86 pop=10000 gen=1000 mut=0.1
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Configuration
#define MAX_V 64
#define MAX_K 10
#define MAX_B 200
#define MAX_M_SUBSETS 50000
#define MAX_K_SUBSETS 500000

// Default parameters
#define DEFAULT_POP_SIZE 8192
#define DEFAULT_GENERATIONS 10000
#define DEFAULT_MUTATION_RATE 0.1f
#define DEFAULT_TOURNAMENT_SIZE 4
#define DEFAULT_ELITE_COUNT 10

typedef uint64_t mask_t;

// Host-side globals
int h_v, h_k, h_m, h_t, h_b;
int h_numMSubsets, h_numKSubsets;
mask_t* h_mSubsetMasks = NULL;
mask_t* h_kSubsetMasks = NULL;

// Device-side constants (in constant memory for fast access)
__constant__ int d_v, d_k, d_m, d_t, d_b;
__constant__ int d_numMSubsets, d_numKSubsets;

// Device-side data
mask_t* d_mSubsetMasks = NULL;
mask_t* d_kSubsetMasks = NULL;
mask_t* d_population = NULL;      // pop_size * b masks
mask_t* d_newPopulation = NULL;   // for double buffering
int* d_fitness = NULL;            // pop_size fitness values
curandState* d_randStates = NULL;

// Host-side results
int* h_fitness = NULL;
mask_t* h_bestSolution = NULL;

//=============================================================================
// Utility Functions
//=============================================================================

// Binomial coefficient
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

// Generate all k-subsets of v elements as bitmasks
void generateSubsetMasks(int v, int k, mask_t* masks, int* count) {
    *count = 0;
    
    // Generate subsets using combinatorial number system
    int subset[MAX_K];
    for (int i = 0; i < k; i++) subset[i] = i;
    
    while (1) {
        // Convert subset to mask
        mask_t mask = 0;
        for (int i = 0; i < k; i++) {
            mask |= (1ULL << subset[i]);
        }
        masks[(*count)++] = mask;
        
        // Generate next subset
        int i = k - 1;
        while (i >= 0 && subset[i] == v - k + i) i--;
        if (i < 0) break;
        subset[i]++;
        for (int j = i + 1; j < k; j++) {
            subset[j] = subset[j-1] + 1;
        }
    }
}

// Parse command line arguments
void parseArgs(int argc, char* argv[], int* v, int* k, int* m, int* t, int* b,
               int* popSize, int* generations, float* mutationRate, int* tournamentSize, int* eliteCount) {
    *v = 27; *k = 6; *m = 4; *t = 3; *b = 86;
    *popSize = DEFAULT_POP_SIZE;
    *generations = DEFAULT_GENERATIONS;
    *mutationRate = DEFAULT_MUTATION_RATE;
    *tournamentSize = DEFAULT_TOURNAMENT_SIZE;
    *eliteCount = DEFAULT_ELITE_COUNT;
    
    for (int i = 1; i < argc; i++) {
        if (sscanf(argv[i], "v=%d", v) == 1) continue;
        if (sscanf(argv[i], "k=%d", k) == 1) continue;
        if (sscanf(argv[i], "m=%d", m) == 1) continue;
        if (sscanf(argv[i], "t=%d", t) == 1) continue;
        if (sscanf(argv[i], "b=%d", b) == 1) continue;
        if (sscanf(argv[i], "pop=%d", popSize) == 1) continue;
        if (sscanf(argv[i], "gen=%d", generations) == 1) continue;
        if (sscanf(argv[i], "mut=%f", mutationRate) == 1) continue;
        if (sscanf(argv[i], "tour=%d", tournamentSize) == 1) continue;
        if (sscanf(argv[i], "elite=%d", eliteCount) == 1) continue;
    }
}

//=============================================================================
// CUDA Kernels
//=============================================================================

// Initialize random states
__global__ void initRandStates(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Initialize population with random k-subsets
__global__ void initPopulation(mask_t* population, mask_t* kSubsetMasks, 
                                curandState* randStates, int popSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;
    
    curandState localState = randStates[idx];
    
    // Each individual gets b random k-subsets (blocks)
    for (int i = 0; i < d_b; i++) {
        int randIdx = curand(&localState) % d_numKSubsets;
        population[idx * d_b + i] = kSubsetMasks[randIdx];
    }
    
    randStates[idx] = localState;
}

// Evaluate fitness of all individuals
// Fitness = number of uncovered m-subsets (lower is better, 0 = perfect)
__global__ void evaluateFitness(mask_t* population, mask_t* mSubsetMasks, 
                                 int* fitness, int popSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;
    
    int uncovered = 0;
    
    // Check each m-subset
    for (int m = 0; m < d_numMSubsets; m++) {
        mask_t mMask = mSubsetMasks[m];
        bool covered = false;
        
        // Check if any block covers this m-subset
        for (int b = 0; b < d_b; b++) {
            mask_t blockMask = population[idx * d_b + b];
            // m-subset is covered if intersection has at least t elements
            if (__popcll(mMask & blockMask) >= d_t) {
                covered = true;
                break;
            }
        }
        
        if (!covered) uncovered++;
    }
    
    fitness[idx] = uncovered;
}

// Tournament selection + crossover + mutation
__global__ void evolve(mask_t* oldPop, mask_t* newPop, int* fitness,
                       mask_t* kSubsetMasks, curandState* randStates,
                       int popSize, int tournamentSize, float mutationRate,
                       int eliteCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;
    
    curandState localState = randStates[idx];
    
    // Elitism: copy best individuals unchanged
    if (idx < eliteCount) {
        // Find the idx-th best individual (simple linear search)
        // For efficiency, we just copy first eliteCount - they'll be replaced if not elite
        // A proper implementation would sort, but this is a simplification
        for (int b = 0; b < d_b; b++) {
            newPop[idx * d_b + b] = oldPop[idx * d_b + b];
        }
        randStates[idx] = localState;
        return;
    }
    
    // Tournament selection for parent 1
    int parent1 = curand(&localState) % popSize;
    int bestFitness1 = fitness[parent1];
    for (int t = 1; t < tournamentSize; t++) {
        int candidate = curand(&localState) % popSize;
        if (fitness[candidate] < bestFitness1) {
            parent1 = candidate;
            bestFitness1 = fitness[candidate];
        }
    }
    
    // Tournament selection for parent 2
    int parent2 = curand(&localState) % popSize;
    int bestFitness2 = fitness[parent2];
    for (int t = 1; t < tournamentSize; t++) {
        int candidate = curand(&localState) % popSize;
        if (fitness[candidate] < bestFitness2) {
            parent2 = candidate;
            bestFitness2 = fitness[candidate];
        }
    }
    
    // Uniform crossover
    for (int b = 0; b < d_b; b++) {
        if (curand(&localState) % 2 == 0) {
            newPop[idx * d_b + b] = oldPop[parent1 * d_b + b];
        } else {
            newPop[idx * d_b + b] = oldPop[parent2 * d_b + b];
        }
    }
    
    // Mutation: replace random blocks with random k-subsets
    for (int b = 0; b < d_b; b++) {
        float r = curand_uniform(&localState);
        if (r < mutationRate) {
            int randIdx = curand(&localState) % d_numKSubsets;
            newPop[idx * d_b + b] = kSubsetMasks[randIdx];
        }
    }
    
    randStates[idx] = localState;
}

// Find minimum fitness (reduction kernel)
__global__ void findBestFitness(int* fitness, int* bestFitness, int* bestIdx, int popSize) {
    __shared__ int sharedFitness[256];
    __shared__ int sharedIdx[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    if (idx < popSize) {
        sharedFitness[tid] = fitness[idx];
        sharedIdx[tid] = idx;
    } else {
        sharedFitness[tid] = INT_MAX;
        sharedIdx[tid] = -1;
    }
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sharedFitness[tid + s] < sharedFitness[tid]) {
                sharedFitness[tid] = sharedFitness[tid + s];
                sharedIdx[tid] = sharedIdx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        // Atomic min to find global best
        atomicMin(bestFitness, sharedFitness[0]);
        // Note: This doesn't correctly track bestIdx across blocks
        // For simplicity, we'll find the actual index on the host
    }
}

//=============================================================================
// Main
//=============================================================================

void printSolution(mask_t* solution, int b, int k) {
    printf("\nSolution found! Blocks:\n");
    for (int i = 0; i < b; i++) {
        mask_t mask = solution[i];
        printf("Block %3d: { ", i + 1);
        int first = 1;
        for (int bit = 0; bit < 64; bit++) {
            if (mask & (1ULL << bit)) {
                if (!first) printf(", ");
                printf("%d", bit + 1);  // 1-indexed
                first = 0;
            }
        }
        printf(" }\n");
    }
}

int main(int argc, char* argv[]) {
    int v, k, m, t, b;
    int popSize, generations;
    float mutationRate;
    int tournamentSize, eliteCount;
    
    parseArgs(argc, argv, &v, &k, &m, &t, &b, &popSize, &generations, 
              &mutationRate, &tournamentSize, &eliteCount);
    
    printf("=== CUDA Genetic Algorithm for Covering Designs ===\n");
    printf("Parameters: v=%d k=%d m=%d t=%d b=%d\n", v, k, m, t, b);
    printf("GA Settings: pop=%d gen=%d mut=%.2f tour=%d elite=%d\n",
           popSize, generations, mutationRate, tournamentSize, eliteCount);
    
    h_v = v; h_k = k; h_m = m; h_t = t; h_b = b;
    
    // Calculate number of subsets
    h_numMSubsets = (int)binomial(v, m);
    h_numKSubsets = (int)binomial(v, k);
    
    printf("M-subsets: %d, K-subsets: %d\n", h_numMSubsets, h_numKSubsets);
    
    if (h_numMSubsets > MAX_M_SUBSETS || h_numKSubsets > MAX_K_SUBSETS) {
        printf("ERROR: Too many subsets. Increase MAX_M_SUBSETS or MAX_K_SUBSETS.\n");
        return 1;
    }
    
    // Allocate and generate subset masks on host
    h_mSubsetMasks = (mask_t*)malloc(h_numMSubsets * sizeof(mask_t));
    h_kSubsetMasks = (mask_t*)malloc(h_numKSubsets * sizeof(mask_t));
    h_fitness = (int*)malloc(popSize * sizeof(int));
    h_bestSolution = (mask_t*)malloc(b * sizeof(mask_t));
    
    printf("Generating m-subset masks...\n");
    int count;
    generateSubsetMasks(v, m, h_mSubsetMasks, &count);
    printf("Generated %d m-subset masks\n", count);
    
    printf("Generating k-subset masks...\n");
    generateSubsetMasks(v, k, h_kSubsetMasks, &count);
    printf("Generated %d k-subset masks\n", count);
    
    // Copy constants to device
    cudaMemcpyToSymbol(d_v, &v, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_m, &m, sizeof(int));
    cudaMemcpyToSymbol(d_t, &t, sizeof(int));
    cudaMemcpyToSymbol(d_b, &b, sizeof(int));
    cudaMemcpyToSymbol(d_numMSubsets, &h_numMSubsets, sizeof(int));
    cudaMemcpyToSymbol(d_numKSubsets, &h_numKSubsets, sizeof(int));
    
    // Allocate device memory
    size_t popBytes = (size_t)popSize * b * sizeof(mask_t);
    size_t mMaskBytes = h_numMSubsets * sizeof(mask_t);
    size_t kMaskBytes = h_numKSubsets * sizeof(mask_t);
    
    printf("Allocating GPU memory: %.2f MB\n", 
           (2 * popBytes + mMaskBytes + kMaskBytes + popSize * sizeof(int) + 
            popSize * sizeof(curandState)) / (1024.0 * 1024.0));
    
    cudaMalloc(&d_mSubsetMasks, mMaskBytes);
    cudaMalloc(&d_kSubsetMasks, kMaskBytes);
    cudaMalloc(&d_population, popBytes);
    cudaMalloc(&d_newPopulation, popBytes);
    cudaMalloc(&d_fitness, popSize * sizeof(int));
    cudaMalloc(&d_randStates, popSize * sizeof(curandState));
    
    // Copy subset masks to device
    cudaMemcpy(d_mSubsetMasks, h_mSubsetMasks, mMaskBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kSubsetMasks, h_kSubsetMasks, kMaskBytes, cudaMemcpyHostToDevice);
    
    // Initialize random states
    int threadsPerBlock = 256;
    int blocks = (popSize + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Initializing random states...\n");
    initRandStates<<<blocks, threadsPerBlock>>>(d_randStates, time(NULL), popSize);
    cudaDeviceSynchronize();
    
    // Initialize population
    printf("Initializing population...\n");
    initPopulation<<<blocks, threadsPerBlock>>>(d_population, d_kSubsetMasks, 
                                                 d_randStates, popSize);
    cudaDeviceSynchronize();
    
    // Evolution loop
    printf("\nStarting evolution...\n");
    clock_t startTime = clock();
    
    int bestFitnessEver = INT_MAX;
    int bestIdxEver = -1;
    int lastImprovement = 0;
    
    for (int gen = 0; gen < generations; gen++) {
        // Evaluate fitness
        evaluateFitness<<<blocks, threadsPerBlock>>>(d_population, d_mSubsetMasks, 
                                                      d_fitness, popSize);
        cudaDeviceSynchronize();
        
        // Copy fitness to host to find best
        cudaMemcpy(h_fitness, d_fitness, popSize * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Find best fitness on host
        int bestFitness = INT_MAX;
        int bestIdx = -1;
        for (int i = 0; i < popSize; i++) {
            if (h_fitness[i] < bestFitness) {
                bestFitness = h_fitness[i];
                bestIdx = i;
            }
        }
        
        // Track best ever
        if (bestFitness < bestFitnessEver) {
            bestFitnessEver = bestFitness;
            bestIdxEver = bestIdx;
            lastImprovement = gen;
            
            // Copy best solution to host
            cudaMemcpy(h_bestSolution, d_population + bestIdx * b, 
                       b * sizeof(mask_t), cudaMemcpyDeviceToHost);
            
            printf("Gen %5d: Best fitness = %d (uncovered m-subsets)\n", 
                   gen, bestFitness);
            
            // Check if we found a perfect solution
            if (bestFitness == 0) {
                printf("\n*** PERFECT SOLUTION FOUND! ***\n");
                break;
            }
        }
        
        // Print progress every 100 generations
        if (gen % 100 == 0 && gen > 0) {
            double elapsed = (double)(clock() - startTime) / CLOCKS_PER_SEC;
            printf("Gen %5d: Best = %d, Best ever = %d (%.1f gen/sec)\n", 
                   gen, bestFitness, bestFitnessEver, gen / elapsed);
        }
        
        // Early stopping if no improvement for many generations
        if (gen - lastImprovement > 1000) {
            printf("No improvement for 1000 generations, trying restart...\n");
            // Reinitialize part of population
            initPopulation<<<blocks, threadsPerBlock>>>(d_population, d_kSubsetMasks, 
                                                         d_randStates, popSize);
            cudaDeviceSynchronize();
            lastImprovement = gen;
        }
        
        // Sort population by fitness for elitism (simple approach: swap best to front)
        // For proper elitism, we'd need a more sophisticated approach
        // Here we just ensure the best individual is at index 0
        if (bestIdx != 0) {
            // Swap best to position 0 on device
            mask_t* tempBlock = (mask_t*)malloc(b * sizeof(mask_t));
            cudaMemcpy(tempBlock, d_population, b * sizeof(mask_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(d_population, d_population + bestIdx * b, b * sizeof(mask_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_population + bestIdx * b, tempBlock, b * sizeof(mask_t), cudaMemcpyHostToDevice);
            
            int tempFitness = h_fitness[0];
            h_fitness[0] = h_fitness[bestIdx];
            h_fitness[bestIdx] = tempFitness;
            cudaMemcpy(d_fitness, h_fitness, popSize * sizeof(int), cudaMemcpyHostToDevice);
            free(tempBlock);
        }
        
        // Evolve population
        evolve<<<blocks, threadsPerBlock>>>(d_population, d_newPopulation, d_fitness,
                                            d_kSubsetMasks, d_randStates,
                                            popSize, tournamentSize, mutationRate,
                                            eliteCount);
        cudaDeviceSynchronize();
        
        // Swap populations
        mask_t* temp = d_population;
        d_population = d_newPopulation;
        d_newPopulation = temp;
    }
    
    double totalTime = (double)(clock() - startTime) / CLOCKS_PER_SEC;
    
    printf("\n=== Results ===\n");
    printf("Best fitness achieved: %d uncovered m-subsets\n", bestFitnessEver);
    printf("Total time: %.2f seconds\n", totalTime);
    printf("Generations per second: %.1f\n", generations / totalTime);
    printf("Total fitness evaluations: %lld\n", (long long)generations * popSize);
    printf("Evaluations per second: %.0f\n", (double)generations * popSize / totalTime);
    
    if (bestFitnessEver == 0) {
        printSolution(h_bestSolution, b, k);
    } else {
        printf("\nBest solution found (not perfect, %d uncovered):\n", bestFitnessEver);
        // Optionally print the best solution even if not perfect
    }
    
    // Cleanup
    cudaFree(d_mSubsetMasks);
    cudaFree(d_kSubsetMasks);
    cudaFree(d_population);
    cudaFree(d_newPopulation);
    cudaFree(d_fitness);
    cudaFree(d_randStates);
    
    free(h_mSubsetMasks);
    free(h_kSubsetMasks);
    free(h_fitness);
    free(h_bestSolution);
    
    return 0;
}

