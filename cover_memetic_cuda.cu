/*
 * CUDA Memetic Algorithm for Covering Designs
 * 
 * Combines Genetic Algorithm with Local Search for better convergence.
 * Uses GPU parallelism for both population evaluation and local search.
 * 
 * Usage: cover_memetic_cuda.exe v=27 k=6 m=4 t=3 b=86 pop=1024 gen=1000 local=100
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
#define MAX_B 200
#define MAX_M_SUBSETS 50000
#define MAX_K_SUBSETS 500000
#define THREADS_PER_BLOCK 128

typedef uint64_t mask_t;

// Device-side constants
__constant__ int d_v, d_k, d_m, d_t, d_b;
__constant__ int d_numMSubsets, d_numKSubsets;

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

//=============================================================================
// CUDA Kernels
//=============================================================================

__global__ void initRandStates(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed + idx * 1000, idx, 0, &states[idx]);
    }
}

// Device function: count uncovered m-subsets for one solution
__device__ int countUncovered(mask_t* solution, mask_t* mSubsetMasks) {
    int uncovered = 0;
    for (int m = 0; m < d_numMSubsets; m++) {
        mask_t mMask = mSubsetMasks[m];
        bool covered = false;
        for (int b = 0; b < d_b; b++) {
            if (__popcll(mMask & solution[b]) >= d_t) {
                covered = true;
                break;
            }
        }
        if (!covered) uncovered++;
    }
    return uncovered;
}

// Initialize population randomly
__global__ void initPopulation(mask_t* population, mask_t* kSubsetMasks, 
                                curandState* randStates, int popSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;
    
    curandState localState = randStates[idx];
    for (int i = 0; i < d_b; i++) {
        int randIdx = curand(&localState) % d_numKSubsets;
        population[idx * d_b + i] = kSubsetMasks[randIdx];
    }
    randStates[idx] = localState;
}

// Evaluate fitness of all individuals
__global__ void evaluateFitness(mask_t* population, mask_t* mSubsetMasks, 
                                 int* fitness, int popSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;
    
    fitness[idx] = countUncovered(population + idx * d_b, mSubsetMasks);
}

// Local search: try to improve each solution by random block swaps
// Each thread works on one individual
__global__ void localSearch(mask_t* population, mask_t* mSubsetMasks, 
                            mask_t* kSubsetMasks, int* fitness,
                            curandState* randStates, int popSize, int numSteps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;
    
    curandState localState = randStates[idx];
    mask_t* solution = population + idx * d_b;
    int currentFitness = fitness[idx];
    
    for (int step = 0; step < numSteps && currentFitness > 0; step++) {
        // Pick a random block to replace
        int blkIdx = curand(&localState) % d_b;
        mask_t oldBlock = solution[blkIdx];
        
        // Try a random new block
        int newBlockIdx = curand(&localState) % d_numKSubsets;
        mask_t newBlock = kSubsetMasks[newBlockIdx];
        
        // Apply the change
        solution[blkIdx] = newBlock;
        
        // Evaluate new fitness
        int newFitness = countUncovered(solution, mSubsetMasks);
        
        if (newFitness <= currentFitness) {
            // Accept improvement or equal
            currentFitness = newFitness;
        } else {
            // Reject: revert
            solution[blkIdx] = oldBlock;
        }
    }
    
    fitness[idx] = currentFitness;
    randStates[idx] = localState;
}

// Smarter local search: hill climbing with steepest descent
__global__ void localSearchSteepest(mask_t* population, mask_t* mSubsetMasks, 
                                     mask_t* kSubsetMasks, int* fitness,
                                     curandState* randStates, int popSize, int numSteps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;
    
    curandState localState = randStates[idx];
    mask_t* solution = population + idx * d_b;
    int currentFitness = fitness[idx];
    
    for (int step = 0; step < numSteps && currentFitness > 0; step++) {
        int bestBlockIdx = -1;
        mask_t bestNewBlock = 0;
        int bestFitness = currentFitness;
        
        // Try replacing each block with a few random alternatives
        for (int tryBlock = 0; tryBlock < d_b; tryBlock++) {
            mask_t oldBlock = solution[tryBlock];
            
            // Try 5 random replacements for this block
            for (int tryNew = 0; tryNew < 5; tryNew++) {
                int newIdx = curand(&localState) % d_numKSubsets;
                mask_t newBlock = kSubsetMasks[newIdx];
                
                solution[tryBlock] = newBlock;
                int newFitness = countUncovered(solution, mSubsetMasks);
                
                if (newFitness < bestFitness) {
                    bestFitness = newFitness;
                    bestBlockIdx = tryBlock;
                    bestNewBlock = newBlock;
                }
                
                solution[tryBlock] = oldBlock;
            }
        }
        
        // Apply best move found
        if (bestBlockIdx >= 0) {
            solution[bestBlockIdx] = bestNewBlock;
            currentFitness = bestFitness;
        } else {
            // No improvement found, try random restart of one block
            int randBlock = curand(&localState) % d_b;
            solution[randBlock] = kSubsetMasks[curand(&localState) % d_numKSubsets];
            currentFitness = countUncovered(solution, mSubsetMasks);
        }
    }
    
    fitness[idx] = currentFitness;
    randStates[idx] = localState;
}

// Tournament selection + crossover + mutation
__global__ void evolve(mask_t* oldPop, mask_t* newPop, int* fitness,
                       mask_t* kSubsetMasks, curandState* randStates,
                       int popSize, int tournamentSize, float mutationRate,
                       int eliteIdx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;
    
    curandState localState = randStates[idx];
    
    // Elitism: copy best individual unchanged to first position
    if (idx == 0) {
        for (int b = 0; b < d_b; b++) {
            newPop[b] = oldPop[eliteIdx * d_b + b];
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
    
    // Mutation
    for (int b = 0; b < d_b; b++) {
        float r = curand_uniform(&localState);
        if (r < mutationRate) {
            int randIdx = curand(&localState) % d_numKSubsets;
            newPop[idx * d_b + b] = kSubsetMasks[randIdx];
        }
    }
    
    randStates[idx] = localState;
}

//=============================================================================
// Main
//=============================================================================

void printSolution(mask_t* solution, int b) {
    printf("\nSolution found! Blocks:\n");
    for (int i = 0; i < b; i++) {
        mask_t mask = solution[i];
        printf("Block %3d: { ", i + 1);
        int first = 1;
        for (int bit = 0; bit < 64; bit++) {
            if (mask & (1ULL << bit)) {
                if (!first) printf(", ");
                printf("%d", bit + 1);
                first = 0;
            }
        }
        printf(" }\n");
    }
}

int main(int argc, char* argv[]) {
    // Default parameters
    int v = 27, k = 6, m = 4, t = 3, b = 86;
    int popSize = 512;
    int generations = 5000;
    float mutationRate = 0.05f;
    int tournamentSize = 3;
    int localSteps = 50;  // Local search steps per individual per generation
    int steepest = 0;     // Use steepest descent local search
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        sscanf(argv[i], "v=%d", &v);
        sscanf(argv[i], "k=%d", &k);
        sscanf(argv[i], "m=%d", &m);
        sscanf(argv[i], "t=%d", &t);
        sscanf(argv[i], "b=%d", &b);
        sscanf(argv[i], "pop=%d", &popSize);
        sscanf(argv[i], "gen=%d", &generations);
        sscanf(argv[i], "mut=%f", &mutationRate);
        sscanf(argv[i], "tour=%d", &tournamentSize);
        sscanf(argv[i], "local=%d", &localSteps);
        sscanf(argv[i], "steepest=%d", &steepest);
    }
    
    int numMSubsets = (int)binomial(v, m);
    int numKSubsets = (int)binomial(v, k);
    
    printf("=== CUDA Memetic Algorithm for Covering Designs ===\n");
    printf("Parameters: v=%d k=%d m=%d t=%d b=%d\n", v, k, m, t, b);
    printf("GA Settings: pop=%d gen=%d mut=%.2f tour=%d local=%d steepest=%d\n",
           popSize, generations, mutationRate, tournamentSize, localSteps, steepest);
    printf("M-subsets: %d, K-subsets: %d\n", numMSubsets, numKSubsets);
    
    // Copy constants to device
    cudaMemcpyToSymbol(d_v, &v, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_m, &m, sizeof(int));
    cudaMemcpyToSymbol(d_t, &t, sizeof(int));
    cudaMemcpyToSymbol(d_b, &b, sizeof(int));
    cudaMemcpyToSymbol(d_numMSubsets, &numMSubsets, sizeof(int));
    cudaMemcpyToSymbol(d_numKSubsets, &numKSubsets, sizeof(int));
    
    // Allocate host memory
    mask_t* h_mSubsetMasks = (mask_t*)malloc(numMSubsets * sizeof(mask_t));
    mask_t* h_kSubsetMasks = (mask_t*)malloc(numKSubsets * sizeof(mask_t));
    int* h_fitness = (int*)malloc(popSize * sizeof(int));
    mask_t* h_bestSolution = (mask_t*)malloc(b * sizeof(mask_t));
    
    // Generate subset masks
    printf("Generating subset masks...\n");
    int count;
    generateSubsetMasks(v, m, h_mSubsetMasks, &count);
    generateSubsetMasks(v, k, h_kSubsetMasks, &count);
    
    // Allocate device memory
    mask_t *d_mSubsetMasks, *d_kSubsetMasks;
    mask_t *d_population, *d_newPopulation;
    int *d_fitness;
    curandState *d_randStates;
    
    size_t popBytes = (size_t)popSize * b * sizeof(mask_t);
    
    cudaMalloc(&d_mSubsetMasks, numMSubsets * sizeof(mask_t));
    cudaMalloc(&d_kSubsetMasks, numKSubsets * sizeof(mask_t));
    cudaMalloc(&d_population, popBytes);
    cudaMalloc(&d_newPopulation, popBytes);
    cudaMalloc(&d_fitness, popSize * sizeof(int));
    cudaMalloc(&d_randStates, popSize * sizeof(curandState));
    
    cudaMemcpy(d_mSubsetMasks, h_mSubsetMasks, numMSubsets * sizeof(mask_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kSubsetMasks, h_kSubsetMasks, numKSubsets * sizeof(mask_t), cudaMemcpyHostToDevice);
    
    // Initialize
    int blocks = (popSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    printf("Initializing...\n");
    initRandStates<<<blocks, THREADS_PER_BLOCK>>>(d_randStates, time(NULL), popSize);
    initPopulation<<<blocks, THREADS_PER_BLOCK>>>(d_population, d_kSubsetMasks, d_randStates, popSize);
    cudaDeviceSynchronize();
    
    // Evolution loop
    printf("\nStarting evolution with local search...\n");
    clock_t startTime = clock();
    
    int bestFitnessEver = INT_MAX;
    int lastImprovement = 0;
    
    for (int gen = 0; gen < generations; gen++) {
        // Evaluate fitness
        evaluateFitness<<<blocks, THREADS_PER_BLOCK>>>(d_population, d_mSubsetMasks, d_fitness, popSize);
        cudaDeviceSynchronize();
        
        // Local search on each individual
        if (localSteps > 0) {
            if (steepest) {
                localSearchSteepest<<<blocks, THREADS_PER_BLOCK>>>(
                    d_population, d_mSubsetMasks, d_kSubsetMasks, d_fitness,
                    d_randStates, popSize, localSteps);
            } else {
                localSearch<<<blocks, THREADS_PER_BLOCK>>>(
                    d_population, d_mSubsetMasks, d_kSubsetMasks, d_fitness,
                    d_randStates, popSize, localSteps);
            }
            cudaDeviceSynchronize();
        }
        
        // Find best on host
        cudaMemcpy(h_fitness, d_fitness, popSize * sizeof(int), cudaMemcpyDeviceToHost);
        
        int bestFitness = INT_MAX;
        int bestIdx = 0;
        for (int i = 0; i < popSize; i++) {
            if (h_fitness[i] < bestFitness) {
                bestFitness = h_fitness[i];
                bestIdx = i;
            }
        }
        
        if (bestFitness < bestFitnessEver) {
            bestFitnessEver = bestFitness;
            lastImprovement = gen;
            
            cudaMemcpy(h_bestSolution, d_population + bestIdx * b, 
                       b * sizeof(mask_t), cudaMemcpyDeviceToHost);
            
            printf("Gen %5d: Best fitness = %d\n", gen, bestFitness);
            
            if (bestFitness == 0) {
                printf("\n*** PERFECT SOLUTION FOUND! ***\n");
                break;
            }
        }
        
        if (gen % 50 == 0 && gen > 0) {
            double elapsed = (double)(clock() - startTime) / CLOCKS_PER_SEC;
            printf("Gen %5d: Best = %d, Best ever = %d (%.1f gen/sec)\n", 
                   gen, bestFitness, bestFitnessEver, gen / elapsed);
        }
        
        // Restart if stuck
        if (gen - lastImprovement > 500) {
            printf("Restarting population (stuck for 500 generations)...\n");
            initPopulation<<<blocks, THREADS_PER_BLOCK>>>(d_population, d_kSubsetMasks, d_randStates, popSize);
            cudaDeviceSynchronize();
            lastImprovement = gen;
        }
        
        // Evolve
        evolve<<<blocks, THREADS_PER_BLOCK>>>(d_population, d_newPopulation, d_fitness,
                                              d_kSubsetMasks, d_randStates,
                                              popSize, tournamentSize, mutationRate, bestIdx);
        cudaDeviceSynchronize();
        
        // Swap
        mask_t* temp = d_population;
        d_population = d_newPopulation;
        d_newPopulation = temp;
    }
    
    double totalTime = (double)(clock() - startTime) / CLOCKS_PER_SEC;
    
    printf("\n=== Results ===\n");
    printf("Best fitness: %d uncovered m-subsets\n", bestFitnessEver);
    printf("Total time: %.2f seconds\n", totalTime);
    printf("Generations per second: %.2f\n", generations / totalTime);
    
    if (bestFitnessEver == 0) {
        printSolution(h_bestSolution, b);
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

