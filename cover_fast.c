/*
** cover_fast.c - Bitmask-optimized covering design finder using simulated annealing
**
** This is a high-performance version of Nurmela's cover program using:
** - Bitmask representation for all subsets (32-bit or 64-bit)
** - Hardware popcount for O(1) intersection checks
** - Gosper's hack for efficient subset enumeration
** - Optimized neighbor generation with bit manipulation
**
** For v <= 32, uses 32-bit masks. For v <= 64, uses 64-bit masks.
** Significantly faster than rank-based approach for coverage checking.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#define GET_TIME_MS() (GetTickCount64())
#else
#include <sys/time.h>
static inline uint64_t GET_TIME_MS(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
#endif

/* Type definitions */
typedef uint64_t mask_t;  /* Supports v up to 64 */
typedef int32_t cost_t;

/* Portable popcount */
#if defined(__GNUC__) || defined(__clang__)
#define POPCOUNT(x) __builtin_popcountll(x)
#elif defined(_MSC_VER)
#include <intrin.h>
#define POPCOUNT(x) (int)__popcnt64(x)
#else
static inline int popcount64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
}
#define POPCOUNT(x) popcount64(x)
#endif

/* Xorshift128+ PRNG for speed */
typedef struct {
    uint64_t s[2];
} xorshift_state;

static inline uint64_t xorshift128plus(xorshift_state* state) {
    uint64_t s1 = state->s[0];
    const uint64_t s0 = state->s[1];
    state->s[0] = s0;
    s1 ^= s1 << 23;
    state->s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return state->s[1] + s0;
}

static inline uint32_t rnd(xorshift_state* state, uint32_t n) {
    return (uint32_t)(xorshift128plus(state) % n);
}

static inline double rnd01(xorshift_state* state) {
    return (double)(xorshift128plus(state) >> 11) * (1.0 / 9007199254740992.0);
}

/* Global parameters */
static int v = 27, k = 6, m = 4, t = 3, b = 86;
static int testCount = 1;
static double coolFact = 0.999;
static double initProb = 0.5;
static double initialT = 0.0;
static int Tset = 0;
static int frozen = 1000000000;
static int endLimit = 0;
static int L = 0;
static double LFact = 1.0;
static int verbose = 2;

/* Precomputed tables */
static mask_t* allMSubsets = NULL;   /* All C(v,m) m-subsets as bitmasks */
static int numMSubsets = 0;
static mask_t fullMask = 0;          /* Mask with v bits set */

/* Current solution */
static mask_t* blocks = NULL;        /* Current b k-subsets as bitmasks */
static uint8_t* covered = NULL;      /* covered[i] = how many blocks cover m-subset i */
static cost_t* costs = NULL;         /* costs[x] = cost for x-times covered */

/* Binomial coefficients */
static uint32_t binCoef[65][65];

static void calculateBinCoefs(void) {
    int n, r;
    for (n = 0; n <= 64; n++) {
        binCoef[n][0] = 1;
        binCoef[n][n] = 1;
        for (r = 1; r < n; r++) {
            uint64_t val = (uint64_t)binCoef[n-1][r-1] + binCoef[n-1][r];
            binCoef[n][r] = (val > 0xFFFFFFFF) ? 0xFFFFFFFF : (uint32_t)val;
        }
    }
}

/* Generate all k-subsets of v elements using Gosper's hack */
static int generateAllSubsets(int n, int r, mask_t** outMasks) {
    int count = binCoef[n][r];
    mask_t* masks = (mask_t*)malloc(count * sizeof(mask_t));
    if (!masks) return 0;
    
    mask_t mask = ((mask_t)1 << r) - 1;  /* First r-subset: bits 0..r-1 */
    int idx = 0;
    
    mask_t limit = (mask_t)1 << n;
    while (mask < limit && idx < count) {
        masks[idx++] = mask;
        /* Gosper's hack: get next r-subset */
        mask_t c = mask & -(int64_t)mask;
        mask_t r2 = mask + c;
        mask = (((r2 ^ mask) >> 2) / c) | r2;
    }
    
    *outMasks = masks;
    return idx;
}

/* Select a random bit from a mask */
static inline int selectRandomBit(xorshift_state* rng, mask_t mask, int numBits) {
    int which = rnd(rng, numBits);
    int count = 0;
    for (int i = 0; i < 64; i++) {
        if (mask & ((mask_t)1 << i)) {
            if (count == which) return i;
            count++;
        }
    }
    return -1;  /* Should never happen */
}

/* Generate a random k-subset mask */
static inline mask_t randomKSubset(xorshift_state* rng) {
    mask_t mask = 0;
    int remaining = k;
    for (int i = 0; i < v && remaining > 0; i++) {
        /* Include bit i with probability remaining/(v-i) */
        if (rnd(rng, v - i) < (uint32_t)remaining) {
            mask |= ((mask_t)1 << i);
            remaining--;
        }
    }
    return mask;
}

/* Generate a random neighbor of a k-subset */
static inline mask_t randomNeighbor(xorshift_state* rng, mask_t curr) {
    mask_t comp = fullMask ^ curr;  /* Complement: elements not in curr */
    
    /* Remove a random element from curr */
    int removeBit = selectRandomBit(rng, curr, k);
    
    /* Add a random element from complement */
    int addBit = selectRandomBit(rng, comp, v - k);
    
    return (curr & ~((mask_t)1 << removeBit)) | ((mask_t)1 << addBit);
}

/* Check if m-subset is covered by block with intersection >= t */
static inline int isCovered(mask_t mSubset, mask_t block) {
    return POPCOUNT(mSubset & block) >= t;
}

/* Calculate initial costs table */
static void calculateCosts(void) {
    costs = (cost_t*)malloc((b + 1) * sizeof(cost_t));
    for (int i = 0; i <= b; i++) {
        costs[i] = (i == 0) ? 1 : 0;  /* Cost 1 if uncovered, 0 otherwise */
    }
}

/* Compute cost of current solution from scratch */
static cost_t computeCostFromScratch(void) {
    cost_t cost = 0;
    
    /* Reset covered array */
    memset(covered, 0, numMSubsets * sizeof(uint8_t));
    
    /* For each m-subset, check if it's covered by any block */
    for (int i = 0; i < numMSubsets; i++) {
        mask_t mSub = allMSubsets[i];
        for (int j = 0; j < b; j++) {
            if (isCovered(mSub, blocks[j])) {
                covered[i] = 1;
                break;
            }
        }
        if (covered[i] == 0) cost++;
    }
    
    return cost;
}

/* Compute cost delta for replacing blocks[blockIdx] with newBlock */
static cost_t computeCostDelta(int blockIdx, mask_t newBlock) {
    mask_t oldBlock = blocks[blockIdx];
    cost_t delta = 0;
    
    /* For each m-subset that might be affected */
    for (int i = 0; i < numMSubsets; i++) {
        mask_t mSub = allMSubsets[i];
        
        int oldCovers = isCovered(mSub, oldBlock);
        int newCovers = isCovered(mSub, newBlock);
        
        if (oldCovers && !newCovers) {
            /* Lost coverage from oldBlock - check if still covered by others */
            int stillCovered = 0;
            for (int j = 0; j < b; j++) {
                if (j != blockIdx && isCovered(mSub, blocks[j])) {
                    stillCovered = 1;
                    break;
                }
            }
            if (!stillCovered) delta++;  /* m-subset becomes uncovered */
        }
        else if (!oldCovers && newCovers) {
            /* Gained coverage from newBlock */
            if (covered[i] == 0) delta--;  /* m-subset becomes covered */
        }
    }
    
    return delta;
}

/* Accept a neighbor move: replace blocks[blockIdx] with newBlock */
static void acceptNeighbor(int blockIdx, mask_t newBlock) {
    mask_t oldBlock = blocks[blockIdx];
    
    /* Update covered array */
    for (int i = 0; i < numMSubsets; i++) {
        mask_t mSub = allMSubsets[i];
        
        int oldCovers = isCovered(mSub, oldBlock);
        int newCovers = isCovered(mSub, newBlock);
        
        if (oldCovers && !newCovers) {
            /* Check if still covered by others */
            int stillCovered = 0;
            for (int j = 0; j < b; j++) {
                if (j != blockIdx && isCovered(mSub, blocks[j])) {
                    stillCovered = 1;
                    break;
                }
            }
            covered[i] = stillCovered;
        }
        else if (!oldCovers && newCovers) {
            covered[i] = 1;
        }
    }
    
    blocks[blockIdx] = newBlock;
}

/* Initialize solution with random blocks */
static cost_t initSolution(xorshift_state* rng) {
    for (int i = 0; i < b; i++) {
        blocks[i] = randomKSubset(rng);
    }
    return computeCostFromScratch();
}

/* Approximate initial temperature */
static double approxInitT(xorshift_state* rng, cost_t currCost) {
    double T = 0.0;
    int m2 = 0;
    
    for (int i = 0; i < 300; i++) {
        int blockIdx = rnd(rng, b);
        mask_t newBlock = randomNeighbor(rng, blocks[blockIdx]);
        cost_t delta = computeCostDelta(blockIdx, newBlock);
        
        if (delta > 0) {
            m2++;
            T += (double)delta;
        }
    }
    
    if (m2 == 0) return 1.0;
    return T / m2 / (-log(initProb));
}

/* Main simulated annealing function */
static cost_t simulatedAnnealing(xorshift_state* rng) {
    cost_t currCost = initSolution(rng);
    cost_t bestCost = currCost;
    
    if (verbose) {
        printf("initCost      = %d\n", currCost);
    }
    
    double T;
    if (Tset) {
        T = initialT;
    } else {
        T = approxInitT(rng, currCost);
    }
    
    if (verbose) {
        printf("initTemp      = %.4f\n", T);
        printf("\nStarting annealing...\n\n");
    }
    
    int iterLength = L;
    int notChanged = 0;
    cost_t lastCost = currCost;
    uint64_t iterCounter = 0;
    
    if (verbose >= 2) {
        printf("      T      cost   best\n");
        printf("    ---------------------\n");
    }
    
    while (notChanged < frozen) {
        int improved = 0;
        
        for (int i = 0; i < iterLength; i++) {
            int blockIdx = rnd(rng, b);
            mask_t newBlock = randomNeighbor(rng, blocks[blockIdx]);
            cost_t delta = computeCostDelta(blockIdx, newBlock);
            iterCounter++;
            
            int accept = 0;
            if (delta <= 0) {
                accept = 1;
                if (delta < 0) {
                    notChanged = 0;
                    improved = 1;
                }
            } else {
                double prob = exp(-delta / T);
                if (rnd01(rng) < prob) {
                    accept = 1;
                }
            }
            
            if (accept) {
                acceptNeighbor(blockIdx, newBlock);
                currCost += delta;
                
                if (currCost < bestCost) {
                    bestCost = currCost;
                }
                
                if (currCost <= endLimit) {
                    if (verbose >= 2) printf("\n");
                    if (verbose) {
                        printf("...annealing accomplished.\n\n");
                        printf("Iterations    = %llu\n", (unsigned long long)iterCounter);
                    }
                    return currCost;
                }
            }
        }
        
        if (!improved) {
            notChanged++;
        }
        
        if (verbose >= 2) {
            printf("    %5.2f   %4d   %4d\n", T, currCost, bestCost);
        }
        
        T *= coolFact;
        lastCost = currCost;
    }
    
    if (verbose >= 2) printf("\n");
    if (verbose) {
        printf("...annealing accomplished.\n\n");
        printf("Iterations    = %llu\n", (unsigned long long)iterCounter);
    }
    
    return currCost;
}

/* Print solution */
static void printSolution(FILE* fp) {
    for (int j = 0; j < b; j++) {
        mask_t mask = blocks[j];
        for (int i = 0; i < v; i++) {
            if (mask & ((mask_t)1 << i)) {
                fprintf(fp, "%d ", i);
            }
        }
        fprintf(fp, "\n");
    }
}

/* Parse command line arguments */
static void parseArguments(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "v=", 2) == 0) v = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "k=", 2) == 0) k = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "m=", 2) == 0) m = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "t=", 2) == 0) t = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "b=", 2) == 0) b = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "TC=", 3) == 0) testCount = atoi(argv[i] + 3);
        else if (strncmp(argv[i], "CF=", 3) == 0 || strncmp(argv[i], "CoolingFactor=", 14) == 0) {
            coolFact = atof(strchr(argv[i], '=') + 1);
        }
        else if (strncmp(argv[i], "IT=", 3) == 0 || strncmp(argv[i], "InitTemp=", 9) == 0) {
            initialT = atof(strchr(argv[i], '=') + 1);
            Tset = 1;
        }
        else if (strncmp(argv[i], "IP=", 3) == 0 || strncmp(argv[i], "InitProb=", 9) == 0) {
            initProb = atof(strchr(argv[i], '=') + 1);
        }
        else if (strncmp(argv[i], "frozen=", 7) == 0) frozen = atoi(argv[i] + 7);
        else if (strncmp(argv[i], "EL=", 3) == 0 || strncmp(argv[i], "EndLimit=", 9) == 0) {
            endLimit = atoi(strchr(argv[i], '=') + 1);
        }
        else if (strncmp(argv[i], "L=", 2) == 0) L = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "LFact=", 6) == 0) LFact = atof(argv[i] + 6);
        else if (strncmp(argv[i], "verbose=", 8) == 0) verbose = atoi(argv[i] + 8);
    }
}

int main(int argc, char** argv) {
    xorshift_state rng;
    
    /* Initialize RNG */
    rng.s[0] = (uint64_t)time(NULL) ^ 0x123456789ABCDEFULL;
    rng.s[1] = (uint64_t)time(NULL) ^ 0xFEDCBA987654321ULL;
    for (int i = 0; i < 20; i++) xorshift128plus(&rng);  /* Warm up */
    
    parseArguments(argc, argv);
    
    if (verbose) {
        printf("\n");
        printf("cover_fast - Bitmask-optimized covering design finder\n");
        printf("=====================================================\n\n");
    }
    
    /* Validate parameters */
    if (v > 64) {
        fprintf(stderr, "ERROR: v=%d exceeds maximum of 64 for bitmask representation\n", v);
        return 1;
    }
    
    /* Initialize */
    calculateBinCoefs();
    
    /* Set full mask */
    fullMask = ((mask_t)1 << v) - 1;
    if (v == 64) fullMask = ~(mask_t)0;
    
    /* Compute iteration length if not set */
    if (L == 0) {
        L = (int)(LFact * k * (v - k) * b + 0.5);
    }
    
    if (verbose) {
        printf("Design parameters:\n");
        printf("------------------\n");
        printf("t - (v,m,k,1) = %d - (%d,%d,%d,1)\n", t, v, m, k);
        printf("b = %d\n\n", b);
        printf("Optimization parameters:\n");
        printf("------------------------\n");
        printf("TestCount     = %d\n", testCount);
        printf("CoolingFactor = %.4f\n", coolFact);
        if (Tset)
            printf("InitTemp      = %.3f\n", initialT);
        else
            printf("InitProb      = %.2f\n", initProb);
        printf("frozen        = %d\n", frozen);
        printf("EndLimit      = %d\n", endLimit);
        printf("L             = %d\n\n", L);
    }
    
    /* Generate all m-subsets */
    numMSubsets = generateAllSubsets(v, m, &allMSubsets);
    if (numMSubsets != (int)binCoef[v][m]) {
        fprintf(stderr, "ERROR: Generated %d m-subsets, expected %d\n", 
                numMSubsets, binCoef[v][m]);
        return 1;
    }
    
    if (verbose) {
        printf("Memory allocation:\n");
        printf("------------------\n");
        printf("m-subsets:     %d masks (%lu bytes)\n", 
               numMSubsets, (unsigned long)(numMSubsets * sizeof(mask_t)));
        printf("blocks:        %d masks (%lu bytes)\n",
               b, (unsigned long)(b * sizeof(mask_t)));
        printf("covered:       %d bytes\n\n", numMSubsets);
    }
    
    /* Allocate solution arrays */
    blocks = (mask_t*)malloc(b * sizeof(mask_t));
    covered = (uint8_t*)calloc(numMSubsets, sizeof(uint8_t));
    calculateCosts();
    
    if (!blocks || !covered || !costs) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return 1;
    }
    
    /* Run simulated annealing */
    uint64_t startTime = GET_TIME_MS();
    
    cost_t bestCost = -1;
    int solFound = 0;
    
    for (int run = 0; run < testCount; run++) {
        cost_t finalCost = simulatedAnnealing(&rng);
        
        if (bestCost == -1 || finalCost < bestCost) {
            bestCost = finalCost;
        }
        
        if (finalCost <= endLimit) {
            solFound = 1;
            
            if (verbose) {
                printf("Result:\n");
                printf("-------\n");
                printf("finalCost     = %d\n\n", finalCost);
                printf("Solution:\n");
                printf("---------\n");
                printSolution(stdout);
            }
            
            /* Write to file */
            FILE* fp = fopen("cover_fast.res", "w");
            if (fp) {
                printSolution(fp);
                fclose(fp);
            }
            
            break;
        } else {
            if (verbose) {
                printf("Result:\n");
                printf("-------\n");
                printf("finalCost     = %d\n", finalCost);
                printf("EndLimit was not reached.\n\n");
            }
        }
    }
    
    uint64_t endTime = GET_TIME_MS();
    
    if (verbose) {
        printf("Statistics:\n");
        printf("-----------\n");
        printf("bestCost      = %d\n", bestCost);
        printf("Wall time     = %.2f sec\n", (endTime - startTime) / 1000.0);
    }
    
    /* Cleanup */
    free(allMSubsets);
    free(blocks);
    free(covered);
    free(costs);
    
    return !solFound;
}

