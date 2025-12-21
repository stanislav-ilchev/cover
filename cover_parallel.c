/*
 * cover_parallel.c - OpenMP-accelerated covering design search
 * 
 * This implements a parallel simulated annealing approach where
 * multiple independent SA processes run simultaneously across CPU cores.
 * Each thread executes a complete SA run, and the best solution wins.
 *
 * Compile with: 
 *   gcc -O3 -fopenmp -o cover_parallel cover_parallel.c -lm
 *   cl /O2 /openmp cover_parallel.c (MSVC)
 *
 * Usage: cover_parallel [options]
 *   All original cover options plus:
 *   TC=N or TestCount=N  - number of parallel runs (default: auto based on CPU cores x 100)
 *   threads=N            - number of threads (default: all cores)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#define GET_TIME() ((double)GetTickCount64() / 1000.0)
#else
#include <sys/time.h>
static double GET_TIME() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}
#endif

// Configuration
#define MAX_V 64
#define MAX_K 20
#define MAX_B 2048
#define MAX_COVER_LEN 32768
#define MAX_COVERED_LEN 100000

typedef unsigned int rankType;
typedef unsigned short coveredType;
typedef int costType;
typedef int costDType;
typedef unsigned char varietyType;
typedef uint64_t maskType;
typedef unsigned int binCoefType;

// Global binomial coefficients
static binCoefType binCoef[MAX_V + 1][MAX_V + 2];

// Problem parameters
static int g_v = 7, g_k = 3, g_t = 2, g_m = 2, g_b = 7;
static int g_coverNumber = 1;
static float g_coolFact = 0.99f, g_initProb = 0.5f;
static float g_initTemp = 0.0f;
static int g_initTempSet = 0;
static int g_frozen = 10, g_endLimit = 0;
static float g_LFact = 1.0f;
static int g_L = 0;
static int g_verbose = 2;
static int g_solX = 0;

// Optional seed solution
static char g_seedPath[512] = {0};
static int g_seedDrop = -1; // 1-based index to drop from seed, or -1
static int g_seedCount = 0;
static maskType* g_seedMasks = NULL;
static rankType* g_seedRanks = NULL;

// Derived parameters (computed once)
static int g_coverLen, g_coveredLen, g_neighborLen;

// Forward declarations
static int popcount_mask(maskType mask);
static rankType rankFromMask(maskType mask, int card, int v);

// Thread-safe random number generator (xorshift128+)
typedef struct {
    uint64_t s[2];
} xorshift128p_state;

static inline uint64_t xorshift128p(xorshift128p_state* state) {
    uint64_t s1 = state->s[0];
    const uint64_t s0 = state->s[1];
    state->s[0] = s0;
    s1 ^= s1 << 23;
    state->s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return state->s[1] + s0;
}

static inline void xorshift128p_init(xorshift128p_state* state, uint64_t seed) {
    state->s[0] = seed;
    state->s[1] = seed ^ 0x123456789ABCDEF0ULL;
    // Warm up
    for (int i = 0; i < 20; i++) xorshift128p(state);
}

static inline int rnd_r(xorshift128p_state* state, int n) {
    return (int)(xorshift128p(state) % n);
}

static inline double random01_r(xorshift128p_state* state) {
    return (double)xorshift128p(state) / (double)UINT64_MAX;
}

// Calculate binomial coefficients
static void calculateBinCoefs(void) {
    for (int v = 0; v <= MAX_V; v++) {
        binCoef[v][0] = binCoef[v][v] = 1;
        binCoef[v][v + 1] = 0;
        for (int k = 1; k <= v - 1; k++) {
            binCoef[v][k] = binCoef[v - 1][k - 1] + binCoef[v - 1][k];
            if (binCoef[v][k] < binCoef[v - 1][k - 1] ||
                binCoef[v][k] < binCoef[v - 1][k])
                binCoef[v][k] = 0;
        }
    }
}

// Load a seed solution from a file (whitespace-separated blocks)
static void loadSeedFile(void) {
    if (g_seedPath[0] == '\0') return;

    FILE* fp = fopen(g_seedPath, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open seed file: %s\n", g_seedPath);
        exit(1);
    }

    int capacity = 256;
    g_seedMasks = (maskType*)malloc(capacity * sizeof(maskType));
    if (!g_seedMasks) {
        fprintf(stderr, "Out of memory loading seed file\n");
        exit(1);
    }

    int count = 0;
    while (1) {
        int nums[MAX_K];
        int nread = 0;
        if (fscanf(fp, "%d", &nums[0]) != 1) break;
        nread = 1;
        for (int i = 1; i < g_k; i++) {
            if (fscanf(fp, "%d", &nums[i]) != 1) {
                fprintf(stderr, "Incomplete block in seed file\n");
                exit(1);
            }
            nread++;
        }
        if (nread != g_k) break;

        // Validate and build mask
        maskType mask = 0;
        for (int i = 0; i < g_k; i++) {
            if (nums[i] < 0 || nums[i] >= g_v) {
                fprintf(stderr, "Seed value %d out of range [0,%d)\n", nums[i], g_v);
                exit(1);
            }
            mask |= ((maskType)1 << nums[i]);
        }
        if (popcount_mask(mask) != g_k) {
            fprintf(stderr, "Duplicate value detected in seed block\n");
            exit(1);
        }

        if (count == capacity) {
            capacity *= 2;
            g_seedMasks = (maskType*)realloc(g_seedMasks, capacity * sizeof(maskType));
            if (!g_seedMasks) {
                fprintf(stderr, "Out of memory growing seed array\n");
                exit(1);
            }
        }
        g_seedMasks[count++] = mask;
    }

    fclose(fp);

    if (count == 0) {
        fprintf(stderr, "Seed file is empty: %s\n", g_seedPath);
        exit(1);
    }

    // If seed has more blocks than requested, drop one
    if (count > g_b) {
        int dropIdx = g_seedDrop - 1;
        if (dropIdx < 0 || dropIdx >= count) {
            dropIdx = count - 1;
        }
        int out = 0;
        for (int i = 0; i < count; i++) {
            if (i == dropIdx) continue;
            g_seedMasks[out++] = g_seedMasks[i];
            if (out == g_b) break;
        }
        count = g_b;
    }

    g_seedCount = count;
    g_seedRanks = (rankType*)malloc(g_seedCount * sizeof(rankType));
    if (!g_seedRanks) {
        fprintf(stderr, "Out of memory allocating seed ranks\n");
        exit(1);
    }
    for (int i = 0; i < g_seedCount; i++) {
        g_seedRanks[i] = rankFromMask(g_seedMasks[i], g_k, g_v);
    }
}

// Unrank subset
static void unrankSubset(rankType rank, varietyType* subset, int card) {
    int m = rank;
    for (int i = card - 1; i >= 0; i--) {
        int p = i;
        while (binCoef[p + 1][i + 1] <= (unsigned)m) p++;
        m -= binCoef[p][i + 1];
        subset[i] = (varietyType)p;
    }
}

// Rank subset
static rankType rankSubset(varietyType* subset, int card) {
    rankType rank = 0;
    for (int i = 0; i < card; i++)
        rank += binCoef[subset[i]][i + 1];
    return rank;
}

// Make complement
static void makeComplement(varietyType* s, varietyType* c, int v) {
    for (int i = 0; i < v; i++)
        if (*s == (varietyType)i)
            s++;
        else
            *c++ = (varietyType)i;
    *c = MAX_V + 1;
}

// Convert rank to bitmask
static maskType maskFromRank(rankType rank, int card) {
    varietyType subset[MAX_K + 1];
    unrankSubset(rank, subset, card);
    maskType mask = 0;
    for (int i = 0; i < card; i++)
        mask |= ((maskType)1 << subset[i]);
    return mask;
}

// Convert bitmask to rank
static rankType rankFromMask(maskType mask, int card, int v) {
    rankType rank = 0;
    int count = 0;
    for (int i = 0; i < v && count < card; i++) {
        if (mask & ((maskType)1 << i)) {
            count++;
            rank += binCoef[i][count];
        }
    }
    return rank;
}

// Select nth set bit
static int selectNthBit(maskType mask, int n, int v) {
    for (int i = 0; i < v; i++) {
        if (mask & ((maskType)1 << i)) {
            if (n == 0) return i;
            n--;
        }
    }
    return -1;
}

static int popcount_mask(maskType mask) {
    int count = 0;
    while (mask) {
        mask &= (mask - 1);
        count++;
    }
    return count;
}

// Random neighbor using bitmask
static rankType randomNeighborMask(maskType currMask, maskType* outMask,
                                    xorshift128p_state* rng, int k, int v) {
    maskType fullMask = (v == 64) ? ~(maskType)0 : (((maskType)1 << v) - 1);
    maskType comp = fullMask ^ currMask;
    
    int removeBit = selectNthBit(currMask, rnd_r(rng, k), v);
    int addBit = selectNthBit(comp, rnd_r(rng, v - k), v);
    
    *outMask = (currMask & ~(((maskType)1) << removeBit)) |
               (((maskType)1) << addBit);
    return rankFromMask(*outMask, k, v);
}

// Comparison for qsort
static int compareRanks(const void* a, const void* b) {
    rankType ra = *(const rankType*)a;
    rankType rb = *(const rankType*)b;
    return (ra > rb) - (ra < rb);
}

// Calculate coverings for one k-set using bitmask
static void calculateOneCoveringMask(maskType mask, rankType* buf, int k, int m, int t, int v) {
    varietyType subset[MAX_K + 1];
    varietyType csubset[MAX_V + 1];
    
    int idx = 0;
    for (int i = 0; i < v; i++)
        if (mask & ((maskType)1 << i))
            subset[idx++] = (varietyType)i;
    subset[k] = MAX_V + 1;
    
    idx = 0;
    for (int i = 0; i < v; i++)
        if (!(mask & ((maskType)1 << i)))
            csubset[idx++] = (varietyType)i;
    csubset[v - k] = MAX_V + 1;
    
    rankType* coverptr = buf;
    int minKM = (k < m) ? k : m;
    
    for (int ti = t; ti <= minKM; ti++) {
        varietyType subsubset[MAX_K + 1];
        for (int i = 0; i < ti; i++) subsubset[i] = (varietyType)i;
        subsubset[ti] = MAX_V + 1;
        
        do {
            varietyType subcsubset[MAX_K + 1];
            int mti = m - ti;
            for (int i = 0; i < mti; i++) subcsubset[i] = (varietyType)i;
            subcsubset[mti] = MAX_V + 1;
            
            do {
                varietyType mergeset[MAX_K + 1];
                int ss = 0, sc = 0;
                subsubset[ti] = (varietyType)k;
                subcsubset[mti] = (varietyType)(v - k);
                
                for (int i = 0; i < m; i++) {
                    if (subset[(int)subsubset[ss]] < csubset[(int)subcsubset[sc]])
                        mergeset[i] = subset[(int)subsubset[ss++]];
                    else
                        mergeset[i] = csubset[(int)subcsubset[sc++]];
                }
                subsubset[ti] = MAX_V + 1;
                subcsubset[mti] = MAX_V + 1;
                
                *coverptr++ = rankSubset(mergeset, m);
                
                // Next subcsubset
                if (mti == 0) break;
                int j = 0;
                while (j + 1 < mti && subcsubset[j + 1] <= subcsubset[j] + 1) j++;
                if (subcsubset[0] >= v - k - mti) break;
                subcsubset[j]++;
                for (int i = 0; i < j; i++) subcsubset[i] = (varietyType)i;
            } while (1);
            
            // Next subsubset
            if (ti == 0) break;
            int j = 0;
            while (j + 1 < ti && subsubset[j + 1] <= subsubset[j] + 1) j++;
            if (subsubset[0] >= k - ti) break;
            subsubset[j]++;
            for (int i = 0; i < j; i++) subsubset[i] = (varietyType)i;
        } while (1);
    }
    
    int len = (int)(coverptr - buf);
    *coverptr = binCoef[v][m]; // sentinel
    qsort(buf, len, sizeof(rankType), compareRanks);
}

// Single SA run structure
typedef struct {
    costType finalCost;
    rankType solution[MAX_B];
    int iterCount;
} SAResult;

// Run single simulated annealing (thread-safe)
static SAResult runSimulatedAnnealing(xorshift128p_state* rng, 
                                       int v, int k, int t, int m, int b,
                                       int coverNumber, float coolFact, float initProb,
                                       float initTemp, int initTempSet,
                                       int L, int frozen, int endLimit, int coverLen) {
    SAResult result;
    memset(&result, 0, sizeof(SAResult));
    result.iterCount = 0;
    result.finalCost = 999999;  // Initialize to high cost
    
    int coveredLen = binCoef[v][m];
    
    // Heap-allocated arrays (too large for stack)
    rankType* kset = (rankType*)malloc(b * sizeof(rankType));
    maskType* ksetMask = (maskType*)malloc(b * sizeof(maskType));
    coveredType* covered = (coveredType*)malloc(coveredLen * sizeof(coveredType));
    costType* costs = (costType*)malloc((b + 1) * sizeof(costType));
    costDType* costds = (costDType*)malloc((b + 1) * sizeof(costDType));
    rankType* currCoverings = (rankType*)malloc(coverLen * sizeof(rankType));
    rankType* nextCoverings = (rankType*)malloc(coverLen * sizeof(rankType));
    
    if (!kset || !ksetMask || !covered || !costs || !costds || !currCoverings || !nextCoverings) {
        result.finalCost = 999999;
        free(kset); free(ksetMask); free(covered); free(costs); free(costds);
        free(currCoverings); free(nextCoverings);
        return result;
    }
    
    // Calculate costs (covering design)
    for (int i = 0; i <= b; i++) {
        if (i < coverNumber)
            costs[i] = (costType)(coverNumber - i);
        else
            costs[i] = 0;
    }
    for (int i = 0; i < b; i++)
        costds[i] = costs[i] - costs[i + 1];
    
    // Initialize covered array
    memset(covered, 0, coveredLen * sizeof(coveredType));
    
    // Generate initial solution (seeded if provided)
    for (int i = 0; i < b; i++) {
        if (g_seedCount > 0 && i < g_seedCount) {
            kset[i] = g_seedRanks[i];
            ksetMask[i] = g_seedMasks[i];
        } else {
            kset[i] = rnd_r(rng, binCoef[v][k]);
            ksetMask[i] = maskFromRank(kset[i], k);
        }
        
        calculateOneCoveringMask(ksetMask[i], currCoverings, k, m, t, v);
        for (int j = 0; currCoverings[j] != binCoef[v][m]; j++)
            covered[currCoverings[j]]++;
    }
    
    costType currCost = 0;
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
            int setNumber = rnd_r(rng, b);
            maskType nextMask;
            randomNeighborMask(ksetMask[setNumber], &nextMask, rng, k, v);
            
            calculateOneCoveringMask(ksetMask[setNumber], currCoverings, k, m, t, v);
            calculateOneCoveringMask(nextMask, nextCoverings, k, m, t, v);
            
            costType costDelta = 0;
            rankType* cp = currCoverings;
            rankType* np = nextCoverings;
            rankType sentinel = binCoef[v][m];
            
            while (*cp != sentinel || *np != sentinel) {
                if (*cp == *np) {
                    if (*cp == sentinel) break;
                    cp++; np++;
                } else if (*cp < *np) {
                    costDelta += costds[covered[*cp] - 1];
                    cp++;
                } else {
                    costDelta -= costds[covered[*np]];
                    np++;
                }
            }
            
            if (costDelta > 0) {
                m2++;
                T += (float)(-costDelta);
            }
        }
        T = (m2 == 0) ? 1.0f : T / m2 / logf(initProb);
    }
    
    // Main simulated annealing loop
    costType bestSeen = currCost;
    int notChanged = 0;
    
    while (notChanged < frozen) {
        costType lastCost = currCost;
        
        for (int iter = 0; iter < L; iter++) {
            result.iterCount++;
            
            int setNumber = rnd_r(rng, b);
            maskType nextMask;
            rankType nextS = randomNeighborMask(ksetMask[setNumber], &nextMask, rng, k, v);
            
            // Compute cost delta
            calculateOneCoveringMask(ksetMask[setNumber], currCoverings, k, m, t, v);
            calculateOneCoveringMask(nextMask, nextCoverings, k, m, t, v);
            
            costType costDelta = 0;
            rankType* cp = currCoverings;
            rankType* np = nextCoverings;
            rankType sentinel = binCoef[v][m];
            
            while (*cp != sentinel || *np != sentinel) {
                if (*cp == *np) {
                    if (*cp == sentinel) break;
                    cp++; np++;
                } else if (*cp < *np) {
                    costDelta += costds[covered[*cp] - 1];
                    cp++;
                } else {
                    costDelta -= costds[covered[*np]];
                    np++;
                }
            }
            
            // Accept or reject
            int accept = 0;
            if (costDelta <= 0) {
                accept = 1;
            } else {
                double r = random01_r(rng);
                if (r < exp(-costDelta / T))
                    accept = 1;
            }
            
            if (accept) {
                // Update covered counts - remove old
                calculateOneCoveringMask(ksetMask[setNumber], currCoverings, k, m, t, v);
                for (int j = 0; currCoverings[j] != binCoef[v][m]; j++)
                    covered[currCoverings[j]]--;
                
                // Add new
                calculateOneCoveringMask(nextMask, nextCoverings, k, m, t, v);
                for (int j = 0; nextCoverings[j] != binCoef[v][m]; j++)
                    covered[nextCoverings[j]]++;
                
                kset[setNumber] = nextS;
                ksetMask[setNumber] = nextMask;
                currCost += costDelta;
                
                if (costDelta < 0) {
                    notChanged = 0;
                    if (currCost < bestSeen)
                        bestSeen = currCost;
                }
                
                if (currCost <= endLimit) {
                    result.finalCost = currCost;
                    memcpy(result.solution, kset, b * sizeof(rankType));
                    goto cleanup;
                }
            }
        }
        
        if (lastCost <= currCost)
            notChanged++;
        T *= coolFact;
    }
    
    result.finalCost = currCost;
    memcpy(result.solution, kset, b * sizeof(rankType));
    
cleanup:
    free(kset);
    free(ksetMask);
    free(covered);
    free(costs);
    free(costds);
    free(currCoverings);
    free(nextCoverings);
    return result;
}

// Print subset
static void printSubset(FILE* fp, rankType r, int card, int v) {
    varietyType set[MAX_K + 1];
    unrankSubset(r, set, card);
    if (g_solX) {
        varietyType* vptr = set;
        for (int i = 0; i < v; i++)
            if (*vptr == i) {
                fprintf(fp, "X");
                vptr++;
            } else
                fprintf(fp, "-");
    } else {
        for (int i = 0; i < card; i++)
            fprintf(fp, "%d ", set[i]);
    }
}

// Parse command line
static void parseArguments(int argc, char** argv, int* testCount, int* numThreads) {
    *testCount = 0;  // Will be auto-set based on cores
    *numThreads = 0; // Will use all cores
    
    for (int i = 1; i < argc; i++) {
        char* arg = argv[i];
        if (strncmp(arg, "v=", 2) == 0) g_v = atoi(arg + 2);
        else if (strncmp(arg, "k=", 2) == 0) g_k = atoi(arg + 2);
        else if (strncmp(arg, "t=", 2) == 0) g_t = atoi(arg + 2);
        else if (strncmp(arg, "m=", 2) == 0) g_m = atoi(arg + 2);
        else if (strncmp(arg, "b=", 2) == 0) g_b = atoi(arg + 2);
        else if (strncmp(arg, "l=", 2) == 0) g_coverNumber = atoi(arg + 2);
        else if (strncmp(arg, "TC=", 3) == 0) *testCount = atoi(arg + 3);
        else if (strncmp(arg, "TestCount=", 10) == 0) *testCount = atoi(arg + 10);
        else if (strncmp(arg, "CF=", 3) == 0) g_coolFact = (float)atof(arg + 3);
        else if (strncmp(arg, "CoolingFactor=", 14) == 0) g_coolFact = (float)atof(arg + 14);
        else if (strncmp(arg, "IP=", 3) == 0) g_initProb = (float)atof(arg + 3);
        else if (strncmp(arg, "InitProb=", 9) == 0) g_initProb = (float)atof(arg + 9);
        else if (strncmp(arg, "IT=", 3) == 0) { g_initTemp = (float)atof(arg + 3); g_initTempSet = 1; }
        else if (strncmp(arg, "InitTemp=", 9) == 0) { g_initTemp = (float)atof(arg + 9); g_initTempSet = 1; }
        else if (strncmp(arg, "frozen=", 7) == 0) g_frozen = atoi(arg + 7);
        else if (strncmp(arg, "EL=", 3) == 0) g_endLimit = atoi(arg + 3);
        else if (strncmp(arg, "EndLimit=", 9) == 0) g_endLimit = atoi(arg + 9);
        else if (strncmp(arg, "LF=", 3) == 0) g_LFact = (float)atof(arg + 3);
        else if (strncmp(arg, "LFact=", 6) == 0) g_LFact = (float)atof(arg + 6);
        else if (strncmp(arg, "L=", 2) == 0) g_L = atoi(arg + 2);
        else if (strncmp(arg, "verbose=", 8) == 0) g_verbose = atoi(arg + 8);
        else if (strncmp(arg, "threads=", 8) == 0) *numThreads = atoi(arg + 8);
        else if (strncmp(arg, "SolX=", 5) == 0) g_solX = atoi(arg + 5);
        else if (strncmp(arg, "SX=", 3) == 0) g_solX = atoi(arg + 3);
        else if (strncmp(arg, "seed=", 5) == 0) {
            strncpy(g_seedPath, arg + 5, sizeof(g_seedPath) - 1);
            g_seedPath[sizeof(g_seedPath) - 1] = '\0';
        }
        else if (strncmp(arg, "seedDrop=", 9) == 0) g_seedDrop = atoi(arg + 9);
        else if (strncmp(arg, "start=", 6) == 0) {
            strncpy(g_seedPath, arg + 6, sizeof(g_seedPath) - 1);
            g_seedPath[sizeof(g_seedPath) - 1] = '\0';
        }
        else if (strncmp(arg, "startDrop=", 10) == 0) g_seedDrop = atoi(arg + 10);
    }
    
    // If m not specified, assume m = t
    if (g_m == 2 && g_t != 2) g_m = g_t;
}

int main(int argc, char** argv) {
    int testCount, numThreads;
    
    parseArguments(argc, argv, &testCount, &numThreads);
    
    // Detect number of cores
#ifdef _OPENMP
    int maxThreads = omp_get_max_threads();
#else
    int maxThreads = 1;
#endif
    
    if (numThreads <= 0) numThreads = maxThreads;
    if (testCount <= 0) testCount = numThreads * 100;  // Default: 100 runs per core
    
#ifdef _OPENMP
    omp_set_num_threads(numThreads);
#endif
    
    printf("\n");
    printf("cover_parallel - Multi-threaded covering design search\n");
    printf("=======================================================\n\n");
    printf("Using %d threads with %d parallel SA runs\n\n", numThreads, testCount);
    printf("Design parameters:\n");
    printf("------------------\n");
    printf("t - (v,m,k,l) = %d - (%d,%d,%d,%d)\n", g_t, g_v, g_m, g_k, g_coverNumber);
    printf("b = %d\n\n", g_b);
    if (g_seedCount > 0) {
        printf("Seed file: %s (%d seeded, %d random)\n\n",
               g_seedPath, g_seedCount, g_b - g_seedCount);
    }
    
    // Calculate binomial coefficients
    calculateBinCoefs();
    loadSeedFile();
    
    // Calculate derived parameters
    g_neighborLen = g_k * (g_v - g_k);
    g_coverLen = 0;
    int minKMT = (g_k - g_t < g_m - g_t) ? g_k - g_t : g_m - g_t;
    for (int i = 0; i <= minKMT; i++)
        g_coverLen += binCoef[g_k][g_t + i] * binCoef[g_v - g_k][g_m - g_t - i];
    g_coverLen++;
    g_coveredLen = binCoef[g_v][g_m];
    
    if (g_L <= 0)
        g_L = (int)(g_LFact * g_k * (g_v - g_k) * g_b + 0.5);
    
    printf("Optimization parameters:\n");
    printf("------------------------\n");
    printf("CoolingFactor = %.4f\n", g_coolFact);
    if (g_initTempSet)
        printf("InitTemp      = %.3f\n", g_initTemp);
    else
        printf("InitProb      = %.2f\n", g_initProb);
    printf("frozen        = %d\n", g_frozen);
    printf("L             = %d\n", g_L);
    printf("EndLimit      = %d\n\n", g_endLimit);
    
    // Allocate results array
    SAResult* results = (SAResult*)calloc(testCount, sizeof(SAResult));
    
    double startTime = GET_TIME();
    
    // Run parallel SA
    int completed = 0;
    int earlyExit = 0;
    
#pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        
        // Initialize thread-local RNG
        xorshift128p_state rng;
        xorshift128p_init(&rng, (uint64_t)time(NULL) ^ ((uint64_t)tid << 32) ^ 
                          ((uint64_t)tid * 0x9E3779B97F4A7C15ULL));
        
#pragma omp for schedule(dynamic)
        for (int run = 0; run < testCount; run++) {
            if (earlyExit) continue;  // Skip if solution found
            
            results[run] = runSimulatedAnnealing(&rng, g_v, g_k, g_t, g_m, g_b,
                                                  g_coverNumber, g_coolFact, g_initProb,
                                                  g_initTemp, g_initTempSet,
                                                  g_L, g_frozen, g_endLimit, g_coverLen);
            
#pragma omp atomic
            completed++;
            
            if (g_verbose >= 2) {
#pragma omp critical
                {
                    printf("Run %d/%d: cost=%d\n", completed, testCount, results[run].finalCost);
                }
            }
            
            // Early exit if perfect solution found
            if (results[run].finalCost <= g_endLimit) {
#pragma omp atomic write
                earlyExit = 1;
            }
        }
    }
    
    double endTime = GET_TIME();
    double elapsed = endTime - startTime;
    
    // Find best result
    int bestIdx = 0;
    costType bestCost = results[0].finalCost;
    long long totalIters = results[0].iterCount;
    
    for (int i = 1; i < testCount; i++) {
        totalIters += results[i].iterCount;
        if (results[i].finalCost < bestCost) {
            bestCost = results[i].finalCost;
            bestIdx = i;
        }
    }
    
    // Count successes
    int successes = 0;
    costType costSum = 0;
    for (int i = 0; i < testCount; i++) {
        costSum += results[i].finalCost;
        if (results[i].finalCost <= g_endLimit)
            successes++;
    }
    
    printf("\nResults:\n");
    printf("--------\n");
    printf("Total runs:      %d\n", testCount);
    printf("Best cost:       %d\n", bestCost);
    printf("Average cost:    %.2f\n", (float)costSum / testCount);
    printf("Success rate:    %d/%d (%.1f%%)\n", successes, testCount, 
           100.0 * successes / testCount);
    printf("Total time:      %.2f seconds\n", elapsed);
    printf("Runs/second:     %.1f\n", testCount / elapsed);
    printf("Total iterations: %lld\n", totalIters);
    printf("Iter/second:     %.0f\n\n", totalIters / elapsed);
    
    if (bestCost <= g_endLimit) {
        printf("Solution found (from run %d):\n", bestIdx);
        printf("-----------------------------\n");
        for (int i = 0; i < g_b; i++) {
            printSubset(stdout, results[bestIdx].solution[i], g_k, g_v);
            printf("\n");
        }
        
        // Save to file
        FILE* fp = fopen("cover.res", "w");
        if (fp) {
            for (int i = 0; i < g_b; i++) {
                printSubset(fp, results[bestIdx].solution[i], g_k, g_v);
                fprintf(fp, "\n");
            }
            fclose(fp);
            printf("\nSolution saved to cover.res\n");
        }
    } else {
        printf("EndLimit not reached.\n");
    }
    
    // Estimate speedup
    printf("\nPerformance estimate:\n");
    printf("---------------------\n");
    printf("Parallel speedup vs single-threaded: ~%dx\n", numThreads);
    printf("Total equivalent sequential runs: %d\n", testCount);
    printf("Effective work speedup: ~%dx\n", testCount);
    
    free(g_seedMasks);
    free(g_seedRanks);
    free(results);
    
    return bestCost <= g_endLimit ? 0 : 1;
}
