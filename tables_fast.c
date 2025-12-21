/*   tables_fast.c
**
**   OPTIMIZED VERSION with precomputed m-subset masks and popcount.
**   
**   Key optimization: calculateOneCoveringMask() uses precomputed m-subset
**   masks and hardware popcount instead of constructing/ranking/sorting.
**   This eliminates qsort and is ~4x faster for on-the-fly mode.
*/

#include <string.h>
#include <stdlib.h>
#include "cover.h"
#include "bincoef.h"
#include "setoper.h"
#include "tables.h"

/* Portable popcount for 64-bit */
#if defined(__GNUC__) || defined(__clang__)
#define POPCOUNT64(x) __builtin_popcountll(x)
#elif defined(_MSC_VER)
#include <intrin.h>
#define POPCOUNT64(x) (int)__popcnt64(x)
#else
static inline int popcount64_fallback(unsigned long long x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
}
#define POPCOUNT64(x) popcount64_fallback(x)
#endif

rankType *kset = NULL;
maskType *ksetMask = NULL;
int neighborLen, coverLen, coveredLen;
rankType *neighbors, *coverings;
rankType *ksetCoverings = NULL;
int onTheFlyCache = 0;
coveredType *covered;
costType *costs;
costDType *costds;
static unsigned long tableBytes = 0;

/* NEW: Precomputed m-subset masks for fast covering computation */
static maskType *mSubsetMasks = NULL;
static int useFastCovering = 0;

/* Flag to use mask-based neighbor generation (avoids neighbors table) */
int useMaskNeighbors = 0;


/*
** generateMSubsetMasks() generates all C(v,m) m-subsets as bitmasks
** in rank order using Gosper's hack.
*/
static void generateMSubsetMasks(void)
{
    int count = binCoef[v][m];
    mSubsetMasks = (maskType *) malloc(count * sizeof(maskType));
    if (!mSubsetMasks) {
        useFastCovering = 0;
        return;
    }
    
    maskType mask = ((maskType)1 << m) - 1;  /* First m-subset: bits 0..m-1 */
    maskType limit = (maskType)1 << v;
    int idx = 0;
    
    while (mask < limit && idx < count) {
        mSubsetMasks[idx++] = mask;
        /* Gosper's hack: get next m-subset in lexicographic order */
        maskType c = mask & -(long long)mask;
        maskType r = mask + c;
        mask = (((r ^ mask) >> 2) / c) | r;
    }
    
    useFastCovering = 1;
    if (verbose)
        printf("mSubsetMasks: %11d   masks (fast covering enabled)\n", count);
}


/*
** allocateMemory() allocates the memory for the tables with given v,k,m,t
** and calculates neighborLen, coverLen, coveredLen
*/
void allocateMemory(void)
{
  unsigned elemCountNeighbors, elemCountCoverings, elemCountCovered;
  unsigned long tmp;
  int i;

  if(verbose)
    printf("Memory allocation:\n"
	   "------------------\n");

  neighborLen = k * (v - k);

  if(overflowBinCoef(v, k) || overflowBinCoef(v, m))
    coverError(BINCOEF_OVERFLOW);
  if(binCoef[v][k] != (rankType) binCoef[v][k] ||
     binCoef[v][m] != (rankType) binCoef[v][m])
    coverError(RANKTYPE_OVERFLOW);
  if(v > maxv)
    coverError(V_TOO_LARGE);
  if(t > k || t > m || k > v || m > v || b <= 0 || t <= 0)
    coverError(INVALID_PARAMETERS);

  tmp = coverLen = 0;
  for(i = 0; i <= min(k - t, m - t); i++) {
    if(overflowBinCoef(k, t + i) || overflowBinCoef(v - k, m - t - i))
      coverError(BINCOEF_OVERFLOW);
    coverLen += binCoef[k][t + i] * binCoef[v - k][m - t - i];
    if(coverLen < tmp)
      coverError(INTERNAL_OVERFLOW);
    tmp = (unsigned long) coverLen;
  }
  coverLen++; /* sentinels after cover sets of each k-set */
  if(coverLen < tmp)
    coverError(INTERNAL_OVERFLOW);

  if(!onTheFly) {
    elemCountNeighbors = neighborLen * binCoef[v][k];
    if(abs(elemCountNeighbors - 
	   (unsigned)((float) neighborLen * (float) binCoef[v][k])) > 1000)
      coverError(INTERNAL_OVERFLOW);
    if(verbose)
      printf("neighbors:%11u   elems\n", elemCountNeighbors);
    elemCountCoverings = coverLen * binCoef[v][k];
    if(abs(elemCountCoverings - 
	   (unsigned)((float) coverLen * (float) binCoef[v][k])) > 1000)
      coverError(INTERNAL_OVERFLOW);
    if(verbose)
      printf("coverings:%11u   elems\n", elemCountCoverings);
  }
  else {
    elemCountNeighbors = 0;
    elemCountCoverings = coverLen * 2;
    if(elemCountCoverings <= coverLen)
      coverError(INTERNAL_OVERFLOW);
    if(verbose)
      printf("coverings:%11u   elems\n", elemCountCoverings);
  }

  elemCountCovered = coveredLen = binCoef[v][m];
  if(verbose)
    printf("covered:  %11u   elems\n", elemCountCovered); 

  {
    unsigned long long bytes;
    bytes = (unsigned long long) elemCountNeighbors * sizeof(rankType) +
            (unsigned long long) elemCountCoverings * sizeof(rankType) +
            (unsigned long long) elemCountCovered * sizeof(coveredType);
    if(bytes > ULONG_MAX)
      coverError(INTERNAL_OVERFLOW);
    tableBytes = (unsigned long) bytes;
  }

  /* are the space demands too much? */
  if(memoryLimit > 0 && tableBytes > memoryLimit)
    coverError(TOO_MUCH_SPACE);
  /* if not, try to get the memory */
  else {
    if(elemCountNeighbors)
      neighbors = (rankType *) calloc(elemCountNeighbors, sizeof(rankType));
    coverings = (rankType *) calloc(elemCountCoverings, sizeof(rankType)); 
    covered = (coveredType *) calloc(elemCountCovered, sizeof(coveredType)); 
  }
    
  /* was the memory allocation OK? */
  if(!covered || !coverings || (!neighbors && !onTheFly))
    coverError(MEM_ALLOC_ERROR);
  
  /* NEW: Generate m-subset masks for fast covering computation (both OF=0 and OF=1) */
  if (v <= 63) {
    generateMSubsetMasks();
  }
}


/*
** updateOnTheFlyCache() allocates cached coverings for the current solution.
*/
static void updateOnTheFlyCache(void)
{
  unsigned long long cacheBytes;
  void *tmp;

  if(!onTheFly || coverLen <= 0 || b <= 0) {
    onTheFlyCache = 0;
    if(ksetCoverings) {
      free((void *) ksetCoverings);
      ksetCoverings = NULL;
    }
    return;
  }

  cacheBytes = (unsigned long long) b * coverLen * sizeof(rankType);
  if(cacheBytes > ULONG_MAX)
    coverError(INTERNAL_OVERFLOW);

  if(memoryLimit > 0 && tableBytes + (unsigned long) cacheBytes > memoryLimit) {
    onTheFlyCache = 0;
    if(ksetCoverings) {
      free((void *) ksetCoverings);
      ksetCoverings = NULL;
    }
    return;
  }

  onTheFlyCache = 1;
  if(ksetCoverings)
    tmp = realloc((void *) ksetCoverings, (size_t) cacheBytes);
  else
    tmp = malloc((size_t) cacheBytes);
  if(!tmp)
    coverError(MEM_ALLOC_ERROR);
  ksetCoverings = (rankType *) tmp;
}


/*
** calculateNeighbors computes the neighbor ranks for each rank of a k-set.
*/
void calculateNeighbors(void)
{
  rankType r;
  varietyType subset[maxv + 1], csubset[maxv + 1];
  varietyType subsubset[maxv + 1], subcsubset[maxv + 1], mergeset[maxv + 1];
  varietyType *ssptr, *scptr, *mptr;
  rankType *nptr;
  int i;

  nptr = neighbors;
  getFirstSubset(subset, k);
  for(r = 0; r < (rankType) binCoef[v][k]; r++) {
    makeComplement(subset, csubset, v);
    getFirstSubset(subsubset, k - 1);
    do {
      getFirstSubset(subcsubset, 1);
      do {
	ssptr = subsubset;
	scptr = subcsubset;
	mptr = mergeset;
	subsubset[k - 1] = (varietyType) k;
	subcsubset[1] = (varietyType) v - k;
	for(i = 0; i < k; i++)
	  if(subset[(int) *ssptr] < csubset[(int) *scptr])
	    *mptr++ = subset[(int) *ssptr++];
	  else
	    *mptr++ = csubset[(int) *scptr++];
	subsubset[k - 1] = (varietyType) maxv + 1;
	subcsubset[1] = (varietyType) maxv + 1;
	*mptr = maxv + 1;
	*nptr++ = rankSubset(mergeset, k);
      } while(getNextSubset(subcsubset, 1, v - k));
    } while(getNextSubset(subsubset, k - 1, k));
    getNextSubset(subset, k, v);
  }
}


int compareRanks(rankType *a, rankType *b)
{
  if(*a < *b)
    return -1;
  else {
    if(*a > *b)
      return 1;
    else
      return 0;
  }
}


/*
** calculateOneCovering() - OPTIMIZED with popcount when masks available
** Uses precomputed m-subset masks to avoid ranking and sorting.
*/
void calculateOneCovering(rankType kRank, rankType *buf)
{
  rankType *coverptr;
  int i;

  /* Fast path: use precomputed m-subset masks with popcount */
  if (useFastCovering && mSubsetMasks && v <= 63) {
    maskType kMask = maskFromRank(kRank, k);
    coverptr = buf;
    for (i = 0; i < coveredLen; i++) {
      if (POPCOUNT64(kMask & mSubsetMasks[i]) >= t) {
        *coverptr++ = (rankType) i;
      }
    }
    *coverptr = binCoef[v][m];  /* sentinel */
    return;
  }

  /* Fallback: original method */
  {
    static varietyType subset[maxv + 1], csubset[maxv + 1];
    static varietyType subsubset[maxv + 1];
    static varietyType subcsubset[maxv + 1], mergeset[maxv + 1];
    static varietyType *ssptr, *scptr, *mptr;
    static int ti;

    coverptr = buf;
    unrankSubset(kRank, subset, k);
    subset[k] = maxv + 1;
    makeComplement(subset, csubset, v);
    for(ti = t; ti <= min(k, m); ti++) {
      getFirstSubset(subsubset, ti);
      do {
        getFirstSubset(subcsubset, m - ti);
        do {
          ssptr = subsubset;
          scptr = subcsubset;
          mptr = mergeset;
          subsubset[ti] = (varietyType) k;
          subcsubset[m - ti] = (varietyType) v - k;
          for(i = 0; i < m; i++)
            if(subset[(int) *ssptr] < csubset[(int) *scptr])
              *mptr++ = subset[(int) *ssptr++];
            else
              *mptr++ = csubset[(int) *scptr++];
          subsubset[ti] = (varietyType) (maxv + 1);
          subcsubset[m - ti] = (varietyType) (maxv + 1);
          *mptr = (varietyType) (maxv + 1);
          *coverptr++ = rankSubset(mergeset, m);
        } while(getNextSubset(subcsubset, m - ti, v - k));
      } while(getNextSubset(subsubset, ti, k));
    }
    *coverptr = binCoef[v][m];
    qsort((char *) buf, coverLen - 1, sizeof(rankType), compareRanks);
  }
}


/*
** calculateOneCoveringMask() - OPTIMIZED VERSION using precomputed masks
**
** Instead of constructing m-subsets, ranking them, and sorting:
** - Iterate through precomputed m-subset masks (already in rank order)
** - Use popcount to check intersection >= t
** - No sorting needed!
*/
void calculateOneCoveringMask(maskType kMask, rankType *buf)
{
  rankType *coverptr;
  int i;

  /* Use fast path if m-subset masks are available */
  if (useFastCovering && mSubsetMasks) {
    coverptr = buf;
    for (i = 0; i < coveredLen; i++) {
      if (POPCOUNT64(kMask & mSubsetMasks[i]) >= t) {
        *coverptr++ = (rankType) i;
      }
    }
    *coverptr = binCoef[v][m];  /* sentinel */
    return;
  }
  
  /* Fallback to original method */
  {
    static varietyType subset[maxv + 1], csubset[maxv + 1];
    static varietyType subsubset[maxv + 1];
    static varietyType subcsubset[maxv + 1], mergeset[maxv + 1];
    static varietyType *ssptr, *scptr, *mptr;
    static int ti;
    int idx;

    idx = 0;
    for(i = 0; i < v; i++)
      if(kMask & ((maskType) 1 << i))
        subset[idx++] = (varietyType) i;
    subset[k] = maxv + 1;

    idx = 0;
    for(i = 0; i < v; i++)
      if(!(kMask & ((maskType) 1 << i)))
        csubset[idx++] = (varietyType) i;
    csubset[v - k] = maxv + 1;

    coverptr = buf;
    for(ti = t; ti <= min(k, m); ti++) {
      getFirstSubset(subsubset, ti);
      do {
        getFirstSubset(subcsubset, m - ti);
        do {
          ssptr = subsubset;
          scptr = subcsubset;
          mptr = mergeset;
          subsubset[ti] = (varietyType) k;
          subcsubset[m - ti] = (varietyType) v - k;
          for(i = 0; i < m; i++)
            if(subset[(int) *ssptr] < csubset[(int) *scptr])
              *mptr++ = subset[(int) *ssptr++];
            else
              *mptr++ = csubset[(int) *scptr++];
          subsubset[ti] = (varietyType) (maxv + 1);
          subcsubset[m - ti] = (varietyType) (maxv + 1);
          *mptr = (varietyType) (maxv + 1);
          *coverptr++ = rankSubset(mergeset, m);
        } while(getNextSubset(subcsubset, m - ti, v - k));
      } while(getNextSubset(subsubset, ti, k));
    }
    *coverptr = binCoef[v][m];
    qsort((char *) buf, coverLen - 1, sizeof(rankType), compareRanks);
  }
}


void calculateCoverings(void)
{
  rankType r;

  for(r = 0; r < (rankType) binCoef[v][k]; r++)
    calculateOneCovering(r, coverings + ((int) r * coverLen));
}


void freeTables(void)
{
  if(!onTheFly) {
    free((void *) neighbors);
  }
  free((void *) coverings);
  free((void *) covered);
  if(ksetCoverings)
    free((void *) ksetCoverings);
  if(ksetMask)
    free((void *) ksetMask);
  if(kset)
    free((void *) kset);
  /* NEW: Free m-subset masks */
  if(mSubsetMasks) {
    free((void *) mSubsetMasks);
    mSubsetMasks = NULL;
  }
}


void computeTables(int tl, int kl, int ml, int vl)
{
  t = tl;
  k = kl;
  m = ml;
  v = vl;
  allocateMemory();
  updateOnTheFlyCache();
  if(!onTheFly) {
    calculateNeighbors();
    calculateCoverings();
  }
}


void bIs(int bl)
{
  b = bl;
  if(b > maxkSetCount)
    coverError(B_TOO_LARGE);
  if(kset) {
    if(!(kset = (rankType *) realloc((char *) kset, b * sizeof(rankType))) ||
       !(ksetMask = (maskType *) realloc((char *) ksetMask,
					 b * sizeof(maskType))) ||
       !(costs = (costType *) realloc((char *) costs,
				      (b + 1) * sizeof(costType))) ||
       !(costds = (costDType *) realloc((char *) costds,
					(b + 1) * sizeof(costDType))))
      coverError(MEM_ALLOC_ERROR);
  }
  else
    if(!(kset = (rankType *) malloc(b * sizeof(rankType))) ||
       !(ksetMask = (maskType *) malloc(b * sizeof(maskType))) ||
       !(costs = (costType *) malloc((b + 1) * sizeof(costType))) ||
       !(costds = (costDType *) malloc((b + 1) * sizeof(costDType))))
      coverError(MEM_ALLOC_ERROR);
  updateOnTheFlyCache();
}


void sortSolution(void)
{
  qsort(kset, b, sizeof(rankType), compareRanks);
}

