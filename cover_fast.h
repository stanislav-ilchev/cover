/*   cover_fast.h
**
**   Fast inline functions and optimized PRNG for cover program.
**
*/

#ifndef _cover_fast_
#define _cover_fast_

#include "cover.h"

/* ============================================================================
** FAST XORSHIFT64* PRNG
** Much faster than rand()/random() and has excellent statistical properties.
** ============================================================================
*/

static unsigned long long xorshift_state = 0x853c49e6748fea9bULL;

static inline void fast_seed(unsigned long long seed) {
  xorshift_state = seed ? seed : 0x853c49e6748fea9bULL;
}

/* Xorshift64* - period 2^64-1, passes BigCrush */
static inline unsigned long long fast_rand64(void) {
  unsigned long long x = xorshift_state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  xorshift_state = x;
  return x * 0x2545F4914F6CDD1DULL;
}

/* Fast modulo for random range [0, n) */
static inline unsigned int fast_rnd(unsigned int n) {
  return (unsigned int)(((unsigned long long)fast_rand64() * n) >> 32);
}

/* Fast random float in [0,1) */
static inline double fast_random01(void) {
  return (double)(fast_rand64() >> 11) * (1.0 / 9007199254740992.0);
}

/* ============================================================================
** FAST BIT MANIPULATION
** ============================================================================
*/

/* Count trailing zeros - position of lowest set bit */
static inline int fast_ctz64(unsigned long long x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_ctzll(x);
#elif defined(_MSC_VER)
  unsigned long idx;
  _BitScanForward64(&idx, x);
  return (int)idx;
#else
  int n = 0;
  if ((x & 0xFFFFFFFF) == 0) { n += 32; x >>= 32; }
  if ((x & 0xFFFF) == 0) { n += 16; x >>= 16; }
  if ((x & 0xFF) == 0) { n += 8; x >>= 8; }
  if ((x & 0xF) == 0) { n += 4; x >>= 4; }
  if ((x & 0x3) == 0) { n += 2; x >>= 2; }
  if ((x & 0x1) == 0) { n += 1; }
  return n;
#endif
}

/* Population count - number of set bits */
static inline int fast_popcount64(unsigned long long x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(x);
#elif defined(_MSC_VER)
  return (int)__popcnt64(x);
#else
  x = x - ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
  return (int)((x * 0x0101010101010101ULL) >> 56);
#endif
}

/* Select the nth set bit (0-indexed) - returns bit position */
static inline int fast_select_bit(unsigned long long mask, int n) {
  while (n > 0) {
    mask &= mask - 1;  /* clear lowest set bit */
    n--;
  }
  return fast_ctz64(mask);
}

/* ============================================================================
** FAST SORTING FOR SMALL ARRAYS
** Insertion sort is faster than qsort for small arrays (< 32 elements)
** ============================================================================
*/

static inline void fast_sort_ranks(rankType *arr, int n) {
  int i, j;
  rankType key;
  for (i = 1; i < n; i++) {
    key = arr[i];
    j = i - 1;
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = key;
  }
}

/* Hybrid sort: insertion for small, qsort for large */
static inline void fast_sort_ranks_hybrid(rankType *arr, int n) {
  if (n <= 32) {
    fast_sort_ranks(arr, n);
  } else {
    extern int compareRanks(rankType *a, rankType *b);
    qsort((char *)arr, n, sizeof(rankType), compareRanks);
  }
}

#endif /* #ifndef _cover_fast_ */

