/*
 * check_redundancy.c
 *
 * Checks whether any block in a given (v,k,m,t) lotto/cover design is redundant.
 * A block is redundant if removing it still covers all m-subsets at least t times.
 *
 * Usage:
 *   check_redundancy.exe v=49 k=6 m=6 t=3 file=163.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cover.h"
#include "bincoef.h"
#include "setoper.h"

int v = 49, k = 6, m = 6, t = 3, b = 0;
int solX = 0;

static int min_int(int a, int b) { return (a < b) ? a : b; }

static int compare_ints(const void *a, const void *b) {
  return (*(const int *)a) - (*(const int *)b);
}

static void parse_args(int argc, char **argv, const char **file_path) {
  int i;

  for (i = 1; i < argc; i++) {
    if (strncmp(argv[i], "v=", 2) == 0) v = atoi(argv[i] + 2);
    else if (strncmp(argv[i], "k=", 2) == 0) k = atoi(argv[i] + 2);
    else if (strncmp(argv[i], "m=", 2) == 0) m = atoi(argv[i] + 2);
    else if (strncmp(argv[i], "t=", 2) == 0) t = atoi(argv[i] + 2);
    else if (strncmp(argv[i], "file=", 5) == 0) *file_path = argv[i] + 5;
    else if (strncmp(argv[i], "input=", 6) == 0) *file_path = argv[i] + 6;
    else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      exit(1);
    }
  }
}

static int read_blocks(const char *path, maskType **out_blocks) {
  FILE *fp = fopen(path, "r");
  int capacity = 128;
  int count = 0;
  int nums[64];

  if (!fp) {
    fprintf(stderr, "Failed to open %s\n", path);
    exit(1);
  }

  *out_blocks = (maskType *)malloc(capacity * sizeof(maskType));
  if (!*out_blocks) {
    fprintf(stderr, "Out of memory allocating blocks\n");
    exit(1);
  }

  for (;;) {
    int i;
    if (fscanf(fp, "%d", &nums[0]) != 1)
      break;
    for (i = 1; i < k; i++) {
      if (fscanf(fp, "%d", &nums[i]) != 1) {
        fprintf(stderr, "Incomplete block in %s\n", path);
        exit(1);
      }
    }

    qsort(nums, k, sizeof(int), compare_ints);
    for (i = 0; i < k; i++) {
      if (nums[i] < 0 || nums[i] >= v) {
        fprintf(stderr, "Value %d out of range [0,%d)\n", nums[i], v);
        exit(1);
      }
      if (i > 0 && nums[i] == nums[i - 1]) {
        fprintf(stderr, "Duplicate value %d in block\n", nums[i]);
        exit(1);
      }
    }

    if (count == capacity) {
      capacity *= 2;
      *out_blocks = (maskType *)realloc(*out_blocks, capacity * sizeof(maskType));
      if (!*out_blocks) {
        fprintf(stderr, "Out of memory growing blocks\n");
        exit(1);
      }
    }

    {
      maskType mask = 0;
      for (i = 0; i < k; i++)
        mask |= ((maskType)1 << nums[i]);
      (*out_blocks)[count++] = mask;
    }
  }

  fclose(fp);
  return count;
}

static void add_coverings(maskType kMask, unsigned short *covered) {
  varietyType subset[maxv + 1], csubset[maxv + 1];
  varietyType subsubset[maxv + 1], subcsubset[maxv + 1], mergeset[maxv + 1];
  varietyType *ssptr, *scptr, *mptr;
  int i, ti, idx;

  idx = 0;
  for (i = 0; i < v; i++)
    if (kMask & ((maskType)1 << i))
      subset[idx++] = (varietyType)i;
  subset[k] = (varietyType)(maxv + 1);

  idx = 0;
  for (i = 0; i < v; i++)
    if (!(kMask & ((maskType)1 << i)))
      csubset[idx++] = (varietyType)i;
  csubset[v - k] = (varietyType)(maxv + 1);

  for (ti = t; ti <= min_int(k, m); ti++) {
    getFirstSubset(subsubset, ti);
    do {
      getFirstSubset(subcsubset, m - ti);
      do {
        ssptr = subsubset;
        scptr = subcsubset;
        mptr = mergeset;
        subsubset[ti] = (varietyType)k;
        subcsubset[m - ti] = (varietyType)(v - k);
        for (i = 0; i < m; i++) {
          if (subset[(int)*ssptr] < csubset[(int)*scptr])
            *mptr++ = subset[(int)*ssptr++];
          else
            *mptr++ = csubset[(int)*scptr++];
        }
        subsubset[ti] = (varietyType)(maxv + 1);
        subcsubset[m - ti] = (varietyType)(maxv + 1);
        mergeset[m] = (varietyType)(maxv + 1);
        covered[rankSubset(mergeset, m)]++;
      } while (getNextSubset(subcsubset, m - ti, v - k));
    } while (getNextSubset(subsubset, ti, k));
  }
}

static int count_unique_coverings(maskType kMask, const unsigned short *covered) {
  varietyType subset[maxv + 1], csubset[maxv + 1];
  varietyType subsubset[maxv + 1], subcsubset[maxv + 1], mergeset[maxv + 1];
  varietyType *ssptr, *scptr, *mptr;
  int i, ti, idx;
  int unique = 0;

  idx = 0;
  for (i = 0; i < v; i++)
    if (kMask & ((maskType)1 << i))
      subset[idx++] = (varietyType)i;
  subset[k] = (varietyType)(maxv + 1);

  idx = 0;
  for (i = 0; i < v; i++)
    if (!(kMask & ((maskType)1 << i)))
      csubset[idx++] = (varietyType)i;
  csubset[v - k] = (varietyType)(maxv + 1);

  for (ti = t; ti <= min_int(k, m); ti++) {
    getFirstSubset(subsubset, ti);
    do {
      getFirstSubset(subcsubset, m - ti);
      do {
        rankType r;
        ssptr = subsubset;
        scptr = subcsubset;
        mptr = mergeset;
        subsubset[ti] = (varietyType)k;
        subcsubset[m - ti] = (varietyType)(v - k);
        for (i = 0; i < m; i++) {
          if (subset[(int)*ssptr] < csubset[(int)*scptr])
            *mptr++ = subset[(int)*ssptr++];
          else
            *mptr++ = csubset[(int)*scptr++];
        }
        subsubset[ti] = (varietyType)(maxv + 1);
        subcsubset[m - ti] = (varietyType)(maxv + 1);
        mergeset[m] = (varietyType)(maxv + 1);
        r = rankSubset(mergeset, m);
        if (covered[r] == 1)
          unique++;
      } while (getNextSubset(subcsubset, m - ti, v - k));
    } while (getNextSubset(subsubset, ti, k));
  }

  return unique;
}

static void print_block(maskType mask) {
  int i;
  for (i = 0; i < v; i++)
    if (mask & ((maskType)1 << i))
      printf("%d ", i);
  printf("\n");
}

int main(int argc, char **argv) {
  const char *file_path = NULL;
  maskType *blocks = NULL;
  unsigned short *covered = NULL;
  int i;
  int redundant = 0;
  int min_unique = -1;
  int min_idx = -1;

  parse_args(argc, argv, &file_path);
  if (!file_path) {
    fprintf(stderr, "Missing file=... argument\n");
    return 1;
  }
  if (v > maxv) {
    fprintf(stderr, "v=%d exceeds maxv=%d in cover.h\n", v, maxv);
    return 1;
  }

  calculateBinCoefs();

  blocks = NULL;
  b = read_blocks(file_path, &blocks);
  printf("Loaded %d blocks from %s\n", b, file_path);

  covered = (unsigned short *)calloc(binCoef[v][m], sizeof(unsigned short));
  if (!covered) {
    fprintf(stderr, "Out of memory allocating covered array\n");
    return 1;
  }

  printf("Computing coverage counts...\n");
  for (i = 0; i < b; i++) {
    add_coverings(blocks[i], covered);
  }

  printf("Checking for redundant blocks...\n");
  for (i = 0; i < b; i++) {
    int uniq = count_unique_coverings(blocks[i], covered);
    if (min_unique == -1 || uniq < min_unique) {
      min_unique = uniq;
      min_idx = i;
    }
    if (uniq == 0) {
      printf("Block %d is redundant:\n", i + 1);
      print_block(blocks[i]);
      redundant++;
    }
  }

  if (!redundant) {
    printf("No redundant blocks found.\n");
    if (min_idx >= 0) {
      printf("Smallest unique coverage: %d (block %d)\n", min_unique, min_idx + 1);
      print_block(blocks[min_idx]);
    }
  }

  free(covered);
  free(blocks);
  return 0;
}
