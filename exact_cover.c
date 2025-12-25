#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>

typedef uint32_t mask_t;

typedef struct {
  mask_t *keys;
  int *values;
  size_t size;
} MaskMap;

typedef struct {
  int index;
  int gain;
} Candidate;

static int v = 27;
static int k = 6;
static int m = 4;
static int t = 3;
static int limit = 86;
static int fixFirst = 1;
static char resultFileName[256] = "exact_solution.txt";

static mask_t *blocks = NULL;
static mask_t *draws = NULL;
static int blockCount = 0;
static int drawCount = 0;
static MaskMap blockMap;
static MaskMap drawMap;

static uint64_t **uncoveredStack = NULL;
static int bitsetWords = 0;
static int *solution = NULL;
static int maxCover = 0;
static int foundDepth = -1;

static int *stamp = NULL;
static int stampValue = 1;

static uint64_t choose_u64(int n, int r)
{
  uint64_t num = 1;
  uint64_t den = 1;
  int i;
  if(r < 0 || r > n)
    return 0;
  if(r > n - r)
    r = n - r;
  for(i = 1; i <= r; i++) {
    num *= (uint64_t) (n - r + i);
    den *= (uint64_t) i;
  }
  return num / den;
}

static size_t next_pow2(size_t n)
{
  size_t p = 1;
  while(p < n)
    p <<= 1;
  return p;
}

static void map_init(MaskMap *map, size_t count)
{
  map->size = next_pow2(count * 2 + 1);
  map->keys = (mask_t *) calloc(map->size, sizeof(mask_t));
  map->values = (int *) calloc(map->size, sizeof(int));
  if(!map->keys || !map->values) {
    fprintf(stderr, "ERROR: out of memory for map\n");
    exit(1);
  }
}

static void map_put(MaskMap *map, mask_t key, int value)
{
  size_t mask = map->size - 1;
  size_t idx = (key * 2654435761u) & mask;
  while(map->keys[idx] != 0 && map->keys[idx] != key)
    idx = (idx + 1) & mask;
  map->keys[idx] = key;
  map->values[idx] = value + 1;
}

static int map_get(const MaskMap *map, mask_t key)
{
  size_t mask = map->size - 1;
  size_t idx = (key * 2654435761u) & mask;
  while(map->keys[idx] != 0) {
    if(map->keys[idx] == key)
      return map->values[idx] - 1;
    idx = (idx + 1) & mask;
  }
  return -1;
}

static int next_combination(int *comb, int r, int n)
{
  int i = r - 1;
  while(i >= 0 && comb[i] == n - r + i)
    i--;
  if(i < 0)
    return 0;
  comb[i]++;
  for(i++; i < r; i++)
    comb[i] = comb[i - 1] + 1;
  return 1;
}

static void build_blocks(void)
{
  int i;
  int *comb;
  mask_t mask;
  blockCount = (int) choose_u64(v, k);
  blocks = (mask_t *) malloc(sizeof(mask_t) * blockCount);
  if(!blocks) {
    fprintf(stderr, "ERROR: out of memory for blocks\n");
    exit(1);
  }
  comb = (int *) malloc(sizeof(int) * k);
  if(!comb) {
    fprintf(stderr, "ERROR: out of memory for comb\n");
    exit(1);
  }
  for(i = 0; i < k; i++)
    comb[i] = i;
  i = 0;
  do {
    int j;
    mask = 0;
    for(j = 0; j < k; j++)
      mask |= (mask_t) 1u << comb[j];
    blocks[i++] = mask;
  } while(next_combination(comb, k, v));
  free(comb);
}

static void build_draws(void)
{
  int i;
  int *comb;
  mask_t mask;
  drawCount = (int) choose_u64(v, m);
  draws = (mask_t *) malloc(sizeof(mask_t) * drawCount);
  if(!draws) {
    fprintf(stderr, "ERROR: out of memory for draws\n");
    exit(1);
  }
  comb = (int *) malloc(sizeof(int) * m);
  if(!comb) {
    fprintf(stderr, "ERROR: out of memory for comb\n");
    exit(1);
  }
  for(i = 0; i < m; i++)
    comb[i] = i;
  i = 0;
  do {
    int j;
    mask = 0;
    for(j = 0; j < m; j++)
      mask |= (mask_t) 1u << comb[j];
    draws[i++] = mask;
  } while(next_combination(comb, m, v));
  free(comb);
}

static int bitset_is_set(const uint64_t *bits, int idx)
{
  return (bits[idx / 64] >> (idx % 64)) & 1u;
}

static void bitset_clear(uint64_t *bits, int idx)
{
  bits[idx / 64] &= ~(1ull << (idx % 64));
}

static int count_uncovered(const uint64_t *bits)
{
  int i;
  int count = 0;
  for(i = 0; i < bitsetWords; i++)
    count += __builtin_popcountll(bits[i]);
  return count;
}

static void block_points(mask_t block, int *pts, int *count)
{
  int i;
  int idx = 0;
  for(i = 0; i < v; i++) {
    if(block & ((mask_t) 1u << i))
      pts[idx++] = i;
  }
  *count = idx;
}

static int count_block_gain(mask_t block, const uint64_t *uncovered)
{
  int pts[32];
  int ptCount = 0;
  int gain = 0;
  int i, j, kidx;
  int outside[32];
  int outCount = 0;

  block_points(block, pts, &ptCount);
  for(i = 0; i < v; i++) {
    if(!(block & ((mask_t) 1u << i)))
      outside[outCount++] = i;
  }

  for(i = 0; i < ptCount - 3; i++) {
    for(j = i + 1; j < ptCount - 2; j++) {
      for(kidx = j + 1; kidx < ptCount - 1; kidx++) {
        int l;
        for(l = kidx + 1; l < ptCount; l++) {
          mask_t dmask = ((mask_t) 1u << pts[i]) |
                         ((mask_t) 1u << pts[j]) |
                         ((mask_t) 1u << pts[kidx]) |
                         ((mask_t) 1u << pts[l]);
          int dindex = map_get(&drawMap, dmask);
          if(dindex >= 0 && bitset_is_set(uncovered, dindex))
            gain++;
        }
      }
    }
  }

  for(i = 0; i < ptCount - 2; i++) {
    for(j = i + 1; j < ptCount - 1; j++) {
      for(kidx = j + 1; kidx < ptCount; kidx++) {
        int o;
        for(o = 0; o < outCount; o++) {
          mask_t dmask = ((mask_t) 1u << pts[i]) |
                         ((mask_t) 1u << pts[j]) |
                         ((mask_t) 1u << pts[kidx]) |
                         ((mask_t) 1u << outside[o]);
          int dindex = map_get(&drawMap, dmask);
          if(dindex >= 0 && bitset_is_set(uncovered, dindex))
            gain++;
        }
      }
    }
  }
  return gain;
}

static void apply_block(mask_t block, uint64_t *uncovered)
{
  int pts[32];
  int ptCount = 0;
  int i, j, kidx;
  int outside[32];
  int outCount = 0;

  block_points(block, pts, &ptCount);
  for(i = 0; i < v; i++) {
    if(!(block & ((mask_t) 1u << i)))
      outside[outCount++] = i;
  }

  for(i = 0; i < ptCount - 3; i++) {
    for(j = i + 1; j < ptCount - 2; j++) {
      for(kidx = j + 1; kidx < ptCount - 1; kidx++) {
        int l;
        for(l = kidx + 1; l < ptCount; l++) {
          mask_t dmask = ((mask_t) 1u << pts[i]) |
                         ((mask_t) 1u << pts[j]) |
                         ((mask_t) 1u << pts[kidx]) |
                         ((mask_t) 1u << pts[l]);
          int dindex = map_get(&drawMap, dmask);
          if(dindex >= 0)
            bitset_clear(uncovered, dindex);
        }
      }
    }
  }

  for(i = 0; i < ptCount - 2; i++) {
    for(j = i + 1; j < ptCount - 1; j++) {
      for(kidx = j + 1; kidx < ptCount; kidx++) {
        int o;
        for(o = 0; o < outCount; o++) {
          mask_t dmask = ((mask_t) 1u << pts[i]) |
                         ((mask_t) 1u << pts[j]) |
                         ((mask_t) 1u << pts[kidx]) |
                         ((mask_t) 1u << outside[o]);
          int dindex = map_get(&drawMap, dmask);
          if(dindex >= 0)
            bitset_clear(uncovered, dindex);
        }
      }
    }
  }
}

static int first_uncovered(const uint64_t *uncovered)
{
  int i;
  for(i = 0; i < drawCount; i++) {
    if(bitset_is_set(uncovered, i))
      return i;
  }
  return -1;
}

static int candidate_compare(const void *a, const void *b)
{
  const Candidate *ca = (const Candidate *) a;
  const Candidate *cb = (const Candidate *) b;
  if(ca->gain != cb->gain)
    return cb->gain - ca->gain;
  return ca->index - cb->index;
}

static int collect_candidates(mask_t drawMask, int lastBlockIndex, Candidate *candidates)
{
  int drawPts[4];
  int drawCountPts = 0;
  int i, j, kidx;
  int count = 0;

  for(i = 0; i < v; i++) {
    if(drawMask & ((mask_t) 1u << i))
      drawPts[drawCountPts++] = i;
  }

  if(stampValue == INT32_MAX) {
    memset(stamp, 0, sizeof(int) * blockCount);
    stampValue = 1;
  }
  stampValue++;

  for(i = 0; i < drawCountPts - 2; i++) {
    for(j = i + 1; j < drawCountPts - 1; j++) {
      for(kidx = j + 1; kidx < drawCountPts; kidx++) {
        int rem[32];
        int remCount = 0;
        int r1, r2, r3;
        mask_t tripleMask = ((mask_t) 1u << drawPts[i]) |
                            ((mask_t) 1u << drawPts[j]) |
                            ((mask_t) 1u << drawPts[kidx]);
        int p;
        for(p = 0; p < v; p++) {
          if(!(tripleMask & ((mask_t) 1u << p)))
            rem[remCount++] = p;
        }
        for(r1 = 0; r1 < remCount - 2; r1++) {
          for(r2 = r1 + 1; r2 < remCount - 1; r2++) {
            for(r3 = r2 + 1; r3 < remCount; r3++) {
              mask_t bmask = tripleMask |
                             ((mask_t) 1u << rem[r1]) |
                             ((mask_t) 1u << rem[r2]) |
                             ((mask_t) 1u << rem[r3]);
              int bindex = map_get(&blockMap, bmask);
              if(bindex < 0 || bindex <= lastBlockIndex)
                continue;
              if(stamp[bindex] != stampValue) {
                stamp[bindex] = stampValue;
                candidates[count++].index = bindex;
              }
            }
          }
        }
      }
    }
  }
  return count;
}

static int search(int depth, int lastBlockIndex)
{
  uint64_t *uncovered = uncoveredStack[depth];
  int remaining = count_uncovered(uncovered);
  int i;

  if(remaining == 0) {
    foundDepth = depth;
    return 1;
  }
  if(depth >= limit)
    return 0;
  if(depth + (remaining + maxCover - 1) / maxCover > limit)
    return 0;

  {
    int dindex = first_uncovered(uncovered);
    mask_t drawMask = draws[dindex];
    int maxCandidates = (int) (choose_u64(4, 3) * choose_u64(v - 3, 3));
    Candidate *candidates = (Candidate *) malloc(sizeof(Candidate) * maxCandidates);
    int candCount;

    if(!candidates) {
      fprintf(stderr, "ERROR: out of memory for candidates\n");
      exit(1);
    }
    candCount = collect_candidates(drawMask, lastBlockIndex, candidates);
    if(candCount == 0) {
      free(candidates);
      return 0;
    }
    for(i = 0; i < candCount; i++) {
      candidates[i].gain = count_block_gain(blocks[candidates[i].index], uncovered);
    }
    qsort(candidates, candCount, sizeof(Candidate), candidate_compare);

    for(i = 0; i < candCount; i++) {
      int bindex = candidates[i].index;
      uint64_t *next = uncoveredStack[depth + 1];
      memcpy(next, uncovered, sizeof(uint64_t) * bitsetWords);
      apply_block(blocks[bindex], next);
      solution[depth] = bindex;
      if(search(depth + 1, bindex)) {
        free(candidates);
        return 1;
      }
    }
    free(candidates);
  }
  return 0;
}

static void print_block(FILE *fp, mask_t block)
{
  int i;
  int first = 1;
  for(i = 0; i < v; i++) {
    if(block & ((mask_t) 1u << i)) {
      if(!first)
        fprintf(fp, " ");
      fprintf(fp, "%d", i + 1);
      first = 0;
    }
  }
  fprintf(fp, "\n");
}

static void write_solution(int depth)
{
  int i;
  FILE *fp = fopen(resultFileName, "w");
  if(!fp) {
    fprintf(stderr, "ERROR: could not write %s\n", resultFileName);
    return;
  }
  for(i = 0; i < depth; i++)
    print_block(fp, blocks[solution[i]]);
  fclose(fp);
}

static void usage(const char *prog)
{
  fprintf(stderr,
          "Usage: %s [v=27 k=6 m=4 t=3 b=86 fixFirst=1 result=exact_solution.txt]\n",
          prog);
}

static void parse_args(int argc, char **argv)
{
  int i;
  for(i = 1; i < argc; i++) {
    if(sscanf(argv[i], "v=%d", &v) == 1)
      continue;
    if(sscanf(argv[i], "k=%d", &k) == 1)
      continue;
    if(sscanf(argv[i], "m=%d", &m) == 1)
      continue;
    if(sscanf(argv[i], "t=%d", &t) == 1)
      continue;
    if(sscanf(argv[i], "b=%d", &limit) == 1)
      continue;
    if(sscanf(argv[i], "fixFirst=%d", &fixFirst) == 1)
      continue;
    if(sscanf(argv[i], "result=%255s", resultFileName) == 1)
      continue;
    usage(argv[0]);
    exit(1);
  }
}

int main(int argc, char **argv)
{
  int i;
  int depth = 0;
  mask_t firstMask = 0;
  uint64_t *base;

  parse_args(argc, argv);

  if(v > 32) {
    fprintf(stderr, "ERROR: v must be <= 32\n");
    return 1;
  }
  if(m != 4 || t != 3) {
    fprintf(stderr, "ERROR: this exact solver supports m=4, t=3 only\n");
    return 1;
  }

  build_blocks();
  build_draws();

  map_init(&blockMap, blockCount);
  for(i = 0; i < blockCount; i++)
    map_put(&blockMap, blocks[i], i);

  map_init(&drawMap, drawCount);
  for(i = 0; i < drawCount; i++)
    map_put(&drawMap, draws[i], i);

  bitsetWords = (drawCount + 63) / 64;
  uncoveredStack = (uint64_t **) malloc(sizeof(uint64_t *) * (limit + 1));
  if(!uncoveredStack) {
    fprintf(stderr, "ERROR: out of memory for bitsets\n");
    return 1;
  }
  for(i = 0; i <= limit; i++) {
    uncoveredStack[i] = (uint64_t *) malloc(sizeof(uint64_t) * bitsetWords);
    if(!uncoveredStack[i]) {
      fprintf(stderr, "ERROR: out of memory for bitsets\n");
      return 1;
    }
  }

  base = uncoveredStack[0];
  for(i = 0; i < bitsetWords; i++)
    base[i] = ~0ull;
  if(drawCount % 64)
    base[bitsetWords - 1] &= (1ull << (drawCount % 64)) - 1;

  solution = (int *) malloc(sizeof(int) * limit);
  stamp = (int *) calloc(blockCount, sizeof(int));
  if(!solution || !stamp) {
    fprintf(stderr, "ERROR: out of memory for solution/stamp\n");
    return 1;
  }

  maxCover = (int) (choose_u64(k, 4) + choose_u64(k, 3) * (v - k));

  if(fixFirst) {
    for(i = 0; i < k; i++)
      firstMask |= (mask_t) 1u << i;
    solution[0] = map_get(&blockMap, firstMask);
    if(solution[0] < 0) {
      fprintf(stderr, "ERROR: failed to find fixed first block\n");
      return 1;
    }
    memcpy(uncoveredStack[1], base, sizeof(uint64_t) * bitsetWords);
    apply_block(firstMask, uncoveredStack[1]);
    depth = 1;
  }

  if(search(depth, depth ? solution[depth - 1] : -1)) {
    printf("Found solution with %d blocks. Writing to %s\n", foundDepth, resultFileName);
    write_solution(foundDepth);
    return 0;
  }

  printf("No solution found with b <= %d\n", limit);
  return 0;
}
