#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <inttypes.h>

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

typedef struct {
  uint64_t **uncoveredStack;
  int *solution;
  int *stamp;
  int stampValue;
  int report;
} SearchContext;

static int v = 27;
static int k = 6;
static int m = 4;
static int t = 3;
static int limit = 86;
static int fixFirst = 1;
static int threads = 1;
static char resultFileName[256] = "exact_solution.txt";

static mask_t *blocks = NULL;
static mask_t *draws = NULL;
static int *blockPoints = NULL;
static int *blockOutside = NULL;
static int blockCount = 0;
static int drawCount = 0;
static MaskMap blockMap;
static MaskMap drawMap;

static int bitsetWords = 0;
static int maxCover = 0;
static int foundDepth = -1;
static int *bestSolution = NULL;

static atomic_uint_fast64_t nodesVisited;
static atomic_uint_fast64_t *depthNodes = NULL;
static atomic_uint_fast64_t *depthCandidates = NULL;
static atomic_int foundFlag;
static pthread_mutex_t solutionMutex = PTHREAD_MUTEX_INITIALIZER;
static time_t startTime = 0;
static time_t lastReport = 0;
static int reportInterval = 5;

static void free_context(SearchContext *ctx);

static void format_duration(double seconds, char *buf, size_t size)
{
  int total = (int) (seconds + 0.5);
  int hours = total / 3600;
  int minutes = (total % 3600) / 60;
  int secs = total % 60;
  if(hours > 0)
    snprintf(buf, size, "%dh%02dm%02ds", hours, minutes, secs);
  else
    snprintf(buf, size, "%dm%02ds", minutes, secs);
}

static void report_progress(int depth, int remaining, int candCount)
{
  time_t now = time(NULL);
  double elapsed;
  double rate;
  int minSteps;
  double estimatedNodes = 1.0;
  double etaSeconds = -1.0;
  char elapsedBuf[32];
  char etaBuf[32];
  int i;
  uint64_t visited = atomic_load(&nodesVisited);

  if(startTime == 0)
    startTime = now;
  if(lastReport != 0 && now - lastReport < reportInterval)
    return;
  lastReport = now;

  elapsed = difftime(now, startTime);
  rate = elapsed > 0.0 ? (double) visited / elapsed : 0.0;
  minSteps = (remaining + maxCover - 1) / maxCover;
  if(minSteps < 0)
    minSteps = 0;
  if(minSteps > 0) {
    for(i = 0; i < minSteps && depth + i <= limit; i++) {
      int d = depth + i;
      double avg = 1.0;
      if(depthNodes && atomic_load(&depthNodes[d]) > 0)
        avg = (double) atomic_load(&depthCandidates[d]) /
              (double) atomic_load(&depthNodes[d]);
      else if(candCount > 0)
        avg = (double) candCount;
      if(avg < 1.0)
        avg = 1.0;
      estimatedNodes *= avg;
      if(estimatedNodes > 1e18)
        break;
    }
  }
  if(rate > 0.0 && estimatedNodes > 0.0)
    etaSeconds = estimatedNodes / rate;

  format_duration(elapsed, elapsedBuf, sizeof(elapsedBuf));
  if(etaSeconds >= 0.0)
    format_duration(etaSeconds, etaBuf, sizeof(etaBuf));
  else
    snprintf(etaBuf, sizeof(etaBuf), "unknown");

  printf("Progress: depth=%d/%d remaining=%d nodes=%" PRIu64 " rate=%.2f/s elapsed=%s ETA~%s\n",
         depth, limit, remaining, visited, rate, elapsedBuf, etaBuf);
  fflush(stdout);
}

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

static void build_block_points(void)
{
  int i;
  int insideCount = k;
  int outsideCount = v - k;

  blockPoints = (int *) malloc(sizeof(int) * blockCount * insideCount);
  blockOutside = (int *) malloc(sizeof(int) * blockCount * outsideCount);
  if(!blockPoints || !blockOutside) {
    fprintf(stderr, "ERROR: out of memory for block points\n");
    exit(1);
  }

  for(i = 0; i < blockCount; i++) {
    mask_t block = blocks[i];
    int insideIndex = 0;
    int outsideIndex = 0;
    int p;
    for(p = 0; p < v; p++) {
      if(block & ((mask_t) 1u << p))
        blockPoints[i * insideCount + insideIndex++] = p;
      else
        blockOutside[i * outsideCount + outsideIndex++] = p;
    }
  }
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

static int count_block_gain(int bindex, const uint64_t *uncovered)
{
  int gain = 0;
  int i, j, kidx;
  int l;
  int o;
  int insideCount = k;
  int outsideCount = v - k;
  const int *pts = blockPoints + bindex * insideCount;
  const int *outside = blockOutside + bindex * outsideCount;

  for(i = 0; i < insideCount - 3; i++) {
    for(j = i + 1; j < insideCount - 2; j++) {
      for(kidx = j + 1; kidx < insideCount - 1; kidx++) {
        for(l = kidx + 1; l < insideCount; l++) {
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

  for(i = 0; i < insideCount - 2; i++) {
    for(j = i + 1; j < insideCount - 1; j++) {
      for(kidx = j + 1; kidx < insideCount; kidx++) {
        for(o = 0; o < outsideCount; o++) {
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

static void apply_block(int bindex, uint64_t *uncovered)
{
  int i, j, kidx;
  int l;
  int o;
  int insideCount = k;
  int outsideCount = v - k;
  const int *pts = blockPoints + bindex * insideCount;
  const int *outside = blockOutside + bindex * outsideCount;

  for(i = 0; i < insideCount - 3; i++) {
    for(j = i + 1; j < insideCount - 2; j++) {
      for(kidx = j + 1; kidx < insideCount - 1; kidx++) {
        for(l = kidx + 1; l < insideCount; l++) {
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

  for(i = 0; i < insideCount - 2; i++) {
    for(j = i + 1; j < insideCount - 1; j++) {
      for(kidx = j + 1; kidx < insideCount; kidx++) {
        for(o = 0; o < outsideCount; o++) {
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
  int word;
  for(word = 0; word < bitsetWords; word++) {
    uint64_t bits = uncovered[word];
    if(bits) {
      int offset = __builtin_ctzll(bits);
      return word * 64 + offset;
    }
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

static int collect_candidates(SearchContext *ctx, mask_t drawMask, int lastBlockIndex,
                              Candidate *candidates)
{
  int drawPts[4];
  int drawCountPts = 0;
  int i, j, kidx;
  int count = 0;

  for(i = 0; i < v; i++) {
    if(drawMask & ((mask_t) 1u << i))
      drawPts[drawCountPts++] = i;
  }

  if(ctx->stampValue == INT32_MAX) {
    memset(ctx->stamp, 0, sizeof(int) * blockCount);
    ctx->stampValue = 1;
  }
  ctx->stampValue++;

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
            if(ctx->stamp[bindex] != ctx->stampValue) {
              ctx->stamp[bindex] = ctx->stampValue;
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

static int count_candidates_for_draw(SearchContext *ctx, mask_t drawMask, int lastBlockIndex,
                                     int bestCount)
{
  int drawPts[4];
  int drawCountPts = 0;
  int i, j, kidx;
  int count = 0;

  for(i = 0; i < v; i++) {
    if(drawMask & ((mask_t) 1u << i))
      drawPts[drawCountPts++] = i;
  }

  if(ctx->stampValue == INT32_MAX) {
    memset(ctx->stamp, 0, sizeof(int) * blockCount);
    ctx->stampValue = 1;
  }
  ctx->stampValue++;

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
              if(ctx->stamp[bindex] != ctx->stampValue) {
                ctx->stamp[bindex] = ctx->stampValue;
                count++;
                if(bestCount > 0 && count >= bestCount)
                  return count;
              }
            }
          }
        }
      }
    }
  }
  return count;
}

static int select_best_draw(SearchContext *ctx, const uint64_t *uncovered, int lastBlockIndex)
{
  int word;
  int bestIndex = -1;
  int bestCount = INT_MAX;

  for(word = 0; word < bitsetWords; word++) {
    uint64_t bits = uncovered[word];
    while(bits) {
      int offset = __builtin_ctzll(bits);
      int dindex = word * 64 + offset;
      if(dindex < drawCount) {
        int candCount = count_candidates_for_draw(ctx, draws[dindex], lastBlockIndex, bestCount);
        if(candCount < bestCount) {
          bestCount = candCount;
          bestIndex = dindex;
          if(bestCount <= 1)
            return bestIndex;
        }
      }
      bits &= bits - 1;
    }
  }
  if(bestIndex < 0)
    return first_uncovered(uncovered);
  return bestIndex;
}

static int search(SearchContext *ctx, int depth, int lastBlockIndex)
{
  uint64_t *uncovered = ctx->uncoveredStack[depth];
  int remaining = count_uncovered(uncovered);
  int i;

  atomic_fetch_add(&nodesVisited, 1);
  if(depthNodes)
    atomic_fetch_add(&depthNodes[depth], 1);

  if(atomic_load(&foundFlag))
    return 1;
  if(remaining == 0) {
    pthread_mutex_lock(&solutionMutex);
    if(!atomic_load(&foundFlag)) {
      foundDepth = depth;
      memcpy(bestSolution, ctx->solution, sizeof(int) * depth);
      atomic_store(&foundFlag, 1);
    }
    pthread_mutex_unlock(&solutionMutex);
    return 1;
  }
  if(depth >= limit)
    return 0;
  if(depth + (remaining + maxCover - 1) / maxCover > limit)
    return 0;

  {
    int dindex = select_best_draw(ctx, uncovered, lastBlockIndex);
    mask_t drawMask = draws[dindex];
    int maxCandidates = (int) (choose_u64(4, 3) * choose_u64(v - 3, 3));
    Candidate *candidates = (Candidate *) malloc(sizeof(Candidate) * maxCandidates);
    int candCount;

    if(!candidates) {
      fprintf(stderr, "ERROR: out of memory for candidates\n");
      exit(1);
    }
    candCount = collect_candidates(ctx, drawMask, lastBlockIndex, candidates);
    if(candCount == 0) {
      free(candidates);
      return 0;
    }
    if(depthCandidates)
      atomic_fetch_add(&depthCandidates[depth], (uint64_t) candCount);
    if(ctx->report)
      report_progress(depth, remaining, candCount);
    for(i = 0; i < candCount; i++) {
      candidates[i].gain = count_block_gain(candidates[i].index, uncovered);
    }
    qsort(candidates, candCount, sizeof(Candidate), candidate_compare);

    for(i = 0; i < candCount; i++) {
      int bindex = candidates[i].index;
      uint64_t *next = ctx->uncoveredStack[depth + 1];
      memcpy(next, uncovered, sizeof(uint64_t) * bitsetWords);
      apply_block(bindex, next);
      ctx->solution[depth] = bindex;
      if(search(ctx, depth + 1, bindex)) {
        free(candidates);
        return 1;
      }
      if(atomic_load(&foundFlag)) {
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
    print_block(fp, blocks[bestSolution[i]]);
  fclose(fp);
}

static void usage(const char *prog)
{
  fprintf(stderr,
          "Usage: %s [v=27 k=6 m=4 t=3 b=86 fixFirst=1 threads=1 result=exact_solution.txt]\n",
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
    if(sscanf(argv[i], "threads=%d", &threads) == 1)
      continue;
    if(sscanf(argv[i], "result=%255s", resultFileName) == 1)
      continue;
    usage(argv[0]);
    exit(1);
  }
}

static SearchContext *create_context(void)
{
  SearchContext *ctx = (SearchContext *) calloc(1, sizeof(SearchContext));
  int i;
  if(!ctx)
    return NULL;
  ctx->uncoveredStack = (uint64_t **) malloc(sizeof(uint64_t *) * (limit + 1));
  if(!ctx->uncoveredStack) {
    free(ctx);
    return NULL;
  }
  for(i = 0; i <= limit; i++) {
    ctx->uncoveredStack[i] = (uint64_t *) malloc(sizeof(uint64_t) * bitsetWords);
    if(!ctx->uncoveredStack[i]) {
      int j;
      for(j = 0; j < i; j++)
        free(ctx->uncoveredStack[j]);
      free(ctx->uncoveredStack);
      free(ctx);
      return NULL;
    }
  }
  ctx->solution = (int *) malloc(sizeof(int) * limit);
  ctx->stamp = (int *) calloc(blockCount, sizeof(int));
  ctx->stampValue = 1;
  if(!ctx->solution || !ctx->stamp) {
    free_context(ctx);
    return NULL;
  }
  return ctx;
}

static void free_context(SearchContext *ctx)
{
  int i;
  if(!ctx)
    return;
  if(ctx->uncoveredStack) {
    for(i = 0; i <= limit; i++)
      free(ctx->uncoveredStack[i]);
    free(ctx->uncoveredStack);
  }
  free(ctx->solution);
  free(ctx->stamp);
  free(ctx);
}

typedef struct {
  SearchContext *ctx;
  const uint64_t *base;
  const int *baseSolution;
  int baseDepth;
  Candidate *candidates;
  int candCount;
  atomic_int *nextIndex;
} ThreadWork;

static void *search_worker(void *arg)
{
  ThreadWork *work = (ThreadWork *) arg;
  SearchContext *ctx = work->ctx;
  int idx;

  if(work->baseDepth > 0)
    memcpy(ctx->solution, work->baseSolution, sizeof(int) * work->baseDepth);

  while((idx = atomic_fetch_add(work->nextIndex, 1)) < work->candCount) {
    int bindex = work->candidates[idx].index;
    uint64_t *start = ctx->uncoveredStack[work->baseDepth];
    uint64_t *next = ctx->uncoveredStack[work->baseDepth + 1];

    if(atomic_load(&foundFlag))
      break;

    memcpy(start, work->base, sizeof(uint64_t) * bitsetWords);
    memcpy(next, start, sizeof(uint64_t) * bitsetWords);
    apply_block(bindex, next);
    ctx->solution[work->baseDepth] = bindex;
    search(ctx, work->baseDepth + 1, bindex);
    if(atomic_load(&foundFlag))
      break;
  }
  return NULL;
}

int main(int argc, char **argv)
{
  int i;
  int depth = 0;
  int threadCount;
  mask_t firstMask = 0;
  uint64_t *base;
  uint64_t *baseUncovered;
  int *baseSolution;
  SearchContext *mainCtx = NULL;

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
  build_block_points();

  map_init(&blockMap, blockCount);
  for(i = 0; i < blockCount; i++)
    map_put(&blockMap, blocks[i], i);

  map_init(&drawMap, drawCount);
  for(i = 0; i < drawCount; i++)
    map_put(&drawMap, draws[i], i);

  bitsetWords = (drawCount + 63) / 64;
  mainCtx = create_context();
  if(!mainCtx) {
    fprintf(stderr, "ERROR: out of memory for bitsets\n");
    return 1;
  }

  base = mainCtx->uncoveredStack[0];
  for(i = 0; i < bitsetWords; i++)
    base[i] = ~0ull;
  if(drawCount % 64)
    base[bitsetWords - 1] &= (1ull << (drawCount % 64)) - 1;

  bestSolution = (int *) malloc(sizeof(int) * limit);
  baseSolution = (int *) calloc(limit, sizeof(int));
  depthNodes = (atomic_uint_fast64_t *) calloc(limit + 1, sizeof(atomic_uint_fast64_t));
  depthCandidates = (atomic_uint_fast64_t *) calloc(limit + 1, sizeof(atomic_uint_fast64_t));
  if(!bestSolution || !baseSolution || !depthNodes || !depthCandidates) {
    fprintf(stderr, "ERROR: out of memory for solution/stats\n");
    return 1;
  }
  for(i = 0; i <= limit; i++) {
    atomic_init(&depthNodes[i], 0);
    atomic_init(&depthCandidates[i], 0);
  }
  atomic_init(&nodesVisited, 0);
  atomic_init(&foundFlag, 0);

  maxCover = (int) (choose_u64(k, 4) + choose_u64(k, 3) * (v - k));

  if(fixFirst) {
    for(i = 0; i < k; i++)
      firstMask |= (mask_t) 1u << i;
    baseSolution[0] = map_get(&blockMap, firstMask);
    if(baseSolution[0] < 0) {
      fprintf(stderr, "ERROR: failed to find fixed first block\n");
      return 1;
    }
    memcpy(mainCtx->uncoveredStack[1], base, sizeof(uint64_t) * bitsetWords);
    apply_block(baseSolution[0], mainCtx->uncoveredStack[1]);
    depth = 1;
  }
  baseUncovered = mainCtx->uncoveredStack[depth];
  mainCtx->report = 1;

  startTime = time(NULL);
  lastReport = 0;
  printf("Starting exact cover search: v=%d k=%d m=%d t=%d b=%d blocks=%d draws=%d\n",
         v, k, m, t, limit, blockCount, drawCount);
  fflush(stdout);

  if(threads < 1)
    threads = 1;

  if(threads == 1) {
    if(depth > 0)
      memcpy(mainCtx->solution, baseSolution, sizeof(int) * depth);
    if(search(mainCtx, depth, depth ? baseSolution[depth - 1] : -1)) {
      printf("Found solution with %d blocks. Writing to %s\n", foundDepth, resultFileName);
      write_solution(foundDepth);
      free_context(mainCtx);
      free(baseSolution);
      return 0;
    }
  } else {
    int dindex = select_best_draw(mainCtx, baseUncovered, depth ? baseSolution[depth - 1] : -1);
    mask_t drawMask = draws[dindex];
    int maxCandidates = (int) (choose_u64(4, 3) * choose_u64(v - 3, 3));
    Candidate *candidates = (Candidate *) malloc(sizeof(Candidate) * maxCandidates);
    int candCount;
    pthread_t *threadIds;
    SearchContext **contexts;
    ThreadWork *workItems;
    atomic_int nextIndex;

    if(!candidates) {
      fprintf(stderr, "ERROR: out of memory for candidates\n");
      return 1;
    }

    candCount = collect_candidates(mainCtx, drawMask, depth ? baseSolution[depth - 1] : -1,
                                   candidates);
    if(candCount == 0) {
      free(candidates);
      printf("No solution found with b <= %d\n", limit);
      free_context(mainCtx);
      free(baseSolution);
      return 0;
    }
    for(i = 0; i < candCount; i++)
      candidates[i].gain = count_block_gain(candidates[i].index, baseUncovered);
    qsort(candidates, candCount, sizeof(Candidate), candidate_compare);

    threadCount = threads < candCount ? threads : candCount;
    threadIds = (pthread_t *) malloc(sizeof(pthread_t) * threadCount);
    contexts = (SearchContext **) malloc(sizeof(SearchContext *) * threadCount);
    workItems = (ThreadWork *) malloc(sizeof(ThreadWork) * threadCount);
    if(!threadIds || !contexts || !workItems) {
      fprintf(stderr, "ERROR: out of memory for threads\n");
      return 1;
    }

    atomic_init(&nextIndex, 0);
    for(i = 0; i < threadCount; i++) {
      contexts[i] = create_context();
      if(!contexts[i]) {
        fprintf(stderr, "ERROR: out of memory for thread context\n");
        return 1;
      }
      contexts[i]->report = (i == 0);
      workItems[i].ctx = contexts[i];
      workItems[i].base = baseUncovered;
      workItems[i].baseSolution = baseSolution;
      workItems[i].baseDepth = depth;
      workItems[i].candidates = candidates;
      workItems[i].candCount = candCount;
      workItems[i].nextIndex = &nextIndex;
      pthread_create(&threadIds[i], NULL, search_worker, &workItems[i]);
    }

    for(i = 0; i < threadCount; i++)
      pthread_join(threadIds[i], NULL);

    for(i = 0; i < threadCount; i++)
      free_context(contexts[i]);

    free(contexts);
    free(workItems);
    free(threadIds);
    free(candidates);

    if(atomic_load(&foundFlag)) {
      printf("Found solution with %d blocks. Writing to %s\n", foundDepth, resultFileName);
      write_solution(foundDepth);
      free_context(mainCtx);
      free(baseSolution);
      return 0;
    }
  }

  printf("No solution found with b <= %d\n", limit);
  free_context(mainCtx);
  free(baseSolution);
  return 0;
}
