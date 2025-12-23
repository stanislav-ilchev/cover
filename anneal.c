/*   anneal.c
**
**   This file contains the functions in simulated annealing process.
**
*/


#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include "cover.h"
#include "bincoef.h"
#include "tables.h"
#include "setoper.h"
#include "exp.h"


int iterCounter = 0;
float endT;

/*
** `setNumber' is the index to the table `kset'. `setNumber' indicates
** the index of the k-set in the proposed change. The current solution
** can be changed to the proposed next solution by assigning 
** kset[setNumber] = nextS. `stored', `storedPtr', `currSto' and
**`nextSto' are for on-the-fly annealing. `costs[x]' holds the difference
** of costs associated with a m-set covered x times and x+1 times.
**
*/

static int setNumber;
static rankType nextS;
static maskType nextMask;
static rankType stored[2];
static int currSto, nextSto;
static int useMask = 0;
static rankType *nextCoveringsPtr = NULL;
rankType *storedPtr[2];

/* Cached values for performance */
static maskType cachedFullMask = 0;
static rankType cachedSentinel = 0;

static int startLoaded = 0;
static int startCount = 0;
static rankType *startRanks = NULL;
static maskType *startMasks = NULL;

static int popcountMask(maskType mask)
{
  int count = 0;
  while(mask) {
    mask &= (mask - 1);
    count++;
  }
  return count;
}

static void loadStartFileOnce(void)
{
  FILE *fp;
  int capacity = 256;
  int count = 0;
  int i;

  if(startLoaded)
    return;
  startLoaded = 1;
  if(startFileName[0] == '\0')
    return;

  fp = fopen(startFileName, "r");
  if(!fp) {
    fprintf(stderr, "Can't open start file %s.\n", startFileName);
    coverError(SEE_ABOVE_ERROR);
  }

  startMasks = (maskType *) malloc(capacity * sizeof(maskType));
  startRanks = (rankType *) malloc(capacity * sizeof(rankType));
  if(!startMasks || !startRanks)
    coverError(MEM_ALLOC_ERROR);

  for(;;) {
    int nums[maxv + 1];
    maskType mask = 0;
    if(fscanf(fp, "%d", &nums[0]) != 1)
      break;
    for(i = 1; i < k; i++) {
      if(fscanf(fp, "%d", &nums[i]) != 1) {
        fprintf(stderr, "Incomplete block in start file.\n");
        coverError(SEE_ABOVE_ERROR);
      }
    }

    for(i = 0; i < k; i++) {
      if(nums[i] < 0 || nums[i] >= v) {
        fprintf(stderr, "Value %d out of range [0,%d) in start file.\n",
                nums[i], v);
        coverError(INVALID_PARAMETERS);
      }
      if(mask & ((maskType) 1 << nums[i])) {
        fprintf(stderr, "Duplicate value %d in start file block.\n", nums[i]);
        coverError(INVALID_PARAMETERS);
      }
      mask |= ((maskType) 1 << nums[i]);
    }
    if(popcountMask(mask) != k) {
      fprintf(stderr, "Invalid block in start file.\n");
      coverError(INVALID_PARAMETERS);
    }

    if(count == capacity) {
      capacity *= 2;
      startMasks = (maskType *) realloc(startMasks, capacity * sizeof(maskType));
      startRanks = (rankType *) realloc(startRanks, capacity * sizeof(rankType));
      if(!startMasks || !startRanks)
        coverError(MEM_ALLOC_ERROR);
    }
    startMasks[count] = mask;
    startRanks[count] = rankFromMask(mask, k);
    count++;
  }

  fclose(fp);

  if(count == 0) {
    fprintf(stderr, "Start file is empty.\n");
    coverError(SEE_ABOVE_ERROR);
  }

  if(startDrop > 0 && startDrop <= count) {
    int dropIdx = startDrop - 1;
    for(i = dropIdx; i + 1 < count; i++) {
      startMasks[i] = startMasks[i + 1];
      startRanks[i] = startRanks[i + 1];
    }
    count--;
  }
  else if(startDrop > count && verbose) {
    printf("StartDrop=%d ignored; file has %d blocks.\n", startDrop, count);
  }

  if(count > b) {
    if(verbose)
      printf("Start file has %d blocks; using first %d.\n", count, b);
    count = b;
  }
  startCount = count;
}

void freeStartSolution(void)
{
  free(startRanks);
  free(startMasks);
  startRanks = NULL;
  startMasks = NULL;
  startCount = 0;
  startLoaded = 0;
}


/*
** `calculateCosts()' calculates the costs and cost differences in the
** tables `costs' and `costds' to be used when calculating the costs of
** the solutions.
**
*/

void calculateCosts(void)
{
  int i;

  if(pack) /* packing design */
    for(i = 0; i <= b; i++)
      if(i < coverNumber)
	costs[i] = (costType) 0;
      else
	costs[i] = (costDType) (i - coverNumber);
  else     /* covering design */
    for(i = 0; i <= b; i++)
      if(i < coverNumber)
	costs[i] = (costType) (coverNumber - i);
      else
	costs[i] = 0;
  for(i = 0; i < b; i++)
    costds[i] = costs[i] - costs[i + 1];
}



/*
** `initSolution' makes an initial solution by selecting `b' random k-sets.
** The table `covered' is initialized to zero for computing the initial
** solution. It then initiates the table `covered' to this initial solution
** and calculates the initial value of the cost function (zeros in `covered').
** 
**
*/

costType initSolution(void)
{
  int i, j;
  costType initCost;
  costType P2plus = (costType) 0;
  coveredType *ptr;
  coveredType *cptr;
  rankType *coveringsPtr;

  useMask = (v <= 63);
  
  /* Initialize cached values for performance */
  cachedSentinel = binCoef[v][m];
  if(v == 64)
    cachedFullMask = (maskType) ~0;
  else
    cachedFullMask = (((maskType) 1) << v) - 1;
  
  loadStartFileOnce();
  if(restrictedNeighbors)
    setNumber = 0;
  for(i = 0; i < 2; i++)
    stored[i] = binCoef[v][k];
  nextSto = currSto = -1;
  nextCoveringsPtr = NULL;
  for(i = binCoef[v][m], cptr = covered; i > 0; i--)
    *cptr++ = (coveredType) 0;
  for(i = 0; i < b; i++) {
    if(i < startCount) {
      kset[i] = startRanks[i];
      if(useMask)
	ksetMask[i] = startMasks[i];
    }
    else {
      kset[i] = rnd(binCoef[v][k]);
      if(useMask)
	ksetMask[i] = maskFromRank(kset[i], k);
    }
    if(onTheFly) {
      if(onTheFlyCache) {
	coveringsPtr = ksetCoverings + (int) i * coverLen;
	if(useMask)
	  calculateOneCoveringMask(ksetMask[i], coveringsPtr);
	else
	  calculateOneCovering(kset[i], coveringsPtr);
      }
      else {
	if(useMask)
	  calculateOneCoveringMask(ksetMask[i], coverings);
	else
	  calculateOneCovering(kset[i], coverings);
	coveringsPtr = coverings;
      }
    }
    else
      coveringsPtr = coverings + (int) kset[i] * coverLen;
    for(j = 0; j < coverLen - 1; j++)
      covered[coveringsPtr[j]]++;
  }
  for(i = 0, initCost = (costType) 0, ptr = covered; i < coveredLen;
      i++, ptr++)
    initCost += costs[*ptr];

  if(verbose) {
    printf("initCost      = %d\n", initCost); 
  }
  return initCost;
}


/*
** compareVarieties is needed for qsort int randomNeighbor()
**
*/

int compareVarieties(varietyType *a, varietyType *b)
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
** randomNeighbor() computes the rank of a random neigbor of the k-set with
** rank `curr'.
*/

static rankType randomNeighbor(rankType curr)
{
  varietyType subset[maxv + 1];
  varietyType csubset[maxv + 1];

  unrankSubset(curr, subset, k);
  makeComplement(subset, csubset, v);
  subset[rnd(k)] = csubset[rnd(v-k)];
  qsort((char *) subset, k, sizeof(varietyType), compareVarieties);
  return rankSubset(subset, k);
}


/*
** selectNthBit() returns the bit index of the nth set bit in mask.
** Optimized version using bit manipulation.
*/

static int selectNthBit(maskType mask, int n)
{
  /* Clear the first n set bits */
  while(n > 0) {
    mask &= mask - 1;  /* clear lowest set bit */
    n--;
  }
  /* Return index of lowest remaining set bit */
  if(mask == 0) return -1;
  
  /* Use __builtin_ctzll if available (count trailing zeros) */
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_ctzll(mask);
#else
{
  int i;
    for(i = 0; i < 64; i++)
      if(mask & ((maskType) 1 << i))
	return i;
  return -1;
  }
#endif
}


/*
** randomNeighborMask() computes a random neighbor rank using bitmasks.
** Uses cached fullMask for performance.
*/

static rankType randomNeighborMask(maskType currMask, maskType *outMask)
{
  maskType comp;
  int removeBit, addBit;

  comp = cachedFullMask ^ currMask;
  removeBit = selectNthBit(currMask, rnd(k));
  addBit = selectNthBit(comp, rnd(v - k));
  *outMask = (currMask & ~(((maskType) 1) << removeBit)) |
             (((maskType) 1) << addBit);
  return rankFromMask(*outMask, k);
}


/*
** computeNeighbor() calculates the cost difference between the current
** solution and a random neighbor of the current solution. It employs
** the sentinels at the end of the covered sets of each k-set. (That's
** why they were put there in calculateNeighbors().)
**
*/

costType computeNeighbor(void)
{
  costType costDelta = 0;
  int i;
  rankType currS;
  rankType *currPtr, *nextPtr;

  if(restrictedNeighbors) {
    if(++setNumber == b)
      setNumber = 0;
  }
  else
    setNumber = rnd(b);

  currS = kset[setNumber];
  if(onTheFly) {
    if(useMask)
      nextS = randomNeighborMask(ksetMask[setNumber], &nextMask);
    else
      nextS = randomNeighbor(currS);

    if(onTheFlyCache) {
      currPtr = ksetCoverings + (int) setNumber * coverLen;
      nextPtr = coverings;
      if(useMask)
	calculateOneCoveringMask(nextMask, nextPtr);
      else
	calculateOneCovering(nextS, nextPtr);
      nextCoveringsPtr = nextPtr;
    }
    else {
      currPtr = NULL;
      for(i = 0; i < 2; i++)
	if(currS == stored[i]) {
	  currPtr = storedPtr[i];
	  currSto = i;
	}
      nextPtr = NULL;
      for(i = 0; i < 2; i++)
	if(nextS == stored[i]) {
	  nextPtr = storedPtr[i];
	  nextSto = i;
	}
      for(i = 0; !currPtr; i++)
	if(nextSto != i) {
	  storedPtr[i] = currPtr = coverings + i * coverLen;
	  currSto = i;
	  stored[i] = currS;
	  calculateOneCovering(currS, currPtr);
	}
      for(i = 0; !nextPtr; i++)
	if(currSto != i) {
	  storedPtr[i] = nextPtr = coverings + i * coverLen;
	  nextSto = i;
	  stored[i] = nextS;
	  calculateOneCovering(nextS, nextPtr);
	}
    }
  }
  else {
    nextS = neighbors[currS * neighborLen + rnd(neighborLen)];
    currPtr = coverings + currS * coverLen;
    nextPtr = coverings + nextS * coverLen;
  }

  /* Use cached sentinel for faster comparison */
  for(;;) {
    if(*currPtr == *nextPtr) {
      if(*currPtr == cachedSentinel)
	break;
      currPtr++;
      nextPtr++;
    }
    else if(*currPtr < *nextPtr)
      costDelta += costds[covered[*currPtr++] - 1];
    else
      costDelta -= costds[covered[*nextPtr++]];
  }
  
  return costDelta;
}


/*
** acceptNeighbor() changes the current solution to the latest solution
** computed.
*/

void acceptNeighbor(void)
{
  int i;
  rankType currS;
  rankType *currCovPtr, *nextCovPtr;
  int len = coverLen - 1;

  currS = kset[setNumber];
  
  if(onTheFly) {
    /* On-the-fly mode: separate loops due to different pointer sources */
    if(onTheFlyCache)
      currCovPtr = ksetCoverings + (int) setNumber * coverLen;
    else
      currCovPtr = coverings + currSto * coverLen;
    for(i = 0; i < len; i++)
      covered[currCovPtr[i]]--;
    
    if(onTheFlyCache)
      nextCovPtr = nextCoveringsPtr;
    else
      nextCovPtr = coverings + nextSto * coverLen;
    for(i = 0; i < len; i++)
      covered[nextCovPtr[i]]++;
    
    if(onTheFlyCache) {
      memcpy(ksetCoverings + (int) setNumber * coverLen, nextCovPtr,
	     coverLen * sizeof(rankType));
    }
    if(useMask)
      ksetMask[setNumber] = nextMask;
  }
  else {
    /* Precomputed mode: fused loop for better ILP */
    currCovPtr = coverings + currS * coverLen;
    nextCovPtr = coverings + nextS * coverLen;
    for(i = 0; i < len; i++) {
      covered[currCovPtr[i]]--;
      covered[nextCovPtr[i]]++;
    }
  }
  
  kset[setNumber] = nextS;
}


/*
** approxInitT() tries to find a good initial value for T, so that the
** probability of accepting a cost increasing move is approximately the
** probability set by the user. T_ITER is the count of iterations when
** calculating the initial temperature from `initProb'. T_LIFESAVER is
** the initial temperature, if no cost increasing moves are found during
** the iteration.
**
*/

#define T_LIFESAVER 1.0
#define T_ITER 300

static float approxInitT(void)
{
  float T;
  int i, m2;
  costType costDelta;

  T = 0.0;
  m2 = 0;
  for(i = 0; i < T_ITER; i++) {
    costDelta = computeNeighbor();
    if(costDelta > 0) {
      m2++;
      T += -costDelta;
    }
  }
  if(m2 == 0)
    T = T_LIFESAVER;
  else
    T = T / m2 / log(initProb);
  return T;
}


/*
** simulatedAnnealing() is the algorithm. `coolFact' is the cooling
** factor, `initProb' the wanted initial probability for accepting a
** cost increasing move, `frozen' is the count of successive temperatures
** without cost decrease, before the iteration is considered frozen.
** `iterLenght' is number of iterations performed at each temperature.
**
*/

#define T_PRINT_ITER 300

costType simulatedAnnealing(double coolFact, double initProb, 
			    int iterLength, int frozen, int endLimit)
{
  float deltaF_;
  float r, D;
  costType costDelta, actCost, currCost, lastCost, bestSeen;
  int notChanged = 0, i, j, k2, m1, m2, m3, m0, l;
  varietyType set[maxv + 1];
  float T;

  if(verbose)
    printf("Starting annealing...\n\n");
  calculateCosts();
  currCost = initSolution();
  bestSeen = lastCost = currCost;
  
  if(Tset)
    T = initialT;                         /* T was given as a parameter */
  else
    T = approxInitT();

  /* tests the probability for cost increasing moves */
  if(verbose) {
    m1 = m2 = 0;
    deltaF_ = 0.0;
    for(i = 0; i < T_PRINT_ITER; i++) {
      costDelta = computeNeighbor();
      if(costDelta > 0) {
	m1++;
	if(random01() < ExpProb(costDelta / T))
	  m2++;
      }
    }
    if(m1 == 0) {
      m1 = 1;
      m2 = 0;
    }
    printf("initial inc%%  = %.2f\n\n", (double) m2 / (double) m1);
  }

  if(verbose >= 2)
    printf("      T      cost   best   inc%%   tot%%   0-m%%\n"
	   "    ------------------------------------------\n");
  while(notChanged < frozen) {
    m1 = m2 = m3 = m0 = 0;
    for(i = 0; i < iterLength; i++) {
      costDelta = computeNeighbor();
      iterCounter++;
      if(costDelta <= 0) {
	m3++;
	acceptNeighbor();
	currCost += costDelta;
	if(currCost <= endLimit) {
	  endT = T;
	  if(verbose >= 2)
	    printf("\n");
	  if(verbose)
	    printf("...annealing accomplished.\n\n");
	  return currCost;       /* a good enough final solution was found */
	}
	if(costDelta < 0) {
	  notChanged = 0;
	  if(currCost < bestSeen)
	    bestSeen = currCost;
	}
	else
	  m0++;
      }
      else {
	r = random01();
	D = costDelta / T;
	if(r < ExpProb(D)) {
	  acceptNeighbor();
	  m1++;
	  currCost += costDelta;
	}
	else
	  m2++;
      }
    }
    if(lastCost <= currCost)
      notChanged++;
    lastCost = currCost;
    if(m2 == 0)
      m2 = 1; /* prevent division by zero */
    if(verbose >= 2)
      printf("    %5.2f   %4d   %4d    %4.1f   %4.1f   %4.3f\n", 
	     T, currCost, bestSeen, (double) m1 / (double) (m1 + m2) * 100.0,
	     (double) (m3 + m1) / (double) (m1 + m2 + m3) * 100.0,
	     (double) (m0) / (double) (m1 + m2 + m3));
    T *= coolFact;
  }
  endT = T;
  if(verbose >= 2)
    printf("\n");
  if(verbose)
    printf("...annealing accomplished.\n\n");
  return currCost;
}


/*
** `thresholdAccepting()' implements the Threshold Accepting algorithm.
** Similar to SA but uses a deterministic threshold instead of probabilistic
** acceptance. Accept if costDelta < threshold.
**
*/

costType thresholdAccepting(double initThreshold, double coolFact,
			    int iterLength, int frozen, int endLimit)
{
  costType costDelta, currCost, lastCost, bestSeen;
  int notChanged = 0, i, m1, m2, m3, m0;
  double threshold;

  if(verbose)
    printf("Starting threshold accepting...\n\n");
  calculateCosts();
  currCost = initSolution();
  bestSeen = lastCost = currCost;
  
  threshold = initThreshold;

  if(verbose >= 2)
    printf("  thresh    cost   best   acc%%   imp%%\n"
	   "  ----------------------------------------\n");

  while(notChanged < frozen) {
    m1 = m2 = m3 = m0 = 0;
    for(i = 0; i < iterLength; i++) {
      costDelta = computeNeighbor();
      iterCounter++;
      
      /* Accept if improvement OR if cost increase is below threshold */
      if(costDelta <= 0) {
	m3++;  /* improving or equal move */
	acceptNeighbor();
	currCost += costDelta;
	if(currCost <= endLimit) {
	  endT = threshold;
	  if(verbose >= 2)
	    printf("\n");
	  if(verbose)
	    printf("...threshold accepting accomplished.\n\n");
	  return currCost;
	}
	if(costDelta < 0) {
	  notChanged = 0;
	  if(currCost < bestSeen)
	    bestSeen = currCost;
	}
	else
	  m0++;  /* equal move */
      }
      else if(costDelta < threshold) {
	/* Accept uphill move below threshold */
	m1++;
	acceptNeighbor();
	currCost += costDelta;
      }
      else {
	/* Reject - cost increase too large */
	m2++;
      }
    }
    if(lastCost <= currCost)
      notChanged++;
    lastCost = currCost;
    if(m2 == 0)
      m2 = 1;
    if(verbose >= 2)
      printf("  %7.2f  %4d   %4d    %4.1f   %4.1f\n", 
	     threshold, currCost, bestSeen,
	     (double) (m3 + m1) / (double) (m1 + m2 + m3) * 100.0,
	     (double) m3 / (double) (m1 + m2 + m3) * 100.0);
    threshold *= coolFact;
  }
  endT = threshold;
  if(verbose >= 2)
    printf("\n");
  if(verbose)
    printf("...threshold accepting accomplished.\n\n");
  return currCost;
}


/*
** `recordToRecordTravel()' implements the Record-to-Record Travel algorithm.
** Accept moves if newCost <= bestSeen + threshold (fixed deviation from best).
** This maintains a "band" around the best solution found so far.
**
*/

costType recordToRecordTravel(int threshold, int frozen, int endLimit)
{
  costType costDelta, currCost, bestSeen, newCost;
  costType acceptLimit;  /* precompute bestSeen + threshold */
  int i, m1, m2;
  int itersSinceImprovement = 0;

  if(verbose)
    printf("Starting Record-to-Record Travel (threshold=%d)...\n\n", threshold);
  calculateCosts();
  currCost = initSolution();
  bestSeen = currCost;
  acceptLimit = bestSeen + threshold;

  if(verbose >= 2)
    printf("      iter     best\n"
	   "   ------------------\n");

  while(itersSinceImprovement < frozen) {
    m1 = m2 = 0;
    
    /* Run a batch of iterations - tight inner loop */
    for(i = 0; i < 10000; i++) {
      costDelta = computeNeighbor();
      newCost = currCost + costDelta;
      
      /* Accept if within threshold of best seen */
      if(newCost <= acceptLimit) {
	acceptNeighbor();
	currCost = newCost;
	m1++;
	
	if(currCost < bestSeen) {
	  bestSeen = currCost;
	  acceptLimit = bestSeen + threshold;  /* update limit */
	  itersSinceImprovement = 0;
	  
	  if(verbose >= 2)
	    printf("  %8d    %5d\n", iterCounter + i + 1, bestSeen);
	  
	  if(currCost <= endLimit) {
	    endT = threshold;
	    iterCounter += i + 1;
	    if(verbose)
	      printf("...Record-to-Record accomplished (found target).\n\n");
	    return currCost;
	  }
	}
      }
      else {
	m2++;
      }
      
      itersSinceImprovement++;
    }
    
    iterCounter += 10000;
  }
  
  endT = threshold;
  if(verbose >= 2)
    printf("\n");
  if(verbose)
    printf("...Record-to-Record finished (best seen: %d, current: %d, deviation: %d).\n\n",
	   bestSeen, currCost, currCost - bestSeen);
  return currCost;
}


/*
** `localOptimization()' is a separate function for efficiency reasons. It
** performs the local optimization or hill-climbing procedure.
**
*/

costType localOptimization(int frozen, int endLimit)
{
  costType costDelta, currCost;
  int notChanged = 0, i, found;

  if(verbose >= 2)
    printf("Starting local optimization...\n");
  calculateCosts();
  currCost = initSolution();
  do {
    while(notChanged < frozen) {
      costDelta = computeNeighbor();
      iterCounter++;
      if(costDelta < 0) {
	acceptNeighbor();
	notChanged = 0;
	currCost += costDelta;
	if(currCost <= endLimit)
	  return currCost;       /* a good enough final solution was found */
      }
      else
	notChanged++;
    }
    found = 0;
    if(!onTheFly)
      for(i = 0; i < neighborLen && !found; i++)
	/* check a neighbor solution */;
  } while(found);
  if(verbose >= 2)
    printf("...local optimization accomplished.\n\n");
  return currCost;
}
