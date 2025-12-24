# Strategies to Increase 3-Set Coverage While Keeping Cost=0

## Current Situation
- Cost = 0 means all m-subsets (6-subsets) are covered
- Current coverage: ~3074 / 18424 (16.68%) of 3-subsets
- Target: >= 3075 for world record
- Need: Increase coverage by at least 1 while maintaining cost=0

## Proposed Strategies

### Strategy 1: Coverage-Aware Acceptance (RECOMMENDED)
**When cost=0, prioritize moves that maintain cost=0 AND increase coverage**

- Current RR: Accepts if `newCost <= bestCost + threshold`
- When `bestCost == 0`: Accept moves with `newCost == 0` (maintains cost=0)
- Enhancement: When cost=0, also check coverage delta
- Accept if: `newCost == 0 AND newCoverage >= currentCoverage`
- This ensures we only accept moves that don't break cost=0 AND improve coverage

### Strategy 2: Two-Phase Search
**Switch to coverage-maximization mode when cost=0**

- Phase 1: Standard RR to reach cost=0
- Phase 2: When cost=0, switch to coverage-focused search:
  - Only accept moves that keep cost=0
  - Among cost=0 moves, prefer those with higher coverage
  - Use greedy local search: try multiple swaps, pick best coverage

### Strategy 3: Targeted Block Swaps
**Focus swaps on blocks that might cover new 3-sets**

- When cost=0, identify which 3-sets are NOT covered
- Try swaps that add elements from uncovered 3-sets
- Prefer swaps that add elements from frequently uncovered 3-sets

### Strategy 4: Block Diversity Maximization
**Ensure blocks are diverse to cover different 3-sets**

- Track which blocks cover which 3-sets
- Prefer swaps that increase diversity
- Avoid redundant blocks (blocks that cover same 3-sets)

### Strategy 5: Greedy Local Search (When Cost=0)
**Try all possible single swaps, pick best**

- When cost=0, for each block:
  - Try all possible single-element swaps
  - Evaluate coverage for each swap
  - Accept swap with highest coverage that maintains cost=0
- More expensive but more thorough

### Strategy 6: Acceptance Relaxation
**Allow temporary cost increases if they lead to better coverage**

- When cost=0, accept moves with `newCost <= smallThreshold` (e.g., 1-5)
- If new solution can be improved back to cost=0 with better coverage
- Risk: Might get stuck with cost>0

## Implementation Priority

1. **Strategy 1 (Coverage-Aware)**: Easiest to implement, good balance
2. **Strategy 2 (Two-Phase)**: More complex but potentially more effective
3. **Strategy 5 (Greedy)**: Most thorough but computationally expensive

## Recommended Approach

Combine Strategy 1 + Strategy 2:
- When cost=0, switch to coverage-aware mode
- Only accept moves that maintain cost=0
- Among cost=0 moves, prefer those with higher coverage
- Use a small threshold (TH=0 or TH=1) when cost=0 to be strict





