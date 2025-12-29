#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from itertools import combinations

try:
    from docplex.mp.model import Model
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: docplex.\n"
        "Install it with:\n"
        "  python -m pip install docplex\n"
    ) from exc

V = 27
K = 6
T = 3
M = 4


INCUMBENT_BLOCKS = [
    (1, 2, 4, 5, 19, 25),
    (1, 2, 6, 9, 11, 19),
    (1, 2, 10, 21, 22, 23),
    (1, 2, 12, 13, 16, 19),
    (1, 2, 14, 18, 19, 26),
    (1, 3, 4, 11, 21, 24),
    (1, 3, 7, 19, 20, 27),
    (1, 3, 8, 10, 17, 22),
    (1, 5, 8, 9, 21, 24),
    (1, 6, 21, 22, 24, 25),
    (1, 7, 8, 20, 23, 24),
    (1, 7, 12, 14, 20, 26),
    (1, 7, 13, 16, 18, 20),
    (1, 7, 15, 17, 20, 22),
    (1, 10, 12, 15, 17, 18),
    (1, 10, 13, 17, 23, 26),
    (1, 10, 14, 16, 17, 27),
    (1, 15, 21, 23, 24, 27),
    (2, 3, 4, 11, 17, 20),
    (2, 3, 7, 8, 22, 24),
    (2, 3, 10, 15, 20, 21),
    (2, 5, 8, 9, 17, 20),
    (2, 6, 17, 20, 22, 25),
    (2, 7, 8, 10, 21, 27),
    (2, 7, 12, 15, 18, 24),
    (2, 7, 13, 23, 24, 26),
    (2, 7, 14, 16, 24, 27),
    (2, 8, 15, 17, 19, 24),
    (2, 10, 12, 14, 21, 26),
    (2, 10, 13, 16, 18, 21),
    (2, 15, 17, 20, 23, 27),
    (3, 4, 5, 11, 14, 18),
    (3, 4, 7, 10, 11, 19),
    (3, 4, 8, 9, 22, 25),
    (3, 5, 6, 12, 13, 27),
    (3, 5, 6, 14, 18, 23),
    (3, 5, 6, 15, 16, 26),
    (3, 6, 11, 22, 23, 25),
    (3, 8, 12, 15, 22, 26),
    (3, 8, 13, 16, 22, 27),
    (3, 8, 19, 20, 21, 22),
    (3, 9, 12, 16, 23, 25),
    (3, 9, 13, 14, 15, 25),
    (3, 9, 18, 25, 26, 27),
    (3, 17, 19, 21, 23, 24),
    (4, 5, 6, 7, 17, 21),
    (4, 5, 10, 20, 24, 25),
    (4, 5, 12, 15, 25, 27),
    (4, 5, 13, 16, 25, 26),
    (4, 6, 8, 12, 16, 23),
    (4, 6, 8, 13, 14, 15),
    (4, 6, 8, 18, 26, 27),
    (4, 6, 9, 10, 20, 24),
    (4, 7, 9, 17, 21, 25),
    (4, 9, 12, 13, 22, 27),
    (4, 9, 14, 18, 22, 23),
    (4, 9, 15, 16, 22, 26),
    (4, 13, 15, 23, 26, 27),
    (5, 6, 10, 11, 20, 24),
    (5, 6, 14, 18, 22, 25),
    (5, 7, 8, 9, 10, 19),
    (5, 7, 11, 17, 21, 25),
    (5, 8, 9, 14, 18, 23),
    (5, 11, 12, 16, 22, 23),
    (5, 11, 13, 14, 15, 22),
    (5, 11, 18, 22, 26, 27),
    (6, 7, 9, 11, 17, 21),
    (6, 7, 10, 19, 22, 25),
    (6, 9, 11, 12, 13, 26),
    (6, 9, 11, 15, 16, 27),
    (7, 10, 15, 19, 23, 27),
    (7, 12, 16, 17, 21, 26),
    (7, 13, 14, 17, 18, 21),
    (8, 11, 12, 13, 25, 27),
    (8, 11, 14, 18, 23, 25),
    (8, 11, 15, 16, 25, 26),
    (9, 10, 11, 20, 24, 25),
    (10, 12, 13, 14, 20, 24),
    (10, 16, 18, 20, 24, 26),
    (10, 17, 19, 22, 24, 27),
    (12, 14, 16, 18, 23, 27),
    (12, 14, 17, 19, 24, 26),
    (12, 15, 18, 19, 20, 21),
    (13, 16, 17, 18, 19, 24),
    (13, 19, 20, 21, 23, 26),
    (14, 16, 19, 20, 21, 27),
]


def verify_design(blocks):
    all_triples = list(combinations(range(1, V + 1), T))
    coverage = Counter()
    for block in blocks:
        for triple in combinations(block, T):
            coverage[triple] += 1

    min_cov = min(coverage.get(triple, 0) for triple in all_triples)
    missing = [(triple, coverage.get(triple, 0)) for triple in all_triples if coverage.get(triple, 0) < M]
    return min_cov, missing


def build_model(max_blocks, use_mip_start):
    points = range(1, V + 1)
    all_blocks = list(combinations(points, K))
    all_triples = list(combinations(points, T))

    mdl = Model(name="C(27,6,3) multicover m=4")

    x = {block: mdl.binary_var(name=f"x_{'_'.join(map(str, block))}") for block in all_blocks}

    mdl.minimize(mdl.sum(x[block] for block in all_blocks))

    triple_to_blocks = defaultdict(list)
    for block in all_blocks:
        for triple in combinations(block, T):
            triple_to_blocks[triple].append(block)

    for triple in all_triples:
        mdl.add_constraint(
            mdl.sum(x[block] for block in triple_to_blocks[triple]) >= M,
            ctname=f"cov_{'_'.join(map(str, triple))}",
        )

    if max_blocks is not None:
        mdl.add_constraint(mdl.sum(x[block] for block in all_blocks) <= max_blocks, ctname="max_blocks")

    if use_mip_start:
        incumbent_set = set(tuple(sorted(block)) for block in INCUMBENT_BLOCKS)
        start_vals = {x[block]: (1 if block in incumbent_set else 0) for block in all_blocks}
        mdl.add_mip_start(start_vals)

    return mdl


def main():
    parser = argparse.ArgumentParser(description="Docplex verification and MIP for C(27,6,3)=4")
    parser.add_argument("--verify", action="store_true", help="Verify the 86-block solution")
    parser.add_argument("--solve", action="store_true", help="Solve the full MIP")
    parser.add_argument("--max-blocks", type=int, default=None, help="Add sum(x) <= max_blocks")
    parser.add_argument("--no-mip-start", action="store_true", help="Disable MIP start")
    parser.add_argument("--time-limit", type=int, default=None, help="Time limit in seconds")
    parser.add_argument("--log", action="store_true", help="Enable CPLEX log output")
    args = parser.parse_args()

    if args.verify:
        min_cov, missing = verify_design(INCUMBENT_BLOCKS)
        print(f"[verify] min_coverage={min_cov}")
        print(f"[verify] bad_triples={len(missing)}")
        if missing:
            print("[verify] example:", missing[:10])

    if not args.solve:
        return

    model = build_model(args.max_blocks, not args.no_mip_start)
    if args.time_limit is not None:
        model.parameters.timelimit = args.time_limit
    model.parameters.mip.tolerances.mipgap = 0

    solution = model.solve(log_output=args.log)
    if solution is None:
        print("[solve] no solution")
    else:
        print(solution)


if __name__ == "__main__":
    main()
