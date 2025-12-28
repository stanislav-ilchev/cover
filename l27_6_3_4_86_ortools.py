#!/usr/bin/env python3
import argparse
import itertools
from pathlib import Path

from ortools.sat.python import cp_model

V = 27
K = 6
B = 86
Msize = 4


def add_lex_leq(model: cp_model.CpModel, a_bits, b_bits, prefix_name: str):
    """
    Enforce lexicographic a_bits <= b_bits (binary vectors).
    """
    n = len(a_bits)
    eq = [model.NewBoolVar(f"{prefix_name}_eq_{i}") for i in range(n + 1)]
    model.Add(eq[0] == 1)

    for p in range(n):
        a = a_bits[p]
        b = b_bits[p]
        eqp = eq[p]
        eqn = eq[p + 1]
        z = model.NewBoolVar(f"{prefix_name}_eqbit_{p}")

        model.Add(a == b).OnlyEnforceIf(z)
        model.Add(a != b).OnlyEnforceIf(z.Not())

        model.Add(a - b <= 0).OnlyEnforceIf(eqp)

        model.AddBoolAnd([eqp, z]).OnlyEnforceIf(eqn)
        model.AddBoolOr([eqn, eqp.Not(), z.Not()])


def build_model():
    model = cp_model.CpModel()

    x = [[model.NewBoolVar(f"x_{j}_{p}") for p in range(V)] for j in range(B)]

    for j in range(B):
        model.Add(sum(x[j]) == K)

    for p in range(V):
        model.Add(x[0][p] == (1 if p < 6 else 0))

    for j in range(1, B - 1):
        add_lex_leq(model, x[j], x[j + 1], f"lex_{j}")

    M_list = list(itertools.combinations(range(V), Msize))
    y = []
    for mi, mset in enumerate(M_list):
        y_mi = []
        for j in range(B):
            y_mi.append([model.NewBoolVar(f"y_{mi}_{j}_{ev}") for ev in range(Msize)])
        y.append(y_mi)

        model.Add(sum(y_mi[j][ev] for j in range(B) for ev in range(Msize)) >= 1)

        for j in range(B):
            for ev in range(Msize):
                required = [mset[t] for t in range(Msize) if t != ev]
                for p in required:
                    model.Add(y_mi[j][ev] <= x[j][p])

    model.Minimize(0)
    return model, x


def extract_blocks(solver: cp_model.CpSolver, x):
    blocks = []
    for j in range(B):
        blocks.append([p for p in range(V) if solver.Value(x[j][p])])
    return blocks


def main():
    ap = argparse.ArgumentParser(description="OR-Tools CP-SAT encoding for L(27,6,4,3)=86")
    ap.add_argument("--solve", action="store_true", help="Solve the model")
    ap.add_argument("--proto", type=Path, help="Write the CP-SAT model proto to this path")
    ap.add_argument("--sol", type=Path, help="Write block solution to this path")
    ap.add_argument("--time-limit", type=float, default=None, help="Time limit in seconds")
    ap.add_argument("--workers", type=int, default=None, help="CP-SAT worker count")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--log", action="store_true", help="Enable CP-SAT search logging")
    args = ap.parse_args()

    model, x = build_model()

    if args.proto:
        model.ExportToFile(str(args.proto))
        print(f"[ok] wrote model proto to {args.proto}")

    if not args.solve:
        return

    solver = cp_model.CpSolver()
    if args.time_limit is not None:
        solver.parameters.max_time_in_seconds = args.time_limit
    if args.workers is not None:
        solver.parameters.num_search_workers = args.workers
    if args.seed is not None:
        solver.parameters.random_seed = args.seed
    if args.log:
        solver.parameters.log_search_progress = True

    status = solver.Solve(model)
    print(f"[result] status={solver.StatusName(status)}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return

    blocks = extract_blocks(solver, x)
    if args.sol:
        with args.sol.open("w", encoding="utf-8") as f:
            for j, bk in enumerate(blocks):
                f.write(f"B{j:02d}: {' '.join(map(str, bk))}\n")
        print(f"[ok] wrote blocks to {args.sol}")


if __name__ == "__main__":
    main()
