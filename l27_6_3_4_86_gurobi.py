#!/usr/bin/env python3
import argparse
import itertools
from pathlib import Path

from gurobipy import GRB, Model, quicksum

V = 27
K = 6
B = 86
Msize = 4


def add_lex_leq(model: Model, a_bits, b_bits, prefix_name: str):
    """
    Enforce lexicographic a_bits <= b_bits (binary vectors).
    """
    eq = model.addVars(len(a_bits) + 1, vtype=GRB.BINARY, name=f"{prefix_name}_eq")
    model.addConstr(eq[0] == 1, name=f"{prefix_name}_eq0")

    for p in range(len(a_bits)):
        a = a_bits[p]
        b = b_bits[p]
        z = model.addVar(vtype=GRB.BINARY, name=f"{prefix_name}_eqbit_{p}")

        # z == (a == b)
        model.addConstr(z <= 1 - a + b, name=f"{prefix_name}_eqbit_ub1_{p}")
        model.addConstr(z <= 1 - b + a, name=f"{prefix_name}_eqbit_ub2_{p}")
        model.addConstr(z >= a + b - 1, name=f"{prefix_name}_eqbit_lb1_{p}")
        model.addConstr(z >= 1 - a - b, name=f"{prefix_name}_eqbit_lb2_{p}")

        # eq[p+1] == eq[p] AND z
        model.addConstr(eq[p + 1] <= eq[p], name=f"{prefix_name}_eq_and1_{p}")
        model.addConstr(eq[p + 1] <= z, name=f"{prefix_name}_eq_and2_{p}")
        model.addConstr(eq[p + 1] >= eq[p] + z - 1, name=f"{prefix_name}_eq_and3_{p}")

        # If prefix equal, enforce a <= b at position p
        model.addConstr(a - b <= 1 - eq[p], name=f"{prefix_name}_lex_{p}")


def build_model():
    model = Model("L27_6_3_4_86")

    x = model.addVars(B, V, vtype=GRB.BINARY, name="x")

    # Block sizes
    for j in range(B):
        model.addConstr(quicksum(x[j, p] for p in range(V)) == K, name=f"block_size_{j}")

    # Fix block 0 to {0,1,2,3,4,5}
    for p in range(V):
        model.addConstr(x[0, p] == (1 if p < 6 else 0), name=f"block0_{p}")

    # Lex order among blocks 1..85
    for j in range(1, B - 1):
        add_lex_leq(model, [x[j, p] for p in range(V)], [x[j + 1, p] for p in range(V)], f"lex_{j}")

    # Coverage constraints
    M_list = list(itertools.combinations(range(V), Msize))
    y = model.addVars(len(M_list), B, Msize, vtype=GRB.BINARY, name="y")

    for mi, M in enumerate(M_list):
        model.addConstr(
            quicksum(y[mi, j, ev] for j in range(B) for ev in range(Msize)) >= 1,
            name=f"cover_{mi}",
        )

        for j in range(B):
            for ev in range(Msize):
                required = [M[t] for t in range(Msize) if t != ev]
                for p in required:
                    model.addConstr(
                        y[mi, j, ev] <= x[j, p],
                        name=f"cover_{mi}_{j}_{ev}_{p}",
                    )

    model.setObjective(0, GRB.MINIMIZE)
    return model, x


def extract_blocks(x):
    blocks = []
    for j in range(B):
        blocks.append([p for p in range(V) if x[j, p].X > 0.5])
    return blocks


def main():
    ap = argparse.ArgumentParser(description="Gurobi MIP encoding for L(27,6,4,3)=86")
    ap.add_argument("--solve", action="store_true", help="Solve the model")
    ap.add_argument("--lp", type=Path, help="Write LP file to this path")
    ap.add_argument("--sol", type=Path, help="Write block solution to this path")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi threads")
    ap.add_argument("--time-limit", type=float, default=None, help="Time limit in seconds")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    args = ap.parse_args()

    model, x = build_model()

    if args.threads is not None:
        model.setParam(GRB.Param.Threads, args.threads)
    if args.time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, args.time_limit)
    if args.seed is not None:
        model.setParam(GRB.Param.Seed, args.seed)

    if args.lp:
        model.write(str(args.lp))
        print(f"[ok] wrote LP to {args.lp}")

    if not args.solve:
        return

    model.optimize()

    if model.SolCount == 0:
        print("[result] no solution")
        return

    print(f"[result] solution status={model.Status}")
    blocks = extract_blocks(x)
    if args.sol:
        with args.sol.open("w", encoding="utf-8") as f:
            for j, bk in enumerate(blocks):
                f.write(f"B{j:02d}: {' '.join(map(str, bk))}\n")
        print(f"[ok] wrote blocks to {args.sol}")


if __name__ == "__main__":
    main()
