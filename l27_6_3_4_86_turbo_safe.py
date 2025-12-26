#!/usr/bin/env python3
import itertools
import argparse
import subprocess
import tempfile
from pathlib import Path

V = 27
K = 6
B = 86
Msize = 4


class CNFStream:
    def __init__(self, tmp_path: Path):
        self.f = open(tmp_path, "w", encoding="utf-8")
        self.num_clauses = 0

    def add_clause(self, *lits: int):
        self.f.write(" ".join(map(str, lits)) + " 0\n")
        self.num_clauses += 1

    def close(self):
        self.f.close()


class VarAlloc:
    def __init__(self):
        self.next_var = 1

    def new(self) -> int:
        v = self.next_var
        self.next_var += 1
        return v

    @property
    def num_vars(self) -> int:
        return self.next_var - 1


def bits_eq_lits(bits, value: int):
    lits = []
    for i, b in enumerate(bits):
        lits.append(b if ((value >> i) & 1) else -b)
    return lits


def at_most_k(cnf: CNFStream, va: VarAlloc, vars_list, k: int):
    n = len(vars_list)
    if k >= n:
        return
    if k == 0:
        for v in vars_list:
            cnf.add_clause(-v)
        return

    s = [[va.new() for _ in range(k)] for _ in range(n - 1)]

    cnf.add_clause(-vars_list[0], s[0][0])
    for j in range(1, k):
        cnf.add_clause(-s[0][j])

    for i in range(1, n - 1):
        cnf.add_clause(-vars_list[i], s[i][0])
        cnf.add_clause(-s[i - 1][0], s[i][0])
        for j in range(1, k):
            cnf.add_clause(-vars_list[i], -s[i - 1][j - 1], s[i][j])
            cnf.add_clause(-s[i - 1][j], s[i][j])
        cnf.add_clause(-vars_list[i], -s[i - 1][k - 1])

    cnf.add_clause(-vars_list[n - 1], -s[n - 2][k - 1])


def exactly_k(cnf: CNFStream, va: VarAlloc, vars_list, k: int):
    at_most_k(cnf, va, vars_list, k)
    negs = [-v for v in vars_list]
    at_most_k(cnf, va, negs, len(vars_list) - k)


def add_lex_leq(cnf: CNFStream, va: VarAlloc, a_bits, b_bits):
    """
    Correct CNF for a <=_lex b over {0,1}^n using prefix equality vars.
    """
    n = len(a_bits)
    assert n == len(b_bits)

    eq = [va.new() for _ in range(n + 1)]
    cnf.add_clause(eq[0])  # eq[0] = True

    for p in range(n):
        a = a_bits[p]
        b = b_bits[p]
        eqp = eq[p]
        eqn = eq[p + 1]

        # lex condition while prefix equal: eqp -> (¬a ∨ b)
        cnf.add_clause(-eqp, -a, b)

        # eqn -> eqp
        cnf.add_clause(-eqn, eqp)

        # eqn -> (a <-> b)
        cnf.add_clause(-eqn, -a, b)  # a -> b
        cnf.add_clause(-eqn, a, -b)  # b -> a

        # (eqp & a & b) -> eqn
        cnf.add_clause(-eqp, -a, -b, eqn)

        # (eqp & ~a & ~b) -> eqn
        cnf.add_clause(-eqp, a, b, eqn)


def generate_cnf(out_cnf: Path):
    va = VarAlloc()

    # x first for decoding: var_id = 1 + j*V + p
    x = [[va.new() for _ in range(V)] for _ in range(B)]

    M_list = list(itertools.combinations(range(V), Msize))
    num_M = len(M_list)

    bbits = [[va.new() for _ in range(7)] for _ in range(num_M)]
    ebits = [[va.new() for _ in range(2)] for _ in range(num_M)]

    with tempfile.TemporaryDirectory() as td:
        body_path = Path(td) / "body.cnf"
        cnf = CNFStream(body_path)

        # 1) block sizes
        for j in range(B):
            exactly_k(cnf, va, x[j], K)

        # 2) fix block 0 to {0,1,2,3,4,5}
        for p in range(V):
            cnf.add_clause(x[0][p] if p < 6 else -x[0][p])

        # 3) TURBO but SAFE: lex order only among blocks 1..85
        for j in range(1, B - 1):
            add_lex_leq(cnf, va, x[j], x[j + 1])

        # 4) restrict witness block index to [0..85]
        for mi in range(num_M):
            bits = bbits[mi]
            for bad in range(B, 128):
                eq = bits_eq_lits(bits, bad)
                cnf.add_clause(*[-lit for lit in eq])

        # 5) coverage constraints
        for mi, M in enumerate(M_list):
            m = list(M)
            b = bbits[mi]
            e = ebits[mi]
            eq_e_lits = [bits_eq_lits(e, ev) for ev in range(4)]

            for j in range(B):
                eq_j_lits = bits_eq_lits(b, j)
                neg_eq_j = [-lit for lit in eq_j_lits]

                for ev in range(4):
                    neg_eq_e = [-lit for lit in eq_e_lits[ev]]
                    required = [m[t] for t in range(4) if t != ev]
                    for p in required:
                        cnf.add_clause(*(neg_eq_j + neg_eq_e + [x[j][p]]))

        cnf.close()

        with open(out_cnf, "w", encoding="utf-8") as out:
            out.write(f"p cnf {va.num_vars} {cnf.num_clauses}\n")
            with open(body_path, "r", encoding="utf-8") as body:
                for line in body:
                    out.write(line)

    return va.num_vars, cnf.num_clauses


def decode_blocks_from_kissat_output(kissat_out: str):
    trues = set()
    for line in kissat_out.splitlines():
        line = line.strip()
        if not line.startswith("v"):
            continue
        for tok in line.split()[1:]:
            lit = int(tok)
            if lit > 0:
                trues.add(lit)

    blocks = []
    for j in range(B):
        base = 1 + j * V
        blocks.append([p for p in range(V) if (base + p) in trues])
    return blocks


def run_kissat(kissat_bin: str, cnf_path: Path):
    proc = subprocess.run(
        [kissat_bin, str(cnf_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.stdout


def main():
    ap = argparse.ArgumentParser(description="Turbo+safe SAT encoding for L(27,6,4,3)=86 (Kissat).")
    ap.add_argument("--cnf", default="L27_6_3_4_eq86_turbo_safe.cnf")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--kissat", default="kissat")
    ap.add_argument("--print-blocks", action="store_true")
    args = ap.parse_args()

    cnf_path = Path(args.cnf)
    nvars, ncls = generate_cnf(cnf_path)
    print(f"[ok] wrote {cnf_path}  vars={nvars} clauses={ncls}")

    if not args.solve:
        return

    out = run_kissat(args.kissat, cnf_path)
    if "UNSATISFIABLE" in out:
        print("[result] UNSAT")
        return
    if "SATISFIABLE" not in out:
        print("[result] no SAT/UNSAT marker; output follows:\n")
        print(out)
        return

    print("[result] SAT")
    blocks = decode_blocks_from_kissat_output(out)
    if args.print_blocks:
        for j, bk in enumerate(blocks):
            print(f"B{j:02d}:", " ".join(map(str, bk)))


if __name__ == "__main__":
    main()
