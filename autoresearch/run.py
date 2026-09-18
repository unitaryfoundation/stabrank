"""One autoresearch iteration: search a cell, log what it cost, submit if it wins.

The loop the board is built for is: pick a cell, spend compute looking for a
decomposition, refit exactly, verify, and record the cost whether or not
anything was found. This script is one iteration of that loop with the
bookkeeping made unavoidable. Every annealing run appends a line to
`autoresearch/runs.jsonl` with its configuration, seed, wall-clock, process
CPU time and residual, so the cost-per-verified-discovery curve can be drawn
from the log rather than reconstructed. When a run reaches the target rank the
decomposition is converted to the witness format, refit in exact arithmetic,
and written as a submission whose `provenance.compute` is the sum over every
run logged for that cell, failed ones included.

Usage:
    run.py ORBIT M RANK [--seeds 4] [--chains 8] [--iters 8000] [--cooling 0.995]
           [--author NAME] [--github HANDLE ...] [--llm MODEL] [--out bounds/X.json]

Runs from the repository root under `uv run --extra challenge`. A found
decomposition that does not refit exactly is logged as a near miss, not
submitted.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

LOG = os.path.join(ROOT, "autoresearch", "runs.jsonl")


def hardware():
    return f"{platform.machine()} {platform.system()}, {os.cpu_count()} cores"


def target(orbit, m):
    from stabrank_verify import target_vector
    v = np.array([complex(x) for x in target_vector(orbit, m)]).ravel()
    return v / np.linalg.norm(v)


def sector_target(m, s):
    """The Z^m eigensector s of |T3>^m, as the (m-1)-qutrit carry state.

    Since Z t_a = t_{a+3}, the three Galois cat states of |T3>^m are its Z^m
    eigensector projections; each lives in a 3^(m-1)-dimensional stabilizer
    code and, after the Clifford (x_1..x_m) -> (x_1..x_{m-1}, sum x), is the
    state psi_s(x) = w3^ceil((sum x - s)/3) on m-1 qutrits. Its stabilizer
    rank r_m bounds chi(|T3>^m) <= 3 r_m, and r_{m+m'-2} <= r_m r_{m'} by
    contraction; the cell that would beat the T3 exponent is r_5 <= 5 or
    r_6 <= 8.
    """
    n = m - 1
    w3 = np.exp(2j * np.pi / 3)
    v = np.empty(3 ** n, dtype=complex)
    for i in range(3 ** n):
        digits = [(i // 3 ** (n - 1 - j)) % 3 for j in range(n)]
        v[i] = w3 ** int(np.ceil((sum(digits) - s) / 3))
    return v / np.linalg.norm(v)


def warm_basis(path, p, m, rank, psi):
    """The terms of a verified witness, pruned to `rank` by dropping the least
    significant term repeatedly; a warm start for annealing one rank below a
    known decomposition."""
    from stabrank import (can_represent_as_linear_combination,
                          prune_least_significant_basis_function)
    from stabrank_verify import stabilizer_vector
    sub = json.load(open(path))
    funcs = [np.array([complex(z) for z in stabilizer_vector(t, p, m)])
             for t in sub["witness"]["terms"]]
    while len(funcs) > rank:
        funcs, _, _ = prune_least_significant_basis_function(
            psi, funcs, can_represent_as_linear_combination)
    return funcs


def anneal_once(orbit, m, rank, seed, chains, iters, cooling, warm=None):
    """One annealing run at fixed rank. Returns (vectors or None, residual, secs, cpu)."""
    from stabrank import generate_random_stabilizer_state
    from stabrank.stabrank_core import run_sa_pauli_expansion
    from stabrank_verify import ORBIT_P
    if orbit.startswith("T3sector"):
        p, psi = 3, sector_target(m + 1, int(orbit[len("T3sector"):]))
    else:
        p = ORBIT_P[orbit]
        psi = target(orbit, m)
    np.random.seed(seed)
    if warm:
        basis = warm_basis(warm, p, m, rank, psi)
    else:
        basis = [generate_random_stabilizer_state(m, p=p) for _ in range(rank)]
    t0, c0 = time.time(), time.process_time()
    _, funcs, _, err, _, _ = run_sa_pauli_expansion(
        target=psi, n_orig=m, p_prime=p, k_subset_size=rank, initial_basis=basis,
        initial_temperature=1.0, cooling_rate=cooling, num_iterations_at_temp=iters,
        min_temperature=1 / 4000, atol=1e-7, two_func_perturb_prob=0.3,
        random_replace_prob=0.05, use_real_qubit_moves=False, clifford_ratio=0.5,
        early_exit_threshold=1e-9, seed=seed, num_chains=chains)
    secs, cpu = time.time() - t0, time.process_time() - c0
    final = [np.asarray(f, dtype=complex) for f in funcs]
    vecs = final if err < 1e-9 else None
    return vecs, float(err), secs, cpu, final


def log_run(rec):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def cell_compute(orbit, m, rank, llm):
    """Sum of every logged run for this cell, for provenance.compute."""
    cpu = wall = 0.0
    runs = 0
    if os.path.exists(LOG):
        for line in open(LOG):
            r = json.loads(line)
            if (r["orbit"], r["m"], r["rank"]) == (orbit, m, rank):
                cpu += r["cpu_s"]
                wall += r["wall_s"]
                runs += 1
    comp = {"cpu_hours": round(cpu / 3600, 4), "wall_clock_hours": round(wall / 3600, 4),
            "runs": runs, "hardware": hardware()}
    if llm:
        comp["llm"] = [{"model": llm, "role": "drove the search"}]
    return comp


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("orbit", help="an orbit name, or T3sector<s> for the Z-eigensector s of "
                                  "|T3>^(m+1) as an m-qutrit carry state (logged, not submitted)")
    ap.add_argument("m", type=int)
    ap.add_argument("rank", type=int)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--cooling", type=float, default=0.995)
    ap.add_argument("--author", default="this repository")
    ap.add_argument("--github", nargs="*", default=[])
    ap.add_argument("--llm", default="", help="model identifier, if an LLM drove this search")
    ap.add_argument("--out", help="submission path; default bounds/ORBIT-mM-upper-RANK.json")
    ap.add_argument("--warm-from", default="", help="a bound file whose witness, pruned to the "
                    "target rank, seeds the annealer instead of random states")
    ap.add_argument("--save-plateaus", default="", help="directory in which to save the final "
                    "basis of every failed run, for completion search (autoresearch/kopt.py)")
    a = ap.parse_args(argv[1:])

    from to_witness import witness_from_vectors, NotStabilizer
    from stabrank_verify import verify, ORBIT_P, implied_gamma

    out = a.out or os.path.join(ROOT, "bounds", f"{a.orbit}-m{a.m}-upper-{a.rank}.json")
    residuals = []
    for seed in range(a.seed0, a.seed0 + a.seeds):
        vecs, err, secs, cpu, final = anneal_once(a.orbit, a.m, a.rank, seed, a.chains,
                                                  a.iters, a.cooling, warm=a.warm_from or None)
        if vecs is None and a.save_plateaus:
            os.makedirs(a.save_plateaus, exist_ok=True)
            np.savez(os.path.join(a.save_plateaus,
                                  f"plateau_{a.orbit}_m{a.m}_r{a.rank}_seed{seed}.npz"),
                     residual=err, *final)
        rec = {"when": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "orbit": a.orbit, "m": a.m, "rank": a.rank, "seed": seed, "chains": a.chains,
               "iters": a.iters, "cooling": a.cooling, "wall_s": round(secs, 1),
               "cpu_s": round(cpu, 1), "residual": err, "solved": vecs is not None,
               "hardware": hardware(), "llm": a.llm or None,
               "warm_from": os.path.relpath(a.warm_from, ROOT) if a.warm_from else None}
        residuals.append(err)
        exact = None
        if vecs is not None and not a.orbit.startswith("T3sector"):
            try:
                w = witness_from_vectors(a.orbit, a.m, vecs)
                exact = True
            except (NotStabilizer, ValueError) as exc:
                w, exact = None, False
                rec["refit"] = f"failed: {exc}"
        rec["exact"] = exact
        log_run(rec)
        if vecs is not None and a.orbit.startswith("T3sector"):
            np.savez(os.path.join(ROOT, "autoresearch",
                                  f"found_{a.orbit}_m{a.m}_r{a.rank}_seed{seed}.npz"), *vecs)
            print(f"sector decomposition found and saved; lift it with the cat construction "
                  f"(see the T3 m=5 bound notes) before submitting")
            return 0
        print(f"seed {seed}: residual {err:.3e} in {secs:.0f}s wall, {cpu:.0f}s CPU"
              + (", refit exact" if exact else (", refit FAILED" if exact is False else "")),
              flush=True)
        if exact and not a.orbit.startswith("T3sector"):
            gamma = implied_gamma(ORBIT_P[a.orbit], a.rank, a.m)
            sub = {
                "schema_version": "0.1", "orbit": a.orbit, "m": a.m, "direction": "upper",
                "rank": a.rank, "witness": w,
                "provenance": {
                    "author": a.author, "reference": "stabrank autoresearch/run.py",
                    "method": f"simulated annealing at rank {a.rank} ({a.chains} chains, "
                              f"{a.iters} iterations per temperature, cooling {a.cooling}), "
                              "then exact refit",
                    "date": datetime.date.today().isoformat(), "github": a.github,
                    "compute": cell_compute(a.orbit, a.m, a.rank, a.llm)},
                "notes": (f"Found by autoresearch/run.py on seed {seed} after "
                          f"{len(residuals) - 1} failed seed(s) at this rank "
                          f"(residuals {', '.join(f'{r:.3f}' for r in residuals[:-1]) or 'none'}). "
                          f"Implied gamma {gamma:.4f}. Every run is logged in autoresearch/runs.jsonl."),
            }
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                json.dump(sub, f, indent=2)
                f.write("\n")
            r = verify(sub)
            print(f"written {os.path.relpath(out, ROOT)}; verifier: {r}")
            return 0 if r.ok else 1
    print(f"no rank-{a.rank} decomposition in {a.seeds} seed(s); best residual "
          f"{min(residuals):.4f}. Runs are logged; nothing submitted.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
