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


def anneal_once(orbit, m, rank, seed, chains, iters, cooling):
    """One annealing run at fixed rank. Returns (vectors or None, residual, secs, cpu)."""
    from stabrank import generate_random_stabilizer_state
    from stabrank.stabrank_core import run_sa_pauli_expansion
    from stabrank_verify import ORBIT_P
    p = ORBIT_P[orbit]
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    psi = target(orbit, m)
    basis = [generate_random_stabilizer_state(m, p=p) for _ in range(rank)]
    t0, c0 = time.time(), time.process_time()
    _, funcs, _, err, _, _ = run_sa_pauli_expansion(
        target=psi, n_orig=m, p_prime=p, k_subset_size=rank, initial_basis=basis,
        initial_temperature=1.0, cooling_rate=cooling, num_iterations_at_temp=iters,
        min_temperature=1 / 4000, atol=1e-7, two_func_perturb_prob=0.3,
        random_replace_prob=0.05, use_real_qubit_moves=False, clifford_ratio=0.5,
        early_exit_threshold=1e-9, seed=seed, num_chains=chains)
    secs, cpu = time.time() - t0, time.process_time() - c0
    vecs = [np.asarray(f, dtype=complex) for f in funcs] if err < 1e-9 else None
    return vecs, float(err), secs, cpu


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
    ap.add_argument("orbit")
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
    a = ap.parse_args(argv[1:])

    from to_witness import witness_from_vectors, NotStabilizer
    from stabrank_verify import verify, ORBIT_P, implied_gamma

    out = a.out or os.path.join(ROOT, "bounds", f"{a.orbit}-m{a.m}-upper-{a.rank}.json")
    residuals = []
    for seed in range(a.seed0, a.seed0 + a.seeds):
        vecs, err, secs, cpu = anneal_once(a.orbit, a.m, a.rank, seed, a.chains, a.iters,
                                           a.cooling)
        rec = {"when": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "orbit": a.orbit, "m": a.m, "rank": a.rank, "seed": seed, "chains": a.chains,
               "iters": a.iters, "cooling": a.cooling, "wall_s": round(secs, 1),
               "cpu_s": round(cpu, 1), "residual": err, "solved": vecs is not None,
               "hardware": hardware(), "llm": a.llm or None}
        residuals.append(err)
        exact = None
        if vecs is not None:
            try:
                w = witness_from_vectors(a.orbit, a.m, vecs)
                exact = True
            except (NotStabilizer, ValueError) as exc:
                w, exact = None, False
                rec["refit"] = f"failed: {exc}"
        rec["exact"] = exact
        log_run(rec)
        print(f"seed {seed}: residual {err:.3e} in {secs:.0f}s wall, {cpu:.0f}s CPU"
              + (", refit exact" if exact else (", refit FAILED" if exact is False else "")),
              flush=True)
        if exact:
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
