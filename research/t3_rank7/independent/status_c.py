"""Progress of the case-C scan from logs/caseC_progress.jsonl.

Prints tasks done of total, inner steps covered of predicted, the step rate of the
current session (since logs/caseC_session.json was written by the launch script), the
projected wall time to completion, and the projected cost at the hourly rate given as
argv[1] (default 0.49 USD/h).  The total task count is cached in logs/caseC_tasks.json
because computing it loads the 640 MB kok pickle.
"""
import json, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
RATE_USD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.49


def task_total():
    cache = os.path.join(LOGS, "caseC_tasks.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    sys.path.insert(0, HERE)
    import caseC
    from setup_m3 import load
    d = load(3)
    z = np.load(os.path.join(LOGS, "costC.npz"))
    koks = pickle.load(open(os.path.join(LOGS, "costC_kok.pkl"), "rb"))
    tasks = caseC.build_tasks(z["reps"], z["jokm"], koks, d["N"])
    out = dict(tasks=len(tasks), steps_predicted=int(z["steps"]))
    json.dump(out, open(cache, "w"))
    return out


def progress():
    prog = os.path.join(LOGS, "caseC_progress.jsonl")
    n = 0; steps = 0; classes = 0; cands = 0
    if os.path.exists(prog):
        for line in open(prog):
            try:
                rec = json.loads(line)
            except ValueError:
                continue          # a partial last line while the run is writing
            n += 1; steps += rec["steps"]; classes += rec["classes"]; cands += len(rec["cands"])
    return n, steps, classes, cands


def main():
    tot = task_total()
    n, steps, classes, cands = progress()
    now = time.time()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now))}")
    print(f"tasks done {n} of {tot['tasks']} ({100 * n / tot['tasks']:.3f} %)")
    print(f"inner steps covered {steps:.4e} of {tot['steps_predicted']:.4e} ({100 * steps / tot['steps_predicted']:.3f} %)")
    print(f"class sets decided {classes}, candidates (rank equality) {cands}")
    sess = os.path.join(LOGS, "caseC_session.json")
    if not os.path.exists(sess):
        print("no session file: rate unknown"); return
    s = json.load(open(sess))
    dt = now - s["start_time"]; ds = steps - s["steps_at_start"]
    if dt <= 0 or ds <= 0:
        print("no progress in this session yet"); return
    rate = ds / dt
    remaining = tot["steps_predicted"] - steps
    eta_s = remaining / rate
    print(f"session: {dt / 3600:.2f} h elapsed, {ds:.4e} steps, {rate:.4e} steps/s "
          f"({1e9 / rate:.1f} ns per step of wall time across all workers)")
    print(f"projected remaining {eta_s / 3600:.1f} h = {eta_s / 86400:.2f} days; "
          f"completion {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(now + eta_s))}; "
          f"remaining cost {eta_s / 3600 * RATE_USD:.0f} USD at {RATE_USD} USD/h")


if __name__ == "__main__":
    main()
