"""Unattended scheduler for autoresearch/run.py: manifest in, checkpointed log out.

`run.py` is one iteration of search, refit, verify and record. This script
runs many of them for days without a person watching. A manifest lists the
cells to attack, the seeds to spend on each, the annealer settings and a
wall-clock cap per job. The loop runs exactly one job at a time, as a
subprocess of `run.py` under `nice -n 19` with `OMP_NUM_THREADS=1`, taking
one seed from every cell in priority order before coming back for the next
seed, so no cell starves. After every job it checkpoints which seeds are done
to `autoresearch/state/<manifest>.json`, so `--resume` picks up where a crash
or a shutdown left off, and `replay` runs the exact job sequence a state file
records. Every job gets one JSON line in `autoresearch/loop.log` with start
and end time, wall clock, CPU time, exit code and a failure class read from
the exit code and the stderr tail. Infrastructure failures are retried once;
a search that simply did not find anything is never retried.

Usage:
    loop.py run MANIFEST [--resume] [--max-hours H] [--max-jobs N] [--dry-run]
    loop.py replay STATE [--max-hours H] [--max-jobs N]
    loop.py status [--since HOURS] [--json]

Runs from the repository root under `uv run --extra challenge`. `run.py`
appends to `autoresearch/runs.jsonl` as usual; this script never writes to it.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import resource
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "autoresearch", "run.py")
STATE_DIR = os.path.join(ROOT, "autoresearch", "state")
LOG = os.path.join(ROOT, "autoresearch", "loop.log")

JOB_DEFAULTS = {"seed0": 1, "seeds": 1, "chains": 8, "iters": 8000, "cooling": 0.995,
                "priority": 0, "cap_s": 4 * 3600, "seeds_per_round": 1,
                "stop_on_solve": True, "llm": "", "warm_from": "", "save_plateaus": "",
                "extra_args": []}

# Outcomes that end a job for good. Everything else is retried once.
SEARCH_OUTCOMES = {"discovery", "miss", "near_miss", "verifier_refusal", "timeout"}
INFRA_EXCEPTIONS = {"ImportError", "ModuleNotFoundError", "OSError", "FileNotFoundError",
                    "PermissionError", "BrokenPipeError", "ConnectionError", "MemoryError",
                    "BlockingIOError", "TimeoutError"}
STDERR_TAIL = 8


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(dt):
    return dt.isoformat(timespec="seconds")


def parse_iso(s):
    return datetime.datetime.fromisoformat(s)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# --- manifest -----------------------------------------------------------------

def load_manifest(path):
    """The manifest with defaults applied to every job and ids filled in."""
    with open(path) as f:
        man = json.load(f)
    defaults = dict(JOB_DEFAULTS)
    defaults.update(man.get("defaults", {}))
    jobs = []
    seen = set()
    for i, raw in enumerate(man.get("jobs", [])):
        for key in ("orbit", "m", "rank"):
            if key not in raw:
                raise ValueError(f"job {i} lacks {key!r}")
        job = dict(defaults)
        job.update(raw)
        job.setdefault("id", f"{job['orbit']}-m{job['m']}-r{job['rank']}")
        if job["id"] in seen:
            raise ValueError(f"duplicate job id {job['id']!r}")
        seen.add(job["id"])
        job["index"] = i
        jobs.append(job)
    if not jobs:
        raise ValueError("manifest lists no jobs")
    name = man.get("name") or os.path.splitext(os.path.basename(path))[0]
    return {"name": name, "path": os.path.abspath(path), "sha256": sha256_file(path),
            "jobs": jobs}


def job_seeds(job):
    return list(range(job["seed0"], job["seed0"] + job["seeds"]))


def schedule(jobs, done):
    """The (job id, seed) sequence still to run: rounds over the jobs in priority
    order (larger first, then manifest order), `seeds_per_round` seeds each."""
    rank = {j["id"]: r for r, j in
            enumerate(sorted(jobs, key=lambda j: (-j["priority"], j["index"])))}
    keyed = []
    for j in jobs:
        d = done.get(j["id"], {})
        if j["stop_on_solve"] and d.get("solved"):
            continue
        finished = set(d.get("seeds_done", []))
        per_round = max(1, int(j["seeds_per_round"]))
        for k, seed in enumerate(job_seeds(j)):
            if seed not in finished:
                # the round is the seed's absolute position, so a resumed loop
                # continues the same global order the fresh one would have run
                keyed.append(((k // per_round, rank[j["id"]], k), (j["id"], seed)))
    keyed.sort()
    return [unit for _, unit in keyed]


def command(job, seed, runner):
    cmd = ["nice", "-n", "19", sys.executable, runner, str(job["orbit"]), str(job["m"]),
           str(job["rank"]), "--seeds", "1", "--seed0", str(seed),
           "--chains", str(job["chains"]), "--iters", str(job["iters"]),
           "--cooling", str(job["cooling"])]
    if job["llm"]:
        cmd += ["--llm", job["llm"]]
    if job["warm_from"]:
        cmd += ["--warm-from", job["warm_from"]]
    if job["save_plateaus"]:
        cmd += ["--save-plateaus", job["save_plateaus"]]
    cmd += [str(x) for x in job["extra_args"]]
    return cmd


# --- diagnosis ----------------------------------------------------------------

def classify(exit_code, stdout, stderr, timed_out=False, interrupted=False):
    """(outcome, retry). Read the exit code first, then the stderr tail."""
    if interrupted:
        return "interrupted", False
    if timed_out:
        return "timeout", False
    if exit_code == 0:
        return "discovery", False
    err = stderr or ""
    out = stdout or ""
    if exit_code is not None and exit_code < 0:
        sig = -exit_code
        if sig == signal.SIGKILL:
            return "oom", True
        return f"killed:{signal.Signals(sig).name}", True
    if exit_code == 137:
        return "oom", True
    if "MemoryError" in err or "Killed" in err or "out of memory" in err.lower():
        return "oom", True
    if "usage:" in err and "error:" in err and "Traceback" not in err:
        return "exception:ArgumentError", False
    if exit_code == 2 and "Traceback" not in err:
        if "refit FAILED" in out:
            return "near_miss", False
        return "miss", False
    if exit_code == 1 and "Traceback" not in err and "verifier:" in out:
        return "verifier_refusal", False
    if "Traceback" in err:
        name = exception_name(err)
        if name in INFRA_EXCEPTIONS:
            return f"infrastructure:{name}", True
        return f"exception:{name}", False
    if exit_code in (126, 127):
        return "infrastructure:spawn", True
    return f"unknown:exit{exit_code}", True


def exception_name(stderr):
    for line in reversed(stderr.strip().splitlines()):
        line = line.strip()
        if not line or line.startswith(("File ", "Traceback", "^", "During handling",
                                         "The above exception")):
            continue
        head = line.split(":", 1)[0].strip()
        if head and all(c.isalnum() or c in "._" for c in head):
            return head.rsplit(".", 1)[-1]
    return "Unknown"


def tail(text, n=STDERR_TAIL):
    lines = (text or "").rstrip().splitlines()
    return lines[-n:]


def parse_residual(stdout):
    for line in reversed((stdout or "").splitlines()):
        toks = line.replace(",", " ").replace(";", " ").split()
        for i, tok in enumerate(toks[:-1]):
            if tok == "residual":
                try:
                    return float(toks[i + 1].rstrip(".:"))
                except ValueError:
                    continue
    return None


# --- state and log ------------------------------------------------------------

def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def append_log(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def read_log(path):
    if not os.path.exists(path):
        return []
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# --- the loop -----------------------------------------------------------------

class Loop:
    def __init__(self, manifest, state_path, log_path, runner=RUNNER, max_hours=None,
                 max_jobs=None, resume=False, replay_sequence=None):
        self.manifest = manifest
        self.state_path = state_path
        self.log_path = log_path
        self.runner = runner
        self.max_s = max_hours * 3600 if max_hours else None
        self.max_jobs = max_jobs
        self.replay_sequence = replay_sequence
        self.jobs = {j["id"]: j for j in manifest["jobs"]}
        self.loop_id = uuid.uuid4().hex[:8]
        self.proc = None
        self.stop = None
        self.signals = 0
        prior = load_json(state_path, None) if resume else None
        if prior and prior.get("manifest_sha256") != manifest["sha256"]:
            print(f"note: manifest changed since {state_path} was written; "
                  f"done seeds are kept, the schedule is recomputed", file=sys.stderr)
        self.state = prior or {
            "manifest": manifest["path"], "manifest_name": manifest["name"],
            "manifest_sha256": manifest["sha256"], "jobs": manifest["jobs"],
            "created": iso(utcnow()), "sequence": [], "done": {}, "loops": []}
        self.state["loops"].append({"loop_id": self.loop_id, "started": iso(utcnow()),
                                    "argv": sys.argv[1:], "host": socket.gethostname(),
                                    "pid": os.getpid()})

    # -- signals --
    def install_signals(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.on_signal)

    def on_signal(self, signum, frame):
        self.signals += 1
        self.stop = f"signal:{signal.Signals(signum).name}"
        if self.proc is not None and self.proc.poll() is None:
            if self.signals == 1:
                self.proc.terminate()
            else:
                self.proc.kill()
        if self.signals >= 3:
            raise SystemExit(128 + signum)

    # -- running --
    def pending(self):
        if self.replay_sequence is not None:
            done = self.state["done"]
            seq = []
            for jid, seed in self.replay_sequence:
                if jid not in self.jobs:
                    continue
                if seed in done.get(jid, {}).get("seeds_done", []):
                    continue
                if (jid, seed) not in seq:
                    seq.append((jid, seed))
            return seq
        return schedule(self.manifest["jobs"], self.state["done"])

    def run(self):
        t_start = time.monotonic()
        started = utcnow()
        append_log(self.log_path, {"event": "start", "loop_id": self.loop_id,
                                   "manifest": self.manifest["name"], "when": iso(started),
                                   "host": socket.gethostname(), "pid": os.getpid(),
                                   "max_hours": self.max_s / 3600 if self.max_s else None,
                                   "replay": self.replay_sequence is not None})
        self.checkpoint()
        n = 0
        while self.stop is None:
            seq = self.pending()
            if not seq:
                self.stop = "exhausted"
                break
            if self.max_jobs is not None and n >= self.max_jobs:
                self.stop = "max-jobs"
                break
            jid, seed = seq[0]
            job = self.jobs[jid]
            if self.max_s is not None:
                remaining = self.max_s - (time.monotonic() - t_start)
                if remaining < job["cap_s"]:
                    self.stop = "max-hours"
                    break
            rec = self.run_job(job, seed, attempt=1)
            n += 1
            if rec["retry"] and self.stop is None:
                rec = self.run_job(job, seed, attempt=2)
                n += 1
        ended = utcnow()
        self.state["loops"][-1]["ended"] = iso(ended)
        self.state["loops"][-1]["stop"] = self.stop
        self.checkpoint()
        append_log(self.log_path, {"event": "stop", "loop_id": self.loop_id,
                                   "manifest": self.manifest["name"], "when": iso(ended),
                                   "reason": self.stop, "jobs": n,
                                   "wall_s": round((ended - started).total_seconds(), 1)})
        print(f"loop {self.loop_id} stopped: {self.stop} after {n} job(s)", flush=True)
        return 0 if self.stop in ("exhausted", "max-hours", "max-jobs") else 130

    def run_job(self, job, seed, attempt):
        cmd = command(job, seed, self.runner)
        env = dict(os.environ)
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMBA_NUM_THREADS"):
            env[v] = "1"
        started = utcnow()
        t0 = time.monotonic()
        ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        timed_out = False
        stdout = stderr = ""
        code = None
        print(f"[{iso(started)}] {job['id']} seed {seed}"
              + (f" (attempt {attempt})" if attempt > 1 else ""), flush=True)
        try:
            self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True)
        except OSError as exc:
            self.proc = None
            stderr = f"Traceback (most recent call last):\n{type(exc).__name__}: {exc}"
            code = 127
        else:
            try:
                stdout, stderr = self.proc.communicate(timeout=job["cap_s"])
            except subprocess.TimeoutExpired:
                timed_out = True
                self.proc.terminate()
                try:
                    stdout, stderr = self.proc.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    stdout, stderr = self.proc.communicate()
            code = self.proc.returncode
            self.proc = None
        wall = time.monotonic() - t0
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
        interrupted = self.stop is not None and not timed_out and code not in (0, 2)
        outcome, retry = classify(code, stdout, stderr, timed_out=timed_out,
                                  interrupted=interrupted)
        retry = retry and attempt == 1
        ended = utcnow()
        rec = {"event": "job", "loop_id": self.loop_id, "manifest": self.manifest["name"],
               "job": job["id"], "orbit": job["orbit"], "m": job["m"], "rank": job["rank"],
               "seed": seed, "attempt": attempt, "started": iso(started), "ended": iso(ended),
               "wall_s": round(wall, 1), "cpu_s": round(cpu, 1), "exit_code": code,
               "outcome": outcome, "retry": retry, "residual": parse_residual(stdout),
               "stderr_tail": tail(stderr), "stdout_last": tail(stdout, 1),
               "cap_s": job["cap_s"], "cmd": cmd[3:]}
        append_log(self.log_path, rec)
        entry = {"job": job["id"], "seed": seed, "attempt": attempt, "outcome": outcome,
                 "started": rec["started"], "ended": rec["ended"], "wall_s": rec["wall_s"]}
        self.state["sequence"].append(entry)
        d = self.state["done"].setdefault(job["id"], {"seeds_done": [], "solved": False,
                                                       "outcomes": {}})
        if outcome != "interrupted" and not retry:
            if seed not in d["seeds_done"]:
                d["seeds_done"].append(seed)
            d["outcomes"][str(seed)] = outcome
            if outcome == "discovery":
                d["solved"] = True
        self.checkpoint()
        print(f"    {outcome} in {wall:.0f}s wall, {cpu:.0f}s CPU"
              + (f", residual {rec['residual']:.4g}" if rec["residual"] is not None else "")
              + (", retrying" if retry else ""), flush=True)
        if outcome not in SEARCH_OUTCOMES and rec["stderr_tail"]:
            for line in rec["stderr_tail"][-3:]:
                print(f"    | {line}", flush=True)
        return rec

    def checkpoint(self):
        self.state["updated"] = iso(utcnow())
        dump_json(self.state_path, self.state)


# --- status -------------------------------------------------------------------

def status(log_path, since_hours=None):
    recs = read_log(log_path)
    now = utcnow()
    cutoff = now - datetime.timedelta(hours=since_hours) if since_hours else None
    jobs = [r for r in recs if r.get("event") == "job"]
    starts = [r for r in recs if r.get("event") == "start"]
    stops = [r for r in recs if r.get("event") == "stop"]
    if cutoff:
        jobs = [r for r in jobs if parse_iso(r["ended"]) >= cutoff]
        starts = [r for r in starts if parse_iso(r["when"]) >= cutoff]
        stops = [r for r in stops if parse_iso(r["when"]) >= cutoff]
    jobs.sort(key=lambda r: r["started"])
    by_class = collections.Counter(r["outcome"] for r in jobs)
    failures = {k: v for k, v in by_class.items()
                if k not in ("discovery", "miss", "near_miss")}
    discoveries = [f"{r['job']} seed {r['seed']}" for r in jobs if r["outcome"] == "discovery"]
    near = [f"{r['job']} seed {r['seed']}" for r in jobs if r["outcome"] == "near_miss"]
    retries = sum(1 for r in jobs if r["attempt"] > 1)
    wall = sum(r["wall_s"] for r in jobs)
    busy = sum((parse_iso(r["ended"]) - parse_iso(r["started"])).total_seconds() for r in jobs)
    cpu = sum(r["cpu_s"] for r in jobs)
    covered = gap = 0.0
    gap_at = None
    if jobs:
        first = parse_iso(jobs[0]["started"])
        last = max(parse_iso(r["ended"]) for r in jobs)
        covered = (last - first).total_seconds()
        end_prev = parse_iso(jobs[0]["ended"])
        for r in jobs[1:]:
            g = (parse_iso(r["started"]) - end_prev).total_seconds()
            if g > gap:
                gap, gap_at = g, (iso(end_prev), r["started"])
            end_prev = max(end_prev, parse_iso(r["ended"]))
    per_cell = collections.defaultdict(lambda: collections.Counter())
    for r in jobs:
        per_cell[r["job"]][r["outcome"]] += 1
    return {"log": log_path, "since_hours": since_hours, "jobs": len(jobs),
            "sessions": len(starts), "stops": collections.Counter(r["reason"] for r in stops),
            "by_outcome": dict(by_class), "failures": failures, "retries": retries,
            "discoveries": discoveries, "near_misses": near,
            "wall_hours": wall / 3600, "cpu_hours": cpu / 3600,
            "covered_hours": covered / 3600,
            "busy_fraction": min(1.0, busy / covered) if covered else None,
            "longest_gap_s": gap, "longest_gap_at": gap_at,
            "first": jobs[0]["started"] if jobs else None,
            "last": max(r["ended"] for r in jobs) if jobs else None,
            "per_cell": {k: dict(v) for k, v in sorted(per_cell.items())}}


def hms(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def print_status(s):
    window = f"last {s['since_hours']:g} h" if s["since_hours"] else "whole log"
    rel = os.path.relpath(s["log"], ROOT)
    print(f"loop status from {s['log'] if rel.startswith('..') else rel} ({window})")
    if not s["jobs"]:
        print("no jobs logged")
        return
    print(f"jobs run: {s['jobs']} in {s['sessions']} session(s); retries {s['retries']}; "
          f"stops {dict(s['stops']) or '{}'}")
    print(f"outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(s["by_outcome"].items())))
    print(f"failures by class: "
          + (", ".join(f"{k} {v}" for k, v in sorted(s["failures"].items())) or "none"))
    print(f"discoveries: {', '.join(s['discoveries']) or 'none'}")
    if s["near_misses"]:
        print(f"near misses (solved numerically, no exact refit): {', '.join(s['near_misses'])}")
    print(f"compute: {s['wall_hours']:.2f} h wall, {s['cpu_hours']:.2f} h CPU")
    for cell, c in s["per_cell"].items():
        print(f"  {cell}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    busy = f", busy {100 * s['busy_fraction']:.0f}%" if s["busy_fraction"] is not None else ""
    gap = f"longest gap between jobs {hms(s['longest_gap_s'])}"
    if s["longest_gap_at"]:
        gap += f" ({s['longest_gap_at'][0]} to {s['longest_gap_at'][1]})"
    print(f"operability: {s['covered_hours']:.1f} h covered from {s['first']} to {s['last']}"
          f"{busy}; {gap}; {sum(s['failures'].values())} failure(s), all classified")


# --- cli ----------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--max-hours", type=float, default=None,
                       help="stop cleanly once no job's cap fits before this wall-clock limit")
        p.add_argument("--max-jobs", type=int, default=None)
        p.add_argument("--state-dir", default=STATE_DIR)
        p.add_argument("--log", default=LOG)
        p.add_argument("--runner", default=RUNNER, help="the script run per job (run.py)")
        p.add_argument("--dry-run", action="store_true", help="print the schedule and exit")

    p_run = sub.add_parser("run", help="run a manifest")
    p_run.add_argument("manifest")
    p_run.add_argument("--resume", action="store_true",
                       help="continue from autoresearch/state/<manifest>.json")
    common(p_run)
    p_rep = sub.add_parser("replay", help="re-run the job sequence a state file records")
    p_rep.add_argument("state")
    p_rep.add_argument("--state-name", default=None,
                       help="name of the replay's own state file (default <name>-replay)")
    common(p_rep)
    p_st = sub.add_parser("status", help="operability summary from the log")
    p_st.add_argument("--since", type=float, default=None, help="hours")
    p_st.add_argument("--log", default=LOG)
    p_st.add_argument("--json", action="store_true")
    a = ap.parse_args(argv[1:])

    if a.cmd == "status":
        s = status(a.log, a.since)
        if a.json:
            print(json.dumps(s, indent=1, sort_keys=True, default=str))
        else:
            print_status(s)
        return 0

    replay_sequence = None
    if a.cmd == "run":
        manifest = load_manifest(a.manifest)
        state_path = os.path.join(a.state_dir, manifest["name"] + ".json")
        if os.path.exists(state_path) and not a.resume and not a.dry_run:
            print(f"{state_path} exists; pass --resume to continue it or remove it to start over",
                  file=sys.stderr)
            return 2
    else:
        prior = load_json(a.state, None)
        if prior is None:
            print(f"no state file at {a.state}", file=sys.stderr)
            return 2
        manifest = {"name": prior["manifest_name"], "path": prior["manifest"],
                    "sha256": prior["manifest_sha256"], "jobs": prior["jobs"]}
        replay_sequence = [(e["job"], e["seed"]) for e in prior["sequence"]]
        name = a.state_name or manifest["name"] + "-replay"
        state_path = os.path.join(a.state_dir, name + ".json")

    if a.dry_run:
        done = load_json(state_path, {}).get("done", {}) if a.cmd == "run" and a.resume else {}
        seq = (replay_sequence if replay_sequence is not None
               else schedule(manifest["jobs"], done))
        jobs = {j["id"]: j for j in manifest["jobs"]}
        for jid, seed in seq:
            print(" ".join(command(jobs[jid], seed, a.runner)[3:]))
        print(f"{len(seq)} job(s)", file=sys.stderr)
        return 0

    loop = Loop(manifest, state_path, a.log, runner=a.runner, max_hours=a.max_hours,
                max_jobs=a.max_jobs, resume=(a.cmd == "run" and a.resume),
                replay_sequence=replay_sequence)
    loop.install_signals()
    return loop.run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
