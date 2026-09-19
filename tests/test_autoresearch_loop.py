"""The unattended scheduler: checkpoint and resume, replay, diagnosis, status."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

pytest.importorskip("sympy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "autoresearch"))

import loop  # noqa: E402

LOOP = os.path.join(ROOT, "autoresearch", "loop.py")

# A stand-in for run.py with the same command line. The orbit name selects the
# behaviour, so a manifest of stub cells exercises every outcome class.
STUB = r'''
import argparse, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("orbit"); ap.add_argument("m", type=int); ap.add_argument("rank", type=int)
ap.add_argument("--seeds", type=int); ap.add_argument("--seed0", type=int)
ap.add_argument("--chains", type=int); ap.add_argument("--iters", type=int)
ap.add_argument("--cooling", type=float); ap.add_argument("--llm", default="")
ap.add_argument("--save-plateaus", default="")
a = ap.parse_args()
print("SA progress line", file=sys.stderr)
if a.orbit == "miss":
    print(f"seed {a.seed0}: residual 1.234e-01 in 1s wall, 1s CPU")
    print(f"no rank-{a.rank} decomposition in 1 seed(s); best residual 0.1234. Runs are logged; nothing submitted.")
    sys.exit(2)
if a.orbit == "hit":
    print(f"seed {a.seed0}: residual 1.000e-12 in 1s wall, 1s CPU, refit exact")
    print("written bounds/hit-m2-upper-2.json; verifier: VerifyResult(ok=True)")
    sys.exit(0)
if a.orbit == "hitseed3":
    if a.seed0 == 3:
        print("written bounds/x.json; verifier: VerifyResult(ok=True)"); sys.exit(0)
    print("no rank-2 decomposition in 1 seed(s); best residual 0.5"); sys.exit(2)
if a.orbit == "near":
    print(f"seed {a.seed0}: residual 1.000e-12 in 1s wall, 1s CPU, refit FAILED")
    print(f"no rank-{a.rank} decomposition in 1 seed(s); best residual 0.0000. Runs are logged; nothing submitted.")
    sys.exit(2)
if a.orbit == "refused":
    print("written bounds/x.json; verifier: VerifyResult(ok=False, reason='identity fails')")
    sys.exit(1)
if a.orbit == "boom":
    raise ValueError("bad basis")
if a.orbit == "infra":
    raise ModuleNotFoundError("No module named 'stabrank_core'")
if a.orbit == "flaky":
    marker = os.path.join(os.environ["STUB_DIR"], f"flaky-{a.seed0}")
    if os.path.exists(marker):
        print("written bounds/x.json; verifier: VerifyResult(ok=True)"); sys.exit(0)
    open(marker, "w").close()
    raise OSError("transient")
if a.orbit == "oom":
    print("Traceback (most recent call last):\n  File x\nMemoryError", file=sys.stderr)
    sys.exit(1)
if a.orbit == "slow":
    time.sleep(float(os.environ.get("STUB_SLEEP", "30")))
    sys.exit(2)
sys.exit(99)
'''


@pytest.fixture
def stub(tmp_path):
    path = tmp_path / "stub_run.py"
    path.write_text(STUB)
    return str(path)


def write_manifest(tmp_path, jobs, name="t", defaults=None):
    d = {"chains": 1, "iters": 1, "cooling": 0.5, "cap_s": 60}
    d.update(defaults or {})
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"name": name, "defaults": d, "jobs": jobs}))
    return str(path)


def run_loop(tmp_path, stub, *args, check=True, timeout=50):
    env = dict(os.environ, STUB_DIR=str(tmp_path))
    cmd = [sys.executable, LOOP, *args, "--runner", stub,
           "--state-dir", str(tmp_path / "state"), "--log", str(tmp_path / "loop.log")]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    if check:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


def log_jobs(tmp_path):
    return [r for r in loop.read_log(str(tmp_path / "loop.log")) if r["event"] == "job"]


def state(tmp_path, name="t"):
    return json.load(open(tmp_path / "state" / f"{name}.json"))


# --- pure functions -----------------------------------------------------------

def test_classify_by_exit_code_and_stderr():
    assert loop.classify(0, "verifier: ok", "") == ("discovery", False)
    assert loop.classify(2, "best residual 0.1", "") == ("miss", False)
    assert loop.classify(2, "seed 1: residual 1e-12, refit FAILED\nno rank", "") == ("near_miss", False)
    assert loop.classify(1, "written x; verifier: VerifyResult(ok=False)", "") == ("verifier_refusal", False)
    assert loop.classify(1, "", "Traceback (most recent call last):\n  File\nValueError: bad") == ("exception:ValueError", False)
    assert loop.classify(1, "", "Traceback ...\nModuleNotFoundError: No module named 'x'") == ("infrastructure:ModuleNotFoundError", True)
    assert loop.classify(1, "", "Traceback ...\nMemoryError") == ("oom", True)
    assert loop.classify(-9, "", "") == ("oom", True)
    assert loop.classify(-15, "", "")[0] == "killed:SIGTERM"
    assert loop.classify(127, "", "") == ("infrastructure:spawn", True)
    assert loop.classify(2, "", "usage: run.py [-h]\nrun.py: error: argument m") == ("exception:ArgumentError", False)
    assert loop.classify(None, "", "", timed_out=True) == ("timeout", False)
    assert loop.classify(-15, "", "", interrupted=True) == ("interrupted", False)
    assert loop.classify(99, "", "") == ("unknown:exit99", True)


def test_schedule_is_round_robin_in_priority_order():
    jobs = [{"id": "a", "seed0": 1, "seeds": 3, "priority": 0, "seeds_per_round": 1,
             "stop_on_solve": True, "index": 0},
            {"id": "b", "seed0": 10, "seeds": 2, "priority": 5, "seeds_per_round": 1,
             "stop_on_solve": True, "index": 1},
            {"id": "c", "seed0": 20, "seeds": 1, "priority": 5, "seeds_per_round": 1,
             "stop_on_solve": True, "index": 2}]
    assert loop.schedule(jobs, {}) == [("b", 10), ("c", 20), ("a", 1), ("b", 11), ("a", 2), ("a", 3)]
    done = {"b": {"seeds_done": [10], "solved": True}, "a": {"seeds_done": [1, 2]}}
    assert loop.schedule(jobs, done) == [("c", 20), ("a", 3)]


def test_parse_residual_takes_the_number_after_the_word():
    assert loop.parse_residual("no rank-1 decomposition in 1 seed(s); best residual 0.5210. x") == 0.521
    assert loop.parse_residual("seed 3: residual 1.234e-01 in 2s") == pytest.approx(0.1234)
    assert loop.parse_residual("nothing here") is None


def test_example_manifest_loads_and_schedules():
    man = loop.load_manifest(os.path.join(ROOT, "autoresearch", "manifests", "example.json"))
    seq = loop.schedule(man["jobs"], {})
    assert len(seq) == 4 and seq[0][0] == "qubit_H-m2-r1"
    plateau = loop.load_manifest(os.path.join(ROOT, "autoresearch", "manifests",
                                              "plateau_cells.json"))
    assert {j["orbit"] for j in plateau["jobs"]} == {"T3sector0", "T3", "qubit_T", "qubit_H"}
    cmd = loop.command(man["jobs"][0], 7, "run.py")
    assert cmd[:3] == ["nice", "-n", "19"] and "--seed0" in cmd and cmd[cmd.index("--seed0") + 1] == "7"


# --- the loop as a process ----------------------------------------------------

def test_checkpoint_and_resume(tmp_path, stub):
    man = write_manifest(tmp_path, [{"id": "x", "orbit": "miss", "m": 2, "rank": 2, "seeds": 3},
                                    {"id": "y", "orbit": "miss", "m": 2, "rank": 2, "seeds": 3,
                                     "seed0": 10}])
    r = run_loop(tmp_path, stub, "run", man, "--max-jobs", "3")
    assert "stopped: max-jobs after 3 job(s)" in r.stdout
    st = state(tmp_path)
    assert st["done"]["x"]["seeds_done"] == [1, 2] and st["done"]["y"]["seeds_done"] == [10]
    # a second start without --resume refuses rather than clobbering the checkpoint
    r = run_loop(tmp_path, stub, "run", man, check=False)
    assert r.returncode == 2 and "--resume" in r.stderr
    r = run_loop(tmp_path, stub, "run", man, "--resume")
    assert "stopped: exhausted after 3 job(s)" in r.stdout
    st = state(tmp_path)
    assert st["done"]["x"]["seeds_done"] == [1, 2, 3] and st["done"]["y"]["seeds_done"] == [10, 11, 12]
    assert [(e["job"], e["seed"]) for e in st["sequence"]] == [
        ("x", 1), ("y", 10), ("x", 2), ("y", 11), ("x", 3), ("y", 12)]
    assert len(st["loops"]) == 2 and st["loops"][0]["stop"] == "max-jobs"
    assert len(log_jobs(tmp_path)) == 6
    # a discovery closes the cell: the remaining seeds are not spent
    man2 = write_manifest(tmp_path, [{"id": "h", "orbit": "hitseed3", "m": 2, "rank": 2,
                                      "seeds": 6, "seed0": 1}], name="hit")
    run_loop(tmp_path, stub, "run", man2)
    st = state(tmp_path, "hit")
    assert st["done"]["h"]["solved"] and st["done"]["h"]["seeds_done"] == [1, 2, 3]


def test_replay_reproduces_the_sequence(tmp_path, stub):
    man = write_manifest(tmp_path, [{"id": "a", "orbit": "miss", "m": 2, "rank": 2, "seeds": 2,
                                     "priority": 1},
                                    {"id": "b", "orbit": "hitseed3", "m": 2, "rank": 2,
                                     "seeds": 4, "seed0": 2}])
    run_loop(tmp_path, stub, "run", man)
    original = state(tmp_path)
    # change the manifest after the fact: the replay follows the state, not the file
    os.remove(man)
    run_loop(tmp_path, stub, "replay", str(tmp_path / "state" / "t.json"))
    replay = state(tmp_path, "t-replay")
    key = lambda st: [(e["job"], e["seed"], e["outcome"]) for e in st["sequence"]]  # noqa: E731
    assert key(replay) == key(original) == [("a", 1, "miss"), ("b", 2, "miss"), ("a", 2, "miss"),
                                             ("b", 3, "discovery")]
    assert replay["manifest_sha256"] == original["manifest_sha256"]


def test_failures_are_classified_and_only_infrastructure_is_retried(tmp_path, stub):
    jobs = [{"id": k, "orbit": k, "m": 2, "rank": 2} for k in
            ("near", "refused", "boom", "infra", "flaky", "oom")]
    jobs.append({"id": "slow", "orbit": "slow", "m": 2, "rank": 2, "cap_s": 1})
    man = write_manifest(tmp_path, jobs)
    r = run_loop(tmp_path, stub, "run", man)
    assert "stopped: exhausted" in r.stdout
    recs = log_jobs(tmp_path)
    by_job = {}
    for rec in recs:
        by_job.setdefault(rec["job"], []).append(rec)
    assert [x["outcome"] for x in by_job["near"]] == ["near_miss"]
    assert [x["outcome"] for x in by_job["refused"]] == ["verifier_refusal"]
    assert [x["outcome"] for x in by_job["boom"]] == ["exception:ValueError"]
    assert by_job["boom"][0]["stderr_tail"][-1].startswith("ValueError")
    assert [x["outcome"] for x in by_job["infra"]] == ["infrastructure:ModuleNotFoundError"] * 2
    assert [x["attempt"] for x in by_job["infra"]] == [1, 2]
    assert [x["outcome"] for x in by_job["flaky"]] == ["infrastructure:OSError", "discovery"]
    assert [x["outcome"] for x in by_job["oom"]] == ["oom", "oom"]
    assert [x["outcome"] for x in by_job["slow"]] == ["timeout"]
    assert by_job["slow"][0]["exit_code"] not in (0, 2) and by_job["slow"][0]["wall_s"] < 20
    st = state(tmp_path)
    for k in ("near", "refused", "boom", "infra", "flaky", "oom", "slow"):
        assert st["done"][k]["seeds_done"] == [1], k
    assert st["done"]["flaky"]["solved"]
    for rec in recs:
        assert rec["started"] <= rec["ended"] and rec["cpu_s"] >= 0
        assert set(rec) >= {"started", "ended", "wall_s", "cpu_s", "exit_code", "outcome",
                            "stderr_tail", "seed", "job", "attempt"}


def test_status_summary(tmp_path, stub):
    jobs = [{"id": k, "orbit": k, "m": 2, "rank": 2} for k in ("miss", "hit", "boom", "infra")]
    man = write_manifest(tmp_path, jobs)
    run_loop(tmp_path, stub, "run", man)
    s = loop.status(str(tmp_path / "loop.log"))
    assert s["jobs"] == 5 and s["sessions"] == 1 and s["retries"] == 1
    assert s["discoveries"] == ["hit seed 1"]
    assert s["failures"] == {"exception:ValueError": 1, "infrastructure:ModuleNotFoundError": 2}
    assert s["longest_gap_s"] >= 0 and s["covered_hours"] >= 0
    r = subprocess.run([sys.executable, LOOP, "status", "--log", str(tmp_path / "loop.log")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    assert "jobs run: 5 in 1 session(s); retries 1" in r.stdout
    assert "discoveries: hit seed 1" in r.stdout
    assert "operability:" in r.stdout and "longest gap between jobs" in r.stdout
    assert "3 failure(s), all classified" in r.stdout
    r = subprocess.run([sys.executable, LOOP, "status", "--log", str(tmp_path / "loop.log"),
                        "--since", "0.0001", "--json"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and json.loads(r.stdout)["jobs"] <= 5
    assert loop.status(str(tmp_path / "nonexistent.log"))["jobs"] == 0


def test_max_hours_stops_before_a_job_that_cannot_fit(tmp_path, stub):
    man = write_manifest(tmp_path, [{"id": "x", "orbit": "miss", "m": 2, "rank": 2, "seeds": 2}],
                         defaults={"cap_s": 3600})
    r = run_loop(tmp_path, stub, "run", man, "--max-hours", "0.5")
    assert "stopped: max-hours after 0 job(s)" in r.stdout
    assert state(tmp_path)["loops"][-1]["stop"] == "max-hours"
    assert log_jobs(tmp_path) == []


def test_sigterm_records_the_interrupted_job_and_exits(tmp_path, stub):
    man = write_manifest(tmp_path, [{"id": "s", "orbit": "slow", "m": 2, "rank": 2, "seeds": 3},
                                    {"id": "x", "orbit": "miss", "m": 2, "rank": 2}])
    env = dict(os.environ, STUB_DIR=str(tmp_path), STUB_SLEEP="30")
    cmd = [sys.executable, LOOP, "run", man, "--runner", stub,
           "--state-dir", str(tmp_path / "state"), "--log", str(tmp_path / "loop.log")]
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    deadline = time.time() + 20
    while time.time() < deadline and not (tmp_path / "loop.log").exists():
        time.sleep(0.1)
    time.sleep(1.0)
    p.send_signal(signal.SIGTERM)
    out, err = p.communicate(timeout=20)
    assert p.returncode == 130, out + err
    assert "stopped: signal:SIGTERM" in out
    recs = loop.read_log(str(tmp_path / "loop.log"))
    jobs = [r for r in recs if r["event"] == "job"]
    assert len(jobs) == 1 and jobs[0]["outcome"] == "interrupted" and jobs[0]["job"] == "s"
    assert recs[-1]["event"] == "stop" and recs[-1]["reason"] == "signal:SIGTERM"
    st = state(tmp_path)
    assert st["done"].get("s", {}).get("seeds_done", []) == []
    # resume picks the interrupted seed up again
    r = subprocess.run([*cmd, "--resume", "--max-jobs", "1"],
                       cwd=ROOT, env=dict(env, STUB_SLEEP="0"), capture_output=True, text=True,
                       timeout=30)
    assert r.returncode == 0 and state(tmp_path)["done"]["s"]["seeds_done"] == [1]
