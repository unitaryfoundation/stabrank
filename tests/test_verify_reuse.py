"""A certificate script runs once per build, and verification is grouped, parallel, and logged."""

import io
import json
import os
import sys
import textwrap

import pytest

sp = pytest.importorskip("sympy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "site_challenge"))

import stabrank_verify as sv  # noqa: E402
import build  # noqa: E402


def _script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return os.path.relpath(str(p), ROOT)


def _counting_script(tmp_path, name, counter, sleep=0):
    return _script(tmp_path, name, f"""
        import time
        open({str(counter)!r}, "a").write("run\\n")
        time.sleep({sleep})
        print("CERTIFIED test")
    """)


def _lower(rel, orbit="S", m=1, rank=1, budget=None):
    cert = {"script": rel, "expect": "CERTIFIED test"}
    if budget is not None:
        cert["budget_s"] = budget
    return {"schema_version": 1, "orbit": orbit, "m": m, "direction": "lower", "rank": rank,
            "provenance": {"author": "test"}, "certificate": cert}


def _runs(counter):
    return counter.read_text().count("run") if counter.exists() else 0


def test_same_script_runs_once(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "_CERT_RUNS", {})
    counter = tmp_path / "runs.txt"
    rel = _counting_script(tmp_path, "cert.py", counter)
    assert sv.verify_lower(_lower(rel)).ok
    assert sv.verify_lower(_lower(rel, orbit="N", m=2, rank=2)).ok
    assert sv.verify_lower(_lower(rel, budget=1800)).ok    # a finished run answers any larger budget
    assert _runs(counter) == 1


def test_timeout_is_reused_only_for_budgets_it_covers(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "_CERT_RUNS", {})
    counter = tmp_path / "runs.txt"
    rel = _counting_script(tmp_path, "slow.py", counter, sleep=3)
    r = sv.verify_lower(_lower(rel, budget=1))
    assert not r.ok and "budget" in r.detail
    r = sv.verify_lower(_lower(rel, budget=1))
    assert not r.ok and "budget" in r.detail
    assert _runs(counter) == 1                               # the timeout was reused
    r = sv.verify_lower(_lower(rel, budget=30))
    assert r.ok and r.tier == "reproduced"                   # a larger budget runs it again
    assert _runs(counter) == 2


def test_verify_subs_groups_by_script_and_writes_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "_CERT_RUNS", {})
    monkeypatch.setattr(build, "CERTS", str(tmp_path / "certs"))
    counter = tmp_path / "runs.txt"
    rel = _counting_script(tmp_path, "cert.py", counter)
    cited = {"schema_version": 1, "orbit": "S", "m": 1, "direction": "upper", "rank": 2,
             "provenance": {"author": "test"}}
    items = [("a", _lower(rel)), ("b", _lower(rel, orbit="N", m=2, rank=2)), ("c", cited)]
    out = io.StringIO()
    res = build.verify_subs(items, jobs=1, log=out)
    assert res["a"]["ok"] and res["a"]["tier"] == "reproduced"
    assert res["b"]["ok"] and res["b"]["tier"] == "reproduced"
    assert res["c"]["ok"] and res["c"]["tier"] == "cited"
    assert _runs(counter) == 1                               # one group, one run of the script
    assert sorted(os.listdir(tmp_path / "certs")) == sorted(f"{build.slug(s)}.json" for _, s in items[:2])
    lines = [l for l in out.getvalue().splitlines() if l.startswith("PASS")]
    assert len(lines) == 3 and all("tier=" in l and "s  " in l for l in lines)
    assert all(res[k]["content_hash"] == build.content_hash(s) for k, s in items)


def test_verify_subs_runs_groups_in_parallel(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CERTS", str(tmp_path / "certs"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [os.path.join(ROOT, "site_challenge"), os.path.join(ROOT, "verify_challenge"),
         os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    c1, c2 = tmp_path / "r1.txt", tmp_path / "r2.txt"
    rel1 = _counting_script(tmp_path, "one.py", c1, sleep=2)
    rel2 = _counting_script(tmp_path, "two.py", c2, sleep=2)
    items = [("a", _lower(rel1)), ("b", _lower(rel2, orbit="N")), ("c", _lower(rel1, orbit="H3"))]
    import time
    t0 = time.monotonic()
    res = build.verify_subs(items, jobs=2, log=io.StringIO())
    wall = time.monotonic() - t0
    assert all(r["ok"] for r in res.values())
    assert _runs(c1) == 1 and _runs(c2) == 1                # a and c share one worker and one run
    assert wall < 7                                          # two groups of 2 s side by side, not 6 s in a row


def test_receipt_hash_ignores_provenance_but_tracks_the_script(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CERTS", str(tmp_path / "certs"))
    rel = _counting_script(tmp_path, "cert.py", tmp_path / "runs.txt")
    sub = _lower(rel)
    h = build.content_hash(sub)
    edited = dict(sub, provenance={"author": "someone else", "notes": "added a reference"},
                  notes="cosmetic")
    assert build.content_hash(edited) == h                  # the board's text is not what a receipt vouches for
    assert build.content_hash(dict(sub, rank=2)) != h        # the claim is
    (tmp_path / "cert.py").write_text('print("CERTIFIED test")\n')
    assert build.content_hash(sub) != h                      # and so is the script that proves it
    # a legacy receipt, hashed over the whole JSON, is still honored
    os.makedirs(tmp_path / "certs")
    legacy = {"ok": True, "tier": "reproduced", "detail": "old", "gamma": None,
              "content_hash": build.legacy_content_hash(sub)}
    with open(tmp_path / "certs" / f"{build.slug(sub)}.json", "w") as f:
        json.dump(legacy, f)
    assert build.receipt_for(sub) == legacy
    assert build.receipt_for(edited) is None                 # the legacy hash does see the edit, as before
