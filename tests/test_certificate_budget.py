"""The declared certificate budget: default, cap, and enforcement."""

import os
import sys
import textwrap

import pytest

sp = pytest.importorskip("sympy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))

import stabrank_verify as sv  # noqa: E402


def test_budget_default_and_cap():
    assert sv.certificate_budget({"certificate": {"script": "x", "expect": "y"}}) == 900
    assert sv.certificate_budget({"certificate": {"budget_s": 1800}}) == 1800
    assert sv.certificate_budget({"certificate": {"budget_s": 99999}}) == sv.BUDGET_CAP_S
    assert sv.certificate_budget({}) == 900


def _sub(script_rel, budget=None):
    cert = {"script": script_rel, "expect": "CERTIFIED test"}
    if budget is not None:
        cert["budget_s"] = budget
    return {"certificate": cert}


def test_declared_budget_is_enforced(tmp_path):
    script = tmp_path / "slow_cert.py"
    script.write_text(textwrap.dedent("""
        import time
        time.sleep(2)
        print("CERTIFIED test")
    """))
    rel = os.path.relpath(str(script), ROOT)
    r = sv.verify_lower(_sub(rel, budget=1))
    assert not r.ok and "budget" in r.detail
    r = sv.verify_lower(_sub(rel, budget=30))
    assert r.ok and r.tier == "reproduced"
    assert "declared" not in r.detail            # 30 s is below the default, nothing to flag


def test_budget_above_default_is_named_in_the_detail(tmp_path):
    script = tmp_path / "fast_cert.py"
    script.write_text('print("CERTIFIED test")\n')
    rel = os.path.relpath(str(script), ROOT)
    r = sv.verify_lower(_sub(rel, budget=3600))
    assert r.ok and "declared 3600s budget" in r.detail
