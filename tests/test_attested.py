"""The attested tier: a batch manifest with hashed outputs, a certificate that
re-runs a declared subset, and the validator's cross-field rules."""

import hashlib
import json
import os
import sys

import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("jsonschema")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "site_challenge"))

import stabrank_verify as sv  # noqa: E402
import validate_bounds as vb  # noqa: E402

EXPECT = "CERTIFIED chi(T3^3) >= 8"


def _rel(path):
    return os.path.relpath(str(path), ROOT)


def _fixture(tmp_path, n=4, break_hash=False, drop_output=False):
    """A manifest of `n` batches with their output files, and a passing script."""
    outdir = tmp_path / "results"
    outdir.mkdir()
    batches = []
    for i in range(n):
        out = outdir / f"batch_{i}.json"
        out.write_text(json.dumps({"index": i, "steps": 1000 * (i + 1), "exceptions": []}))
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        if break_hash and i == n - 1:
            digest = "0" * 64
        batches.append({"id": f"b{i}", "params": {"block": 0, "j": [i, i + 1]},
                        "output": _rel(out), "sha256": digest})
    if drop_output:
        (outdir / f"batch_{n - 1}.json").unlink()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"batches": batches}))
    script = tmp_path / "cert_attested.py"
    script.write_text(f'print("seed: 20260919")\nprint("{EXPECT}")\n')
    return manifest, script


def _sub(manifest, script, recomputed=2, exact=None, compute=True, budget=3600):
    cert = {"script": _rel(script), "expect": EXPECT, "budget_s": budget,
            "attested": {"batches": _rel(manifest), "recomputed": recomputed,
                         "compute_hours": 91, "hardware": "test box",
                         "note": "completeness rests on the stored outputs"}}
    if exact is not None:
        cert["exact"] = exact
    prov = {"author": "test", "date": "2026-09-19"}
    if compute:
        prov["compute"] = {"cpu_hours": 91, "hardware": "test box"}
    return {"schema_version": "0.1", "orbit": "T3", "m": 3, "direction": "lower",
            "rank": 8, "certificate": cert, "provenance": prov}


def test_correct_hashes_earn_attested(tmp_path):
    manifest, script = _fixture(tmp_path)
    r = sv.verify_lower(_sub(manifest, script))
    assert r.ok and r.tier == "attested"
    assert "re-ran 2 of 4" in r.detail
    assert "seed 20260919" in r.detail
    assert "only checked the SHA-256 of the other 2" in r.detail
    assert "was not re-run" in r.detail


def test_wrong_hash_fails(tmp_path):
    manifest, script = _fixture(tmp_path, break_hash=True)
    r = sv.verify_lower(_sub(manifest, script))
    assert not r.ok and r.tier is None
    assert "SHA-256" in r.detail and "b3" in r.detail


def test_missing_output_and_missing_manifest_fail(tmp_path):
    manifest, script = _fixture(tmp_path, drop_output=True)
    r = sv.verify_lower(_sub(manifest, script))
    assert not r.ok and "output not found" in r.detail
    sub = _sub(manifest, script)
    sub["certificate"]["attested"]["batches"] = _rel(tmp_path / "nowhere.json")
    r = sv.verify_lower(sub)
    assert not r.ok and "manifest not found" in r.detail


def test_recomputed_must_fit_the_manifest(tmp_path):
    manifest, script = _fixture(tmp_path, n=3)
    r = sv.verify_lower(_sub(manifest, script, recomputed=5))
    assert not r.ok and "lists 3 batches" in r.detail


def test_exact_with_attested_is_rejected(tmp_path):
    manifest, script = _fixture(tmp_path)
    sub = _sub(manifest, script, exact=True)
    errors, _ = vb.tier_requirements(sub)
    assert any("mutually exclusive" in e for e in errors)
    schema = json.load(open(os.path.join(ROOT, "schema", "bound.schema.json")))
    import jsonschema
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(sub))
    r = sv.verify_lower(sub)
    assert not r.ok and "both exact and attested" in r.detail


def test_validator_rules(tmp_path):
    manifest, script = _fixture(tmp_path)
    schema = json.load(open(os.path.join(ROOT, "schema", "bound.schema.json")))
    import jsonschema
    ok = _sub(manifest, script)
    assert not list(jsonschema.Draft202012Validator(schema).iter_errors(ok))
    errors, notes = vb.tier_requirements(ok)
    assert not errors and any("attested" in n for n in notes)
    errors, _ = vb.tier_requirements(_sub(manifest, script, compute=False))
    assert any("provenance.compute" in e for e in errors)
    # exact alone is still fine
    errors, _ = vb.tier_requirements({**ok, "certificate": {"script": "x", "expect": "y",
                                                              "exact": True}})
    assert not errors


def test_manifest_shape_is_checked(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps([{"id": "a", "output": "x"}]))
    batches, err = sv.load_batch_manifest(_rel(bad))
    assert batches is None and "lacks params, sha256" in err
    bad.write_text(json.dumps([{"id": "a", "params": {}, "output": "x", "sha256": "0"},
                               {"id": "a", "params": {}, "output": "y", "sha256": "0"}]))
    batches, err = sv.load_batch_manifest(_rel(bad))
    assert batches is None and "listed twice" in err


def test_site_ranks_and_pill(tmp_path):
    pytest.importorskip("latex2mathml")
    import build as site
    assert site.TIER_RANK["cited"] < site.TIER_RANK["attested"] < site.TIER_RANK["reproduced"]
    assert "attested" in site.RECORD_TIERS
    manifest, script = _fixture(tmp_path, n=5)
    sub = _sub(manifest, script, recomputed=1)
    pill = site.tier_pill("attested", True, sub)
    assert "t-attested" in pill and "1/5" in pill and "20%" in pill
    assert "title=" in pill and "not re-run" in pill
    row = site.attested_summary(sub)
    assert row["total"] == 5 and row["recomputed"] == 1
    assert ".t-attested" in site.CSS and "dashed" in site.CSS
