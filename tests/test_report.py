"""The progress report's pure functions, on fixtures rather than the repository.

Exponent comparison from ledger rows, the evidence index from a fake git log,
and the cost aggregation from a fake run log joined with compute blocks.
"""

import os
import sys

import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("latex2mathml")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.join(ROOT, "site_challenge"))

import report  # noqa: E402


def row(slug, orbit, m, direction, rank, tier, gamma=None, date="2026-09-16",
        compute=None, reference="stabrank", ok=True):
    return {"slug": slug, "orbit": orbit, "m": m, "direction": direction, "rank": rank,
            "tier": tier, "ok": ok, "gamma": gamma, "date": date, "compute": compute,
            "reference": reference}


BASE = {"S": (0.3, "0.3"), "T3": (0.5, "1/2")}


def test_exponent_verdicts():
    rows = [
        row("S-m2-upper-2", "S", 2, "upper", 2, "lean", gamma=0.3),           # matches
        row("S-m4-upper-4", "S", 4, "upper", 4, "verified", gamma=0.3),       # tie, lower tier
        row("S-m6-upper-7", "S", 6, "upper", 7, "cited", gamma=0.29),        # cited never counts
        row("T3-m2-upper-3", "T3", 2, "upper", 3, "verified", gamma=0.5 + 1e-3),
        row("T3-m3-lower-6", "T3", 3, "lower", 6, "reproduced"),
    ]
    out = report.exponent_comparison(rows, baseline=BASE, orbits=["S", "T3"])
    assert [r["verdict"] for r in out] == ["matches", "trails"]
    assert out[0]["best"]["slug"] == "S-m2-upper-2"       # tie on gamma goes to the higher tier
    assert out[0]["reference"] == "labib2026stabilizer"
    assert out[1]["gap"] == pytest.approx(1e-3)
    out = report.exponent_comparison(
        rows + [row("S-m6-upper-7v", "S", 6, "upper", 7, "verified", gamma=0.29)],
        baseline=BASE, orbits=["S"])
    assert out[0]["verdict"] == "beats" and out[0]["best"]["rank"] == 7


def test_exponent_open_orbit():
    out = report.exponent_comparison([], baseline=BASE, orbits=["S"])
    assert out[0]["best"] is None and out[0]["verdict"] == "open"


def test_cell_intervals_pick_the_tightest_side_and_date_it():
    rows = [
        row("S-m3-lower-3", "S", 3, "lower", 3, "reproduced", date="2026-09-12"),
        row("S-m3-lower-4", "S", 3, "lower", 4, "reproduced", date="2026-09-16"),
        row("S-m3-upper-4", "S", 3, "upper", 4, "lean", gamma=0.42, date="2026-05-27"),
        row("S-m3-upper-5", "S", 3, "upper", 5, "verified", gamma=0.49, date="2026-09-17"),
        row("S-m4-upper-9", "S", 4, "upper", 9, "verified", gamma=0.5, date="2026-09-01"),
        row("S-m4-upper-8", "S", 4, "upper", 8, "verified", gamma=0.47, date="2026-09-02",
            ok=False),
    ]
    out = report.cell_intervals(rows, board_dates={"S-m3-upper-4": "2026-09-12"},
                                orbits=["S"], since="2026-09-15")
    assert [(c["orbit"], c["m"]) for c in out] == [("S", 3), ("S", 4)]
    c3 = out[0]
    assert c3["settled"] and c3["lower"]["rank"] == 4 and c3["upper"]["rank"] == 4
    assert c3["lower"]["moved_since"] and not c3["upper"]["moved_since"]
    assert c3["upper"]["board_date"] == "2026-09-12"
    assert c3["moved_since"]
    c4 = out[1]
    assert c4["lower"] is None and c4["upper"]["rank"] == 9   # the failed bound never counts
    assert not c4["moved_since"]


GIT_ADDING = ("f44f13ea8b1688f96ab04aee6eba4d3e6e1fa611\t2026-09-18\tchi(S^5) >= 5 by "
              "slice-and-lift twice\n")
GIT_MERGES = ("1111111111111111111111111111111111111111\t2026-09-18\tMerge branch 'main' "
              "into s-m5-lift\n"
              "a3d18db43a39d6fe00d2220e3c03e0ca30822be9\t2026-09-18\tMerge pull request #40 "
              "from unitaryfoundation/s-m5-lift\n"
              "4c0d35fc401543fb9201713d805310ca2674d3d5\t2026-09-18\tMerge pull request #42 "
              "from unitaryfoundation/certificate-budget\n")


def test_evidence_from_fake_git_log():
    adding = report.parse_git_log(GIT_ADDING)
    merges = report.parse_git_log(GIT_MERGES)
    assert adding[0]["sha"].startswith("f44f13ea") and adding[0]["date"] == "2026-09-18"
    h = report.merge_for(adding, merges)
    assert h["pull_request"] == 40                       # the first PR merge, not a branch merge
    assert h["merge_commit"].startswith("a3d18db4")
    assert h["added_commit"].startswith("f44f13ea")
    # no pull request on the path: the adding commit stands in
    h = report.merge_for(adding, report.parse_git_log(GIT_MERGES.splitlines()[0]))
    assert h["pull_request"] is None and h["merge_commit"] is None
    assert h["added_commit"].startswith("f44f13ea")
    assert report.merge_for([], merges)["added_commit"] is None
    assert report.pr_number("Merge pull request #7 from x/y") == 7
    assert report.pr_number("Add a bound") is None


def test_evidence_entry_fields():
    sub = {"orbit": "S", "m": 5, "direction": "lower", "rank": 5,
           "certificate": {"script": "verify_challenge/cert_s_m5_lift.py",
                           "expect": "CERTIFIED chi(S^5) >= 5", "budget_s": 900},
           "lean": {"module": "LeanProofs.StrangeM2Lower", "theorem": "strange_m2_lower"},
           "provenance": {"author": "this repository", "date": "2026-09-18",
                          "reference": "stabrank slice_lift.py",
                          "compute": {"cpu_hours": 2.7, "runs": 3}}}
    h = report.merge_for(report.parse_git_log(GIT_ADDING), report.parse_git_log(GIT_MERGES))
    e = report.evidence_entry(sub, "S-m5-lower-5", {"ok": True, "tier": "reproduced"}, h,
                              ("certs/S-m5-lower-5.json", True), True)
    assert e["file"] == "bounds/S-m5-lower-5.json"
    assert e["pull_request"] == 40 and e["receipt"] == "certs/S-m5-lower-5.json"
    assert e["receipt_matches"] is True
    assert e["lean_receipt"] == "certs/lean-LeanProofs-StrangeM2Lower.json"
    assert e["certificate_script"] == "verify_challenge/cert_s_m5_lift.py"
    assert e["compute"]["cpu_hours"] == 2.7 and e["tier"] == "reproduced"
    # a bound this build did not verify has no tier and no receipt
    e = report.evidence_entry(sub, "S-m5-lower-5", None, h, None, None)
    assert e["tier"] is None and e["ok"] is None and e["receipt"] is None


RUNS = [
    {"when": "2026-09-18T09:24:25+00:00", "orbit": "qubit_H", "m": 4, "rank": 4,
     "cpu_s": 3600.0, "exact": True},
    {"when": "2026-09-18T09:27:28+00:00", "orbit": "T3", "m": 4, "rank": 8,
     "cpu_s": 1800.0, "exact": None},
    {"when": "2026-09-18T09:30:56+00:00", "orbit": "T3", "m": 4, "rank": 8,
     "cpu_s": 1800.0, "exact": None},
]


def test_cost_table_joins_log_and_compute_blocks():
    rows = [
        row("qubit_H-m4-upper-4", "qubit_H", 4, "upper", 4, "verified", gamma=0.5,
            date="2026-05-27", compute={"cpu_hours": 0.5, "runs": 2}),
        row("T3-m4-lower-6", "T3", 4, "lower", 6, "reproduced", date="2026-09-16",
            compute={"cpu_hours": 0.2, "runs": 1}),
        row("T3-m4-upper-9", "T3", 4, "upper", 9, "verified", gamma=0.5, date="2026-05-27"),
        # written by run.py: its compute block is the log again, so it adds nothing
        row("T3-m4-upper-8", "T3", 4, "upper", 8, "verified", gamma=0.47, date="2026-09-18",
            compute={"cpu_hours": 1.0, "runs": 2}, reference=report.RUNPY_REFERENCE),
    ]
    out = {(c["orbit"], c["m"], c["rank"]): c for c in report.cost_table(rows, RUNS)}
    assert set(out) == {("qubit_H", 4, 4), ("T3", 4, 6), ("T3", 4, 8)}
    q = out[("qubit_H", 4, 4)]
    assert q["runs"] == 3 and q["cpu_hours"] == pytest.approx(1.5) and q["solutions"] == 2
    assert q["cpu_hours_per_solution"] == pytest.approx(0.75)
    t8 = out[("T3", 4, 8)]
    assert t8["runs"] == 2 and t8["cpu_hours"] == pytest.approx(1.0)
    assert t8["solutions"] == 0 and t8["cpu_hours_per_solution"] is None
    assert t8["board"] == ["verified"]
    assert out[("T3", 4, 6)]["cpu_hours_per_solution"] == pytest.approx(0.2)


def test_cost_curve_is_cumulative_and_folds_the_seed():
    rows = [
        row("a", "S", 2, "upper", 2, "lean", gamma=0.3, date="2026-05-27"),
        row("b", "S", 3, "lower", 4, "reproduced", date="2026-05-27",
            compute={"cpu_hours": 0.05}),
        row("c", "S", 3, "upper", 4, "cited", gamma=0.42, date="2026-09-12"),   # not a discovery
        row("d", "T3", 4, "lower", 6, "reproduced", date="2026-09-16",
            compute={"cpu_hours": 0.2}),
        row("e", "T3", 5, "lower", 6, "reproduced", date="2026-09-18",
            compute={"cpu_hours": 0.15}),
    ]
    pts = report.cost_curve(rows, RUNS, start="2026-09-01")
    assert [p["date"] for p in pts] == ["2026-09-01", "2026-09-16", "2026-09-18"]
    assert pts[0]["seeded"] and pts[0]["cum_discoveries"] == 2
    assert pts[0]["cum_cpu_hours"] == pytest.approx(0.05)
    assert pts[1]["discoveries"] == 1 and pts[1]["cum_cpu_hours"] == pytest.approx(0.25)
    assert pts[2]["cum_discoveries"] == 4
    assert pts[2]["cum_cpu_hours"] == pytest.approx(0.25 + 0.15 + 2.0)
    assert pts[2]["median_declared"] == pytest.approx(0.15)
    assert report.cost_curve([], [], start="2026-09-01") == []
    svg = report.cost_chart(pts)
    assert svg.startswith("<svg class=chart") and svg.count("<path class=ln") == 2
