"""Generate the stabilizer-rank challenge site into docs/.

The leaderboard is earned, not declared. Every bound is run through
verify_challenge/stabrank_verify.py and displayed at the tier it earns:

    verified    the pipeline rebuilt the decomposition and confirmed it here
    reproduced  a certificate script ran and asserted the bound
    attested    an exact argument over an offline enumeration too large for
                any budget; the certificate hashed every stored batch output,
                re-decided the stored exceptions and re-ran a declared subset
    cited       attributed to the literature, not machine-checked

Every tier but `cited` may hold a record. A cited value is shown so the picture
is complete, but it never crowns a cell, so the only way to take a record is to
submit something the pipeline can check. `attested` ranks below `reproduced`,
so a certificate that re-runs the whole argument at the same rank displaces it.

Expensive certificates are not re-run on every build: a cached result in
certs/<slug>.json is trusted if it matches the submission's content hash.
`--no-verify` goes one step further and runs no certificate at all: a bound
without a matching receipt takes its tier from docs/ledger.json, and one absent
from both is left off the board for that build.
"""

from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import html
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stabrank_verify import (ORBIT_LABEL, ORBIT_P, exponent_copies, implied_exponent,  # noqa: E402
                             load_batch_manifest,
                             verify)
from _assets import GHICON, UFLOGO  # noqa: E402

import latex2mathml.converter as _l2m

_MCACHE = {}


def _lines(tex):
    """ORBIT_TEX entries are one string or a tuple of display lines."""
    return tex if isinstance(tex, (list, tuple)) else (tex,)


def M(tex, block=False):
    """Render LaTeX to MathML at build time.

    Done here rather than at page load so the site ships static markup: no
    runtime script, no CDN, and nothing that breaks when a stylesheet or font
    file cannot be fetched. Browsers render MathML natively.
    """
    key = (tex, block)
    if key not in _MCACHE:
        try:
            out = _l2m.convert(tex)
            if block:
                out = out.replace('display="inline"', 'display="block"', 1)
        except Exception:
            out = f"<code>{E(tex)}</code>"
        _MCACHE[key] = out
    return _MCACHE[key]

DOCS = os.path.join(ROOT, "docs")
BOUNDS = os.path.join(ROOT, "bounds")
CERTS = os.path.join(ROOT, "certs")
REPO = "https://github.com/unitaryfoundation/stabrank"

BASELINE = {
    "S": (math.log(2, 3) / 2, "log₃2/2"),
    "N": (math.log(4, 3) / 3, "log₃4/3"),
    "H3": (math.log(4, 3) / 3, "log₃4/3"),
    "T3": (0.5, "1/2"),
    "qubit_H": (math.log(3, 2) / 4, "log₂3/4"),
    "qubit_T": (math.log(3, 2) / 4, "log₂3/4"),
    # No exponent has been published for any p = 5 state. The baseline is the
    # single-copy product bound from chi(T5) = 3, and every page says so.
    "T5": (math.log(3, 5), "log₅3"),
    # The cat family is measured against the qubit exponent it implies:
    # log_2 chi(cat_m)/(m - 2), see exponent_copies in stabrank_verify.
    "cat": (math.log(3, 2) / 4, "log₂3/4"),
}
# Orbits whose baseline is a product bound rather than a published exponent.
UNPUBLISHED = {"T5"}
ORBIT_ORDER = ["S", "N", "H3", "T3", "qubit_H", "qubit_T", "T5", "cat"]
SYSTEM = {"S": "qutrit", "N": "qutrit", "H3": "qutrit", "T3": "qutrit",
          "qubit_H": "qubit", "qubit_T": "qubit", "T5": "ququint",
          "cat": "m qubits"}


def base_word(orbit):
    """'published' or, for an orbit with no published exponent, 'baseline'."""
    return "baseline" if orbit in UNPUBLISHED else "published"


def base_note(orbit):
    """The qualifier shown next to a baseline that is not a published exponent."""
    if orbit in UNPUBLISHED:
        return " (single-copy product bound; no published exponent)"
    if orbit == "cat":
        return (" through the implied H-type exponent log&#8322;&chi;(cat<sub>m</sub>)"
                "/(m&minus;2)")
    return ""
ORBIT_TEX = {
 # \left|...\right\rangle rather than |...\rangle: the bare form renders the
 # ket as an identifier (italic, symbol spacing), the fenced form as an
 # operator, which is what gives correct spacing.
 "S":  r"\left|S\right\rangle = \frac{\left|1\right\rangle - \left|2\right\rangle}{\sqrt{2}}",
 "N":  r"\left|N\right\rangle = \frac{\left|0\right\rangle + \left|1\right\rangle - 2\left|2\right\rangle}{\sqrt{6}}",
 "H3": r"\left|H_3\right\rangle = \sqrt{\tfrac{3+\sqrt{3}}{6}}\,\left|0\right\rangle + "
       r"\sqrt{\tfrac{3-\sqrt{3}}{12}}\,\bigl(\left|1\right\rangle + \left|2\right\rangle\bigr)",
 "T3": r"\left|T_3\right\rangle = \frac{\left|0\right\rangle + \omega_9\left|1\right\rangle + "
       r"\omega_9^{2}\left|2\right\rangle}{\sqrt{3}},\qquad \omega_9 = e^{2\pi i/9}",
 "qubit_H": r"\left|H\right\rangle = \cos(\pi/8)\,\left|0\right\rangle + \sin(\pi/8)\,\left|1\right\rangle",
 "qubit_T": r"\left|T\right\rangle = \cos\beta\,\left|0\right\rangle + e^{i\pi/4}\sin\beta\,\left|1\right\rangle,"
            r"\qquad \cos 2\beta = \tfrac{1}{\sqrt{3}}",
 "T5": r"\left|T_5\right\rangle = \frac{1}{\sqrt{5}} \sum_{x \in \mathbb{F}_5} "
       r"\omega_5^{x^3} \left|x\right\rangle,\qquad \omega_5 = e^{2\pi i/5}",
 # A tuple renders as stacked displays: one line would overflow the box.
 "cat": (r"\left|\text{cat}_m\right\rangle = \frac{\left|T\right\rangle^{\otimes m} + "
         r"\left|T_\perp\right\rangle^{\otimes m}}{\sqrt{2}} = 2^{-(m-1)/2} "
         r"\sum_{|x|\ \mathrm{even}} i^{|x|/2} \left|x\right\rangle",
         r"\left|T\right\rangle = \frac{\left|0\right\rangle + e^{i\pi/4}\left|1\right\rangle}"
         r"{\sqrt{2}},\qquad \left|T_\perp\right\rangle = Z\left|T\right\rangle"),
}

BASE_TEX = {
 "S": r"\gamma \le \tfrac{\log_3 2}{2} \approx 0.3155",
 "N": r"\gamma \le \tfrac{\log_3 4}{3} \approx 0.4206",
 "H3": r"\gamma \le \tfrac{\log_3 4}{3} \approx 0.4206",
 "T3": r"\gamma \le \tfrac{1}{2} = 0.5000",
 "qubit_H": r"\gamma \le \tfrac{\log_2 3}{4} \approx 0.3963",
 "qubit_T": r"\gamma \le \tfrac{\log_2 3}{4} \approx 0.3963",
 "T5": r"\gamma \le \log_5 3 \approx 0.6826",
 "cat": r"\gamma \le \tfrac{\log_2 3}{4} \approx 0.3963",
}

ORBIT_DEF = {
 "S": ("",
   "The Strange state. Its support misses |0&rang; entirely, which is what "
   "makes it unusual among the qutrit magic states: two of the nine amplitudes "
   "of |S&rang;<sup>&otimes;2</sup> vanish, and the missing support is exactly "
   "what the rank-2 decomposition exploits.",
   "Smallest known exact-rank exponent of the four qutrit orbits. Its m=3 and "
   "m=4 cells are settled at 4, so the exponent cannot be improved there; "
   "beating it needs m=6 at rank 7, where search terminates at exactly "
   "&radic;(8/27) and never below."),
 "N": ("(|0&rang; + |1&rang; &minus; 2|2&rang;)/&radic;6",
   "The Norrell state. Real amplitudes, one of them twice the others in "
   "magnitude, so unlike T3 its amplitudes are not all of equal modulus.",
   "Shares the published exponent log&#8323;4/3 with H&#8323;. Its m=4 cell "
   "sits at 7; rank 6 bottoms out at &radic;(211/4043) across five of six "
   "independent anneals, with a deeper basin at &radic;(419/8559)."),
 "H3": ("&radic;((3+&radic;3)/6)&nbsp;|0&rang; + "
   "&radic;((3&minus;&radic;3)/12)&nbsp;(|1&rang; + |2&rang;)",
   "The eigenvector of the qutrit Fourier transform with eigenvalue 1. The "
   "qutrit analogue of the qubit H-type state, and like it the natural target "
   "of distillation.",
   "Its m=4 cell sits at 8; rank 6 terminates at exactly "
   "&radic;(70&minus;37&radic;3)/12, identified to fifteen digits across five "
   "runs on two independent engines."),
 "T3": ("(|0&rang; + &omega;&#8329;|1&rang; + &omega;&#8329;&sup2;|2&rang;)/&radic;3, "
   "&nbsp;&omega;&#8329; = e<sup>2&pi;i/9</sup>",
   "The qutrit T-type or face-centre state. All three amplitudes have equal "
   "modulus 1/&radic;3 and differ only by phase, which is precisely why the "
   "subset-sum lower-bound technique cannot reach this orbit: that argument "
   "needs an exponentially increasing subsequence of amplitude moduli.",
   "The only orbit whose amplitudes leave Q(&omega;&#8323;), living instead in "
   "Q(&omega;&#8329;). That extra field degree is what a Galois argument can "
   "exploit, and it is where the certified &chi; &ge; 6 at m=3 comes from."),
 "qubit_H": ("cos(&pi;/8)|0&rang; + sin(&pi;/8)|1&rang;",
   "The qubit H-type state, the edge centre of the stabilizer octahedron. "
   "Clifford-equivalent to the phase state (|0&rang; + "
   "e<sup>i&pi;/4</sup>|1&rang;)/&radic;2, which is what the literature "
   "usually means by 'the T state'.",
   "The most studied cell in the field: &chi; &le; 7 at six copies underpins "
   "the standard 2<sup>0.47n</sup> figure, and the cat-state rank 6 there "
   "(exponent 0.431, Qassim, Pashayan, and Gosset, arXiv:2106.07740) refuted "
   "the conjecture that 7 was optimal. The published exponent log&#8322;3/4 "
   "comes from their asymptotic contracted-cat-state family rather than from any "
   "single m, so every finite-m entry here sits above it. The ZX restatement "
   "of Kissinger, van de Wetering, and Vilmart (arXiv:2202.09202) adds a "
   "partial decomposition of |T&rang;<sup>&otimes;5</sup> into three terms "
   "that each keep one |T&rang;, so &chi;(T<sup>t</sup>) &le; "
   "3&chi;(T<sup>t&minus;4</sup>); it gives 9 at m=7 and 12 at m=8, and the "
   "glued |cat&#8321;&#8320;&rang; gives 18 at m=10 (exponent 0.417)."),
 "qubit_T": ("cos(&beta;)|0&rang; + e<sup>i&pi;/4</sup>sin(&beta;)|1&rang;, "
   "&nbsp;cos(2&beta;) = 1/&radic;3",
   "The qubit Bravyi-Kitaev T-type state, the face centre of the stabilizer "
   "octahedron. A different Clifford orbit from the H-type despite the "
   "overloaded name.",
   "Rank 5 at six copies is conjectured impossible, with search terminating at "
   "exactly &radic;(5/6)&nbsp;sin(&pi;/12)."),
 "T5": ("5<sup>&minus;1/2</sup> &sum;<sub>x</sub> &omega;&#8325;<sup>x&sup3;</sup>|x&rang;",
   "The ququint T-type state, 5<sup>&minus;1/2</sup> &sum;<sub>x</sub> "
   "&omega;&#8325;<sup>x&sup3;</sup>|x&rang;: the qudit &pi;/8 gate of Howard and "
   "Vala applied to |+&rang;. Every cubic phase over F&#8325; is Clifford-equivalent "
   "to x&sup3;, since the quadratic and linear parts are Clifford phases and "
   "x &rarr; ax scales the cubic coefficient by a&sup3;, which ranges over all of "
   "F&#8325;<sup>*</sup>; so the Howard-Vala family, the Campbell-Anwar-Browne "
   "M state and the |XV<sub>s</sub>, 1&rang; state of Jain and Prakash are one "
   "Clifford orbit, this one. Its amplitudes lie in Q(&omega;&#8325;), the same "
   "field as the stabilizer states, so the Galois argument that works for T&#8323; "
   "has no analogue here.",
   "&chi;(T&#8325;) = 3: rank 2 is excluded exactly over the 30 single-ququint "
   "stabilizer states and a three-term decomposition (two basis states and one "
   "full-support state) verifies symbolically, so &chi; is below the local "
   "dimension, unlike T&#8323; where &chi; = 3 = p. At two copies rank 3 and "
   "rank 4 are excluded over the 3,900 two-ququint states and the annealer found "
   "a rank-8 decomposition, so 5 &le; &chi;(T&#8325;<sup>&otimes;2</sup>) &le; 8 "
   "against the product bound 9. No exponent has been published for any p = 5 "
   "state; the baseline log&#8325;3 = 0.6826 is the single-copy product bound, "
   "and the rank-8 cell already sits below it at 0.6460."),
 "cat": ("(|T&rang;<sup>&otimes;m</sup> + |T<sub>&perp;</sub>&rang;<sup>&otimes;m</sup>)/&radic;2",
   "The magic cat state of Qassim, Pashayan, and Gosset (arXiv:2106.07740, "
   "Eq. 3): the equal superposition of |T&rang;<sup>&otimes;m</sup> and "
   "|T<sub>&perp;</sub>&rang;<sup>&otimes;m</sup> with |T<sub>&perp;</sub>&rang; = "
   "Z|T&rang;, which in the computational basis is supported on the even-weight "
   "strings with phase i<sup>|x|/2</sup>. This is a family indexed by the qubit "
   "count m, not a tensor power: each m is one state, and the schema's m is the "
   "number of qubits. |cat&#8322;&rang; = (|00&rang; + i|11&rang;)/&radic;2 is a "
   "stabilizer state, so the track starts at m = 2.",
   "The family carries the published qubit exponent. Gluing copies of "
   "|cat<sub>m</sub>&rang; through the stabilizer bra &lang;cat&#8322;| gives "
   "|cat<sub>l(m&minus;2)+2</sub>&rang; in &chi;(cat<sub>m</sub>)<sup>l</sup> terms, "
   "and &chi;(T<sup>&otimes;m</sup>)/2 &le; &chi;(cat<sub>m</sub>) &le; "
   "&chi;(T<sup>&otimes;m</sup>) (their Eq. 4), so a rank-r decomposition of "
   "|cat<sub>m</sub>&rang; gives &gamma; &le; log&#8322;(r)/(m&minus;2) for the "
   "H-type orbit; that is the number in the &gamma; column here, and at m = 6, "
   "rank 3, it is the published log&#8322;3/4. The values are their Table 1: "
   "&chi; = 1, 2, 2, 3, 3 at m = 2 to 6 (exact) and &chi; &le; 6 at m = 7, 8. "
   "The decompositions the papers write out (m = 2, 4, 6, the &lang;cat&#8322;| "
   "contraction at m = 8, and the &lang;0| projections to m = 3, 5, 7, also in "
   "Kissinger, van de Wetering, and Vilmart, arXiv:2202.09202) are replayed here "
   "as verified witnesses; the lower bounds at m = 5 to 8 rest on their appendix "
   "and are cited. &chi;(cat&#8328;) &le; 5 would give &gamma; = 0.3870 and "
   "&chi;(T<sup>&otimes;8</sup>) &le; 10."),
}

COLOR = {"S": "#6d28d9", "N": "#0369a1", "H3": "#059669",
         "T3": "#b45309", "qubit_H": "#be185d", "qubit_T": "#128081",
         "T5": "#b91c1c", "cat": "#7c2d12"}
TIER_RANK = {"lean": 5, "verified": 4, "reproduced": 3, "attested": 2, "cited": 1, None: 0}
# `attested` holds records: the pipeline checked every stored batch output
# against its hash, the exact re-decision of the exceptions and a bit-for-bit
# re-run of a declared subset, and the committed runner regenerates the rest.
# That is machine-checked evidence, unlike a citation, and leaving it out would
# show a weaker bound on the board than the repository holds. It ranks below
# `reproduced` so that a full re-run at the same rank replaces it. The argument
# is written out in CONTRIBUTING.md, "Attested offline enumerations".
RECORD_TIERS = ("lean", "verified", "reproduced", "attested")

E = html.escape


def slug(sub):
    return f"{sub['orbit']}-m{sub['m']}-{sub['direction']}-{sub['rank']}"


def content_hash(sub):
    return hashlib.sha256(
        json.dumps(sub, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def ledger_rows():
    """docs/ledger.json as {slug: row}, or {} when there is no ledger yet."""
    path = os.path.join(DOCS, "ledger.json")
    if not os.path.exists(path):
        return {}
    return {r["slug"]: r for r in json.load(open(path)).get("rows", [])}


def load_bounds(no_verify=False):
    out = []
    ledger = ledger_rows() if no_verify else {}
    for path in sorted(glob.glob(os.path.join(BOUNDS, "*.json"))):
        sub = json.load(open(path))
        s, h = slug(sub), content_hash(sub)
        cpath = os.path.join(CERTS, f"{s}.json")
        res = None
        if os.path.exists(cpath):
            cached = json.load(open(cpath))
            if cached.get("content_hash") == h:
                res = cached
        # A pure citation (no witness, no certificate, no Lean claim) runs
        # nothing when verified, so --no-verify can classify it directly
        # instead of skipping a bound that has no receipt and no ledger row.
        citation = not (sub.get("witness") or sub.get("certificate") or sub.get("lean"))
        if res is None and no_verify and not citation:
            row = ledger.get(s)
            if row is None:
                print(f"--no-verify: skipping {s}, no receipt and not in docs/ledger.json",
                      file=sys.stderr)
                continue
            res = {"ok": row["ok"], "tier": row["tier"],
                   "detail": "tier taken from docs/ledger.json; not re-verified in "
                             "this build (--no-verify)",
                   "gamma": row.get("gamma"), "content_hash": h}
        if res is None:
            r = verify(sub)
            res = {"ok": r.ok, "tier": r.tier, "detail": r.detail,
                   "gamma": r.gamma, "content_hash": h}
            if r.ok and r.tier in ("verified", "reproduced", "attested"):
                os.makedirs(CERTS, exist_ok=True)
                with open(cpath, "w") as f:
                    json.dump(res, f, indent=2)
                    f.write("\n")
        if res.get("gamma") is None and sub["direction"] == "upper":
            res["gamma"] = implied_exponent(sub["orbit"], sub["rank"], sub["m"])
        out.append({"sub": sub, "res": res, "slug": s})
    return out


def best_by_cell(entries):
    cells = {}
    for e in entries:
        if not e["res"]["ok"]:
            continue
        s = e["sub"]
        key = (s["orbit"], int(s["m"]), s["direction"])
        cur = cells.get(key)
        better = (s["direction"] == "upper" and (cur is None or s["rank"] < cur["sub"]["rank"])) or \
                 (s["direction"] == "lower" and (cur is None or s["rank"] > cur["sub"]["rank"]))
        same = cur is not None and s["rank"] == cur["sub"]["rank"]
        if better or (same and TIER_RANK[e["res"]["tier"]] > TIER_RANK[cur["res"]["tier"]]):
            cells[key] = e
    return cells


def leaderboard(entries):
    rows = []
    for orbit in ORBIT_ORDER:
        base, base_txt = BASELINE[orbit]
        best = None
        for e in entries:
            s, r = e["sub"], e["res"]
            if s["orbit"] != orbit or s["direction"] != "upper":
                continue
            if not r["ok"] or r["tier"] not in RECORD_TIERS:
                continue
            g = r.get("gamma")
            if g is not None and (best is None or g < best["res"]["gamma"]):
                best = e
        rows.append({"orbit": orbit, "baseline": base, "baseline_txt": base_txt,
                     "best": best})
    return rows


def next_target(orbit, base, cells=None, lead="needs "):
    """The cell with the lowest implied exponent that would beat the published one.

    An empty progress bar on every row carries no information; naming a rank
    that would move the number is actionable. At each m the candidate is the
    largest rank below p^(base m), and among the m the one with the smallest
    exponent log_p(r)/m is shown. A rank below the board's own lower bound at
    that m is not a target, so when the per-cell records are given those m are
    skipped.
    """
    p = ORBIT_P[orbit]
    best = None
    for m in range(2, 11):     # the schema's cap on m
        # For a copy orbit the exponent at m has denominator m; for the cat
        # family it is m - 2 (the implied H-type exponent), and the m = 2
        # cell implies no exponent at all.
        n = exponent_copies(orbit, m)
        if n is None:
            continue
        r = math.floor(p ** (base * n) - 1e-9)
        if abs(p ** (base * n) - r) < 1e-9:
            r -= 1
        if r < 2:
            continue
        low = cells.get((orbit, m, "lower")) if cells else None
        if low is not None and r < int(low["sub"]["rank"]):
            continue
        g = math.log(r, p) / n
        if g < base - 1e-12 and (best is None or g < best[2]):
            best = (m, r, g)
    if best is None:
        return "<span class=none>&mdash;</span>"
    m, r, g = best
    return (f"<span class=tgt>{lead}&chi; &le; {r} at m={m}"
            f"<span class=tgtg>&rarr; {g:.4f}</span></span>")


def contributors(entries):
    """Rank submitters. Records are what count; volume breaks ties."""
    # best_by_cell reports the tightest bound known for a cell, which is what the
    # ledger should show even when it is a literature value. A record is a
    # stronger claim: the pipeline had to be able to check it.
    cells = best_by_cell(entries)
    holders = {id(e) for e in cells.values()
               if e["res"]["tier"] in RECORD_TIERS}
    agg = {}
    for e in entries:
        s, r = e["sub"], e["res"]
        who = s["provenance"].get("author", "unknown")
        a = agg.setdefault(who, {"who": who, "links": gh_links(s["provenance"]),
                                 "n": 0, "verified": 0, "records": 0,
                                 "best": None, "cells": set()})
        a["n"] += 1
        if r["ok"] and r["tier"] == "verified":
            a["verified"] += 1
        if id(e) in holders:
            a["records"] += 1
            a["cells"].add((s["orbit"], int(s["m"])))
        if s["direction"] == "upper" and r["ok"] and r.get("gamma") is not None:
            if a["best"] is None or r["gamma"] < a["best"]:
                a["best"] = r["gamma"]
    rows = sorted(agg.values(),
                  key=lambda a: (-a["records"], -a["verified"], a["best"] if a["best"] else 9))
    return rows


# ------------------------------------------------------------------ chart ---

def progress_chart(entries):
    """Best-known exponent per orbit over time, as a step chart.

    Each orbit's line steps down whenever a submission improved its best gamma.
    The axis ticks name values the chart actually reaches, and the viewBox
    leaves room for the outermost labels rather than clipping them.
    """
    series = {}
    for orbit in ORBIT_ORDER:
        pts = []
        best = None
        rows = sorted(
            [e for e in entries
             if e["sub"]["orbit"] == orbit and e["sub"]["direction"] == "upper"
             and e["res"]["ok"] and e["res"].get("gamma") is not None
             and int(e["sub"]["m"]) > 1],   # m=1 is not an asymptotic record
            key=lambda e: e["sub"]["provenance"].get("date", "2026-01-01"))
        for e in rows:
            yr = int(e["sub"]["provenance"].get("date", "2026")[:4])
            g = e["res"]["gamma"]
            if best is None or g < best:
                best = g
                pts.append((yr, g, e))
        if pts:
            series[orbit] = pts
    if not series:
        return ""

    years = sorted({y for pts in series.values() for y, _, _ in pts})
    y0, y1 = min(years), max(years) + 1
    gs = [g for pts in series.values() for _, g, _ in pts] + \
         [b for b, _ in BASELINE.values()]
    lo, hi = min(gs) * 0.96, max(gs) * 1.04

    W, H = 900, 330
    L, R, T, B = 62, 150, 18, 40
    def X(yr): return L + (yr - y0) / max(1, (y1 - y0)) * (W - L - R)
    def Y(g):  return T + (hi - g) / (hi - lo) * (H - T - B)

    o = [f"<svg class=chart viewBox='0 0 {W} {H}' role='img' "
         f"aria-label='best known per-copy exponent by orbit over time'>"]
    # horizontal gridlines at real gamma values
    ticks = [round(lo + i * (hi - lo) / 4, 3) for i in range(5)]
    for t in ticks:
        o.append(f"<line class=grid x1='{L}' x2='{W-R}' y1='{Y(t):.1f}' y2='{Y(t):.1f}'/>")
        o.append(f"<text class=ax x='{L-9}' y='{Y(t)+4:.1f}' text-anchor='end'>{t:.3f}</text>")
    for yr in years:
        o.append(f"<text class=ax x='{X(yr):.1f}' y='{H-14}' text-anchor='middle'>{yr}</text>")
    o.append(f"<text class=axl x='14' y='{T+(H-T-B)/2:.0f}' "
             f"transform='rotate(-90 14 {T+(H-T-B)/2:.0f})' text-anchor='middle'>"
             f"per-copy exponent &gamma;</text>")

    for orbit, pts in series.items():
        c = COLOR[orbit]
        d = []
        for i, (yr, g, _) in enumerate(pts):
            if i == 0:
                d.append(f"M{X(yr):.1f},{Y(g):.1f}")
            else:
                d.append(f"L{X(yr):.1f},{Y(pts[i-1][1]):.1f}")
                d.append(f"L{X(yr):.1f},{Y(g):.1f}")
        d.append(f"L{X(y1):.1f},{Y(pts[-1][1]):.1f}")
        o.append(f"<path class=ln d='{' '.join(d)}' stroke='{c}'/>")
        for yr, g, e in pts:
            o.append(f"<circle cx='{X(yr):.1f}' cy='{Y(g):.1f}' r='4' "
                     f"fill='#fff' stroke='{c}' stroke-width='2'/>")
    # Hit targets last so they sit on top of every line: an invisible larger
    # circle per point carries a one-line tooltip (the bound, its copies and
    # its exponent; date, tier and author are on the page a click opens) and
    # the page's script positions one shared tooltip element on hover.
    for orbit, pts in series.items():
        for yr, g, e in pts:
            sub = e["sub"]
            tip = f"{ORBIT_LABEL[orbit]} · χ ≤ {sub['rank']} at m={sub['m']} · γ = {g:.4f}"
            o.append(f"<circle class=hit cx='{X(yr):.1f}' cy='{Y(g):.1f}' r='11' "
                     f"fill='transparent' data-tip='{E(tip)}' "
                     f"data-href='bounds/{e['slug']}.html'/>")
    # right-edge labels collide when orbits share an exponent, so lay them out
    # in one pass with a minimum vertical gap and a leader line back to the line
    ends = sorted(((pts[-1][1], ob) for ob, pts in series.items()),
                  key=lambda t: -t[0])
    placed, prev = [], None
    for g, ob in ends:
        y = Y(g)
        if prev is not None and y - prev < 15:
            y = prev + 15
        placed.append((ob, g, y))
        prev = y
    for ob, g, y in placed:
        c = COLOR[ob]
        if abs(y - Y(g)) > 1:
            o.append(f"<path class=leader d='M{X(y1):.1f},{Y(g):.1f} "
                     f"L{X(y1)+6:.1f},{y:.1f}' stroke='{c}'/>")
        o.append(f"<a href='orbits/{ob}.html'>"
                 f"<text class=lbl x='{X(y1)+10:.1f}' y='{y+4:.1f}' fill='{c}'>"
                 f"{ORBIT_LABEL[ob]} &middot; {g:.4f}</text></a>")
    o.append("</svg>")
    return "".join(o)


# ------------------------------------------------------------------- HTML ---

CSS = """
/* Theme B: paper hero with ink type, one plum accent, Fraunces display over
   Source Sans 3 with JetBrains Mono labels. Every color used below is a token
   on :root so the palette can be swapped in one place. The UF logo sits on a
   small dark badge so its yellow mark stays legible on the light hero. */
:root{--ink:#1d1a16;--mut:#6a635a;--ln:#ddd5c8;--bd:#1d1a16;--ac:#6e1e3c;
--achov:#4d1329;--acsoft:#f5e9ee;--ex:#1f6b48;--exb:#ffff00;--dark:#1d1a16;
--bg:#faf7f1;--paper:#f3ede2;--card:#fffdf8;--soft:#f3ede2;--bad:#a3262b;
--lean:#2f3f8f;--attest:#8a5a12;--attestbg:#faf0dc;--badge:#141414;--r:6px;
--font:"Source Sans 3",system-ui,-apple-system,"Segoe UI",sans-serif;
--serif:Fraunces,"Iowan Old Style",Georgia,serif;
--mono:"JetBrains Mono",ui-monospace,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);
font-size:16.5px;line-height:1.6}
.wrap{max-width:1060px;margin:0 auto;padding:0 20px}
a{color:var(--ac)}
code,.mono{font-family:var(--mono)}
h1,h2{font-family:var(--serif);font-weight:600;letter-spacing:-.005em}

header.hero{background:var(--paper);color:var(--ink);padding:34px 0 30px;
position:relative;overflow:hidden;border-bottom:1px solid var(--bd)}
header.hero>.wrap{position:relative;z-index:1}
.heroflow{position:absolute;inset:0;width:100%;height:100%;z-index:0;pointer-events:none}
.heroflow .arcs circle{fill:none;stroke:var(--ac);stroke-width:1;opacity:.10}
.heroflow .arcs circle:nth-child(2n){opacity:.06}
.brand{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:0 0 18px}
.brandmark{display:flex;align-items:center;gap:16px}
.brandmark a{display:inline-flex;background:var(--badge);padding:7px 10px;border-radius:4px}
.uflogo{height:34px;width:auto;display:block}
header.hero h1{font-size:clamp(32px,6vw,46px);margin:0 0 14px;line-height:1.1;
font-variation-settings:"opsz" 144;position:relative;padding-bottom:12px}
header.hero h1::after{content:"";position:absolute;left:0;bottom:0;width:64px;height:3px;
background:var(--ac)}
header.hero p{font-size:19px;max-width:640px;margin:0;color:var(--mut)}
header.hero p a{color:var(--ac);text-decoration:underline}
header.hero p a:hover{background:var(--ac);color:#fff;text-decoration:none}
.topnav{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px}
.topnav a{display:inline-flex;align-items:center;gap:7px;color:var(--ink);
font-family:var(--mono);font-size:13px;font-weight:600;
padding:7px 14px;border:1px solid var(--bd);border-radius:4px;
background:transparent;text-decoration:none}
.topnav a:hover{background:var(--ink);color:var(--paper)}
.lbcta{flex:0 0 auto;font-size:13px;font-weight:600;color:#fff;
font-family:var(--mono);background:var(--ac);border:1px solid var(--ac);
border-radius:4px;padding:9px 16px;text-decoration:none;cursor:pointer}
.lbcta:hover{background:var(--achov);border-color:var(--achov)}

h2{font-size:26px;margin:44px 0 4px}
h3{font-family:var(--font);font-size:17px;margin:26px 0 4px;font-weight:700}
.h2sub{color:var(--mut);font-size:14.5px;margin:0 0 14px}
section p{max-width:74ch}

.chartbox{border:1px solid var(--bd);border-radius:var(--r);padding:8px 10px 2px;background:var(--card)}
.chart{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--ln);stroke-width:1}
.chart .ax{font-family:var(--mono);font-size:11px;fill:var(--mut)}
.chart .axl{font-family:var(--mono);font-size:11px;fill:var(--mut)}
.chart .ln{fill:none;stroke-width:2.5;stroke-linejoin:round}
.chart .lbl{font-family:var(--mono);font-size:12px;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;padding:10px 4px 12px;
font-family:var(--mono);font-size:12px;color:var(--mut)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}

.lb{border:1px solid var(--bd);border-radius:var(--r);background:var(--card);overflow:hidden}
.lbhead{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;
gap:12px;padding:14px 18px;border-bottom:1px solid var(--bd);background:var(--paper)}
.lbhead .t{font-family:var(--mono);font-size:12px;letter-spacing:.12em;
text-transform:uppercase;color:var(--ink);font-weight:600}
.lbhead .s{color:var(--mut);font-size:13.5px}
.lbhead .big{font-family:var(--serif);font-size:32px;font-weight:600;
line-height:1;text-align:right}
.lbhead .bigk{font-family:var(--mono);font-size:10px;letter-spacing:.1em;
text-transform:uppercase;color:var(--mut);text-align:right}
.lbrow{display:grid;grid-template-columns:34px 1fr repeat(3,88px);gap:10px;
align-items:center;padding:11px 18px;border-bottom:1px solid var(--ln);background:var(--card)}
.lbrow:last-child{border-bottom:0}
.lbrow .rk{font-family:var(--mono);color:var(--mut);font-size:14px}
.lbrow .who{font-weight:700}
.lbrow .m{text-align:center}
.lbrow .m b{display:block;font-family:var(--serif);font-size:18px;line-height:1.1}
.lbrow .m span{font-family:var(--mono);font-size:10px;
letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}

table{border-collapse:collapse;width:100%;font-size:14.5px}
.tw{overflow-x:auto}
th,td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--ln);white-space:nowrap}
thead th{font-family:var(--mono);font-size:11px;letter-spacing:.07em;
text-transform:uppercase;color:var(--mut);font-weight:400;border-bottom:1px solid var(--bd)}
td.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
tr.rec td{background:var(--acsoft)}

.pill{display:inline-block;font-family:var(--mono);font-size:11px;line-height:1.5;
padding:1px 8px;border-radius:4px;border:1px solid currentColor}
.t-lean{color:var(--lean);text-decoration:none;font-weight:600}
.t-leanbad{color:var(--bad);text-decoration:none;font-weight:600}
.t-leanpend{color:var(--mut);text-decoration:none}
.t-lean:hover{background:var(--lean);color:#fff}
.t-verified{color:var(--ex)}.t-reproduced{color:var(--ac)}
.t-attested{color:var(--attest);border-style:dashed;background:var(--attestbg)}
.t-cited{color:var(--mut)}.t-failed{color:var(--bad)}
a.pill{text-decoration:none}
a.t-cited:hover{color:var(--ac);border-color:var(--ac)}
.gain{color:var(--ex);font-weight:700}
.none{color:var(--mut)}
.bar{position:relative;height:9px;background:var(--ln);border-radius:2px;min-width:120px}
.bar i{position:absolute;top:0;bottom:0;left:0;background:var(--ac);border-radius:2px}

.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.orb{border:1px solid var(--bd);border-radius:var(--r);padding:14px 16px;background:var(--card)}
.orb h3{font-family:var(--serif);font-weight:600;margin:0 0 2px;font-size:20px}
.orb .sub{font-family:var(--mono);font-size:11px;color:var(--mut);
text-transform:uppercase;letter-spacing:.07em}
.cells{margin-top:10px;font-family:var(--mono);font-size:13px}
.cells div{display:flex;justify-content:space-between;align-items:center;gap:10px;
padding:5px 0;border-bottom:1px dotted var(--ln);white-space:nowrap}
.cells div>span:first-child{flex:0 0 auto}
.cells div>span:last-child{display:inline-flex;align-items:center;gap:7px;
flex:0 0 auto}
.cells div:last-child{border-bottom:0}

details{border:1px solid var(--bd);border-radius:var(--r);padding:10px 14px;background:var(--card)}
details summary{cursor:pointer;font-family:var(--mono);font-size:13px;font-weight:600}
pre{background:var(--card);border:1px solid var(--ln);border-radius:var(--r);padding:12px 14px;
overflow-x:auto;font-size:12.5px;line-height:1.5}
.refs{list-style:none;padding:0;counter-reset:r}
.refs li{counter-increment:r;padding:12px 0 12px 42px;border-bottom:1px solid var(--ln);
position:relative;max-width:80ch}
.refs li:before{content:"[" counter(r) "]";position:absolute;left:0;top:12px;
font-family:var(--mono);font-size:12px;color:var(--mut)}
.refs .ti{font-weight:700}
.refs .au{color:var(--ink)}
.refs .vn{color:var(--mut)}
.refs .nt{color:var(--mut);font-size:13.5px;margin-top:3px}
.chart .leader{fill:none;stroke-width:1.2;opacity:.55}
.prose code,p code{font-family:var(--mono);font-size:.88em;background:var(--soft);
padding:1px 4px;border-radius:3px}
.chart .hit{cursor:pointer}
.chart .hit:hover{fill:rgba(29,26,22,.06)}
#tip{position:fixed;z-index:50;max-width:310px;padding:7px 10px;border-radius:4px;
background:var(--ink);color:var(--paper);font-family:var(--mono);font-size:11.5px;
line-height:1.45;pointer-events:none;opacity:0;transition:opacity .08s}
#tip.show{opacity:1}
.h2note{font-family:var(--font);font-weight:400;font-size:14px;color:var(--mut);letter-spacing:0}

.modal{border:1px solid var(--bd);border-radius:var(--r);padding:0;max-width:620px;width:calc(100% - 32px);
box-shadow:0 24px 70px rgba(29,26,22,.35);background:var(--card)}
.modal::backdrop{background:rgba(29,26,22,.55)}
.modal h3{margin:0 0 4px;font-size:24px;font-family:var(--serif);font-weight:600}
.modal>*:not(form){padding:0 26px}
.modal h3{padding-top:24px}
.modal p:last-of-type{padding-bottom:24px}
.modal p{color:var(--mut);max-width:none}
.modal form{display:flex;justify-content:flex-end;padding:10px 12px 0}
.modal .x{background:none;border:none;font-size:24px;line-height:1;color:var(--mut);
cursor:pointer;padding:0 6px}
.codewrap{position:relative;margin:14px 26px}
.codewrap pre{background:var(--dark);color:#efe9dd;border:none;margin:0;
padding:16px 18px;border-radius:var(--r)}
.codewrap code{color:#efe9dd}
.copy{position:absolute;top:10px;right:10px;font-family:var(--mono);
font-size:11px;background:var(--ac);color:#fff;border:none;border-radius:4px;
padding:4px 10px;cursor:pointer}
.copy:hover{background:var(--achov)}

.recent{display:flex;flex-direction:column;gap:6px;border:1px solid var(--bd);
border-radius:var(--r);padding:8px;background:var(--card)}
.rrow{display:flex;align-items:center;gap:12px;flex-wrap:nowrap;white-space:nowrap;
padding:9px 14px;border-radius:4px;background:var(--card)}
.rrow:hover{background:var(--acsoft)}
.rb{font-size:13.5px;font-weight:700;text-decoration:none;color:var(--ink);flex:0 0 auto}
.rb:hover{color:var(--ac)}
.star{color:var(--ac);flex:0 0 auto}
.rt{color:var(--mut);font-size:13.5px;flex:0 0 auto}
.rw{font-size:13.5px;flex:0 0 auto}
.rd{font-family:var(--mono);font-size:12px;color:var(--mut);
margin-left:auto;flex:0 0 auto}
.tp{font-family:"Cambria Math","STIX Two Math","DejaVu Math TeX Gyre",
"Latin Modern Math",serif;font-size:.86em;vertical-align:.02em}
.warn{border:1px solid var(--bd);border-left:4px solid var(--bad);border-radius:var(--r);
padding:12px 16px;background:var(--card);margin:22px 0 0;font-size:14.5px}
.warn b{color:var(--bad)}
.tgt{font-family:var(--mono);font-size:11.5px;color:var(--mut);
white-space:nowrap}
.tgtg{color:var(--ac);margin-left:8px;font-weight:600}
a.lgd{color:var(--mut);text-decoration:none;display:inline-flex;align-items:center}
a.lgd:hover{color:var(--ac)}
a.m{text-decoration:none;color:inherit;border-radius:4px;padding:4px 0}
a.m:hover{background:var(--soft);color:var(--ac)}
.lbrow .who a{text-decoration:none;color:var(--ink)}
.lbrow .who a:hover{color:var(--ac)}
.orb h3 a,.exp a{text-decoration:none;color:inherit}
.orb h3 a:hover{color:var(--ac)}
.statebox{border:1px solid var(--bd);border-left:4px solid var(--ac);
border-radius:var(--r);padding:16px 20px;background:var(--card);margin:4px 0 14px}
.stateeq{font-size:19px;line-height:1.8;overflow-x:auto}
.stateeq sup{font-size:.7em}
.ket{font-family:var(--font);font-weight:600;white-space:nowrap}
.ket math{font-size:1.02em}
math{font-family:"STIX Two Math","Cambria Math","Latin Modern Math",serif}
.stateeq math{font-size:1.5em}
.h2sub math,.sub math,thead math{font-size:1em}
.sub math{color:var(--mut)}
header.hero p br{line-height:2}
@media(max-width:820px){.rrow .rt,.rrow .rw{display:none}}

.scroll{max-height:520px;overflow-y:auto;border:1px solid var(--bd);border-radius:var(--r);
background:var(--card)}
.scroll table{font-size:13.5px}
.scroll thead th{position:sticky;top:0;background:var(--card);z-index:1;
box-shadow:inset 0 -1px 0 var(--bd)}
a.gh{text-decoration:none;font-weight:600}
td.mono a{text-decoration:none;border-bottom:1px dotted var(--ln);padding-bottom:1px}
td.mono a:hover{border-bottom-color:var(--ac)}
a.gh:hover{text-decoration:underline}

footer.foot{margin:64px 0 0;border-top:1px solid var(--bd);background:var(--paper)}
.footmain{max-width:1060px;margin:0 auto;padding:26px 20px 30px}
.footbrand{max-width:420px}
.footbrand .fb{display:flex;align-items:center;gap:12px;margin-bottom:8px}
.footbrand .fb span{font-family:var(--serif);font-size:19px;font-weight:600;color:var(--ink)}
.footbrand p{margin:0;color:var(--mut);font-size:14.5px}
.footlinks{display:flex;flex-wrap:wrap;gap:8px 22px;align-items:center;margin-top:16px}
.footlinks a{display:inline-flex;align-items:center;gap:7px;color:var(--ink);
text-decoration:none;font-size:14.5px}
.footlinks a:hover{color:var(--ac);text-decoration:underline}
.footlinks svg{width:16px;height:16px}
@media(max-width:700px){.lbrow{grid-template-columns:28px 1fr 70px;}
.lbrow .m:nth-child(n+4){display:none}
/* a long bound title plus the date exceeds a phone width, so let the row wrap */
.rrow{flex-wrap:wrap;white-space:normal;row-gap:2px}
.rrow .rb{flex:1 1 auto;min-width:0}}
"""

HEROSVG = """<svg class=heroflow viewBox="0 0 1200 360" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
<g class=arcs>
<circle cx="1180" cy="40" r="150"/><circle cx="1180" cy="40" r="230"/>
<circle cx="1180" cy="40" r="310"/><circle cx="1180" cy="40" r="390"/>
<circle cx="1180" cy="40" r="470"/>
</g>
</svg>"""


def head(title, rel=""):
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{E(title)}</title>"
        f"<link rel=icon type='image/svg+xml' href='{rel}favicon.svg'>"
        "<link rel=preconnect href='https://fonts.googleapis.com'>"
        "<link rel=preconnect href='https://fonts.gstatic.com' crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600"
        "&family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap' rel=stylesheet>"
        f"<link rel=stylesheet href='{rel}style.css'></head><body>"
    )


def hero(title, tagline, rel=""):
    return (
        "<header class=hero>" + HEROSVG + "<div class=wrap>"
        "<div class=brand><span class=brandmark>"
        f"<a href='https://unitary.foundation' aria-label='Unitary Foundation'>{UFLOGO}</a>"
        "</span>"
        "<button class=lbcta type=button onclick=\"document.getElementById('participate').showModal()\">"
        "Participate</button></div>"
        f"<h1>{title}</h1><p>{tagline}</p>"
        "<nav class=topnav>"
        f"<a href='{rel}report/index.html'>Report</a>"
        f"<a href='{rel}references.html'>References</a>"
        f"<a href='{REPO}'>{GHICON}GitHub</a>"
        "</nav></div></header>"
    )



# The mark: a magic state (yellow) decomposing into stabilizer terms
# (white). One source, used for both the favicon and the footer brand.
MARK_BODY = '<rect x="1" y="1" width="62" height="62" rx="14" fill="#111111" stroke="rgba(255,255,255,0.16)" stroke-width="1.5"/><g stroke="#ffffff" stroke-width="3.4" stroke-linecap="round" opacity="0.9"><line x1="32" y1="16" x2="17" y2="40"/><line x1="32" y1="16" x2="47" y2="40"/><line x1="32" y1="16" x2="32" y2="46"/></g><g fill="#ffffff"><circle cx="17" cy="42" r="5"/><circle cx="32" cy="47" r="5"/><circle cx="47" cy="42" r="5"/></g><circle cx="32" cy="16" r="6" fill="#ffff00"/>'

FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
           + MARK_BODY + '</svg>')

FOOTMARK = ('<svg width=34 height=34 viewBox="0 0 64 64" aria-hidden="true">'
            + MARK_BODY + '</svg>')


CHART_SCRIPT = """<script>
(function(){
 const tip=document.getElementById('tip');if(!tip)return;
 document.addEventListener('mouseover',e=>{
  const c=e.target.closest('.hit[data-tip]');if(!c)return;
  tip.textContent=c.getAttribute('data-tip');tip.classList.add('show');});
 document.addEventListener('mousemove',e=>{
  if(!tip.classList.contains('show'))return;
  let x=e.clientX+14,y=e.clientY+14;
  if(x+310>innerWidth)x=e.clientX-tip.offsetWidth-14;
  if(y+tip.offsetHeight+8>innerHeight)y=e.clientY-tip.offsetHeight-14;
  tip.style.left=x+'px';tip.style.top=y+'px';});
 document.addEventListener('mouseout',e=>{
  if(e.target.closest('.hit[data-tip]'))tip.classList.remove('show');});
 document.addEventListener('click',e=>{
  const c=e.target.closest('.hit[data-href]');
  if(c)location.href=c.getAttribute('data-href');});
})();
</script>"""


def footer(rel=""):
    links = [
        (REPO, "GitHub", True),
        (REPO + "/blob/main/CONTRIBUTING.md", "Contribute", False),
        (REPO + "/blob/main/schema/bound.schema.json", "Schema", False),
        (rel + "references.html", "References", False),
        ("https://arxiv.org/abs/2605.28586", "Paper", False),
    ]
    out = ["<footer class=foot><div class=footmain><div class=footbrand><div class=fb>",
           FOOTMARK, "<span>Stabilizer Rank Challenge</span></div>",
           "<p>An open, automatically verified leaderboard for exact stabilizer "
           "decompositions of magic states.</p></div><nav class=footlinks>"]
    for href, label, icon in links:
        out.append(f"<a href='{href}'>" + (GHICON if icon else "") + label + "</a>")
    out.append("</nav></div></footer>")
    return "".join(out)


PARTICIPATE = """<dialog id=participate class=modal>
<form method=dialog><button class=x aria-label=Close>&times;</button></form>
<h3>Participate</h3>
<p>Run it yourself, or point a coding agent at it.</p>
<div class=codewrap><button class=copy type=button
onclick="navigator.clipboard.writeText(this.parentNode.querySelector('code').innerText)">copy</button>
<pre><code>git clone https://github.com/unitaryfoundation/stabrank
cd stabrank
# describe your decomposition in bounds/mybound.json
make fit BOUND=bounds/mybound.json ARGS=--write
make verify BOUND=bounds/mybound.json</code></pre></div>
<p><code>fit</code> solves for the exact coefficients, or tells you no exact
combination of your terms reproduces the target. <code>verify</code> rebuilds
the identity in exact arithmetic. If it passes, open a pull request adding the
file and the leaderboard picks it up on the next build.</p>
<p>A bound the pipeline cannot check is recorded as <span class='pill t-cited'>cited</span>
and never takes a record.</p>
</dialog>"""


ARXIV = re.compile(r"arXiv:\s*([0-9]{4}\.[0-9]{4,5})", re.I)


def arxiv_url(prov):
    m = ARXIV.search(prov.get("reference", "") or "")
    return f"https://arxiv.org/abs/{m.group(1)}" if m else None


def source_link(e, rel=""):
    """Where a reader should go to check this bound themselves.

    A Lean proof beats a paper, a paper beats our own detail page, because the
    point of a link here is to leave the leaderboard and see the evidence.
    """
    sub = e["sub"]
    ln = sub.get("lean")
    if ln:
        return (REPO + "/blob/main/lean_proofs/" + ln["module"].replace(".", "/")
                + ".lean", "Lean proof")
    u = arxiv_url(sub["provenance"])
    if u:
        return u, "paper"
    return f"{rel}bounds/{e['slug']}.html", "details"


PR_REF = re.compile(r"(?<![\w#])#(\d{1,5})\b")
IDENT = re.compile(
    r"(?<![\w/`.])("
    r"(?:[\w\-]+/)+[\w\-]+\.\w+"                                 # paths: docs/notes/x.md
    r"|[A-Za-z][\w]*\.(?:py|lean|json|jsonl|md|cpp|hpp)"         # file names
    r"|[A-Za-z][\w]*(?:\.[A-Za-z][\w]*)+"                        # dotted names: Stabilizer.stabRank
    r"|[A-Za-z][\w]*(?:_[\w]+)+"                                 # snake_case identifiers
    r"|[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+(?:[A-Z]\w*)?"              # CamelCase modules: StrangeM2Pointwise
    r")(?![\w/])")
SUBSCRIPT = re.compile(r"\b(psi|phi|[A-Za-z])_([0-9a-z])\b")
KET = re.compile(r"\|([\w+\-]+)&gt;(?:\^\{?(?:ot\s*)?([0-9a-z]+)\}?|\^\(([^)]+)\))?")
BRA = re.compile(r"&lt;([\w+\-]+)\|")
SYMBOLS = [(re.compile(r"\bchi\b"), "&chi;"), (re.compile(r"\bgamma\b"), "&gamma;"),
           (re.compile(r"\bomega\b"), "&omega;"), (re.compile(r"\bw3\b"), "&omega;<sub>3</sub>"),
           (re.compile(r"\bw9\b"), "&omega;<sub>9</sub>"), (re.compile(r"\bpi\b"), "&pi;"),
           (re.compile(r"\bsqrt\((\d+)\)"), r"&radic;\1"), (re.compile(r"\bsqrt(\d+)\b"), r"&radic;\1"),
           (re.compile(r"\bsqrt\b"), "&radic;"), (re.compile(r"\bpsi\b"), "&psi;"), (re.compile(r"\bphi\b"), "&phi;"),
           (re.compile(r"\bbeta\b"), "&beta;"), (re.compile(r"\balpha\b"), "&alpha;")]


def fmt_prose(text):
    """Render a submission's free text (notes, method) as HTML.

    The files are written in plain ASCII: |S>^2, chi(T3^3) >= 8, <= and >=,
    (x) for the tensor product, w3 for a cube root of unity, snake_case and
    dotted Lean names, file names, and #41 for a pull request. This turns
    each into the typographic or linked form so the page does not show
    "IS>^2 <= 2" for a ket, and code names read as code. Escaping happens
    first, so nothing in the text can inject markup.
    """
    t = E(text or "")
    # comparison operators and arrows (after escaping, < is &lt; and > is &gt;)
    t = t.replace("&lt;=", "&le;").replace("&gt;=", "&ge;").replace("!=", "&ne;")
    t = t.replace("-&gt;", "&rarr;").replace(" (x) ", " &otimes; ")
    # kets and bras, matched on the escaped text so the tags emitted survive
    def ket(m):
        exp = m.group(2) or m.group(3)
        sup = f"<sup>&otimes;{exp}</sup>" if exp else ""
        name = re.sub(r"^(\w+?)_(\w+)$", r"\1<sub>\2</sub>", m.group(1))
        return f"|{name}&rang;{sup}"
    t = KET.sub(ket, t)
    t = BRA.sub(lambda m: f"&lang;{m.group(1)}|", t)
    # links
    t = ARXIV.sub(lambda m: f"<a href='https://arxiv.org/abs/{m.group(1)}'>{m.group(0)}</a>", t)
    t = PR_REF.sub(lambda m: f"<a href='{REPO}/pull/{m.group(1)}'>#{m.group(1)}</a>", t)
    # code-like tokens, protecting what is already inside a tag or link
    parts = re.split(r"(<[^>]+>[^<]*</a>|<[^>]+>)", t)
    out = []
    for part in parts:
        if part.startswith("<"):
            out.append(part)
            continue
        for rx, rep in SYMBOLS:
            part = rx.sub(rep, part)
        part = SUBSCRIPT.sub(lambda m: (f"&{m.group(1)};" if len(m.group(1)) > 1 else m.group(1))
                             + f"<sub>{m.group(2)}</sub>", part)
        part = IDENT.sub(lambda m: f"<code>{m.group(1)}</code>", part)
        out.append(part)
    return "".join(out)


def ref_link(prov):
    """Render a reference string, linking the arXiv id if there is one."""
    txt = prov.get("reference", "") or ""
    u = arxiv_url(prov)
    if not u:
        return E(txt)
    m = ARXIV.search(txt)
    return (E(txt[:m.start()]) + f"<a href='{u}'>{E(m.group(0))}</a>"
            + E(txt[m.end():]))


TENS = "<span class=tp>&otimes;</span>"

ORB_TEX_NAME = {"S": "S", "N": "N", "H3": "H_3", "T3": "T_3",
                "qubit_H": r"H", "qubit_T": r"T", "T5": "T_5"}

# A family cell is one m-qubit state, so its ket carries m as an index
# rather than as a tensor power: |cat_6>, not |cat>^{(x) 6}.
FAMILY_KET = {"cat": r"\text{cat}_{%d}"}


def ket(orbit, m, sign=None, rank=None):
    r"""\chi_R(|M\rangle^{\otimes m}) with an optional bound, as MathML."""
    if orbit in FAMILY_KET:
        tex = rf"\chi_R\bigl(\left|{FAMILY_KET[orbit] % int(m)}\right\rangle\bigr)"
    else:
        nm = ORB_TEX_NAME.get(orbit, orbit)
        tex = rf"\chi_R\bigl(\left|{nm}\right\rangle^{{\otimes {m}}}\bigr)"
    if sign and rank is not None:
        tex += (r" \le " if sign in ("&le;", "<=") else r" \ge ") + str(rank)
    return f"<span class=ket>{M(tex)}</span>"


def lean_ok(sub):
    """True/False if a build receipt exists for this bound's module, else None."""
    ln = sub.get("lean")
    if not ln:
        return None
    rec = os.path.join(CERTS, "lean-" + ln["module"].replace(".", "-") + ".json")
    if not os.path.isfile(rec):
        return None
    return bool(json.load(open(rec)).get("ok"))


def lean_badge(sub, rel="", compact=True):
    """Link a bound to the Lean theorem that proves it, when there is one.

    Shown independently of the tier: the badge says a proof exists, the tier
    says whether a build receipt confirmed it compiles.
    """
    ln = sub.get("lean")
    if not ln:
        return ""
    url = (REPO + "/blob/main/lean_proofs/"
           + ln["module"].replace(".", "/") + ".lean")
    ok = lean_ok(sub)
    cls = "t-lean" if ok else ("t-leanbad" if ok is False else "t-leanpend")
    tip = ("machine-checked" if ok else
           ("the recorded lake build of this module fails" if ok is False
            else "no build receipt recorded yet"))
    mark = "" if ok else (" &#9888;" if ok is False else " &middot; unbuilt")
    label = "lean" if compact else f"Lean &middot; {E(ln['theorem'])}"
    return (f"<a class='pill {cls}' href='{url}' "
            f"title='{E(ln['theorem'])} in {E(ln['module'])}: {tip}'>"
            f"{label}{'' if compact else mark}</a>")


def compute_summary(comp):
    """One sentence of what the search cost, from provenance.compute."""
    parts = []
    if comp.get("runs") is not None:
        parts.append(f"{comp['runs']} run{'s' if comp['runs'] != 1 else ''}")
    if comp.get("cpu_hours") is not None:
        parts.append(f"{comp['cpu_hours']:g} CPU-hours")
    if comp.get("gpu_hours"):
        parts.append(f"{comp['gpu_hours']:g} GPU-hours")
    if comp.get("wall_clock_hours") is not None:
        parts.append(f"{comp['wall_clock_hours']:g} h wall-clock")
    if comp.get("hardware"):
        parts.append("on " + E(comp["hardware"]))
    for u in comp.get("llm") or []:
        toks = []
        if u.get("input_tokens") is not None:
            toks.append(f"{u['input_tokens']:,} in")
        if u.get("output_tokens") is not None:
            toks.append(f"{u['output_tokens']:,} out")
        piece = f"<span class=mono>{E(u['model'])}</span>"
        if toks:
            piece += " (" + ", ".join(toks) + " tokens)"
        if u.get("role"):
            piece += ", " + E(u["role"])
        parts.append(piece)
    if comp.get("cost_usd") is not None:
        parts.append(f"${comp['cost_usd']:,.2f}")
    return "; ".join(parts) + "." if parts else "reported, but empty."


def attested_summary(sub):
    """The `certificate.attested` block with the manifest's batch count added
    as `total` (None when the manifest cannot be read), or None when the bound
    does not declare one. Cheap: the manifest is small and read once per path.
    """
    att = (sub.get("certificate") or {}).get("attested")
    if not att:
        return None
    batches, _ = _manifest(att["batches"])
    return {**att, "total": len(batches) if batches else None}


@functools.lru_cache(maxsize=None)
def _manifest(rel):
    return load_batch_manifest(rel)


def write_ledger(entries):
    """docs/ledger.json: every bound with its tier and what finding it cost.

    The board shows results; the ledger is the data behind a cost-per-discovery
    curve, one row per submission, machine-readable and regenerated on every
    build so nothing has to be back-filled later. An attested bound's row
    carries its `attested` block plus the manifest's batch count, so the
    re-run fraction behind the tier is on record.
    """
    rows = []
    for e in entries:
        s, r = e["sub"], e["res"]
        prov = s["provenance"]
        rows.append({
            "slug": e["slug"], "orbit": s["orbit"], "m": int(s["m"]),
            "direction": s["direction"], "rank": int(s["rank"]),
            "tier": r["tier"], "ok": r["ok"],
            "gamma": r.get("gamma") if s["direction"] == "upper" else None,
            "date": prov.get("date"), "author": prov.get("author"),
            "github": prov.get("github") or [], "method": prov.get("method"),
            "reference": prov.get("reference"),
            "compute": prov.get("compute"),
            "budget_s": (s.get("certificate") or {}).get("budget_s"),
            "attested": attested_summary(s),
        })
    rows.sort(key=lambda x: (x["date"] or "", x["slug"]))
    with open(os.path.join(DOCS, "ledger.json"), "w") as f:
        json.dump({"generated_from": "bounds/", "rows": rows}, f, indent=1)
        f.write("\n")
    return rows


def gh_links(prov):
    """Render submitters as linked GitHub handles where we know them."""
    hs = prov.get("github") or []
    if not hs:
        return E(prov.get("author", ""))
    return " ".join(f"<a class=gh href='https://github.com/{E(h)}'>@{E(h)}</a>" for h in hs)


def tier_pill(tier, ok=True, sub=None, compact=True):
    """One badge per bound.

    A lean-tier bound shows only its Lean badge, which already says the tier and
    links to the proof; showing both a `lean` pill and a `Lean - theorem` pill
    was two tags for one fact. A cited bound links to what it is cited from.
    """
    if not ok:
        return "<span class='pill t-failed'>failed</span>"
    if tier == "lean" and sub is not None and sub.get("lean"):
        return lean_badge(sub, compact=compact)
    if tier == "cited" and sub is not None:
        u = arxiv_url(sub["provenance"])
        if u:
            return (f"<a class='pill t-cited' href='{u}' "
                    f"title='cited from {E(sub[chr(39)+chr(39)] if False else sub['provenance'].get('reference',''))}'>"
                    f"cited</a>")
    budget = ((sub or {}).get("certificate") or {}).get("budget_s")
    if tier == "attested" and sub is not None:
        a = attested_summary(sub)
        if a:
            frac = f"{a['recomputed']}/{a['total']}" if a["total"] else str(a["recomputed"])
            pct = (f" ({100 * a['recomputed'] / a['total']:.0f}%)" if a["total"] else "")
            return (f"<span class='pill t-attested' title='offline enumeration, "
                    f"{a['compute_hours']:g} CPU-h on {E(a['hardware'])}, not re-run; the "
                    f"certificate re-ran {frac} batches{pct} from scratch and checked the "
                    f"hashes of the rest'>attested &middot; {frac} re-run</span>")
    if tier == "reproduced" and budget and int(budget) > 900:
        return (f"<span class='pill t-{tier}' title='the certificate ran within a declared "
                f"{int(budget)} s budget, above the 900 s default'>{tier} &middot; {int(budget)} s</span>")
    return f"<span class='pill t-{tier}'>{tier}</span>"


# -------------------------------------------------------------- references --

def parse_bib(path):
    try:
        text = open(path).read()
    except OSError:
        return []
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        i = text.index("{", m.start())
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[i + 1:j]
        e = {"key": m.group(2).strip(), "type": m.group(1).lower()}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            k = fm.group(1).lower()
            p = fm.end()
            if p < len(body) and body[p] == "{":
                d, q = 0, p
                while q < len(body):
                    if body[q] == "{":
                        d += 1
                    elif body[q] == "}":
                        d -= 1
                        if d == 0:
                            break
                    q += 1
                e[k] = body[p + 1:q].strip()
        out.append(e)
    return out


def references_page(refs):
    o = [head("References — Stabilizer Rank Challenge")]
    o.append(hero("References",
                  "Work the bounds on this leaderboard are measured against."))
    o.append(PARTICIPATE)
    o.append("<div class=wrap><ol class=refs>")
    for r in sorted(refs, key=lambda r: (r.get("year", ""), r.get("author", ""))):
        au = E(r.get("author", "")).replace(" and ", ", ")
        ti = E(r.get("title", ""))
        ven = ", ".join(x for x in [E(r.get("journal", "")),
                                    (f"vol. {E(r['volume'])}" if r.get("volume") else ""),
                                    (f"p. {E(r['pages'])}" if r.get("pages") else ""),
                                    E(r.get("year", ""))] if x)
        links = []
        if r.get("eprint"):
            links.append(f"<a href='https://arxiv.org/abs/{E(r['eprint'])}'>"
                         f"arXiv:{E(r['eprint'])}</a>")
        if r.get("doi"):
            links.append(f"<a href='https://doi.org/{E(r['doi'])}'>DOI</a>")
        o.append(f"<li><span class=au>{au}</span>. <span class=ti>{ti}</span>. "
                 f"<span class=vn>{ven}</span>"
                 + (" &middot; " + " &middot; ".join(links) if links else "")
                 + (f"<div class=nt>{E(r['note'])}</div>" if r.get("note") else "")
                 + "</li>")
    o.append("</ol></div>" + footer() + "</body></html>")
    with open(os.path.join(DOCS, "references.html"), "w") as f:
        f.write("".join(o))


# ------------------------------------------------------------------ build ---

def build(no_verify=False):
    entries = load_bounds(no_verify)
    cells = best_by_cell(entries)
    board = leaderboard(entries)
    people = contributors(entries)
    lean_n = sum(1 for e in entries if e["res"]["ok"] and e["res"]["tier"] == "lean")
    leanclaim_n = sum(1 for e in entries if e["sub"].get("lean"))
    ver_n = sum(1 for e in entries if e["res"]["ok"] and e["res"]["tier"] == "verified")
    rep_n = sum(1 for e in entries if e["res"]["ok"] and e["res"]["tier"] == "reproduced")
    att_n = sum(1 for e in entries if e["res"]["ok"] and e["res"]["tier"] == "attested")
    # Only orbits with a published exponent count towards "beaten"; a product
    # bound standing in for an unpublished one is not a record to beat.
    moved = [r for r in board if r["orbit"] not in UNPUBLISHED and r["best"]
             and r["best"]["res"]["gamma"] < r["baseline"] - 1e-12]
    published_n = sum(1 for ob in ORBIT_ORDER if ob not in UNPUBLISHED)
    tight = min((r for r in board if r["best"]),
                key=lambda r: r["best"]["res"]["gamma"], default=None)

    o = [head("Stabilizer Rank Challenge")]
    o.append(hero("Stabilizer Rank Challenge",
                  "Find smaller exact stabilizer decompositions of magic states."))
    o.append(PARTICIPATE)
    o.append("<div class=wrap>")

    # ---- the graph, first thing on the page
    o.append("<h2>Record progress</h2>")
    o.append("<p class=h2sub>Best known per-copy exponent &gamma; for each orbit. "
             "Lower is better; a line steps down when a submission improved that "
             "orbit's best bound.</p>")
    o.append("<div class=chartbox>" + progress_chart(entries) + "</div>")
    o.append("<div id=tip role=tooltip></div>")
    o.append(CHART_SCRIPT)
    o.append("<div class=legend>" + "".join(
        f"<a class=lgd href='orbits/{ob}.html'><i style='background:{COLOR[ob]}'></i>"
        f"{ORBIT_LABEL[ob]} ({SYSTEM[ob]})</a>" for ob in ORBIT_ORDER) + "</div>")

    # ---- contributor leaderboard
    o.append("<h2>Leaderboard</h2>")
    o.append("<p class=h2sub>Ranked by records held. A record needs a bound the "
             "pipeline could check, so cited literature values never take one.</p>")
    o.append("<div class=lb><div class=lbhead><div>"
             f"<div class=t>Leaderboard</div><div class=s>{len(people)} contributors "
             f"&middot; {len(entries)} bounds submitted through the challenge</div></div>")
    if tight:
        o.append(f"<div><div class=bigk>Tightest &gamma;</div>"
                 f"<div class=big>{tight['best']['res']['gamma']:.4f}</div>"
                 f"<div class=s style='text-align:right'>{ORBIT_LABEL[tight['orbit']]}"
                 f" &middot; {E(tight['best']['sub']['provenance'].get('author',''))}</div></div>")
    o.append("</div>")
    bestlink = {}
    for e in entries:
        s_, r = e["sub"], e["res"]
        w = s_["provenance"].get("author")
        if s_["direction"] == "upper" and r["ok"] and r.get("gamma") is not None:
            cur = bestlink.get(w)
            if cur is None or r["gamma"] < cur[0]:
                bestlink[w] = (r["gamma"], source_link(e)[0])
    bestlink = {k: v[1] for k, v in bestlink.items()}
    for i, a in enumerate(people, 1):
        best = f"{a['best']:.4f}" if a["best"] is not None else "&mdash;"
        pg = f"people/{person_slug(a['who'])}.html"
        bl = (f"<a href='{bestlink[a['who']]}'>{best}</a>"
              if a["who"] in bestlink else best)
        o.append(f"<div class=lbrow><div class=rk>{i}</div>"
                 f"<div class=who><a href='{pg}'>{a['links']}</a>"
                 f"{' &#128081;' if i == 1 and a['records'] else ''}</div>"
                 f"<a class=m href='{pg}#bounds'><b>{a['n']}</b>"
                 f"<span>bound{'' if a['n'] == 1 else 's'}</span></a>"
                 f"<a class=m href='{pg}#records'><b>{a['records']}</b>"
                 f"<span>record{'' if a['records'] == 1 else 's'}</span></a>"
                 f"<div class=m><b>{bl}</b><span>best &gamma;</span></div></div>")
    o.append("</div>")

    # ---- exponent table
    o.append("<h2>Exponents</h2>")
    unpub = [ORBIT_LABEL[ob] for ob in ORBIT_ORDER if ob in UNPUBLISHED]
    o.append("<p class=h2sub>Every &gamma; below is published"
             + (", except for " + ", ".join(unpub) + ", where no exponent has been "
                "published for any state of that dimension and the single-copy "
                "product bound stands in" if unpub else "")
             + f". {len(moved)} of the {published_n} published exponents have been beaten here.</p>"
             "<div class=tw><table>")
    o.append("<thead><tr><th>orbit</th><th></th><th class=num>published "
             + M(r"\gamma") + " &le;</th>"
             "<th class=num>best here</th><th>progress</th><th>record held by</th>"
             "</tr></thead><tbody>")
    for r in board:
        orbit, base, best = r["orbit"], r["baseline"], r["best"]
        if best:
            g = best["res"]["gamma"]
            beat = g < base - 1e-12
            s = best["sub"]
            url, what = source_link(best)
            held = (f"<a href='{url}'>&chi; &le; {s['rank']} at m={s['m']}</a> "
                    f"{tier_pill(best['res']['tier'], True, best['sub'])}")
            gtxt = f"<span class=gain>{g:.4f}</span>" if beat else f"{g:.4f}"
            bar = (f"<span class=gain>beaten by {base - g:.4f}</span>" if beat
                   else next_target(orbit, base, cells))
        else:
            gtxt, held = "<span class=none>&mdash;</span>", "<span class=none>open</span>"
            bar = next_target(orbit, base, cells)
        o.append(f"<tr{' class=rec' if best and best['res']['gamma'] < base - 1e-12 else ''}>"
                 f"<td><b><a href='orbits/{orbit}.html'>{ORBIT_LABEL[orbit]}</a></b></td>"
                 f"<td class=mono style='color:var(--mut)'>{SYSTEM[orbit]}</td>"
                 f"<td class=num>{base:.4f}</td><td class=num>{gtxt}</td>"
                 f"<td>{bar}</td><td>{held}</td></tr>")
    o.append("</tbody></table></div>")

    # ---- cell ledger
    o.append("<h2>Cell ledger</h2>")
    o.append("<p class=h2sub>Best bound on "
             + M(r"\chi_R\bigl(\left|M\right\rangle^{\otimes m}\bigr)")
             + " per orbit and copy count, or per family and qubit count. "
             "Matching upper and lower bounds settle a cell.</p>")
    o.append("<div class=grid3>")
    for orbit in ORBIT_ORDER:
        base, base_txt = BASELINE[orbit]
        o.append(f"<div class=orb><h3><a href='orbits/{orbit}.html'>"
                 f"{ORBIT_LABEL[orbit]}</a></h3>"
                 f"<div class=sub>{SYSTEM[orbit]} &middot; {base_word(orbit)} "
                 f"{M(BASE_TEX[orbit])}</div><div class=cells>")
        ms = sorted({int(k[1]) for k in cells if k[0] == orbit})
        if not ms:
            o.append("<div><span class=none>no bounds yet</span></div>")
        for m in ms:
            up, lo = cells.get((orbit, m, "upper")), cells.get((orbit, m, "lower"))
            settled = up and lo and up["sub"]["rank"] == lo["sub"]["rank"]
            u = f"&le;{up['sub']['rank']}" if up else "&mdash;"
            l = f"&ge;{lo['sub']['rank']}, " if lo else ""
            val = f"<b>= {up['sub']['rank']}</b>" if settled else f"{l}{u}"
            tgt = up or lo
            if tgt:
                url, _ = source_link(tgt)
                val = f"<a href='{url}'>{val}</a>"
            o.append(f"<div><span>m = {m}</span><span>{val} "
                     f"{tier_pill(up['res']['tier'], True, up['sub'], compact=True) if up else ''}</span></div>")
        o.append("</div></div>")
    o.append("</div>")

    # ---- recently added
    # Most of the literature arrived in two batches on one day each, so the date
    # alone would hand nearly every slot to whichever orbit sorts first; ties go
    # to record-holding tiers, then across orbits.
    recent = sorted(
        entries,
        key=lambda e: (e["sub"]["provenance"].get("date", ""),
                       e["res"]["tier"] in RECORD_TIERS,
                       -ORBIT_ORDER.index(e["sub"]["orbit"]),
                       -int(e["sub"]["m"])),
        reverse=True)[:10]
    o.append("<h2>Recently added <span class=h2note>&middot; newest first; a "
             "literature entry is dated by its arXiv v1</span></h2>"
             "<div class=recent>")
    for e in recent:
        s_, r = e["sub"], e["res"]
        sign = "&le;" if s_["direction"] == "upper" else "&ge;"
        star = "&#9733; " if e["res"]["tier"] in RECORD_TIERS else ""
        o.append(f"<div class=rrow><a class=rb href='bounds/{e['slug']}.html'>"
                 f"{ket(s_['orbit'], s_['m'], sign, s_['rank'])}</a>"
                 f"{'<span class=star>&#9733;</span>' if star else ''}"
                 f"<span class=rt>{ORBIT_LABEL[s_['orbit']]}</span>"
                 f"<span class=rw>{gh_links(s_['provenance'])}</span>"
                 f"{tier_pill(r['tier'], r['ok'], s_)}"
                 f"<span class=rd>{E(s_['provenance'].get('date','')[:10])}</span></div>")
    o.append("</div>")

    # ---- all submissions
    o.append("<h2>All submissions</h2><div class='tw scroll'><table>")
    o.append("<thead><tr><th>bound</th><th>orbit</th><th class=num>m</th>"
             "<th class=num>rank</th><th class=num>&gamma;</th><th>tier</th>"
             "<th>attribution</th></tr></thead><tbody>")
    for e in sorted(entries, key=lambda e: (ORBIT_ORDER.index(e["sub"]["orbit"]),
                                            int(e["sub"]["m"]), e["sub"]["direction"])):
        s, r = e["sub"], e["res"]
        sign = "&le;" if s["direction"] == "upper" else "&ge;"
        g = r.get("gamma")
        gt = f"{g:.4f}" if (g is not None and s["direction"] == "upper") else "&mdash;"
        o.append(f"<tr><td><a href='bounds/{e['slug']}.html'>"
                 f"{ket(s['orbit'], s['m'], sign, s['rank'])}</a></td>"
                 f"<td><a href='orbits/{s['orbit']}.html'>{ORBIT_LABEL[s['orbit']]}</a></td>"
                 f"<td class=num>{s['m']}</td><td class=num>{s['rank']}</td>"
                 f"<td class=num>{gt}</td>"
                 f"<td>{tier_pill(r['tier'], r['ok'], s_)}</td>"
                 f"<td style='white-space:normal'>{gh_links(s['provenance'])}"
                 f" &middot; <span class=mono style='font-size:12px'>"
                 f"{ref_link(s['provenance'])}</span></td></tr>")
    o.append("</tbody></table></div>")

    o.append("</div>" + footer() + "</body></html>")

    os.makedirs(os.path.join(DOCS, "bounds"), exist_ok=True)
    with open(os.path.join(DOCS, "index.html"), "w") as f:
        f.write("".join(o))
    with open(os.path.join(DOCS, "style.css"), "w") as f:
        f.write(CSS)
    with open(os.path.join(DOCS, "favicon.svg"), "w") as f:
        f.write(FAVICON)
    references_page(parse_bib(os.path.join(DOCS, "refs.bib")))
    rows = write_ledger(entries)
    from report import write_report  # noqa: E402  (report imports this module)
    write_report(entries, rows)
    for e in entries:
        detail_page(e)
    for orbit in ORBIT_ORDER:
        orbit_page(orbit, entries, cells)
    for a in people:
        person_page(a, entries, cells)
    print(f"docs/ written: {len(entries)} bounds, {lean_n} lean-certified "
          f"({leanclaim_n} claim a proof), "
          f"{ver_n} verified, {rep_n} reproduced, {att_n} attested, "
          f"{len(moved)} exponents beaten, "
          f"{len(people)} contributors")


def orbit_page(orbit, entries, cells):
    """A dedicated page per magic-state orbit: what the state is, and every
    bound anyone has submitted against it."""
    defn, blurb, note = ORBIT_DEF[orbit]
    base, base_txt = BASELINE[orbit]
    mine = [e for e in entries if e["sub"]["orbit"] == orbit]
    o = [head(f"{ORBIT_LABEL[orbit]} — Stabilizer Rank Challenge", rel="../")]
    o.append(hero(f"{ORBIT_LABEL[orbit]}",
                  f"The {SYSTEM[orbit]} {ORBIT_LABEL[orbit]} orbit. "
                  f"{base_word(orbit).capitalize()} {M(BASE_TEX[orbit])}"
                  f"{base_note(orbit)}.", rel="../"))
    o.append(PARTICIPATE)
    o.append("<div class=wrap><p><a href='../index.html'>&larr; back to the board</a></p>")
    o.append("<h2>The state</h2><div class=statebox><div class=stateeq>"
             + "".join(M(t, block=True) for t in _lines(ORBIT_TEX[orbit]))
             + "</div></div>")
    o.append(f"<p>{blurb}</p><p>{note}</p>")
    o.append("<h2>Bounds on this orbit</h2><div class=tw><table>"
             "<thead><tr><th>bound</th><th class=num>m</th><th class=num>rank</th>"
             "<th class=num>&gamma;</th><th>tier</th><th>attribution</th>"
             "</tr></thead><tbody>")
    for e in sorted(mine, key=lambda e: (int(e["sub"]["m"]), e["sub"]["direction"])):
        s_, r = e["sub"], e["res"]
        sign = "&le;" if s_["direction"] == "upper" else "&ge;"
        g = r.get("gamma")
        gt = f"{g:.4f}" if (g is not None and s_["direction"] == "upper") else "&mdash;"
        o.append(f"<tr><td><a href='../bounds/{e['slug']}.html'>"
                 f"{ket(orbit, s_['m'], sign, s_['rank'])}</a></td>"
                 f"<td class=num>{s_['m']}</td><td class=num>{s_['rank']}</td>"
                 f"<td class=num>{gt}</td>"
                 f"<td>{tier_pill(r['tier'], r['ok'], s_)}</td>"
                 f"<td style='white-space:normal'>{gh_links(s_['provenance'])} &middot; "
                 f"<span class=mono style='font-size:12px'>{ref_link(s_['provenance'])}"
                 f"</span></td></tr>")
    o.append("</tbody></table></div>")
    # Once the baseline is beaten the target is the board's own best, not
    # the baseline: naming a cell the board already holds is not a target.
    best = min((e for e in mine
                if e["sub"]["direction"] == "upper" and e["res"]["ok"]
                and e["res"]["tier"] in RECORD_TIERS and e["res"].get("gamma") is not None
                and int(e["sub"]["m"]) > 1),
               key=lambda e: e["res"]["gamma"], default=None)
    if best is not None and best["res"]["gamma"] < base - 1e-12:
        g, s_ = best["res"]["gamma"], best["sub"]
        o.append(f"<h2>What would move it</h2><p>The {base_word(orbit)} exponent"
                 f"{base_note(orbit)} is already beaten: &chi; &le; {s_['rank']} at "
                 f"m={s_['m']} gives &gamma; = {g:.4f}. Lowering it further needs "
                 f"{next_target(orbit, g, cells, lead='')}.</p>")
    else:
        o.append(f"<h2>What would move it</h2><p>Beating the {base_word(orbit)} exponent"
                 f"{base_note(orbit)} needs {next_target(orbit, base, cells, lead='')}.</p>")
    o.append("</div>" + footer(rel="../") + "</body></html>")
    os.makedirs(os.path.join(DOCS, "orbits"), exist_ok=True)
    with open(os.path.join(DOCS, "orbits", f"{orbit}.html"), "w") as f:
        f.write("".join(o))


def person_slug(who):
    return re.sub(r"[^a-z0-9]+", "-", who.lower()).strip("-")


def person_page(a, entries, cells):
    """A page per contributor, so the leaderboard's counts are all clickable."""
    holders = {id(e) for e in cells.values()}
    mine = [e for e in entries if e["sub"]["provenance"].get("author") == a["who"]]
    o = [head(f"{a['who']} — Stabilizer Rank Challenge", rel="../")]
    o.append(hero(E(a["who"]),
                  f"{a['n']} bounds submitted &middot; {a['records']} records held.",
                  rel="../"))
    o.append(PARTICIPATE)
    o.append("<div class=wrap><p><a href='../index.html'>&larr; back to the board</a></p>")
    for title, anchor, sel in (
        ("Records held", "records", lambda e: id(e) in holders),
        ("All bounds", "bounds", lambda e: True)):
        rows = [e for e in mine if sel(e)]
        o.append(f"<h2 id='{anchor}'>{title} <span class=h2note>&middot; "
                 f"{len(rows)}</span></h2>")
        if not rows:
            o.append("<p class=none>none yet</p>")
            continue
        o.append("<div class=tw><table><thead><tr><th>bound</th><th>orbit</th>"
                 "<th class=num>&gamma;</th><th>tier</th></tr></thead><tbody>")
        for e in sorted(rows, key=lambda e: (ORBIT_ORDER.index(e["sub"]["orbit"]),
                                             int(e["sub"]["m"]))):
            s_, r = e["sub"], e["res"]
            sign = "&le;" if s_["direction"] == "upper" else "&ge;"
            g = r.get("gamma")
            gt = f"{g:.4f}" if (g is not None and s_["direction"] == "upper") else "&mdash;"
            o.append(f"<tr><td><a href='../bounds/{e['slug']}.html'>"
                     f"{ket(s_['orbit'], s_['m'], sign, s_['rank'])}</a></td>"
                     f"<td><a href='../orbits/{s_['orbit']}.html'>"
                     f"{ORBIT_LABEL[s_['orbit']]}</a></td>"
                     f"<td class=num>{gt}</td>"
                     f"<td>{tier_pill(r['tier'], r['ok'], s_)}</td></tr>")
        o.append("</tbody></table></div>")
    o.append("</div>" + footer(rel="../") + "</body></html>")
    os.makedirs(os.path.join(DOCS, "people"), exist_ok=True)
    with open(os.path.join(DOCS, "people", f"{person_slug(a['who'])}.html"), "w") as f:
        f.write("".join(o))


def detail_page(e):
    s, r = e["sub"], e["res"]
    sign = "&le;" if s["direction"] == "upper" else "&ge;"
    o = [head(f"chi({s['orbit']}^{s['m']}) {sign} {s['rank']}", rel="../")]
    o.append(hero(f"&chi;<sub>R</sub>(&#8739;{E(s['orbit'])}&rang;"
                  f"<sup>&otimes;{s['m']}</sup>) {sign} {s['rank']}",
                  f"{ORBIT_LABEL[s['orbit']]} orbit, {SYSTEM[s['orbit']]}.", rel="../"))
    o.append(PARTICIPATE)
    o.append("<div class=wrap><p><a href='../index.html'>&larr; back to the board</a></p>")
    o.append(f"<h2>Verification</h2><p>{tier_pill(r['tier'], r['ok'], s)} &nbsp; "
             f"{E(r['detail'])}</p>")
    if r.get("gamma") is not None and s["direction"] == "upper":
        base = BASELINE[s["orbit"]][0]
        beat = r["gamma"] < base - 1e-12
        o.append(f"<p>Implied per-copy exponent &gamma; &le; <b>{r['gamma']:.4f}</b> "
                 f"against a {base_word(s['orbit'])} {base:.4f}{base_note(s['orbit'])}, "
                 + ("<span class=gain>an improvement</span>." if beat else "no improvement.")
                 + "</p>")
    if s.get("lean"):
        ln = s["lean"]
        url = (REPO + "/blob/main/lean_proofs/" + ln["module"].replace(".", "/") + ".lean")
        o.append("<h2>Lean proof</h2><p>This bound is backed by a machine-checked "
                 f"theorem: <a class=mono href='{url}'>{E(ln['theorem'])}</a> in "
                 f"<span class=mono>{E(ln['module'])}</span>. "
                 + ("The recorded <code>lake build</code> confirms it compiles."
                    if r["tier"] == "lean" else
                    "No build receipt is recorded here yet, so this bound is shown "
                    "at the tier the other checks earn it; the Lean claim does not "
                    "inflate a tier on its own.") + "</p>")
    comp = s["provenance"].get("compute")
    if comp:
        o.append("<h2>Compute</h2><p>" + compute_summary(comp) + "</p>")
    if s.get("notes"):
        o.append(f"<h2>Notes</h2><p class=prose>{fmt_prose(s['notes'])}</p>")
    o.append("<h2>Attribution</h2><p>" + E(s["provenance"].get("author", ""))
             + " &middot; <span class=mono>" + ref_link(s["provenance"])
             + "</span>" + (" &middot; " + fmt_prose(s["provenance"]["method"])
                            if s["provenance"].get("method") else "") + "</p>")
    o.append("<h2>Submission</h2><details><summary>JSON</summary><pre>"
             + E(json.dumps(s, indent=2)) + "</pre></details>")
    o.append("</div>" + footer(rel="../") + "</body></html>")
    with open(os.path.join(DOCS, "bounds", f"{e['slug']}.html"), "w") as f:
        f.write("".join(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the challenge site into docs/.")
    ap.add_argument("--no-verify", action="store_true",
                    help="run no certificate: use certs/ receipts and docs/ledger.json "
                         "as they are, and leave a bound with neither off the board")
    build(ap.parse_args().no_verify)
