"""Generate the stabilizer-rank challenge site into docs/.

The leaderboard is earned, not declared. Every bound is run through
verify_challenge/stabrank_verify.py and displayed at the tier it earns:

    verified    the pipeline rebuilt the decomposition and confirmed it here
    reproduced  a certificate script ran and asserted the bound
    cited       attributed to the literature, not machine-checked

Only `verified` and `reproduced` bounds may hold a record. A cited value is
shown so the picture is complete, but it never crowns a cell, so the only way
to take a record is to submit something the pipeline can check.

Expensive certificates are not re-run on every build: a cached result in
certs/<slug>.json is trusted if it matches the submission's content hash.
"""

from __future__ import annotations

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
from stabrank_verify import ORBIT_LABEL, ORBIT_P, implied_gamma, verify  # noqa: E402
from _assets import GHICON, UFLOGO  # noqa: E402

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
}
ORBIT_ORDER = ["S", "N", "H3", "T3", "qubit_H", "qubit_T"]
SYSTEM = {"S": "qutrit", "N": "qutrit", "H3": "qutrit", "T3": "qutrit",
          "qubit_H": "qubit", "qubit_T": "qubit"}
ORBIT_DEF = {
 "S": ("(|1&rang; &minus; |2&rang;)/&radic;2",
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
   "The most studied cell in the field: &chi; &le; 6 at six copies underpins "
   "the standard 2<sup>0.47n</sup> figure. The published exponent log&#8322;3/4 "
   "comes from an asymptotic contracted-cat-state family rather than from any "
   "single m, so every finite-m entry here sits above it."),
 "qubit_T": ("cos(&beta;)|0&rang; + e<sup>i&pi;/4</sup>sin(&beta;)|1&rang;, "
   "&nbsp;cos(2&beta;) = 1/&radic;3",
   "The qubit Bravyi-Kitaev T-type state, the face centre of the stabilizer "
   "octahedron. A different Clifford orbit from the H-type despite the "
   "overloaded name.",
   "Rank 5 at six copies is conjectured impossible, with search terminating at "
   "exactly &radic;(5/6)&nbsp;sin(&pi;/12)."),
}

COLOR = {"S": "#6d28d9", "N": "#0369a1", "H3": "#059669",
         "T3": "#b45309", "qubit_H": "#be185d", "qubit_T": "#128081"}
TIER_RANK = {"lean": 4, "verified": 3, "reproduced": 2, "cited": 1, None: 0}
RECORD_TIERS = ("lean", "verified", "reproduced")

E = html.escape


def slug(sub):
    return f"{sub['orbit']}-m{sub['m']}-{sub['direction']}-{sub['rank']}"


def content_hash(sub):
    return hashlib.sha256(
        json.dumps(sub, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def load_bounds():
    out = []
    for path in sorted(glob.glob(os.path.join(BOUNDS, "*.json"))):
        sub = json.load(open(path))
        s, h = slug(sub), content_hash(sub)
        cpath = os.path.join(CERTS, f"{s}.json")
        res = None
        if os.path.exists(cpath):
            cached = json.load(open(cpath))
            if cached.get("content_hash") == h:
                res = cached
        if res is None:
            r = verify(sub)
            res = {"ok": r.ok, "tier": r.tier, "detail": r.detail,
                   "gamma": r.gamma, "content_hash": h}
            if r.ok and r.tier in ("verified", "reproduced"):
                os.makedirs(CERTS, exist_ok=True)
                with open(cpath, "w") as f:
                    json.dump(res, f, indent=2)
                    f.write("\n")
        if res.get("gamma") is None and sub["direction"] == "upper":
            res["gamma"] = implied_gamma(ORBIT_P[sub["orbit"]], sub["rank"], sub["m"])
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


def next_target(orbit, base):
    """The cheapest cell that would beat the published exponent.

    An empty progress bar on every row carries no information; naming the
    smallest rank at each m that would move the number is actionable.
    """
    p = ORBIT_P[orbit]
    best = None
    for m in range(2, 9):
        r = math.floor(p ** (base * m) - 1e-9)
        if abs(p ** (base * m) - r) < 1e-9:
            r -= 1
        if r < 2:
            continue
        g = math.log(r, p) / m
        if g < base - 1e-12 and (best is None or g < best[2]):
            best = (m, r, g)
    if best is None:
        return "<span class=none>&mdash;</span>"
    m, r, g = best
    return (f"<span class=tgt>needs &chi; &le; {r} at m={m}"
            f"<span class=tgtg>&rarr; {g:.4f}</span></span>")


def contributors(entries):
    """Rank submitters. Records are what count; volume breaks ties."""
    cells = best_by_cell(entries)
    holders = {id(e) for e in cells.values()}
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
                pts.append((yr, g))
        if pts:
            series[orbit] = pts
    if not series:
        return ""

    years = sorted({y for pts in series.values() for y, _ in pts})
    y0, y1 = min(years), max(years) + 1
    gs = [g for pts in series.values() for _, g in pts] + \
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
        for i, (yr, g) in enumerate(pts):
            if i == 0:
                d.append(f"M{X(yr):.1f},{Y(g):.1f}")
            else:
                d.append(f"L{X(yr):.1f},{Y(pts[i-1][1]):.1f}")
                d.append(f"L{X(yr):.1f},{Y(g):.1f}")
        d.append(f"L{X(y1):.1f},{Y(pts[-1][1]):.1f}")
        o.append(f"<path class=ln d='{' '.join(d)}' stroke='{c}'/>")
        for yr, g in pts:
            o.append(f"<circle cx='{X(yr):.1f}' cy='{Y(g):.1f}' r='4' "
                     f"fill='#fff' stroke='{c}' stroke-width='2'/>")
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
        o.append(f"<text class=lbl x='{X(y1)+10:.1f}' y='{y+4:.1f}' fill='{c}'>"
                 f"{ORBIT_LABEL[ob]} &middot; {g:.4f}</text>")
    o.append("</svg>")
    return "".join(o)


# ------------------------------------------------------------------- HTML ---

CSS = """
:root{--ink:#0f172a;--mut:#64748b;--ln:#e2e8f0;--ac:#36006c;--ex:#059669;
--exb:#ffff00;--dark:#111111;--bg:#fff;--soft:#f8fafc;--bad:#be185d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:Manrope,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.6}
.wrap{max-width:1060px;margin:0 auto;padding:0 20px}
a{color:var(--ac)}
code,.mono{font-family:"Space Mono",ui-monospace,Menlo,monospace}
h1,h2{font-family:Manrope,system-ui,sans-serif;letter-spacing:-.01em}

header.hero{background:
radial-gradient(115% 130% at 50% -25%,rgba(255,255,0,.18),transparent 60%),
repeating-linear-gradient(0deg,transparent 0 27px,rgba(255,255,255,.05) 27px 28px),
repeating-linear-gradient(90deg,transparent 0 27px,rgba(255,255,255,.05) 27px 28px),
var(--dark);color:#fff;padding:34px 0 30px;
position:relative;overflow:hidden;border-bottom:4px solid #ffff00}
header.hero>.wrap{position:relative;z-index:1}
.heroflow{position:absolute;inset:0;width:100%;height:100%;z-index:0;pointer-events:none}
.heroflow path{fill:none;stroke:#ffff00;stroke-width:1.5;opacity:.10;
stroke-linecap:round;stroke-dasharray:130 420;animation:flowtrail 15s linear infinite}
.heroflow path:nth-child(2){opacity:.07;animation-duration:21s;animation-delay:-4s}
.heroflow path:nth-child(3){opacity:.08;animation-duration:26s;animation-delay:-9s}
.heroflow path:nth-child(4){opacity:.06;animation-duration:18s;animation-delay:-2s}
.heroflow path:nth-child(5){opacity:.05;animation-duration:30s;animation-delay:-13s}
@keyframes flowtrail{to{stroke-dashoffset:-1100}}
@media(prefers-reduced-motion:reduce){.heroflow path{animation:none}}
.brand{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:0 0 18px}
.brandmark{display:flex;align-items:center;gap:16px}
.uflogo{height:38px;width:auto;display:block;filter:drop-shadow(0 4px 14px rgba(0,0,0,.35))}
header.hero h1{font-size:clamp(30px,6vw,44px);margin:0;letter-spacing:-1px}
header.hero p{font-size:18px;max-width:640px;margin:0;color:#e4e4e7}
header.hero p a{color:#ffff00;text-decoration:underline}
header.hero p a:hover{background:#ffff00;color:#111;text-decoration:none}
.topnav{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px}
.topnav a{display:inline-flex;align-items:center;gap:7px;color:#e4e4e7;
font-family:"Space Mono",ui-monospace,monospace;font-size:14px;font-weight:700;
padding:7px 14px;border:1px solid rgba(255,255,255,.18);border-radius:8px;
background:rgba(255,255,255,.06);text-decoration:none}
.topnav a:hover{background:#ffff00;color:#111;border-color:#ffff00}
.lbcta{flex:0 0 auto;font-size:14px;font-weight:700;color:#fff;
font-family:"Space Mono",ui-monospace,monospace;background:var(--ac);border:none;
border-radius:8px;padding:9px 16px;text-decoration:none;cursor:pointer;
box-shadow:0 4px 14px rgba(0,0,0,.35)}
.lbcta:hover{background:#5b21b6}

h2{font-size:22px;font-weight:700;margin:44px 0 4px;letter-spacing:-.01em}
.h2sub{color:var(--mut);font-size:14px;margin:0 0 14px}
section p{max-width:74ch}

.chartbox{border:1px solid var(--ln);border-radius:12px;padding:8px 10px 2px;background:#fff}
.chart{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--ln);stroke-width:1}
.chart .ax{font-family:"Space Mono",monospace;font-size:11px;fill:var(--mut)}
.chart .axl{font-family:"Space Mono",monospace;font-size:11px;fill:var(--mut)}
.chart .ln{fill:none;stroke-width:2.5;stroke-linejoin:round}
.chart .lbl{font-family:"Space Mono",monospace;font-size:12px;font-weight:700}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;padding:10px 4px 12px;
font-family:"Space Mono",monospace;font-size:12px;color:var(--mut)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}

.lb{border:1px solid var(--ln);border-radius:12px;background:var(--soft);overflow:hidden}
.lbhead{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;
gap:12px;padding:14px 18px;border-bottom:1px solid var(--ln);background:#fff}
.lbhead .t{font-family:"Space Mono",monospace;font-size:12px;letter-spacing:.12em;
text-transform:uppercase;color:var(--ink);font-weight:700}
.lbhead .s{color:var(--mut);font-size:13px}
.lbhead .big{font-family:Manrope,sans-serif;font-size:30px;font-weight:700;
line-height:1;text-align:right}
.lbhead .bigk{font-family:"Space Mono",monospace;font-size:10px;letter-spacing:.1em;
text-transform:uppercase;color:var(--mut);text-align:right}
.lbrow{display:grid;grid-template-columns:34px 1fr repeat(3,88px);gap:10px;
align-items:center;padding:11px 18px;border-bottom:1px solid var(--ln);background:#fff}
.lbrow:last-child{border-bottom:0}
.lbrow .rk{font-family:"Space Mono",monospace;color:var(--mut);font-size:14px}
.lbrow .who{font-weight:700}
.lbrow .m{text-align:center}
.lbrow .m b{display:block;font-family:Manrope;font-size:17px;line-height:1.1}
.lbrow .m span{font-family:"Space Mono",monospace;font-size:10px;
letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}

table{border-collapse:collapse;width:100%;font-size:14px}
.tw{overflow-x:auto}
th,td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--ln);white-space:nowrap}
thead th{font-family:"Space Mono",monospace;font-size:11px;letter-spacing:.07em;
text-transform:uppercase;color:var(--mut);font-weight:400}
td.num{text-align:right;font-family:"Space Mono",monospace;font-variant-numeric:tabular-nums}
tr.rec td{background:#fbfaff}

.pill{display:inline-block;font-family:"Space Mono",monospace;font-size:11px;
padding:2px 8px;border-radius:999px;border:1px solid currentColor}
.t-lean{color:#6d28d9;text-decoration:none;font-weight:700}
.t-leanbad{color:var(--bad);text-decoration:none;font-weight:700}
.t-leanpend{color:var(--mut);text-decoration:none}
.t-lean:hover{background:#6d28d9;color:#fff}
.t-verified{color:var(--ex)}.t-reproduced{color:var(--ac)}
.t-cited{color:var(--mut)}.t-failed{color:var(--bad)}
.gain{color:var(--ex);font-weight:700}
.none{color:var(--mut)}
.bar{position:relative;height:9px;background:var(--ln);border-radius:999px;min-width:120px}
.bar i{position:absolute;top:0;bottom:0;left:0;background:var(--ac);border-radius:999px}

.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.orb{border:1px solid var(--ln);border-radius:10px;padding:14px 16px}
.orb h3{font-family:Manrope;margin:0 0 2px;font-size:18px}
.orb .sub{font-family:"Space Mono",monospace;font-size:11px;color:var(--mut);
text-transform:uppercase;letter-spacing:.07em}
.cells{margin-top:10px;font-family:"Space Mono",monospace;font-size:13px}
.cells div{display:flex;justify-content:space-between;gap:10px;padding:3px 0;
border-bottom:1px dotted var(--ln)}
.cells div:last-child{border-bottom:0}

details{border:1px solid var(--ln);border-radius:10px;padding:10px 14px;background:var(--soft)}
details summary{cursor:pointer;font-family:"Space Mono",monospace;font-size:13px;font-weight:700}
pre{background:#fff;border:1px solid var(--ln);border-radius:8px;padding:12px 14px;
overflow-x:auto;font-size:12.5px;line-height:1.5}
.refs{list-style:none;padding:0;counter-reset:r}
.refs li{counter-increment:r;padding:12px 0 12px 42px;border-bottom:1px solid var(--ln);
position:relative;max-width:80ch}
.refs li:before{content:"[" counter(r) "]";position:absolute;left:0;top:12px;
font-family:"Space Mono",monospace;font-size:12px;color:var(--mut)}
.refs .ti{font-weight:700}
.refs .au{color:var(--ink)}
.refs .vn{color:var(--mut)}
.refs .nt{color:var(--mut);font-size:13.5px;margin-top:3px}
.chart .leader{fill:none;stroke-width:1.2;opacity:.55}
.h2note{font-weight:400;font-size:14px;color:var(--mut);letter-spacing:0}

.modal{border:none;border-radius:14px;padding:0;max-width:620px;width:calc(100% - 32px);
box-shadow:0 24px 70px rgba(0,0,0,.35)}
.modal::backdrop{background:rgba(15,23,42,.55)}
.modal h3{margin:0 0 4px;font-size:22px}
.modal>*:not(form){padding:0 26px}
.modal h3{padding-top:24px}
.modal p:last-of-type{padding-bottom:24px}
.modal p{color:var(--mut);max-width:none}
.modal form{display:flex;justify-content:flex-end;padding:10px 12px 0}
.modal .x{background:none;border:none;font-size:24px;line-height:1;color:var(--mut);
cursor:pointer;padding:0 6px}
.codewrap{position:relative;margin:14px 26px}
.codewrap pre{background:var(--dark);color:#e4e4e7;border:none;margin:0;
padding:16px 18px;border-radius:10px}
.codewrap code{color:#e4e4e7}
.copy{position:absolute;top:10px;right:10px;font-family:"Space Mono",monospace;
font-size:11px;background:var(--ac);color:#fff;border:none;border-radius:6px;
padding:4px 10px;cursor:pointer}
.copy:hover{background:#5b21b6}

.recent{display:flex;flex-direction:column;gap:6px;border:1px solid var(--ln);
border-radius:12px;padding:8px;background:var(--soft)}
.rrow{display:flex;align-items:center;gap:12px;flex-wrap:nowrap;white-space:nowrap;
padding:9px 14px;border-radius:8px;background:#fff}
.rrow:hover{background:#fbfaff}
.rb{font-size:13px;font-weight:700;text-decoration:none;color:var(--ink);flex:0 0 auto}
.rb:hover{color:var(--ac)}
.star{color:var(--ac);flex:0 0 auto}
.rt{color:var(--mut);font-size:13px;flex:0 0 auto}
.rw{font-size:13px;flex:0 0 auto}
.rd{font-family:"Space Mono",monospace;font-size:12px;color:var(--mut);
margin-left:auto;flex:0 0 auto}
.tp{font-family:"Cambria Math","STIX Two Math","DejaVu Math TeX Gyre",
"Latin Modern Math",serif;font-size:.86em;vertical-align:.02em}
.tgt{font-family:"Space Mono",monospace;font-size:11.5px;color:var(--mut);
white-space:nowrap}
.tgtg{color:var(--ac);margin-left:8px;font-weight:700}
a.lgd{color:var(--mut);text-decoration:none;display:inline-flex;align-items:center}
a.lgd:hover{color:var(--ac)}
a.m{text-decoration:none;color:inherit;border-radius:8px;padding:4px 0}
a.m:hover{background:var(--soft);color:var(--ac)}
.lbrow .who a{text-decoration:none;color:var(--ink)}
.lbrow .who a:hover{color:var(--ac)}
.orb h3 a,.exp a{text-decoration:none;color:inherit}
.orb h3 a:hover{color:var(--ac)}
.statebox{border:1px solid var(--ln);border-left:4px solid var(--ac);
border-radius:10px;padding:16px 20px;background:var(--soft);margin:4px 0 14px}
.stateeq{font-size:19px;line-height:1.8}
.stateeq sup{font-size:.7em}
.ket{font-family:Manrope,system-ui,sans-serif;font-weight:600;white-space:nowrap}
.ket sup{font-size:.72em}
header.hero p br{line-height:2}
@media(max-width:820px){.rrow .rt,.rrow .rw{display:none}}

.scroll{max-height:520px;overflow-y:auto;border:1px solid var(--ln);border-radius:10px}
.scroll table{font-size:13.5px}
.scroll thead th{position:sticky;top:0;background:#fff;z-index:1;
box-shadow:inset 0 -1px 0 var(--ln)}
a.gh{text-decoration:none;font-weight:600}
td.mono a{text-decoration:none;border-bottom:1px dotted var(--ln);padding-bottom:1px}
td.mono a:hover{border-bottom-color:var(--ac)}
a.gh:hover{text-decoration:underline}

footer.foot{margin:64px 0 0;border-top:1px solid var(--ln);background:var(--soft)}
.footmain{max-width:1060px;margin:0 auto;padding:26px 20px 30px}
.footbrand{max-width:420px}
.footbrand .fb{display:flex;align-items:center;gap:12px;margin-bottom:8px}
.footbrand .fb span{font-size:18px;font-weight:700;color:var(--ink)}
.footbrand p{margin:0;color:var(--mut);font-size:14px}
.footlinks{display:flex;flex-wrap:wrap;gap:8px 22px;align-items:center;margin-top:16px}
.footlinks a{display:inline-flex;align-items:center;gap:7px;color:var(--ink);
text-decoration:none;font-size:14px}
.footlinks a:hover{color:var(--ac);text-decoration:underline}
.footlinks svg{width:16px;height:16px}
@media(max-width:700px){.lbrow{grid-template-columns:28px 1fr 70px;}
.lbrow .m:nth-child(n+4){display:none}}
"""

HEROSVG = """<svg class=heroflow viewBox="0 0 1200 360" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
<path d="M-60,90 C250,20 450,260 760,150 S1160,40 1300,120"/>
<path d="M-60,205 C200,300 500,80 790,225 S1110,265 1300,180"/>
<path d="M-60,300 C300,180 520,360 820,255 S1130,120 1300,300"/>
<path d="M-60,40 C220,140 480,-20 760,90 S1090,180 1300,50"/>
<path d="M-60,260 C280,360 540,200 800,320 S1190,220 1300,260"/>
</svg>"""


def head(title, rel=""):
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{E(title)}</title>"
        "<link rel=preconnect href='https://fonts.googleapis.com'>"
        "<link rel=preconnect href='https://fonts.gstatic.com' crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700"
        "&family=Space+Mono:wght@400;700&display=swap' rel=stylesheet>"
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
        f"<a href='{rel}state-of-the-art.html'>State of the art</a>"
        f"<a href='{rel}references.html'>References</a>"
        f"<a href='{REPO}'>{GHICON}GitHub</a>"
        "</nav></div></header>"
    )



FOOTMARK = '<svg width=34 height=34 viewBox="0 0 64 64" aria-hidden="true"><rect x="1" y="1" width="62" height="62" rx="14" fill="#111111" stroke="rgba(255,255,255,0.16)" stroke-width="1.5"/><g stroke="#ffffff" stroke-width="3.4" stroke-linecap="round" opacity="0.9"><line x1="32" y1="16" x2="17" y2="40"/><line x1="32" y1="16" x2="47" y2="40"/><line x1="32" y1="16" x2="32" y2="46"/></g><g fill="#ffffff"><circle cx="17" cy="42" r="5"/><circle cx="32" cy="47" r="5"/><circle cx="47" cy="42" r="5"/></g><circle cx="32" cy="16" r="6" fill="#ffff00"/></svg>'


def footer(rel=""):
    links = [
        (REPO, "GitHub", True),
        (REPO + "/blob/main/CONTRIBUTING.md", "Contribute", False),
        (REPO + "/blob/main/schema/bound.schema.json", "Schema", False),
        (rel + "state-of-the-art.html", "State of the art", False),
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


def ket(orbit, m, sign=None, rank=None):
    """chi_R(|M>^{ot m}). The tensor glyph gets its own font stack and size:
    left in a monospace superscript it substitutes to an oversized circled x."""
    body = (f"<span class=ket>&chi;<sub>R</sub>(|{E(orbit)}&rang;"
            f"<sup>{TENS}{m}</sup>)</span>")
    if sign and rank is not None:
        body += f" {sign} <b>{rank}</b>"
    return body


def lean_ok(sub):
    """True/False if a build receipt exists for this bound's module, else None."""
    ln = sub.get("lean")
    if not ln:
        return None
    rec = os.path.join(CERTS, "lean-" + ln["module"].replace(".", "-") + ".json")
    if not os.path.isfile(rec):
        return None
    return bool(json.load(open(rec)).get("ok"))


def lean_badge(sub, rel=""):
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
    return (f"<a class='pill {cls}' href='{url}' title='{E(ln['module'])} &mdash; {tip}'>"
            f"Lean &middot; {E(ln['theorem'])}{mark}</a>")


def gh_links(prov):
    """Render submitters as linked GitHub handles where we know them."""
    hs = prov.get("github") or []
    if not hs:
        return E(prov.get("author", ""))
    return " ".join(f"<a class=gh href='https://github.com/{E(h)}'>@{E(h)}</a>" for h in hs)


def tier_pill(tier, ok=True, sub=None):
    """One badge per bound.

    A lean-tier bound shows only its Lean badge, which already says the tier and
    links to the proof; showing both a `lean` pill and a `Lean - theorem` pill
    was two tags for one fact. A cited bound links to what it is cited from.
    """
    if not ok:
        return "<span class='pill t-failed'>failed</span>"
    if tier == "lean" and sub is not None and sub.get("lean"):
        return lean_badge(sub)
    if tier == "cited" and sub is not None:
        u = arxiv_url(sub["provenance"])
        if u:
            return (f"<a class='pill t-cited' href='{u}' "
                    f"title='cited from {E(sub[chr(39)+chr(39)] if False else sub['provenance'].get('reference',''))}'>"
                    f"cited</a>")
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

def build():
    entries = load_bounds()
    cells = best_by_cell(entries)
    board = leaderboard(entries)
    people = contributors(entries)
    lean_n = sum(1 for e in entries if e["res"]["ok"] and e["res"]["tier"] == "lean")
    leanclaim_n = sum(1 for e in entries if e["sub"].get("lean"))
    ver_n = sum(1 for e in entries if e["res"]["ok"] and e["res"]["tier"] == "verified")
    rep_n = sum(1 for e in entries if e["res"]["ok"] and e["res"]["tier"] == "reproduced")
    moved = [r for r in board if r["best"] and r["best"]["res"]["gamma"] < r["baseline"] - 1e-12]
    tight = min((r for r in board if r["best"]),
                key=lambda r: r["best"]["res"]["gamma"], default=None)

    o = [head("Stabilizer Rank Challenge")]
    o.append(hero("Stabilizer Rank Challenge",
                  "Find smaller exact stabilizer decompositions of magic states.<br>"
                  f"<a href='state-of-the-art.html'>Read the state of the art.</a>"))
    o.append(PARTICIPATE)
    o.append("<div class=wrap>")

    # ---- the graph, first thing on the page
    o.append("<h2>Record progress</h2>")
    o.append("<p class=h2sub>Best known per-copy exponent &gamma; for each orbit. "
             "Lower is better; a line steps down when a submission improved that "
             "orbit's best bound.</p>")
    o.append("<div class=chartbox>" + progress_chart(entries) + "</div>")
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
                 f"<div class=who><a href='{pg}'>{E(a['who'])}</a> {a['links']}"
                 f"{' &#128081;' if i == 1 and a['records'] else ''}</div>"
                 f"<a class=m href='{pg}#bounds'><b>{a['n']}</b><span>bounds</span></a>"
                 f"<a class=m href='{pg}#records'><b>{a['records']}</b>"
                 f"<span>records</span></a>"
                 f"<div class=m><b>{bl}</b><span>best &gamma;</span></div></div>")
    o.append("</div>")

    # ---- exponent table
    o.append("<h2>Exponents</h2>")
    o.append("<p class=h2sub>Every &gamma; below is published. "
             f"{len(moved)} of {len(ORBIT_ORDER)} have been beaten here.</p><div class=tw><table>")
    o.append("<thead><tr><th>orbit</th><th></th><th class=num>published &gamma; &le;</th>"
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
                   else next_target(orbit, base))
        else:
            gtxt, held = "<span class=none>&mdash;</span>", "<span class=none>open</span>"
            bar = next_target(orbit, base)
        o.append(f"<tr{' class=rec' if best and best['res']['gamma'] < base - 1e-12 else ''}>"
                 f"<td><b><a href='orbits/{orbit}.html'>{ORBIT_LABEL[orbit]}</a></b></td>"
                 f"<td class=mono style='color:var(--mut)'>{SYSTEM[orbit]}</td>"
                 f"<td class=num>{base:.4f}</td><td class=num>{gtxt}</td>"
                 f"<td>{bar}</td><td>{held}</td></tr>")
    o.append("</tbody></table></div>")

    # ---- cell ledger
    o.append("<h2>Cell ledger</h2>")
    o.append("<p class=h2sub>Best bound on &chi;(&#8739;M&rang;<sup>&otimes;m</sup>) "
             "per orbit and copy count. Matching upper and lower bounds settle a cell.</p>")
    o.append("<div class=grid3>")
    for orbit in ORBIT_ORDER:
        base, base_txt = BASELINE[orbit]
        o.append(f"<div class=orb><h3><a href='orbits/{orbit}.html'>"
                 f"{ORBIT_LABEL[orbit]}</a></h3>"
                 f"<div class=sub>{SYSTEM[orbit]} &middot; published &gamma; &le; "
                 f"{base_txt} &asymp; {base:.4f}</div><div class=cells>")
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
                     f"{tier_pill(up['res']['tier'], True, up['sub']) if up else ''}</span></div>")
        o.append("</div></div>")
    o.append("</div>")

    # ---- recently added
    recent = sorted(entries,
                    key=lambda e: e["sub"]["provenance"].get("date", ""),
                    reverse=True)[:10]
    o.append("<h2>Recently added <span class=h2note>&middot; last 10 by submission "
             "date</span></h2><div class=recent>")
    for e in recent:
        s_, r = e["sub"], e["res"]
        sign = "&le;" if s_["direction"] == "upper" else "&ge;"
        star = "&#9733; " if e["res"]["tier"] in RECORD_TIERS else ""
        url, _ = source_link(e)
        o.append(f"<div class=rrow><a class=rb href='{url}'>"
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
        url, _ = source_link(e)
        o.append(f"<tr><td><a href='{url}'>{ket(s['orbit'], s['m'], sign, s['rank'])}"
                 f"</a></td><td>{ORBIT_LABEL[s['orbit']]}</td>"
                 f"<td class=num>{s['m']}</td><td class=num>{s['rank']}</td>"
                 f"<td class=num>{gt}</td>"
                 f"<td>{tier_pill(r['tier'], r['ok'], s_)} {lean_badge(s)}</td>"
                 f"<td style='white-space:normal'>{gh_links(s['provenance'])}"
                 f" &middot; <span class=mono style='font-size:12px'>"
                 f"{ref_link(s['provenance'])}</span></td></tr>")
    o.append("</tbody></table></div>")

    # ---- lean corpus
    leaned = [e for e in entries if e["sub"].get("lean")]
    if leaned:
        o.append("<h2>Lean corpus <span class=h2note>&middot; "
                 f"{len(leaned)} bounds with machine-checked proofs</span></h2>")
        o.append("<p class=h2sub>A Lean theorem is the strongest thing a bound can "
                 "carry: it does not ask you to trust this pipeline. These are the "
                 "decomposition identities proved in <span class=mono>lean_proofs/"
                 "</span>, and they are built in CI against mathlib.</p>")
        o.append("<div class=tw><table><thead><tr><th>bound</th><th>theorem</th>"
                 "<th>orbit</th></tr></thead><tbody>")
        for e in sorted(leaned, key=lambda e: (ORBIT_ORDER.index(e["sub"]["orbit"]),
                                               int(e["sub"]["m"]))):
            s_, ln = e["sub"], e["sub"]["lean"]
            url = (REPO + "/blob/main/lean_proofs/"
                   + ln["module"].replace(".", "/") + ".lean")
            o.append(f"<tr><td><a href='{url}'>"
                     f"{ket(s_['orbit'], s_['m'], '&le;', s_['rank'])}</a></td>"
                     f"<td class=mono><a href='{url}'>{E(ln['theorem'])}</a></td>"
                     f"<td><a href='orbits/{s_['orbit']}.html'>"
                     f"{ORBIT_LABEL[s_['orbit']]}</a></td></tr>")
        o.append("</tbody></table></div>")

    o.append("</div>" + footer() + "</body></html>")

    os.makedirs(os.path.join(DOCS, "bounds"), exist_ok=True)
    with open(os.path.join(DOCS, "index.html"), "w") as f:
        f.write("".join(o))
    with open(os.path.join(DOCS, "style.css"), "w") as f:
        f.write(CSS)
    references_page(parse_bib(os.path.join(DOCS, "refs.bib")))
    for e in entries:
        detail_page(e)
    for orbit in ORBIT_ORDER:
        orbit_page(orbit, entries, cells)
    for a in people:
        person_page(a, entries, cells)
    print(f"docs/ written: {len(entries)} bounds, {lean_n} lean-certified "
          f"({leanclaim_n} claim a proof), "
          f"{ver_n} verified, {rep_n} reproduced, {len(moved)} exponents beaten, "
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
                  f"Published &gamma; &le; {base_txt} &asymp; {base:.4f}.", rel="../"))
    o.append(PARTICIPATE)
    o.append("<div class=wrap><p><a href='../index.html'>&larr; back to the board</a></p>")
    o.append(f"<h2>The state</h2><div class=statebox><div class=stateeq>"
             f"|{E(orbit)}&rang; = {defn}</div></div>")
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
        url, _ = source_link(e, rel="../")
        o.append(f"<tr><td><a href='{url}'>{ket(orbit, s_['m'], sign, s_['rank'])}</a></td>"
                 f"<td class=num>{s_['m']}</td><td class=num>{s_['rank']}</td>"
                 f"<td class=num>{gt}</td>"
                 f"<td>{tier_pill(r['tier'], r['ok'], s_)}</td>"
                 f"<td style='white-space:normal'>{gh_links(s_['provenance'])} &middot; "
                 f"<span class=mono style='font-size:12px'>{ref_link(s_['provenance'])}"
                 f"</span></td></tr>")
    o.append("</tbody></table></div>")
    o.append(f"<h2>What would move it</h2><p>Beating the published exponent needs "
             f"{next_target(orbit, base)}.</p>")
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
            url, _ = source_link(e, rel="../")
            o.append(f"<tr><td><a href='{url}'>"
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
        o.append(f"<p>Implied per-copy exponent &gamma; &le; <b>{r['gamma']:.4f}</b>, "
                 f"against a published {base:.4f} &mdash; "
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
    if s.get("notes"):
        o.append(f"<h2>Notes</h2><p>{E(s['notes'])}</p>")
    o.append("<h2>Attribution</h2><p>" + E(s["provenance"].get("author", ""))
             + " &middot; <span class=mono>" + E(s["provenance"].get("reference", ""))
             + "</span>" + (" &middot; " + E(s["provenance"]["method"])
                            if s["provenance"].get("method") else "") + "</p>")
    o.append("<h2>Submission</h2><details><summary>JSON</summary><pre>"
             + E(json.dumps(s, indent=2)) + "</pre></details>")
    o.append("</div>" + footer(rel="../") + "</body></html>")
    with open(os.path.join(DOCS, "bounds", f"{e['slug']}.html"), "w") as f:
        f.write("".join(o))


if __name__ == "__main__":
    build()
