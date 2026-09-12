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
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify_challenge"))
from stabrank_verify import ORBIT_LABEL, ORBIT_P, implied_gamma, verify  # noqa: E402

DOCS = os.path.join(ROOT, "docs")
BOUNDS = os.path.join(ROOT, "bounds")
CERTS = os.path.join(ROOT, "certs")

# Published per-copy exponents these bounds are measured against.
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
TIER_RANK = {"verified": 3, "reproduced": 2, "cited": 1, None: 0}
RECORD_TIERS = ("verified", "reproduced")

E = html.escape


def slug(sub):
    return f"{sub['orbit']}-m{sub['m']}-{sub['direction']}-{sub['rank']}"


def content_hash(sub):
    return hashlib.sha256(
        json.dumps(sub, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def load_bounds():
    """Verify every submission, using a cached certificate when it still applies."""
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
            if r.ok and r.tier == "verified":
                os.makedirs(CERTS, exist_ok=True)
                with open(cpath, "w") as f:
                    json.dump(res, f, indent=2)
                    f.write("\n")
        if res.get("gamma") is None and sub["direction"] == "upper":
            res["gamma"] = implied_gamma(ORBIT_P[sub["orbit"]], sub["rank"], sub["m"])
        out.append({"sub": sub, "res": res, "slug": s,
                    "file": os.path.basename(path)})
    return out


def best_by_cell(entries):
    """Best bound per (orbit, m, direction). Ties break toward the higher tier."""
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
    """Best record-eligible exponent per orbit, and what the baseline is."""
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
            if g is None:
                continue
            if best is None or g < best["res"]["gamma"]:
                best = e
        rows.append({"orbit": orbit, "baseline": base, "baseline_txt": base_txt,
                     "best": best})
    return rows


# ------------------------------------------------------------------ HTML ----

CSS = """
:root{--ink:#0f172a;--mut:#64748b;--ln:#e2e8f0;--ac:#36006c;--ex:#059669;
--exb:#ffff00;--dark:#111111;--bg:#fff;--soft:#f8fafc;--warn:#b45309;--bad:#be185d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:Manrope,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.6}
.wrap{max-width:1040px;margin:0 auto;padding:0 20px}
a{color:var(--ac)}
code,.mono{font-family:"Space Mono",ui-monospace,Menlo,monospace}

header.hero{position:relative;overflow:hidden;background:var(--dark);color:#fff;
padding:54px 0 46px;margin-bottom:34px}
.heroflow{position:absolute;inset:0;width:100%;height:100%;opacity:.32}
.heroflow path{fill:none;stroke:var(--exb);stroke-width:1.1}
header.hero .wrap{position:relative}
h1{font-family:"Space Grotesk",sans-serif;font-weight:700;font-size:40px;
letter-spacing:-.02em;margin:0 0 10px;text-wrap:balance}
.tag{color:#cbd5e1;max-width:68ch;margin:0;font-size:16px}
.hero .meta{margin-top:18px;display:flex;flex-wrap:wrap;gap:10px 22px;
font-family:"Space Mono",monospace;font-size:12px;color:#94a3b8}

h2{font-family:"Space Grotesk",sans-serif;font-size:13px;font-weight:700;
letter-spacing:.1em;text-transform:uppercase;color:var(--mut);
margin:40px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--ln)}
section p{max-width:72ch}

.lead{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:0 0 6px}
.card{border:1px solid var(--ln);border-radius:10px;padding:14px 16px;background:var(--soft)}
.card .k{font-family:"Space Mono",monospace;font-size:11px;letter-spacing:.08em;
text-transform:uppercase;color:var(--mut)}
.card .v{font-family:"Space Grotesk",sans-serif;font-size:27px;font-weight:700;
margin-top:2px;font-variant-numeric:tabular-nums}
.card .s{font-size:13px;color:var(--mut)}

table{border-collapse:collapse;width:100%;font-size:14px}
.tw{overflow-x:auto}
th,td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--ln);white-space:nowrap}
thead th{font-family:"Space Mono",monospace;font-size:11px;letter-spacing:.07em;
text-transform:uppercase;color:var(--mut);font-weight:400}
td.num{text-align:right;font-family:"Space Mono",monospace;font-variant-numeric:tabular-nums}
tr.rec td{background:#fbfaff}

.pill{display:inline-block;font-family:"Space Mono",monospace;font-size:11px;
padding:2px 8px;border-radius:999px;border:1px solid currentColor}
.t-verified{color:var(--ex)}
.t-reproduced{color:var(--ac)}
.t-cited{color:var(--mut)}
.t-failed{color:var(--bad)}

.bar{position:relative;height:9px;background:var(--ln);border-radius:999px;min-width:130px}
.bar i{position:absolute;top:0;bottom:0;left:0;background:var(--ac);border-radius:999px;display:block}
.gain{color:var(--ex);font-weight:700}
.none{color:var(--mut)}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.orb{border:1px solid var(--ln);border-radius:10px;padding:14px 16px}
.orb h3{font-family:"Space Grotesk",sans-serif;margin:0 0 2px;font-size:18px}
.orb .sub{font-family:"Space Mono",monospace;font-size:11px;color:var(--mut);
text-transform:uppercase;letter-spacing:.07em}
.cells{margin-top:10px;font-family:"Space Mono",monospace;font-size:13px}
.cells div{display:flex;justify-content:space-between;gap:10px;padding:3px 0;
border-bottom:1px dotted var(--ln)}
.cells div:last-child{border-bottom:0}

pre{background:var(--soft);border:1px solid var(--ln);border-radius:8px;
padding:12px 14px;overflow-x:auto;font-size:13px}
footer{margin:56px 0 40px;padding-top:18px;border-top:1px solid var(--ln);
color:var(--mut);font-size:13px}
@media(max-width:640px){h1{font-size:29px}}
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
        "&family=Space+Grotesk:wght@500;700&family=Space+Mono:wght@400;700&display=swap'"
        " rel=stylesheet>"
        f"<link rel=stylesheet href='{rel}style.css'></head><body>"
    )


def tier_pill(tier, ok=True):
    if not ok:
        return "<span class='pill t-failed'>failed</span>"
    return f"<span class='pill t-{tier}'>{tier}</span>"


def build():
    entries = load_bounds()
    cells = best_by_cell(entries)
    board = leaderboard(entries)
    ok_n = sum(1 for e in entries if e["res"]["ok"])
    ver_n = sum(1 for e in entries if e["res"]["ok"] and e["res"]["tier"] == "verified")
    rep_n = sum(1 for e in entries if e["res"]["ok"] and e["res"]["tier"] == "reproduced")
    moved = [r for r in board if r["best"] and r["best"]["res"]["gamma"] < r["baseline"] - 1e-12]

    o = [head("Stabilizer Rank Challenge")]
    o.append("<header class=hero>" + HEROSVG + "<div class=wrap>")
    o.append("<h1>Stabilizer Rank Challenge</h1>")
    o.append("<p class=tag>How many stabilizer states does it take to write a magic "
             "state exactly? Submit a decomposition, the pipeline rebuilds it in exact "
             "arithmetic, and the leaderboard moves only if it checks out.</p>")
    o.append(f"<div class=meta><span>{len(entries)} submissions</span>"
             f"<span>{ver_n} verified</span><span>{rep_n} reproduced</span>"
             f"<span>{len(ORBIT_ORDER)} orbits</span>"
             f"<span>{len(moved)} published exponents beaten</span></div>")
    o.append("</div></header><div class=wrap>")

    # ---- headline
    o.append("<div class=lead>")
    o.append(f"<div class=card><div class=k>Exponents beaten</div>"
             f"<div class=v>{len(moved)}</div>"
             f"<div class=s>of {len(ORBIT_ORDER)} orbits</div></div>")
    o.append(f"<div class=card><div class=k>Machine-checked</div>"
             f"<div class=v>{ver_n + rep_n}</div>"
             f"<div class=s>of {len(entries)} submissions</div></div>")
    tightest = min((r for r in board if r["best"]),
                   key=lambda r: r["best"]["res"]["gamma"], default=None)
    if tightest:
        o.append(f"<div class=card><div class=k>Tightest exponent</div>"
                 f"<div class=v>{tightest['best']['res']['gamma']:.4f}</div>"
                 f"<div class=s>{ORBIT_LABEL[tightest['orbit']]}, "
                 f"{SYSTEM[tightest['orbit']]}</div></div>")
    o.append("</div>")

    # ---- leaderboard
    o.append("<section><h2>Leaderboard &mdash; per-copy exponent &gamma;</h2>")
    o.append("<p>A record needs a bound the pipeline could check. Cited literature "
             "values appear in the ledger below but never hold a record, so topping a "
             "cell means submitting something verifiable.</p><div class=tw><table>")
    o.append("<thead><tr><th>orbit</th><th></th><th class=num>published &gamma; &le;</th>"
             "<th class=num>best here</th><th>progress</th><th>record held by</th>"
             "</tr></thead><tbody>")
    for r in board:
        orbit, base = r["orbit"], r["baseline"]
        best = r["best"]
        lab = f"{ORBIT_LABEL[orbit]}"
        if best:
            g = best["res"]["gamma"]
            beat = g < base - 1e-12
            s = best["sub"]
            held = (f"<a href='bounds/{best['slug']}.html'>&chi;(&#8739;{E(orbit)}&rang;"
                    f"<sup>&otimes;{s['m']}</sup>) &le; {s['rank']}</a> "
                    f"{tier_pill(best['res']['tier'])}")
            gtxt = (f"<span class=gain>{g:.4f}</span>" if beat else f"{g:.4f}")
            frac = max(0.0, min(1.0, (base - g) / base)) if base else 0
            bar = f"<div class=bar><i style='width:{frac*100:.1f}%'></i></div>"
        else:
            gtxt, held, bar = "<span class=none>&mdash;</span>", \
                "<span class=none>open</span>", "<div class=bar></div>"
        o.append(f"<tr{' class=rec' if best and best['res']['gamma'] < base - 1e-12 else ''}>"
                 f"<td><b>{lab}</b></td><td class=mono style='color:var(--mut)'>{SYSTEM[orbit]}</td>"
                 f"<td class=num>{base:.4f}</td><td class=num>{gtxt}</td>"
                 f"<td>{bar}</td><td>{held}</td></tr>")
    o.append("</tbody></table></div></section>")

    # ---- per-orbit cells
    o.append("<section><h2>Cell ledger</h2>")
    o.append("<p>Best bound on &chi;(&#8739;M&rang;<sup>&otimes;m</sup>) for each orbit "
             "and copy count. A cell with matching upper and lower bounds is settled.</p>")
    o.append("<div class=grid>")
    for orbit in ORBIT_ORDER:
        base, base_txt = BASELINE[orbit]
        o.append(f"<div class=orb><h3>{ORBIT_LABEL[orbit]}</h3>"
                 f"<div class=sub>{SYSTEM[orbit]} &middot; published &gamma; &le; {base_txt} "
                 f"&asymp; {base:.4f}</div><div class=cells>")
        ms = sorted({int(k[1]) for k in cells if k[0] == orbit})
        if not ms:
            o.append("<div><span class=none>no bounds yet</span></div>")
        for m in ms:
            up = cells.get((orbit, m, "upper"))
            lo = cells.get((orbit, m, "lower"))
            u = f"&le;{up['sub']['rank']}" if up else "&mdash;"
            l = f"&ge;{lo['sub']['rank']}" if lo else ""
            settled = up and lo and up["sub"]["rank"] == lo["sub"]["rank"]
            val = (f"<b>= {up['sub']['rank']}</b>" if settled
                   else f"{l + ', ' if l else ''}{u}")
            tier = tier_pill(up["res"]["tier"]) if up else ""
            o.append(f"<div><span>m = {m}</span><span>{val} {tier}</span></div>")
        o.append("</div></div>")
    o.append("</div></section>")

    # ---- all submissions
    o.append("<section><h2>All submissions</h2><div class=tw><table>")
    o.append("<thead><tr><th>bound</th><th>orbit</th><th class=num>m</th>"
             "<th class=num>rank</th><th class=num>&gamma;</th><th>tier</th>"
             "<th>attribution</th></tr></thead><tbody>")
    for e in sorted(entries, key=lambda e: (ORBIT_ORDER.index(e["sub"]["orbit"]),
                                            int(e["sub"]["m"]), e["sub"]["direction"])):
        s, r = e["sub"], e["res"]
        sign = "&le;" if s["direction"] == "upper" else "&ge;"
        g = r.get("gamma")
        gt = f"{g:.4f}" if (g is not None and s["direction"] == "upper") else "&mdash;"
        o.append(f"<tr><td><a href='bounds/{e['slug']}.html'>&chi;<sub>R</sub>"
                 f"(&#8739;{E(s['orbit'])}&rang;<sup>&otimes;{s['m']}</sup>) {sign} "
                 f"{s['rank']}</a></td>"
                 f"<td>{ORBIT_LABEL[s['orbit']]}</td><td class=num>{s['m']}</td>"
                 f"<td class=num>{s['rank']}</td><td class=num>{gt}</td>"
                 f"<td>{tier_pill(r['tier'], r['ok'])}</td>"
                 f"<td style='white-space:normal'>{E(s['provenance'].get('author',''))}"
                 f" &middot; <span class=mono style='font-size:12px'>"
                 f"{E(s['provenance'].get('reference',''))}</span></td></tr>")
    o.append("</tbody></table></div></section>")

    # ---- submit
    o.append("<section><h2>Submit a bound</h2>")
    o.append("<p>Add one JSON file to <code>bounds/</code> and open a pull request. "
             "An upper bound carries its decomposition; every term is given by its "
             "stabilizer parametrisation, so a term that is not a stabilizer state "
             "cannot be written down in the first place. The pipeline rebuilds the "
             "identity in exact arithmetic and a floating-point near-miss earns "
             "nothing.</p>")
    o.append("<pre>" + E(json.dumps({
        "schema_version": "0.1", "orbit": "S", "m": 2,
        "direction": "upper", "rank": 2,
        "witness": {
            "terms": [{"k": 2, "x0": [0, 0], "W": [[1, 0], [0, 1]],
                       "Q": [[1, 1], [0, 1]], "l": [0, 0]}, "..."],
            "coeffs": ["3/4 + sqrt(3)*I/4", "..."]},
        "provenance": {"author": "you", "reference": "arXiv:...", "method": "..."},
        "notes": "what is new about it",
    }, indent=2)) + "</pre>")
    o.append("<p>Do not hand-compute the coefficients. Supply the terms and run "
             "<code>make fit BOUND=bounds/your.json</code>, which solves for them "
             "exactly or tells you no exact combination of those terms works &mdash; "
             "before you spend a review.</p>")
    o.append("<p>Lower bounds cannot be checked from a static witness, so they carry "
             "a certificate script that must run and assert. Anything else is recorded "
             "as <span class='pill t-cited'>cited</span> and cannot take a record.</p>")
    o.append("</section>")

    o.append("<section><h2>Background</h2>")
    o.append("<p>The <a href='state-of-the-art.html'>state-of-the-art notes</a> carry "
             "the material this leaderboard does not: the optimality conjectures and "
             "their exact plateau residuals, the algebraic floors below the m=4 cells, "
             "the T3 orbit's certificate status, and the reference list. A bound here "
             "is a claim the pipeline can check; the notes are where the reasoning "
             "lives.</p></section>")
    o.append("<footer>Bounds are re-verified on every build; nothing here is taken on "
             "the submitter's word. Source and submission guide in the "
             "<a href='https://github.com/unitaryfoundation/stabrank'>stabrank "
             "repository</a>. Run by the "
             "<a href='https://unitary.foundation'>Unitary Foundation</a>.</footer>")
    o.append("</div></body></html>")

    os.makedirs(DOCS, exist_ok=True)
    os.makedirs(os.path.join(DOCS, "bounds"), exist_ok=True)
    with open(os.path.join(DOCS, "index.html"), "w") as f:
        f.write("".join(o))
    with open(os.path.join(DOCS, "style.css"), "w") as f:
        f.write(CSS)

    for e in entries:
        detail_page(e)

    print(f"docs/index.html written: {len(entries)} bounds, "
          f"{ver_n} verified, {rep_n} reproduced, {len(moved)} exponents beaten")
    return entries


def detail_page(e):
    s, r = e["sub"], e["res"]
    sign = "&le;" if s["direction"] == "upper" else "&ge;"
    o = [head(f"chi({s['orbit']}^{s['m']}) {s['direction']} {s['rank']}", rel="../")]
    o.append("<header class=hero>" + HEROSVG + "<div class=wrap>")
    o.append(f"<h1>&chi;<sub>R</sub>(&#8739;{E(s['orbit'])}&rang;"
             f"<sup>&otimes;{s['m']}</sup>) {sign} {s['rank']}</h1>")
    o.append(f"<p class=tag>{ORBIT_LABEL[s['orbit']]} orbit, {SYSTEM[s['orbit']]}. "
             f"{E(s.get('notes','') or '')}</p>")
    o.append("</div></header><div class=wrap>")
    o.append(f"<p><a href='../index.html'>&larr; leaderboard</a></p>")
    o.append("<section><h2>Verification</h2>")
    o.append(f"<p>{tier_pill(r['tier'], r['ok'])} &nbsp; {E(r['detail'])}</p>")
    if r.get("gamma") is not None and s["direction"] == "upper":
        base = BASELINE[s["orbit"]][0]
        beat = r["gamma"] < base - 1e-12
        o.append(f"<p>Implied per-copy exponent &gamma; &le; "
                 f"<b>{r['gamma']:.4f}</b>, against a published {base:.4f} &mdash; "
                 + ("<span class=gain>an improvement</span>." if beat
                    else "no improvement.") + "</p>")
    o.append("</section>")
    o.append("<section><h2>Attribution</h2><p>"
             + E(s["provenance"].get("author", "")) + " &middot; "
             + f"<span class=mono>{E(s['provenance'].get('reference',''))}</span>"
             + (" &middot; " + E(s["provenance"]["method"])
                if s["provenance"].get("method") else "") + "</p></section>")
    o.append("<section><h2>Submission</h2><pre>" + E(json.dumps(s, indent=2))
             + "</pre></section>")
    o.append("</div></body></html>")
    with open(os.path.join(DOCS, "bounds", f"{e['slug']}.html"), "w") as f:
        f.write("".join(o))


if __name__ == "__main__":
    build()
