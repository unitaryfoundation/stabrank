"""The progress report: docs/report/index.html and docs/evidence_index.json.

Written by build.py after the ledger. Four sections, each backed by a pure
function over data the repository already keeps, so a reviewer can recompute
any number on the page from the committed files:

    exponent_comparison   docs/ledger.json against the published exponents
    cell_intervals        lower <= chi <= upper per cell, and when each side moved
    cost_table/cost_curve autoresearch/runs.jsonl joined with the compute blocks
    evidence_index        bounds/ joined with the git history, certs/ and lean_proofs/

The pure functions take rows and log lines rather than file paths, so the tests
drive them with fixtures; the git and filesystem access is confined to
`git_history` and `write_report`.
"""

from __future__ import annotations

import collections
import datetime
import glob
import json
import math
import os
import re
import statistics
import subprocess

import build as site

E = site.E
ROOT = site.ROOT
REPO = site.REPO

# The bibliography key behind each published exponent (docs/refs.bib).
PUBLISHED_REF = {
    "S": "labib2026stabilizer", "N": "labib2026stabilizer",
    "H3": "labib2026stabilizer", "T3": "labib2026stabilizer",
    "qubit_H": "qassim2021improved", "qubit_T": "qassim2021improved",
}

# A bound written by autoresearch/run.py carries, as its compute block, the sum
# of every run logged for its cell, so those runs are not added a second time.
RUNPY_REFERENCE = "stabrank autoresearch/run.py"

# The date the milestone measures progress from.
SINCE = "2026-09-15"

# The board opened in September 2026. Literature values dated before this are
# seeded rather than found, and the cost chart folds them into its starting level.
CHART_START = "2026-09-01"

EPS = 1e-9

PR_RE = re.compile(r"Merge pull request #(\d+)")


# ------------------------------------------------------------- exponents ---

def _tier_rank(tier):
    return site.TIER_RANK.get(tier, 0)


def exponent_comparison(rows, baseline=None, orbits=None, record_tiers=None):
    """One row per orbit: the published exponent, the best on the board, and
    whether the board matches, beats or trails it.

    `rows` are ledger rows. Only record tiers compete, as on the board itself.
    Ties on gamma go to the higher tier, then to the smaller m.
    """
    baseline = baseline or site.BASELINE
    orbits = orbits or site.ORBIT_ORDER
    record_tiers = record_tiers or site.RECORD_TIERS
    out = []
    for orbit in orbits:
        base, base_txt = baseline[orbit]
        cands = [r for r in rows
                 if r["orbit"] == orbit and r["direction"] == "upper" and r["ok"]
                 and r["tier"] in record_tiers and r.get("gamma") is not None]
        row = {"orbit": orbit, "published": base, "published_txt": base_txt,
               "reference": PUBLISHED_REF.get(orbit), "best": None, "verdict": "open",
               "gap": None}
        if cands:
            best = min(cands, key=lambda r: (r["gamma"], -_tier_rank(r["tier"]), r["m"]))
            g = best["gamma"]
            if g < base - EPS:
                verdict = "beats"
            elif g <= base + EPS:
                verdict = "matches"
            else:
                verdict = "trails"
            row.update({"best": {"gamma": g, "tier": best["tier"], "m": best["m"],
                                 "rank": best["rank"], "slug": best["slug"],
                                 "date": best.get("date")},
                        "verdict": verdict, "gap": g - base})
        out.append(row)
    return out


# ----------------------------------------------------------------- cells ---

def cell_intervals(rows, board_dates=None, orbits=None, since=SINCE):
    """Per (orbit, m): the best lower and upper bound, with the date each side
    last moved.

    `board_dates` maps a slug to the date it reached the main branch, when the
    caller has the git history; the provenance date is always reported too,
    since a literature value is dated by its arXiv v1 and can predate the board
    by a decade. A side "moved since" the milestone date when the bound holding
    it is dated on or after that date by either clock.
    """
    orbits = orbits or site.ORBIT_ORDER
    board_dates = board_dates or {}
    best = {}
    for r in rows:
        if not r["ok"]:
            continue
        key = (r["orbit"], int(r["m"]), r["direction"])
        cur = best.get(key)
        if r["direction"] == "upper":
            better = cur is None or r["rank"] < cur["rank"]
        else:
            better = cur is None or r["rank"] > cur["rank"]
        same = cur is not None and r["rank"] == cur["rank"]
        if better or (same and _tier_rank(r["tier"]) > _tier_rank(cur["tier"])):
            best[key] = r

    def side(r):
        if r is None:
            return None
        board = board_dates.get(r["slug"])
        moved = max(d for d in (r.get("date"), board) if d) if (r.get("date") or board) else None
        return {"rank": r["rank"], "tier": r["tier"], "slug": r["slug"],
                "date": r.get("date"), "board_date": board,
                "moved_since": bool(moved and moved >= since)}

    out = []
    for orbit in orbits:
        ms = sorted({k[1] for k in best if k[0] == orbit})
        for m in ms:
            lo = side(best.get((orbit, m, "lower")))
            up = side(best.get((orbit, m, "upper")))
            out.append({"orbit": orbit, "m": m, "lower": lo, "upper": up,
                        "settled": bool(lo and up and lo["rank"] == up["rank"]),
                        "moved_since": bool((lo and lo["moved_since"])
                                            or (up and up["moved_since"]))})
    return out


# ------------------------------------------------------------------ cost ---

def load_runs(path):
    """The autoresearch run log, one dict per line; missing file is no runs."""
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def _declared(rows, record_tiers):
    """Ledger rows whose compute block counts: a record-tier bound with a
    compute block, not written by run.py (whose cost is already in the log)."""
    return [r for r in rows
            if r["ok"] and r["tier"] in record_tiers and r.get("compute")
            and r.get("reference") != RUNPY_REFERENCE]


def cost_table(rows, runs, record_tiers=None):
    """Per cell (orbit, m, rank): runs, CPU-hours, solutions and CPU-hours per
    solution, from the run log and the compute blocks together.

    A solution is an exact refit in the log or a record-tier bound with a
    compute block; a bound with no compute block reports no search and so sits
    outside this table. Cells with no cost data are omitted.
    """
    record_tiers = record_tiers or site.RECORD_TIERS
    cells = collections.defaultdict(lambda: {"runs": 0, "cpu_hours": 0.0,
                                             "solutions": 0, "log_runs": 0,
                                             "bounds": []})
    for r in runs:
        c = cells[(r["orbit"], int(r["m"]), int(r["rank"]))]
        c["runs"] += 1
        c["log_runs"] += 1
        c["cpu_hours"] += float(r.get("cpu_s") or 0) / 3600
        c["solutions"] += int(bool(r.get("exact")))
    for r in _declared(rows, record_tiers):
        comp = r["compute"]
        c = cells[(r["orbit"], int(r["m"]), int(r["rank"]))]
        c["runs"] += int(comp.get("runs") or 0)
        c["cpu_hours"] += float(comp.get("cpu_hours") or 0)
        c["solutions"] += 1
        c["bounds"].append(r["slug"])
    tiers = {}
    for r in rows:
        if r["ok"]:
            tiers.setdefault((r["orbit"], int(r["m"]), int(r["rank"])), []).append(r["tier"])
    out = []
    for (orbit, m, rank), c in sorted(cells.items(),
                                      key=lambda kv: (site.ORBIT_ORDER.index(kv[0][0])
                                                      if kv[0][0] in site.ORBIT_ORDER else 99,
                                                      kv[0])):
        per = c["cpu_hours"] / c["solutions"] if c["solutions"] else None
        out.append({"orbit": orbit, "m": m, "rank": rank, "runs": c["runs"],
                    "log_runs": c["log_runs"], "cpu_hours": c["cpu_hours"],
                    "solutions": c["solutions"], "cpu_hours_per_solution": per,
                    "bounds": c["bounds"],
                    "board": sorted(set(tiers.get((orbit, m, rank), [])),
                                    key=_tier_rank, reverse=True)})
    return out


def cost_curve(rows, runs, record_tiers=None, start=CHART_START):
    """Cumulative CPU-hours and discoveries by date.

    A discovery is a record-tier bound, dated by its provenance. Cost comes from
    the compute blocks (dated by the bound) and from every logged run (dated by
    the run). Events before `start` are folded into the first point, so the
    curve begins at the level the board was seeded with.
    """
    record_tiers = record_tiers or site.RECORD_TIERS
    days = collections.defaultdict(lambda: {"cpu_hours": 0.0, "discoveries": 0,
                                            "declared": []})
    for r in rows:
        if r["ok"] and r["tier"] in record_tiers:
            d = (r.get("date") or "")[:10]
            days[d]["discoveries"] += 1
    for r in _declared(rows, record_tiers):
        d = (r.get("date") or "")[:10]
        h = float(r["compute"].get("cpu_hours") or 0)
        days[d]["cpu_hours"] += h
        days[d]["declared"].append(h)
    for r in runs:
        d = (r.get("when") or "")[:10]
        days[d]["cpu_hours"] += float(r.get("cpu_s") or 0) / 3600
    if not days:
        return []
    seeded = {"cpu_hours": 0.0, "discoveries": 0, "declared": []}
    later = {}
    for d, v in days.items():
        if start and d < start:
            seeded["cpu_hours"] += v["cpu_hours"]
            seeded["discoveries"] += v["discoveries"]
            seeded["declared"] += v["declared"]
        else:
            later[d] = v
    pts = []
    cum_h, cum_n = 0.0, 0
    if seeded["discoveries"] or seeded["cpu_hours"]:
        cum_h, cum_n = seeded["cpu_hours"], seeded["discoveries"]
        pts.append({"date": start if later else max(days), "seeded": True,
                    "cpu_hours": seeded["cpu_hours"], "discoveries": seeded["discoveries"],
                    "median_declared": (statistics.median(seeded["declared"])
                                        if seeded["declared"] else None),
                    "cum_cpu_hours": cum_h, "cum_discoveries": cum_n})
    for d in sorted(later):
        v = later[d]
        cum_h += v["cpu_hours"]
        cum_n += v["discoveries"]
        pts.append({"date": d, "seeded": False, "cpu_hours": v["cpu_hours"],
                    "discoveries": v["discoveries"],
                    "median_declared": (statistics.median(v["declared"])
                                        if v["declared"] else None),
                    "cum_cpu_hours": cum_h, "cum_discoveries": cum_n})
    return pts


def cost_chart(points):
    """Two panels on one time axis: cumulative discoveries above, cumulative
    CPU-hours below. Step lines, one series per panel, in the style of the
    board's progress chart. A dual axis was the alternative and reads worse."""
    pts = [p for p in points]
    if len(pts) < 1:
        return ""
    def day(s):
        return datetime.date.fromisoformat(s)
    d0 = day(pts[0]["date"])
    d1 = day(pts[-1]["date"]) + datetime.timedelta(days=1)
    span = max(1, (d1 - d0).days)

    W, H = 900, 420
    L, R, T, B, GAP = 62, 30, 18, 40, 34
    PH = (H - T - B - GAP) / 2
    def X(d):
        return L + (day(d) - d0).days / span * (W - L - R)

    o = [f"<svg class=chart viewBox='0 0 {W} {H}' role='img' "
         f"aria-label='cumulative verified discoveries and cumulative CPU-hours over time'>"]
    panels = [("cum_discoveries", "verified bounds", site.COLOR["S"], T, "{:.0f}"),
              ("cum_cpu_hours", "CPU-hours", site.COLOR["T3"], T + PH + GAP, "{:.2f}")]
    for key, label, color, top, fmt in panels:
        vals = [p[key] for p in pts]
        ticks = _nice_ticks(max(vals))
        hi = ticks[-1]
        tick_fmt = "{:.0f}" if all(t == int(t) for t in ticks) else "{:.1f}"
        def Y(v, top=top, hi=hi):
            return top + (hi - v) / hi * PH
        for t in ticks:
            o.append(f"<line class=grid x1='{L}' x2='{W-R}' y1='{Y(t):.1f}' y2='{Y(t):.1f}'/>")
            o.append(f"<text class=ax x='{L-9}' y='{Y(t)+4:.1f}' text-anchor='end'>"
                     f"{tick_fmt.format(t)}</text>")
        o.append(f"<text class=axl x='14' y='{top+PH/2:.0f}' "
                 f"transform='rotate(-90 14 {top+PH/2:.0f})' text-anchor='middle'>"
                 f"{label}</text>")
        d = []
        for i, p in enumerate(pts):
            x, y = X(p["date"]), Y(p[key])
            if i == 0:
                d.append(f"M{x:.1f},{y:.1f}")
            else:
                d.append(f"L{x:.1f},{Y(pts[i-1][key]):.1f}")
                d.append(f"L{x:.1f},{y:.1f}")
        d.append(f"L{W-R},{Y(pts[-1][key]):.1f}")
        o.append(f"<path class=ln d='{' '.join(d)}' stroke='{color}'/>")
        for p in pts:
            o.append(f"<circle cx='{X(p['date']):.1f}' cy='{Y(p[key]):.1f}' r='4' "
                     f"fill='#fff' stroke='{color}' stroke-width='2'/>")
        last = pts[-1]
        o.append(f"<text class=lbl x='{W-R-4}' y='{Y(last[key])-8:.1f}' text-anchor='end' "
                 f"fill='{color}'>{fmt.format(last[key])} {label}</text>")
    # date ticks: every point, thinned so labels do not collide
    minpx = 44
    lastx = -1e9
    for p in pts:
        x = X(p["date"])
        if x - lastx < minpx:
            continue
        lastx = x
        o.append(f"<text class=ax x='{x:.1f}' y='{H-14}' text-anchor='middle'>"
                 f"{p['date'][5:]}{' (seeded)' if p.get('seeded') else ''}</text>")
    o.append("</svg>")
    return "".join(o)


def _nice_ticks(top):
    """Gridline values from 0 to just above `top`, at a round step."""
    if top <= 0:
        return [0, 1]
    raw = top / 4
    mag = 10 ** math.floor(math.log10(raw))
    step = next(k * mag for k in (1, 2, 2.5, 5, 10) if k * mag >= raw)
    n = math.ceil(top / step)
    return [round(i * step, 6) for i in range(n + 1)]


# -------------------------------------------------------------- evidence ---

def parse_git_log(text):
    """Lines of `%H%x09%cs%x09%s` into dicts; tolerant of a missing date."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) == 3:
            sha, date, subject = parts
        elif len(parts) == 2:
            sha, subject, date = parts[0], parts[1], None
        else:
            sha, subject, date = parts[0], "", None
        out.append({"sha": sha.strip(), "date": date, "subject": subject.strip()})
    return out


def pr_number(subject):
    m = PR_RE.search(subject or "")
    return int(m.group(1)) if m else None


def merge_for(adding, merges):
    """The pull request that brought a bound to the main branch.

    `adding` are the commits that added the file, newest first, as git prints
    them; `merges` are the merge commits on the ancestry path from the oldest
    adding commit to HEAD, oldest first. The first one whose subject names a
    pull request is the merge. With none, the adding commit stands in.
    """
    if not adding:
        return {"pull_request": None, "merge_commit": None, "merge_date": None,
                "added_commit": None, "added_date": None, "added_subject": None}
    added = adding[-1]
    for m in merges:
        n = pr_number(m["subject"])
        if n is not None:
            return {"pull_request": n, "merge_commit": m["sha"], "merge_date": m["date"],
                    "added_commit": added["sha"], "added_date": added["date"],
                    "added_subject": added["subject"]}
    return {"pull_request": None, "merge_commit": None, "merge_date": None,
            "added_commit": added["sha"], "added_date": added["date"],
            "added_subject": added["subject"]}


def _git(args, root):
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_history(rel_path, root=ROOT, _cache={}):
    """(adding commits, merges on the path to HEAD) for one tracked file."""
    fmt = "--format=%H%x09%cs%x09%s"
    adding = parse_git_log(_git(["log", fmt, "--diff-filter=A", "--", rel_path], root))
    merges = []
    if adding:
        sha = adding[-1]["sha"]
        if sha not in _cache:
            _cache[sha] = parse_git_log(_git(
                ["log", "--merges", "--ancestry-path", "--reverse", fmt, f"{sha}..HEAD"], root))
        merges = _cache[sha]
    return adding, merges


def evidence_entry(sub, slug, res, history, cert, lean_receipt_ok):
    """One evidence-index row from a submission and what the repository holds
    for it. `history` is merge_for()'s result; `cert` is (path, matches) or None.
    """
    ln = sub.get("lean") or {}
    cert_block = sub.get("certificate") or {}
    return {
        "slug": slug, "file": f"bounds/{slug}.json",
        "orbit": sub["orbit"], "m": int(sub["m"]), "direction": sub["direction"],
        "rank": int(sub["rank"]),
        "tier": res["tier"] if res else None, "ok": res["ok"] if res else None,
        "date": sub["provenance"].get("date"),
        "author": sub["provenance"].get("author"),
        "reference": sub["provenance"].get("reference"),
        "method": sub["provenance"].get("method"),
        **history,
        "receipt": cert[0] if cert else None,
        "receipt_matches": cert[1] if cert else None,
        "certificate_script": cert_block.get("script"),
        "certificate_budget_s": cert_block.get("budget_s"),
        "certificate_attested": site.attested_summary(sub),
        "lean_module": ln.get("module"), "lean_theorem": ln.get("theorem"),
        "lean_receipt": (f"certs/lean-{ln['module'].replace('.', '-')}.json"
                         if ln.get("module") else None),
        "lean_receipt_ok": lean_receipt_ok,
        "compute": sub["provenance"].get("compute"),
    }


TIER_ORDER = ("lean", "verified", "reproduced", "attested", "cited")


def tier_counts(evidence):
    """How many evidence rows sit at each tier, plus `failed` and `unverified`."""
    out = collections.Counter()
    for e in evidence:
        if e["tier"] is None and e["ok"] is None:
            out["unverified"] += 1
        elif not e["ok"]:
            out["failed"] += 1
        else:
            out[e["tier"]] += 1
    return dict(out)


def evidence_index(entries, root=ROOT):
    """Every file in bounds/, whether or not this build verified it."""
    by_slug = {e["slug"]: e for e in entries}
    out = []
    for path in sorted(glob.glob(os.path.join(root, "bounds", "*.json"))):
        sub = json.load(open(path))
        slug = site.slug(sub)
        e = by_slug.get(slug)
        res = e["res"] if e else None
        cpath = os.path.join(root, "certs", f"{slug}.json")
        cert = None
        if os.path.exists(cpath):
            cached = json.load(open(cpath))
            cert = (f"certs/{slug}.json", cached.get("content_hash") == site.content_hash(sub))
        adding, merges = git_history(os.path.relpath(path, root), root)
        out.append(evidence_entry(sub, slug, res, merge_for(adding, merges), cert,
                                  site.lean_ok(sub)))
    return out


# ------------------------------------------------------------------ page ---

def _sha(sha):
    if not sha:
        return "<span class=none>&mdash;</span>"
    return f"<a class=mono href='{REPO}/commit/{sha}'>{sha[:10]}</a>"


def _pr(n):
    if n is None:
        return "<span class=none>&mdash;</span>"
    return f"<a href='{REPO}/pull/{n}'>#{n}</a>"


def _blob(rel, label=None):
    return f"<a class=mono href='{REPO}/blob/main/{rel}'>{E(label or rel)}</a>"


def _ref(refs, key):
    r = refs.get(key)
    if not r:
        return E(key or "")
    au = E(r.get("author", "")).replace(" and ", ", ")
    yr = E(r.get("year", ""))
    link = (f" <a href='https://arxiv.org/abs/{E(r['eprint'])}'>arXiv:{E(r['eprint'])}</a>"
            if r.get("eprint") else "")
    return f"{au} ({yr}){link}"


def _h(x, digits=2):
    return f"{x:.{digits}f}" if x is not None else "<span class=none>&mdash;</span>"


def report_page(comparison, cells, table, curve, evidence, refs, generated, head_sha):
    rel = "../"
    o = [site.head("Stabilizer Rank Challenge progress report", rel=rel)]
    o.append(site.hero(
        "Progress report",
        "Exponents against the literature, the interval on every cell, what each "
        "discovery cost, and the evidence behind every bound. Generated "
        f"{E(generated)} from <code>bounds/</code>, <code>autoresearch/runs.jsonl</code> "
        f"and the git history at <code>{E(head_sha[:10])}</code>.", rel=rel))
    o.append(site.PARTICIPATE)
    o.append(f"<div class=wrap><p><a href='{rel}index.html'>&larr; back to the board</a></p>")

    # ---- exponents
    beaten = sum(1 for r in comparison if r["verdict"] == "beats")
    matched = sum(1 for r in comparison if r["verdict"] == "matches")
    o.append("<h2>Exponent comparison</h2>")
    o.append(f"<p class=h2sub>Best record-tier exponent on the board against the published "
             f"one. {beaten} beaten, {matched} matched, "
             f"{len(comparison) - beaten - matched} trailing or open.</p>")
    o.append("<div class=tw><table><thead><tr><th>orbit</th><th></th>"
             "<th class=num>published &gamma;</th><th>reference</th>"
             "<th class=num>best here</th><th>cell</th><th>tier</th>"
             "<th class=num>gap</th><th>verdict</th></tr></thead><tbody>")
    for r in comparison:
        b = r["best"]
        cls = {"beats": "gain", "matches": "", "trails": "none", "open": "none"}[r["verdict"]]
        if b:
            cell = (f"<a href='{rel}bounds/{b['slug']}.html'>&chi; &le; {b['rank']} "
                    f"at m={b['m']}</a>")
            tier = f"<span class='pill t-{b['tier']}'>{b['tier']}</span>"
            best = f"{b['gamma']:.4f}"
            gap = "0.0000" if abs(r["gap"]) < EPS else f"{r['gap']:+.4f}"
        else:
            cell, tier, best, gap = "<span class=none>&mdash;</span>", "", \
                "<span class=none>&mdash;</span>", "<span class=none>&mdash;</span>"
        o.append(f"<tr{' class=rec' if r['verdict'] == 'beats' else ''}>"
                 f"<td><b><a href='{rel}orbits/{r['orbit']}.html'>"
                 f"{site.ORBIT_LABEL[r['orbit']]}</a></b></td>"
                 f"<td class=mono style='color:var(--mut)'>{site.SYSTEM[r['orbit']]}</td>"
                 f"<td class=num>{r['published']:.4f}</td>"
                 f"<td style='white-space:normal'>{_ref(refs, r['reference'])}</td>"
                 f"<td class=num>{best}</td><td>{cell}</td><td>{tier}</td>"
                 f"<td class=num>{gap}</td>"
                 f"<td><span class='{cls}'>{r['verdict']}</span></td></tr>")
    o.append("</tbody></table></div>")
    o.append("<p>The published exponents are the four qutrit values of "
             f"{_ref(refs, 'labib2026stabilizer')} and, for both qubit orbits, the "
             f"contracted-cat-state exponent of {_ref(refs, 'qassim2021improved')}. "
             "The gap is best minus published; negative would be an improvement. "
             "A match at the same rank replaces a cited value with a machine-checked "
             "one; it does not move the exponent.</p>")

    # ---- cells
    moved = [c for c in cells if c["moved_since"]]
    settled = sum(1 for c in cells if c["settled"])
    o.append("<h2>Cell intervals</h2>")
    o.append(f"<p class=h2sub>{len(cells)} cells, {settled} settled. "
             f"{len(moved)} moved on or after {SINCE} (highlighted). "
             "Dates: provenance date, then the date the bound reached main when "
             "the two differ.</p>")
    o.append("<div class=tw><table><thead><tr><th>orbit</th><th class=num>m</th>"
             "<th>interval</th><th>lower tier</th><th>lower since</th>"
             "<th>upper tier</th><th>upper since</th><th>status</th></tr></thead><tbody>")

    def when(s):
        if not s:
            return "<span class=none>&mdash;</span>"
        d = s.get("date") or ""
        b = s.get("board_date")
        txt = E(d)
        if b and b != d:
            txt += f" <span class=none>&middot; {E(b)}</span>"
        return f"<span class=mono style='font-size:12px'>{txt}</span>"

    def pill(s):
        if not s:
            return "<span class=none>&mdash;</span>"
        return (f"<a class='pill t-{s['tier']}' href='{rel}bounds/{s['slug']}.html' "
                f"style='text-decoration:none'>{s['tier']}</a>")

    for c in cells:
        lo, up = c["lower"], c["upper"]
        if c["settled"]:
            iv = f"<b>&chi; = {up['rank']}</b>"
        else:
            iv = ((f"{lo['rank']} &le; " if lo else "") + "&chi;"
                  + (f" &le; {up['rank']}" if up else ""))
        status = "settled" if c["settled"] else "open"
        o.append(f"<tr{' class=rec' if c['moved_since'] else ''}>"
                 f"<td><a href='{rel}orbits/{c['orbit']}.html'>"
                 f"{site.ORBIT_LABEL[c['orbit']]}</a></td><td class=num>{c['m']}</td>"
                 f"<td>{iv}</td><td>{pill(lo)}</td><td>{when(lo)}</td>"
                 f"<td>{pill(up)}</td><td>{when(up)}</td>"
                 f"<td class=mono style='font-size:12px'>{status}</td></tr>")
    o.append("</tbody></table></div>")

    # ---- cost
    o.append("<h2>Cost per discovery</h2>")
    o.append("<p class=h2sub>Cumulative record-tier bounds and cumulative CPU-hours, "
             "by date. Cost is the sum of the compute blocks on the bounds and of "
             "every run in the autoresearch log, failed runs included.</p>")
    o.append("<div class=chartbox>" + cost_chart(curve) + "</div>")
    if curve:
        o.append("<div class=legend>"
                 f"<span><i style='background:{site.COLOR['S']}'></i>verified bounds, cumulative</span>"
                 f"<span><i style='background:{site.COLOR['T3']}'></i>CPU-hours, cumulative</span>"
                 "</div>")
    o.append("<div class=tw><table><thead><tr><th>date</th><th class=num>bounds</th>"
             "<th class=num>CPU-h</th><th class=num>CPU-h per bound</th>"
             "<th class=num>median declared CPU-h</th>"
             "<th class=num>cumulative bounds</th><th class=num>cumulative CPU-h</th>"
             "</tr></thead><tbody>")
    for p in curve:
        per = p["cpu_hours"] / p["discoveries"] if p["discoveries"] else None
        label = E(p["date"]) + (" and earlier (seeded)" if p.get("seeded") else "")
        o.append(f"<tr><td class=mono style='font-size:12px'>{label}</td>"
                 f"<td class=num>{p['discoveries']}</td><td class=num>{_h(p['cpu_hours'])}</td>"
                 f"<td class=num>{_h(per, 3)}</td><td class=num>{_h(p['median_declared'], 3)}</td>"
                 f"<td class=num>{p['cum_discoveries']}</td>"
                 f"<td class=num>{_h(p['cum_cpu_hours'])}</td></tr>")
    o.append("</tbody></table></div>")
    o.append("<p>Median declared CPU-hours is the median of the compute blocks on the "
             "bounds dated that day. Bounds without a compute block count as "
             "discoveries at no recorded cost; the literature values seeded before "
             f"{CHART_START} are folded into the first row.</p>")

    o.append("<h3>Per cell</h3>")
    o.append("<p class=h2sub>Keyed by orbit, copies and rank, as in "
             "<code>autoresearch/summary.py</code>. A solution is an exact refit in the "
             "log or a record-tier bound with a compute block.</p>")
    o.append("<div class='tw scroll'><table><thead><tr><th>orbit</th><th class=num>m</th>"
             "<th class=num>rank</th><th class=num>runs</th><th class=num>CPU-h</th>"
             "<th class=num>solutions</th><th class=num>CPU-h per solution</th>"
             "<th>on the board</th></tr></thead><tbody>")
    for c in table:
        label = site.ORBIT_LABEL.get(c["orbit"], c["orbit"])
        board = " ".join(f"<span class='pill t-{t}'>{t}</span>" for t in c["board"]) \
            or "<span class=none>&mdash;</span>"
        o.append(f"<tr><td>{E(label)}</td><td class=num>{c['m']}</td>"
                 f"<td class=num>{c['rank']}</td><td class=num>{c['runs']}</td>"
                 f"<td class=num>{_h(c['cpu_hours'], 3)}</td>"
                 f"<td class=num>{c['solutions']}</td>"
                 f"<td class=num>{_h(c['cpu_hours_per_solution'], 3)}</td>"
                 f"<td>{board}</td></tr>")
    o.append("</tbody></table></div>")

    # ---- evidence
    with_pr = sum(1 for e in evidence if e["pull_request"] is not None)
    with_receipt = sum(1 for e in evidence if e["receipt"] and e["receipt_matches"])
    tally = tier_counts(evidence)
    by_tier = ", ".join(f"{tally[t]} {t}" for t in TIER_ORDER if tally.get(t))
    o.append("<h2>Evidence index</h2>")
    o.append(f"<p class=h2sub>{len(evidence)} bounds in <code>bounds/</code>: {with_pr} "
             f"reached main through a pull request, {with_receipt} carry a verification "
             "receipt matching the file's content hash. "
             f"By tier: {by_tier or 'none verified at this build'}. Machine-readable copy: "
             f"<a href='{rel}evidence_index.json'>evidence_index.json</a>.</p>")
    o.append("<div class='tw scroll'><table><thead><tr><th>bound</th><th>tier</th>"
             "<th>PR</th><th>merge</th><th>added</th><th>receipt</th><th>Lean</th>"
             "<th>compute</th></tr></thead><tbody>")
    for e in sorted(evidence, key=lambda e: (site.ORBIT_ORDER.index(e["orbit"]),
                                             e["m"], e["direction"], e["rank"])):
        sign = "&le;" if e["direction"] == "upper" else "&ge;"
        if e["tier"] is None and e["ok"] is None:
            tier = "<span class='pill t-cited' title='no receipt and not in the ledger " \
                   "at this build'>unverified</span>"
        elif not e["ok"]:
            tier = "<span class='pill t-failed'>failed</span>"
        else:
            tier = f"<span class='pill t-{e['tier']}'>{e['tier']}</span>"
        if e["receipt"]:
            rc = _blob(e["receipt"], e["receipt"].split("/")[-1])
            if not e["receipt_matches"]:
                rc += " <span class=none title='content hash differs from the file'>stale</span>"
        else:
            rc = "<span class=none>&mdash;</span>"
        if e["lean_module"]:
            ln = _blob("lean_proofs/" + e["lean_module"].replace(".", "/") + ".lean",
                       e["lean_module"].split(".")[-1])
            if e["lean_receipt_ok"] is False:
                ln += " <span class=none>build fails</span>"
            elif e["lean_receipt_ok"] is None:
                ln += " <span class=none>unbuilt</span>"
        else:
            ln = "<span class=none>&mdash;</span>"
        comp = e["compute"]
        if comp:
            bits = []
            if comp.get("runs") is not None:
                bits.append(f"{comp['runs']} runs")
            if comp.get("cpu_hours") is not None:
                bits.append(f"{comp['cpu_hours']:g} CPU-h")
            ct = ", ".join(bits) or "reported"
        else:
            ct = "<span class=none>&mdash;</span>"
        added = _sha(e["added_commit"])
        if e["added_date"]:
            added += f" <span class=none>{E(e['added_date'])}</span>"
        o.append(f"<tr><td><a href='{rel}bounds/{e['slug']}.html'>"
                 f"{site.ket(e['orbit'], e['m'], sign, e['rank'])}</a></td>"
                 f"<td>{tier}</td><td>{_pr(e['pull_request'])}</td>"
                 f"<td>{_sha(e['merge_commit'])}</td><td>{added}</td>"
                 f"<td>{rc}</td><td>{ln}</td>"
                 f"<td class=mono style='font-size:12px'>{ct}</td></tr>")
    o.append("</tbody></table></div>")
    o.append("<p>A bound with no pull request was added to main directly; its adding "
             "commit is the provenance. A receipt is <code>certs/&lt;slug&gt;.json</code>, "
             "written by the verifier and keyed to the content hash of the submission "
             "file, so an edited bound shows a stale receipt until it is re-verified. "
             "Lean-tier bounds have their receipt under "
             "<code>certs/lean-&lt;module&gt;.json</code> instead.</p>")

    # ---- reproduce
    o.append("<h2>Reproducing this page</h2>")
    o.append("<pre>make build                       # re-verifies every bound without a receipt\n"
             "uv run --extra challenge python site_challenge/build.py --no-verify\n"
             "uv run --extra challenge python autoresearch/summary.py --since "
             f"{SINCE}</pre>")
    o.append("<p>The second command rebuilds the pages from <code>docs/ledger.json</code> "
             "and the receipts in <code>certs/</code> without running any certificate; a "
             "bound with neither is listed above as unverified and left off the board.</p>")

    o.append("</div>" + site.footer(rel=rel) + "</body></html>")
    return "".join(o)


def write_report(entries, rows, root=ROOT, docs=None):
    """docs/report/index.html and docs/evidence_index.json."""
    docs = docs or site.DOCS
    runs = load_runs(os.path.join(root, "autoresearch", "runs.jsonl"))
    refs = {r["key"]: r for r in site.parse_bib(os.path.join(docs, "refs.bib"))}
    evidence = evidence_index(entries, root)
    board_dates = {e["slug"]: e["merge_date"] or e["added_date"] for e in evidence}
    comparison = exponent_comparison(rows)
    cells = cell_intervals(rows, board_dates)
    table = cost_table(rows, runs)
    curve = cost_curve(rows, runs)
    head_sha = _git(["rev-parse", "HEAD"], root).strip() or "unknown"
    generated = datetime.date.today().isoformat()

    os.makedirs(os.path.join(docs, "report"), exist_ok=True)
    with open(os.path.join(docs, "report", "index.html"), "w") as f:
        f.write(report_page(comparison, cells, table, curve, evidence, refs,
                            generated, head_sha))
    with open(os.path.join(docs, "evidence_index.json"), "w") as f:
        json.dump({"generated_from": "bounds/", "head": head_sha, "generated": generated,
                   "since": SINCE,
                   "exponents": comparison, "cells": cells, "cost_by_cell": table,
                   "cost_curve": curve, "bounds": evidence}, f, indent=1)
        f.write("\n")
    return {"comparison": comparison, "cells": cells, "table": table,
            "curve": curve, "evidence": evidence}
