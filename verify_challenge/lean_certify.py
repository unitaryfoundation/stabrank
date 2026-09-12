"""Build the Lean modules that bounds depend on and write build receipts.

Building mathlib is far too slow to run on every site build, so the site trusts
a receipt rather than rebuilding. This is the only thing that writes one, and it
writes ok=true only after a real `lake build` of that module exits zero.
"""
import glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAN = os.path.join(ROOT, "lean_proofs")
CERTS = os.path.join(ROOT, "certs")


def receipt(module):
    return os.path.join(CERTS, f"lean-{module.replace('.', '-')}.json")


def main():
    mods = sorted({json.load(open(p)).get("lean", {}).get("module")
                   for p in glob.glob(os.path.join(ROOT, "bounds", "*.json"))} - {None})
    if not mods:
        print("no bounds claim a Lean proof")
        return 0
    os.makedirs(CERTS, exist_ok=True)
    bad = 0
    for m in mods:
        print(f"lake build {m} ...", flush=True)
        try:
            r = subprocess.run(["lake", "build", m], cwd=LEAN,
                               capture_output=True, text=True, timeout=5400)
            ok = r.returncode == 0
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        except FileNotFoundError:
            print("  lake not found on PATH; install elan or run this in CI")
            return 2
        except subprocess.TimeoutExpired:
            ok, tail = False, ["timed out after 5400s"]
        with open(receipt(m), "w") as f:
            json.dump({"module": m, "ok": ok, "log": tail}, f, indent=2)
            f.write("\n")
        print(f"  {'ok' if ok else 'FAILED'}  -> {os.path.relpath(receipt(m), ROOT)}")
        bad += 0 if ok else 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
