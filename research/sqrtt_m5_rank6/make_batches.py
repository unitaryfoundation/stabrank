"""Canonical stored batches for the attested bound: one file per census block in the form census.block_record
writes (timing removed, sorted keys, one line), plus the manifest the verifier checks.

    python make_batches.py SRC_BLOCKS_DIR DEST_RESEARCH_DIR
"""
import sys, os, json, glob, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from census import block_record

src, dest = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(dest, 'results'), exist_ok=True)
root = os.path.abspath(os.path.join(dest, '..', '..'))
batches = []; tot_sec = 0.0; npiv = 0; ncand = 0; sha = None
for f in sorted(glob.glob(os.path.join(src, 'block_*.json'))):
    d = json.load(open(f))
    for r in d['results']:
        tot_sec += r['seconds']; npiv += 1; ncand += len(r['candidates'])
    sha = d['codes_sha256']
    txt = block_record(d['block'], d['m'], d['pivots'], d['results'], d['codes_sha256'])
    name = f"block_{d['block']:05d}.json"
    out = os.path.join(dest, 'results', name)
    open(out, 'w').write(txt)
    batches.append({'id': name[:-5], 'params': {'block': d['block'], 'pivots': d['pivots']},
                    'output': os.path.relpath(out, start=root).replace(os.sep, '/'),
                    'sha256': hashlib.sha256(txt.encode()).hexdigest()})
manifest = {'source': 'Perlmutter job 60159879, 2026-10-23, one CPU node, 128 workers: research/sqrtt_m5_rank6/census.py reps 5; census.py run 5 --workers 128 --block 10',
            'm': 5, 'group': 'S_5 x <S^(x)5>, order 480', 'representatives': npiv, 'blocks': len(batches), 'block_size': 10,
            'pivot_seconds_total': round(tot_sec, 1), 'candidate_records': ncand, 'codes_sha256': sha, 'batches': batches}
json.dump(manifest, open(os.path.join(dest, 'batch_manifest.json'), 'w'), indent=1)
print(f"{len(batches)} batches, {npiv} pivots, {ncand} candidate records, {tot_sec/3600:.1f} CPU-h")
