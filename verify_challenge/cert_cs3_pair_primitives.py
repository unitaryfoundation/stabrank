"""Portable staging copy of accepted exact routines; see PROVENANCE.json.
Not executed or certified by source staging.
"""
from itertools import combinations,product
from math import comb,isqrt
import time
import cert_cs3_dictionary as prior
np=prior.np; exact=prior.exact
P,I,VALUES=prior.P,prior.I,prior.VALUES
N=36720; GROUP=1152


def check_time(deadline, where):
    if time.perf_counter() >= deadline: raise TimeoutError(where)

def coded(values): return np.asarray(values, dtype=np.int8)

def ray(ds): return coded([ds.get(x, -1) for x in range(16)])

def serial(cc): return [exact.serial_coeff(c) for c in cc]

def canonical(row):
    row = row.copy(); support = np.flatnonzero(row >= 0)
    assert len(support)
    gamma = int(row[support[0]])
    row[support] = (row[support] - gamma) % 4
    return row, gamma

def cover_controls(pool, target):
    """Construct identities without scanning the dictionary for completions."""
    ids = {row.tobytes(): j for j, row in enumerate(pool)}
    a, f, g = prior.known_rays()
    halves, gammas = [], []
    for bit in range(2):
        h = f.copy(); h[[x for x in range(16) if (x & 1) != bit]] = -1
        h, gamma = canonical(h); halves.append(h); gammas.append(gamma)
    cc3 = prior.exact_cover([a, f, g], target)
    split = coded([a, g, *halves])
    split_cc = [cc3[0], cc3[2]] + [exact.gmul(cc3[1], exact.PHASES[z]) for z in gammas]
    full = ray({x: 0 for x in range(16)})
    e15 = ray({15: 0})
    low_e = ray({x: 0 for x in range(16) if (x & 3) == 3})
    high_e = ray({x: 0 for x in range(16) if (x >> 2) == 3})
    grid = coded([full, e15, low_e, high_e])
    grid_cc = [(1, 0), (0, -2), (-1, 1), (-1, 1)]
    out = []
    for name, rows, cc in [('split_known_full_ray', split, split_cc), ('non_refinement_product_grid', grid, grid_cc)]:
        assert len(prior.pivots(VALUES[rows.astype(np.int64).T + 1])) == 4
        checked = prior.exact_cover(rows, target)
        assert checked == tuple(cc) and all(c != exact.GZERO for c in checked)
        rid = [ids[r.tobytes()] for r in rows]
        out.append({'name': name, 'ids': rid, 'coefficients': serial(checked),
                    'retained_ids': rid[:2], 'free_ids': rid[2:],
                    'support_sizes': np.sum(rows >= 0, axis=1).astype(int).tolist()})
    return out

def generate_group(meta, gm, deadline):
    """U_q U_p has pullback permutation p[q], ray map gm_q[gm_p]."""
    perms = [np.arange(16, dtype=np.int8)]
    seen = {perms[0].tobytes(): 0}
    actions = np.empty((GROUP, N), dtype=np.int32)
    actions[0] = np.arange(N, dtype=np.int32)
    generators = [coded(g['permutation']) for g in meta['generators']]
    cursor = 0; edges = 0
    while cursor < len(perms):
        check_time(deadline, 'group closure time limit')
        for perm, mapping in zip(generators, gm):
            pp = perms[cursor][perm]; key = pp.tobytes()
            composed = mapping[actions[cursor]].astype(np.int32, copy=False)
            if key not in seen:
                assert len(perms) < GROUP
                seen[key] = len(perms); perms.append(pp)
                actions[len(perms) - 1] = composed
            else:
                assert np.array_equal(actions[seen[key]], composed)
            edges += 1
        cursor += 1
    assert len(perms) == GROUP and edges == GROUP * 7
    inverses = np.array([seen[np.argsort(p).astype(np.int8).tobytes()] for p in perms], dtype=np.int32)
    assert np.array_equal(inverses[inverses], np.arange(GROUP))
    return actions, inverses, {'group_order': GROUP, 'all_generator_edges_checked': edges,
                              'action_array_bytes': actions.nbytes}

def canonical_pair(actions, u, v):
    x, y = actions[:, u].astype(np.int64), actions[:, v].astype(np.int64)
    return int(np.min(np.minimum(x, y) * N + np.maximum(x, y)))
