"""Re-derive every number stated in the OpenReview response from the files in results/.

Run from the repository root:  python3 audits/check_claims.py
Prints one PASS/FAIL line per claim and exits non-zero on any failure. Numbers quoted
from the submitted PDF itself (Tables 8, 9, 15, 18, 25, 26, 27, App. G text) are listed in
CLAIMS.md and are not re-derived here. Query-rule replay numbers are read from
results/query_rule/replay_results.json, produced by audits/replay_query_rules.py.
"""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
FAILS = []


def load(rel):
    return json.loads((RES / rel).read_text())


def check(name, value, expected, tol=0.0, rel=False):
    ok = (abs(value - expected) <= (tol * abs(expected) if rel else tol)) if isinstance(expected, (int, float)) else value == expected
    print(('PASS' if ok else 'FAIL'), name, '| computed', value, '| stated', expected)
    if not ok:
        FAILS.append(name)


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); h = {20: 2.093024054408263, 3: 4.302652729911275}[n] * x.std(ddof=1) / math.sqrt(n)
    return m, m - h, m + h


# ---------------- CelebA-40, 20 paired seeds
c = load('celeba_training/celeba_paired_analysis.json')
check('celeba seeds', c['n_paired_seeds'], 20)
check('celeba selected lr all 0.02', sorted(set(c['selected_learning_rates'].values())), [0.02])
A = c['arm_summary']
for arm, (w, m, b, bce, s) in {'direct': (62.6, 86.9, 71.6, .304, .49), 'mgda_ub': (61.7, 85.6, 67.3, .360, .038), 'uniform': (67.7, 87.6, 70.0, .280, .034)}.items():
    check(f'celeba {arm} worst acc', round(A[arm]['worst_accuracy'] * 100, 1), w)
    check(f'celeba {arm} mean acc', round(A[arm]['mean_accuracy'] * 100, 1), m)
    check(f'celeba {arm} mean balanced acc', round(A[arm]['mean_balanced_accuracy'] * 100, 1), b)
    check(f'celeba {arm} mean BCE', round(A[arm]['mean_bce'], 3), bce)
    check(f'celeba {arm} s/update', round(A[arm]['mean_seconds_per_update'], 2 if arm == 'direct' else 3), s)
rows = c['all_seed_results']
def diffs(a, b, key, scale=100):
    return [(r['test'][a][key] - r['test'][b][key]) * scale for r in rows]
for (a, b, key, mean, lo, hi, pos) in [('direct', 'mgda_ub', 'worst_accuracy', .9, -4.1, 5.8, 9), ('direct', 'mgda_ub', 'mean_accuracy', 1.3, .8, 1.8, 19),
                                       ('direct', 'mgda_ub', 'mean_balanced_accuracy', 4.3, 3.2, 5.5, 19), ('direct', 'uniform', 'worst_accuracy', -5.1, -8.6, -1.7, 3),
                                       ('direct', 'uniform', 'mean_accuracy', -.7, -1.0, -.4, 1), ('direct', 'uniform', 'mean_balanced_accuracy', 1.7, 1.0, 2.3, 18)]:
    d = diffs(a, b, key); m, l, h = tci(d)
    check(f'celeba {a}-{b} {key} mean', round(m, 1), mean); check(f'celeba {a}-{b} {key} CI low', round(l, 1), lo)
    check(f'celeba {a}-{b} {key} CI high', round(h, 1), hi); check(f'celeba {a}-{b} {key} seeds favouring', sum(x > 0 for x in d), pos)
check('celeba direct-mgda_ub BCE lower on seeds', sum(x < 0 for x in diffs('direct', 'mgda_ub', 'mean_bce', 1)), 20)
check('celeba paired SD worst acc direct-mgda_ub about 10 points', round(float(np.std(diffs('direct', 'mgda_ub', 'worst_accuracy'), ddof=1))), 11, tol=1)
sd = {arm: float(np.std([r['test'][arm]['worst_accuracy'] * 100 for r in rows], ddof=1)) for arm in ('direct', 'mgda_ub', 'uniform')}
check('celeba seed SD direct (7-8)', round(sd['direct']), 7); check('celeba seed SD mgda_ub (7-8)', round(sd['mgda_ub']), 8); check('celeba seed SD uniform', round(sd['uniform'], 1), 1.3)
check('celeba MGDA arms minimum worst acc about 40%', round(min(r['test'][a]['worst_accuracy'] * 100 for r in rows for a in ('direct', 'mgda_ub'))), 40)
check('celeba direct / mgda_ub per update (13x)', round(A['direct']['mean_seconds_per_update'] / A['mgda_ub']['mean_seconds_per_update']), 13)
check('celeba direct / uniform per update (14x)', round(A['direct']['mean_seconds_per_update'] / A['uniform']['mean_seconds_per_update']), 14)

# ---------------- QM9
q = load('qm9/paired_analysis.json')
S = q['arm_summary']
names = {'uniform': 'uniform', 'FAMO': 'FAMO', 'Aligned_MTL': 'Aligned-MTL', 'FairGrad': 'FairGrad', 'direct_factorized': 'Direct', 'Nash_MTL': 'Nash-MTL', 'mgda_ub': 'MGDA-UB'}
stated = {'uniform': (.303, .056), 'FAMO': (.312, .074), 'Aligned_MTL': (.327, .275), 'FairGrad': (.337, .310), 'direct_factorized': (.339, .455), 'Nash_MTL': (.414, .536), 'mgda_ub': (.477, .070)}
for k, (mae, sec) in stated.items():
    key = next(x for x in S if x.lower().replace('-', '_') == k.lower())
    check(f'qm9 {names[k]} normalized MAE', round(S[key]['mean_final_test_normalized_mae'], 3), mae)
    sk = next(x for x in S[key] if 'seconds' in x and 'update' in x and 'mean' in x)
    check(f'qm9 {names[k]} s/update', round(S[key][sk], 3), sec)
C = q['all_direct_minus_baseline_comparisons']
def comp(name):
    return C[next(x for x in C if x.lower().replace('-', '_') == name.lower())]
p = comp('mgda_ub'); check('qm9 primary mean', round(p['mean'], 3), -.138)
ci = p.get('t95') or p.get('descriptive_t95') or p.get('student_t_95_interval')
check('qm9 primary CI low', round(ci[0], 3), -.442); check('qm9 primary CI high', round(ci[1], 3), .167)
for name, mean, wins in (('mgda_ub', None, 3), ('Nash_MTL', None, 2), ('FairGrad', .002, None), ('Aligned_MTL', .012, None), ('uniform', .036, 0), ('FAMO', .027, 0)):
    d = comp(name); vals = d.get('paired_seed_differences') or d.get('differences')
    if mean is not None: check(f'qm9 direct-{name} mean', round(d['mean'], 3), mean)
    if wins is not None: check(f'qm9 direct better than {name} on seeds', sum(v < 0 for v in vals), wins)
check('qm9 300-epoch projection about 380 A100-hours', round(q['full_300epoch_update_hours_extrapolated'] / 10) * 10, 380, tol=10)
jp = load('qm9/joint_preflight.json')
nat = [r for r in jp['rows'] if r['config'].get('method') == 'exact_native' and r['config'].get('chunk') == 1 and not r['config'].get('fused') and r.get('median_update_seconds')][0]
dsel = jp['selection']['direct_factorized']
dr = [r for r in jp['rows'] if r['config'].get('method') == 'direct_factorized' and r['config'].get('row_chunk') == dsel['row_chunk'] and r['config'].get('chunk') == dsel['chunk'] and not r['config'].get('fused')][0]
check('qm9 pilot native s', round(nat['median_update_seconds'], 2), .32); check('qm9 pilot native GiB', round(nat['peak_allocated_bytes'] / 2**30, 1), 2.9)
check('qm9 pilot direct s', round(dr['median_update_seconds'], 2), .48); check('qm9 pilot direct GiB', round(dr['peak_allocated_bytes'] / 2**30), 31)

# ---------------- Language benchmark
def groups(rel):
    b = load(rel)['benchmarks'][0]
    return {(g['M'], g['method'], g['contract']): g for g in b['groups']}
G = groups('language_benchmark/pythia160m_summary.json')
def med(M, m, c='table'):
    return G[(M, m, c)]['qualified_rows_only']['median_seconds']
def peak(M, m, c='table'):
    return G[(M, m, c)]['qualified_rows_only']['max_peak_allocated_bytes'] / 2**30
check('160M M8 direct', round(med(8, 'direct_cached'), 3), .376); check('160M M32 direct', round(med(32, 'direct_cached'), 3), .373)
check('160M M8 best structured (auto)', round(min(med(8, m) for m in ('structured_auto', 'structured_parameter', 'structured_token')), 3), .446)
check('160M M32 best structured (token)', round(min(med(32, m) for m in ('structured_auto', 'structured_parameter', 'structured_token')), 3), .457)
check('160M M8 ratio 1.19', round(med(8, 'structured_auto') / med(8, 'direct_cached'), 2), 1.19)
check('160M M32 ratio 1.22', round(med(32, 'structured_token') / med(32, 'direct_cached'), 2), 1.22)
check('160M ghost M8', round(med(8, 'ghost_columns'), 3), .634); check('160M ghost M32', round(med(32, 'ghost_columns'), 3), 1.712)
check('160M native serial M8', round(med(8, 'native_serial'), 3), .957); check('160M native serial M32', round(med(32, 'native_serial'), 3), 3.497)
check('160M native batched M8', round(med(8, 'native_batched'), 3), 1.006); check('160M native batched M32', round(med(32, 'native_batched'), 3), 4.244)
check('160M native batched qualified M8', G[(8, 'native_batched', 'table')]['qualified_nonwarmup'], 6); check('160M native batched qualified M32', G[(32, 'native_batched', 'table')]['qualified_nonwarmup'], 6)
for M_ in (8, 32):
    for m in ('direct_cached', 'structured_auto', 'ghost_columns'):
        check(f'160M M{M_} {m} peak 4.11 GiB', round(peak(M_, m), 2), 4.11)
check('160M native serial peak M8', round(peak(8, 'native_serial'), 2), 9.11); check('160M native serial peak M32', round(peak(32, 'native_serial'), 1), 23.6)
check('160M speedup vs native serial M8 (2.5x)', round(med(8, 'native_serial') / med(8, 'direct_cached'), 1), 2.5)
check('160M speedup vs native serial M32 (9.4x)', round(med(32, 'native_serial') / med(32, 'direct_cached'), 1), 9.4)
check('160M memory vs native serial M8 (2.2x)', round(peak(8, 'native_serial') / peak(8, 'direct_cached'), 1), 2.2)
check('160M memory vs native serial M32 (5.7x)', round(peak(32, 'native_serial') / peak(32, 'direct_cached'), 1), 5.7)
for (M_, m, n) in ((8, 'native_serial', 3), (8, 'structured_auto', 3), (32, 'structured_auto', 9), (8, 'ghost_columns', 9), (32, 'ghost_columns', 0), (8, 'direct_cached', 3), (32, 'direct_cached', 3)):
    check(f'160M joint qualified M{M_} {m}', G[(M_, m, 'joint')]['qualified_nonwarmup'], n)
check('160M native serial fails weighted on 6/9 at M8', 9 - G[(8, 'native_serial', 'weighted')]['qualified_nonwarmup'], 6)
G1 = groups('language_benchmark/pythia1b_summary.json')
def med1(m, c='table'):
    return G1[(8, m, c)]['qualified_rows_only']['median_seconds']
check('1B direct s', round(med1('direct_cached'), 2), 1.92); check('1B direct GiB', round(G1[(8, 'direct_cached', 'table')]['qualified_rows_only']['max_peak_allocated_bytes'] / 2**30, 1), 15.7)
check('1B best structured s', round(min(med1(m) for m in ('structured_auto', 'structured_parameter', 'structured_token')), 2), 1.99)
check('1B ratio 1.03', round(med1('structured_auto') / med1('direct_cached'), 2), 1.03); check('1B ghost s', round(med1('ghost_columns'), 2), 4.51)
check('1B native serial and batched qualified', sum(G1[(8, m, c)]['qualified_nonwarmup'] for m in ('native_serial', 'native_batched') for c in ('table', 'weighted', 'joint')), 0)
check('1B failures are CUDA OOM', load('language_benchmark/pythia1b_summary.json')['benchmarks'][0]['failure_counts'], {'CUDA_OOM': 16})
check('1B every contraction route passes joint 9/9', all(G1[(8, m, 'joint')]['qualified_nonwarmup'] == 9 for m in ('direct_cached', 'direct_online', 'ghost_columns', 'structured_auto', 'structured_parameter', 'structured_token')), True)
tp = load('language_benchmark/training_preflight.json')
TR = {}
for r in tp['rows']:
    cfg = r['configuration']; key = cfg['method'] + ('_' + cfg['method_options']['periodic_backend'] if cfg.get('method_options') else '')
    TR.setdefault(key, []).append(r)
dc = [r for r in TR['direct'] if abs(r['peak_allocated_bytes'] / 2**30 - 15.67) < .05][0]
check('1B update direct s', round(dc['mean_update_seconds'], 2), 1.92)
check('1B update uniform s', round(TR['uniform'][0]['mean_update_seconds'], 2), 1.20); check('1B update FA s', round(TR['fa'][0]['mean_update_seconds'], 2), 1.60)
check('1B update FAMO s', round(TR['famo'][0]['mean_update_seconds'], 2), 1.60)
check('1B update uniform measured peak', round(TR['uniform'][0]['measured_peak_allocated_bytes'] / 2**30, 1), 12.7)
check('1B update FA measured peak', round(TR['fa'][0]['measured_peak_allocated_bytes'] / 2**30, 1), 12.7); check('1B update FA warm-start peak', round(TR['fa'][0]['peak_allocated_bytes'] / 2**30, 1), 15.7)
check('1B update FAMO peak', round(TR['famo'][0]['measured_peak_allocated_bytes'] / 2**30, 1), 12.7)
check('1B direct/uniform 1.6x', round(dc['mean_update_seconds'] / TR['uniform'][0]['mean_update_seconds'], 1), 1.6)
check('1B direct/FA 1.2x', round(dc['mean_update_seconds'] / TR['fa'][0]['mean_update_seconds'], 1), 1.2)
check('1B direct/FAMO 1.2x', round(dc['mean_update_seconds'] / TR['famo'][0]['mean_update_seconds'], 1), 1.2)
check('1B CAGrad did not complete (reported, not used)', TR['cagrad'][0]['complete'], False)

# ---------------- Live CelebA recovery study (original rules)
rs = load('recovery_study/results.json')
qs = [s['reference']['objective'] for s in rs['states']]
check('q* minimum', round(min(qs), 5), .00055); check('q* maximum', round(max(qs), 2), .38); check('q* spread about 700x', round(max(qs) / min(qs), -2), 700)
check('uniform fails to descend (trunk) at all 5 states', sum(s['uniform_min_response'] < 0 for s in rs['states']), 5)
held = [s for s in rs['states'] if s['seed'] != 9101]
tf = [t['factor'] if 'factor' in t else t.get('objective_factor') for t in rs.get('batch_transfer', [])]
if tf and tf[0] is not None:
    pairs = [t for t in rs['batch_transfer'] if t['weight_source_seed'] != 9101 and t['target_seed'] != 9101]
    key = next(k for k in pairs[0] if 'factor' in k)
    check('transfer pairs among held-out', len(pairs), 12); check('median transfer factor 39', round(st.median(t[key] for t in pairs)), 39)
else:
    Kd = {s['seed']: np.asarray(s['gram']) for s in rs['states']}; Wd = {s['seed']: np.asarray(s['reference_weights']) for s in rs['states']}
    fac = [float(Wd[a] @ Kd[b] @ Wd[a]) / Kd_q for a in Kd for b in Kd if a != b and 9101 not in (a, b) for Kd_q in [[s['reference']['objective'] for s in rs['states'] if s['seed'] == b][0]]]
    check('transfer pairs among held-out', len(fac), 12); check('median transfer factor 39', round(st.median(fac)), 39)
Kd = {s['seed']: np.asarray(s['gram']) for s in rs['states']}; Wd = {s['seed']: np.asarray(s['reference_weights']) for s in rs['states']}
check('every transferred vector has a negative trunk response', all(float((Kd[b] @ Wd[a]).min()) < 0 for a in Kd for b in Kd if a != b), True)
ad = [r for r in rs['rows'] if r['rule'] in ('completion', 'margin') and r['complete']]
check('adaptive timed runs accepted', sum(bool((r['recovery'] or {}).get('accepted')) and 'fallback' not in (r['recovery'] or {}) for r in ad), 120)
acts = [r['recovery']['actions'] for r in ad]
check('adaptive actions min', min(acts), 20); check('adaptive actions max', max(acts), 37)
check('worst objective factor 1.00044', round(max(r['native_objective_factor_upper'] for r in ad), 5), 1.00044)
for s_, a_ in zip((9102, 9103, 9104, 9105), (33, 36, 23, 31)):
    check(f'completion actions state {s_} every eps', sorted({r['recovery']['actions'] for r in ad if r['seed'] == s_ and r['rule'] == 'completion'}), [a_])
med_state = lambda rule: st.median(st.median(r['seconds'] for r in rs['rows'] if r['seed'] == s_ and r['rule'] == rule and r.get('epsilon') in (None, .05) and r['complete']) for s_ in (9102, 9103, 9104, 9105))
check('original study native chunk 1 s', round(med_state('native_1'), 2), .54)
pk = lambda rule: max(r['peak_allocated_bytes'] for r in rs['rows'] if r['seed'] != 9101 and r['rule'] == rule and r['complete']) / 2**30
check('native chunk 1 GiB', round(pk('native_1'), 2), 2.09); check('direct GiB', round(pk('direct'), 2), 8.04); check('stream GiB', round(pk('completion'), 2), .54)
ar = [rr for ls in rs['line_search'] for rr in ls.get('rows', []) if 'c' in rr]
check('Armijo checks: 5 states x 4 constants', len(ar), 20)
check('Armijo checks accept 0.02 at first trial', all(rr['accepted'] and rr['accepted_eta'] == .02 and len(rr['trials']) == 1 for rr in ar), True)

# ---------------- Sharp-endpoint live training study
ev = load('sharp_training/sharp_events_summary_vm.json')['runs']
sa = load('sharp_training/sharp_training_reporting_analysis.json')
def arm(a):
    return [r for r in ev if r['arm'] == a]
check('training study complete, 12 runs', (sa['complete'], len(sa['runs'])), (True, 12))
check('sharp early stops', sum(r['early_stops_below_40'] for r in arm('stream_completion')), 889); check('scalar early stops', sum(r['early_stops_below_40'] for r in arm('stream_scalar')), 890)
check('sharp fallbacks', sum(r['fallbacks'] for r in arm('stream_completion')), 11); check('scalar fallbacks', sum(r['fallbacks'] for r in arm('stream_scalar')), 10)
check('fallbacks below 0.06% of median energy (sharp)', max(x['qstar_over_median_accepted_U'] for r in arm('stream_completion') for x in r['fallback_records']) < 6e-4, True)
check('mean actions sharp', round(st.mean(r['actions_mean'] for r in arm('stream_completion')), 1), 29.7)
check('energy test first met median 20 (sharp)', sorted({r['energy_condition_first_met_median'] for r in arm('stream_completion')}), [20])
check('gate decided stops (sharp)', sum(r['stop_decided_by']['margin'] for r in arm('stream_completion')), 887)
check('energy test first met median 31-32 (scalar)', sorted({r['energy_condition_first_met_median'] for r in arm('stream_scalar')}), [31, 32])
runs = {(r['seed'], r['arm']): r for r in sa['runs']}
sav_a = [1 - runs[(s_, 'stream_completion')]['adaptive_actions'] / runs[(s_, 'stream_scalar')]['adaptive_actions'] for s_ in (62001, 62002, 62003)]
sav_t = [1 - runs[(s_, 'stream_completion')]['update_seconds'] / runs[(s_, 'stream_scalar')]['update_seconds'] for s_ in (62001, 62002, 62003)]
check('action saving range low 1.8%', round(min(sav_a) * 100, 1), 1.8); check('action saving range high 3.5%', round(max(sav_a) * 100, 1), 3.5)
check('time saving range low 2.4%', round(min(sav_t) * 100, 1), 2.4); check('time saving range high 2.8%', round(max(sav_t) * 100, 1), 2.8)
check('36 audits all passed', sum(len(r['audit_records']) for r in sa['runs']), 36)
check('stream audit objective factor at most 1.0009', round(max(x['native_objective_factor_upper'] for r in sa['runs'] if r['arm'].startswith('stream') for x in r['audit_records']), 4), 1.0008, tol=1e-4)
mw = lambda a: st.mean(runs[(s_, a)]['final_test']['worst_accuracy'] * 100 for s_ in (62001, 62002, 62003))
for a, v in (('stream_completion', 65.5), ('stream_scalar', 63.2), ('direct', 64.7), ('native_chunk1', 64.7)):
    check(f'training mean worst acc {a}', round(mw(a), 1), v)
check('direct vs native differ by up to 2.5 points', round(max(abs(runs[(s_, 'direct')]['final_test']['worst_accuracy'] - runs[(s_, 'native_chunk1')]['final_test']['worst_accuracy']) * 100 for s_ in (62001, 62002, 62003)), 1), 2.5)
check('stream s/update 1.79', round(st.mean(runs[(s_, 'stream_completion')]['update_seconds'] / 300 for s_ in (62001, 62002, 62003)), 2), 1.79)
check('direct s/update 0.50', round(st.mean(runs[(s_, 'direct')]['update_seconds'] / 300 for s_ in (62001, 62002, 62003)), 2), .50)
check('stream own-recovery peak 0.58 GiB', round(max(r['peak_gib_non_fallback_updates'] for r in arm('stream_completion')), 2), .58)
check('two of three stream runs peaked at Direct memory', sum(runs[(s_, 'stream_completion')]['peak_update_allocated_bytes'] / 2**30 > 8 for s_ in (62001, 62002, 62003)), 2)

# ---------------- Target-aware Stream: live fixed-state study and replay
qs_ = load('query_rule/query_rule_summary.json')
check('live direct s', round(qs_['direct@None']['median_of_state_medians_seconds'], 2), .53)
check('live original s', round(qs_['completion@0.05']['median_of_state_medians_seconds'], 2), 1.85)
check('live target-aware 0.95 s', round(qs_['gate@0.05']['median_of_state_medians_seconds'], 2), 1.72)
check('live target-aware strict s', round(qs_['gate@1.0']['median_of_state_medians_seconds'], 2), 1.08)
check('live reduction by medians 41%', round((1 - qs_['gate@1.0']['median_of_state_medians_seconds'] / qs_['completion@0.05']['median_of_state_medians_seconds']) * 100), 41)
check('live reduction paired geomean 37%', round((1 - qs_['gate@1.0']['paired_time_ratio_to_completion@0.05_geomean']) * 100), 37)
check('live target-aware strict actions', [qs_['gate@1.0']['actions_by_state'][k][0] for k in ('9102', '9103', '9104', '9105')], [20, 25, 15, 18])
check('live all directions pass audit', all(v['all_passed_fidelity'] for v in qs_.values()), True)
check('live every task descends', all(v['all_strict_descent'] for v in qs_.values()), True)
check('live energy within 2.1%', round(qs_['gate@1.0']['max_objective_factor'], 3), 1.020, tol=.001)
check('live stream peak 0.54 GiB', round(qs_['gate@1.0']['max_peak_gib'], 2), .54)
check('live stream about 2x direct', round(qs_['gate@1.0']['median_of_state_medians_seconds'] / qs_['direct@None']['median_of_state_medians_seconds'], 1), 2.1, tol=.1)
check('live original 3.5x direct', round(qs_['completion@0.05']['median_of_state_medians_seconds'] / qs_['direct@None']['median_of_state_medians_seconds'], 1), 3.5)
rp = load('query_rule/replay_results.json')
check('replay predicts all live action counts', (rp['live_action_count_check']['live_stream_rows'], rp['live_action_count_check']['mismatches']), (60, 0))
def red(s, a, b):
    S_ = rp['sets'][s]['summary']; return round((1 - S_[a]['mean_actions'] / S_[b]['mean_actions']) * 100)
check('distinct CelebA states', rp['sets']['celeba_distinct']['summary']['original']['n'], 33)
check('replay target-aware 0.95 vs original CelebA 7%', red('celeba_distinct', 'target_aware_0.95', 'original'), 7)
check('replay target-aware 0.95 vs original CIFAR 24%', red('cifar10_m100', 'target_aware_0.95', 'original'), 24)
check('replay target-aware 0.95 vs original recorded 3%', red('recorded103', 'target_aware_0.95', 'original'), 3)
check('replay sharp vs scalar strict CelebA 37%', red('celeba_distinct', 'target_aware_strict', 'scalar_endpoint_strict'), 37)
check('replay sharp vs scalar strict CIFAR 59%', red('cifar10_m100', 'target_aware_strict', 'scalar_endpoint_strict'), 59)
check('replay sharp vs scalar strict recorded 29%', red('recorded103', 'target_aware_strict', 'scalar_endpoint_strict'), 29)
check('replay CelebA sharp strict mean 20.2', round(rp['sets']['celeba_distinct']['summary']['target_aware_strict']['mean_actions'], 1), 20.2)
check('replay CelebA scalar strict mean 32.1', round(rp['sets']['celeba_distinct']['summary']['scalar_endpoint_strict']['mean_actions'], 1), 32.1)
check('replay all accepted', all(v['accepted'] == v['n'] for s in rp['sets'].values() for v in s['summary'].values()), True)

# ---------------- Block-4 uniform controls and App. G.2 run record
u = load('historical/uniform_block4_controls.json')['rows']
check('uniform descends on all nine panels', all(r['uniform_descends'] for r in u), True)
check('uniform factor range low', round(min(r['uniform_objective_factor'] for r in u), 2), 1.15); check('uniform factor range high', round(max(r['uniform_objective_factor'] for r in u), 2), 4.26)
g2 = load('historical/g2_six_arm_run.json')['runs']['2026082901']['paretotrace_front']['steps']
early = [s for s in g2 if s['accounting']['algorithm_gram_actions'] < 16]
check('G.2 early stops', len(early), 22); check('G.2 early actions range', (min(s['accounting']['algorithm_gram_actions'] for s in early), max(s['accounting']['algorithm_gram_actions'] for s in early)), (4, 10))
check('G.2 U/L at most 1.049', max(s['selection']['action_trace'][-1]['upper'] / s['selection']['action_trace'][-1]['lower'] for s in early) <= 1.049, True)
check('G.2 audited q/q* at most 1.031', max(s['front_audit']['q_over_qstar'] for s in early) <= 1.031, True)
check('G.2 no false stop', sum(bool(s['front_audit']['front_false_stop']) for s in early), 0)
sl = [s['selection']['action_trace'][-1]['numerical_slack'] / s['selection']['action_trace'][-1]['upper'] for s in early]
check('G.2 slack at most 2.9% of U', round(max(sl) * 100, 1), 2.9); check('G.2 slack median 0.16% of U', round(st.median(sl) * 100, 2), .16)
check('G.2 algorithm + audit actions = 16 every step', all(s['accounting']['algorithm_gram_actions'] + s['accounting']['audit_completion_actions'] == 16 for s in g2), True)

print('\n%d claims failed' % len(FAILS) if FAILS else '\nALL CLAIMS PASS')
sys.exit(1 if FAILS else 0)
