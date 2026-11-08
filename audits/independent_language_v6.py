"""Read-only independent audit of the completed Pythia-160m v6 artifact.

Run: .venv-v46/bin/python rebuttal/20261107/audits/independent_language_v6_20261108.py
No project numerical implementation is imported. QPs use SciPy SLSQP for support
discovery followed by a NumPy equality-constrained solve and full KKT checks.
The project analyzer is run separately solely to compare its saved summary.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results_completed/baseline_v6"
EXPECTED_SHA = "db3d551249940ae5b3e5a73f637be77b1a251b8779dbeb7dcefa811e6ac31ce4"
METHODS = ("direct_cached", "direct_online", "ghost_columns", "structured_auto",
           "structured_parameter", "structured_token", "native_serial", "native_batched")
CONTRACTS = ("table", "weighted", "joint")
SEEDS = (5011, 5012, 5013)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for k, v in items:
            assert k not in out, (path, "duplicate key", k)
            out[k] = v
        return out
    def bad(x):
        raise ValueError(x)
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=bad)


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def near(a, b, scale=1, rtol=2e-8):
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert np.allclose(a, b, rtol=rtol, atol=2e-12 * max(abs(scale), 1e-100)), (a, b)


def configurations(method):
    if method.startswith(("direct_", "structured_")):
        base = [dict(row_chunk=r, scratch_mib=s) for r, s in ((64, 32), (256, 32), (64, 128), (256, 128))]
    elif method == "native_serial":
        base = [dict(gram_precision=p) for p in ("fp64_tiled", "fp32")]
    elif method == "native_batched":
        base = [dict(native_chunk=n, gram_precision=p) for p in ("fp64_tiled", "fp32") for n in (2, 4, 8)]
    else:
        base = [{}]
    recon = ("weighted_native",) if method == "direct_online" else ("operator", "weighted_native")
    return [{**c, "reconstruction": r} for c in base for r in recon]


def qp_independent(K):
    n = len(K)
    scale = np.abs(K).max()
    assert np.linalg.eigvalsh(K).min() > 0, "This audit uses positive-definite retained Grams"
    f = minimize(lambda w: w @ (K / scale) @ w, np.ones(n) / n,
                 jac=lambda w: 2 * (K / scale) @ w, bounds=[(0, None)] * n,
                 constraints=[dict(type="eq", fun=lambda w: w.sum() - 1,
                                   jac=lambda w: np.ones(n))], method="SLSQP",
                 options=dict(ftol=1e-14, maxiter=3000))
    assert f.success, f.message
    support = f.x > 1e-7
    z = np.linalg.solve(K[np.ix_(support, support)], np.ones(support.sum()))
    w = np.zeros(n)
    w[support] = z / z.sum()
    q = float(w @ K @ w)
    u = K @ w
    lo = max(0.0, min(q, 2 * float(u.min()) - q))
    assert w.min() >= 0 and abs(w.sum() - 1) < 1e-12
    assert np.max(np.abs(u[support] - q)) < scale * 1e-12
    assert np.all(u[~support] >= q - scale * 1e-12)
    return w, q, lo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON output; refuses overwrite")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output already exists: " + str(args.output))
    assert digest(BASE / "pythia160m.json") == EXPECTED_SHA
    d = read(BASE / "pythia160m.json")
    parity = read(BASE / "audit_gpu_parity.json")
    saved = read(BASE / "pythia160m_summary.json")
    manifest = read(BASE / "source_identity.json")
    assert set(manifest) == {p.name for p in (BASE / "executed_source").glob("*.py")}
    assert set(manifest) == set(d["sources"])
    for name, sha in manifest.items():
        assert digest(BASE / "executed_source" / name) == sha == d["sources"][name]
    for name, sha in d["executed_imports"].items():
        assert manifest[name] == sha
    for name, sha in parity["sources"].items():
        assert manifest[name] == sha
    assert d["complete"] and not d["quick"] and d["mode"] == "reporting"
    assert d["development_seed"] == 5000 and d["reporting_seeds"] == list(SEEDS)
    assert d["objectives"] == [8, 32] and d["batch"] == 32
    assert d["data"] == read(ROOT / "language_data_manifest.json")
    assert d["model"]["model"] == "EleutherAI/pythia-160m-deduped"
    assert d["model"]["revision"] == "c54a0e0b28cc667b6f278803024438d57f847b5d"
    assert d["model_dtype"] == "float32" and d["tf32"] is False
    assert parity["complete"] and parity["seed"] == 4999
    assert parity["gpu"] == d["gpu"] and parity["torch"] == d["torch"]
    assert set(d["native_references"]) == {f"{m}:{s}" for m in (8, 32) for s in (5000, *SEEDS)}

    fits, refs = {}, []
    for key, ref in list(d["native_references"].items()) + [("parity", parity["reference"])]:
        assert key == "parity" or ref["complete"]
        K = np.asarray(ref["gram"])
        near(K, K.T)
        w, q, lo = qp_independent(K)
        scale = np.abs(K).max()
        near(ref["objective"], q, scale)
        near(ref["lower"], lo, scale)
        near(ref["qp_gap"], q - lo, scale)
        assert ref["host_table_bytes"] == len(K) * 162322944 * 4
        fits[key] = K, w, q, lo
        refs.append(dict(key=key, min_eigenvalue=float(np.linalg.eigvalsh(K).min()),
                         support=int((w > 0).sum()), independent_objective=q,
                         independent_lower=lo, objective_delta=q-ref["objective"],
                         lower_delta=lo-ref["lower"], dual_gap=q-lo))

    def audit_diagnostics(r, K, w, qref):
        scale = np.abs(K).max()
        wc = w.astype(np.float32).astype(np.float64)
        near(r["executed_coefficients"], wc)
        near(r["executed_coefficient_sum"], wc.sum())
        near(r["solver_coefficient_sum"], w.sum())
        near(r["cast_table_objective"], wc @ K @ wc, scale)
        norm, err = r["table_direction_norm"], r["emitted_table_direction_absolute_error"]
        gradnorms = np.sqrt(np.diag(K))
        near(r["emitted_table_direction_relative_error"], err / norm)
        near(r["task_gradient_norm_max"], gradnorms.max(), gradnorms.max())
        near(r["mixture_cancellation_condition"], np.abs(w) @ gradnorms / norm)
        responses = np.asarray(r["executed_native_table_responses"])
        near(r["executed_native_table_min_response"], responses.min(), scale)
        near(r["table_response_error_bound_diagnostic"], K @ w-gradnorms*err, scale)
        assert abs(norm**2 - w @ K @ w) <= 1e-10 * scale
        assert np.all(np.abs(responses-K@w) <= gradnorms*err+1e-10*scale)
        assert abs(r["emitted_direction_norm"] - norm) <= err + 1e-10*norm
        near(r["normalized_gain_to_reference_primal"], responses.min()/(r["emitted_direction_norm"]*np.sqrt(qref)))
        td = r["emitted_table_direction_relative_error"] <= 2e-5
        wd = r["emitted_weighted_backward_relative_error"] <= 2e-5
        assert r["passes_table_direction"] == td and r["passes_weighted_backward_direction"] == wd
        return td, wd

    parity_rows = []
    K, wqp, qp, lo = fits["parity"]
    assert [r["weighting"] for r in parity["rows"]] == ["uniform", "qp", "onehot"]
    for r in parity["rows"]:
        assert r["complete"]
        w = {"uniform": np.ones(8)/8, "qp": wqp, "onehot": np.eye(8)[0]}[r["weighting"]]
        max_scaled = 0.0
        numeric = []
        for key, a in r["cpu"].items():
            b = r["gpu"][key]
            if isinstance(a, bool):
                assert a == b
            elif isinstance(a, (int, float, list)):
                a, b = np.asarray(a), np.asarray(b)
                assert np.allclose(a, b, rtol=1e-8, atol=1e-10)
                max_scaled = max(max_scaled, float(np.max(np.abs(a-b)/(1e-10+1e-8*np.abs(b)))))
                numeric.append(key)
        assert numeric == r["numeric_fields"]
        for diag in (r["cpu"], r["gpu"]):
            td, wd = audit_diagnostics(diag, K, w, qp)
            geom = w @ K @ w / lo <= 1.001
            assert r["contracts"] == dict(passes_table_contract=bool(geom and td),
                passes_weighted_contract=bool(geom and wd), passes_joint_contract=bool(geom and td and wd))
        parity_rows.append(dict(weighting=r["weighting"], numeric_fields=len(numeric),
                                max_fraction_of_allclose_tolerance=max_scaled, contracts=r["contracts"]))

    rows, by_seed, gates = {}, {}, Counter()
    for phase, field in (("calibration", "calibration"), ("reporting", "rows")):
        for r in d[field]:
            key = (phase, r["M"], r["seed"], r["method"], canon(r["configuration"]), r["repeat"])
            assert key not in rows
            rows[key] = r
            assert r["warmup"] == (r["repeat"] == 0)
            assert len(r["identities"]) == len(set(r["identities"])) == 32
            assert by_seed.setdefault(r["seed"], r["identities"]) == r["identities"]
            assert "audit_failure" not in r and r["timed_recovery_complete"]
            K, _, qref, lo = fits[f'{r["M"]}:{r["seed"]}']
            H, w = np.asarray(r["gram"]), np.asarray(r["weights"])
            assert np.linalg.eigvalsh(H).min() > 0
            scale = np.abs(K).max()
            near(H, H.T)
            u = H @ w
            q = w @ u
            lower = max(0, min(q, 2*u.min()-q))
            near(r["objective"], q, scale)
            near(r["qp_gap"], q-lower, scale)
            assert q-lower <= 2e-12*max(np.trace(H)/len(H), q, 1e-100)*1.05 + 2e-14*np.abs(H).max()
            near(r["native_reference_lower"], lo, scale)
            near(r["native_reference_qp_gap"], fits[f'{r["M"]}:{r["seed"]}'][2]-lo, scale)
            nativeq = w @ K @ w
            gramerr = np.linalg.norm(H-K)/np.linalg.norm(K)
            near(r["native_gram_relative_error"], gramerr)
            near(r["native_objective_excess_above_lower"], nativeq-lo, scale)
            near(r["native_table_min_response"], (K@w).min(), scale)
            assert lo > scale*1e-10
            near(r["native_objective_factor_upper"], nativeq/lo)
            feasible = w.min() >= -1e-12 and abs(w.sum()-1) <= 1e-10
            gramok, objok = gramerr <= 2e-5, nativeq/lo <= 1.001
            td, wd = audit_diagnostics(r, K, w, qref)
            checks = dict(passed_weight_feasibility_gate=feasible, passed_native_gram_gate=gramok,
                passed_native_objective_gate=objok, passes_table_direction=td, passes_weighted_backward_direction=wd)
            for name, ok in checks.items():
                assert r[name] == ok
                if not ok:
                    gates[phase + ":" + name] += 1
            geom = feasible and gramok and objok
            for c, expected in zip(CONTRACTS, (geom and td, geom and wd, geom and td and wd)):
                assert r["passes_"+c+"_contract"] == expected
            phases = ({"capture_and_gram_seconds"} if r["method"] == "direct_online" else {"capture_seconds", "gram_seconds"}) | {"transfer_and_qp_seconds", "reconstruction_seconds"}
            assert set(r["times"]) == phases
            assert all(x >= 0 for x in r["times"].values())
            assert sum(r["times"].values()) <= r["seconds"] and r["seconds"] > 0
            assert r["peak_allocated_bytes"] <= d["gpu_total_bytes"]
            assert r["peak_allocated_bytes"] - r["incremental_peak_bytes"] == 667744256

    failure_keys = set()
    failure_counts = Counter()
    for f in d["failures"]:
        assert f["kind"] == "contract_gate_failure"
        key = (f["phase"], f["M"], f["seed"], f["method"], canon(f["configuration"]), f["repeat"])
        assert key not in failure_keys and key in rows
        failure_keys.add(key)
        r = rows[key]
        assert not r["passes_joint_contract"]
        assert (f["table"], f["weighted"]) == (r["passes_table_contract"], r["passes_weighted_contract"])
        failure_counts[f["phase"] + (":warmup" if f["repeat"] == 0 else ":nonwarmup")] += 1
    assert failure_keys == {k for k, r in rows.items() if not r["passes_joint_contract"]}

    expected_keys, development, groups = set(), {}, []
    for m in (8, 32):
        for method in METHODS:
            for cfg in configurations(method):
                expected_keys.update(("calibration", m, 5000, method, canon(cfg), rep) for rep in range(3))
            for c in CONTRACTS:
                candidates = []
                for cfg in configurations(method):
                    part = [rows[("calibration", m, 5000, method, canon(cfg), rep)] for rep in range(3)]
                    if all(r["passes_"+c+"_contract"] for r in part):
                        candidates.append((float(np.median([r["seconds"] for r in part[1:]])), cfg))
                choice = min(candidates, key=lambda z: z[0]) if candidates else (None, None)
                assert d["selected"][str(m)][method][c] == choice[1]
                development[m, method, c] = choice
            unique = {}
            for c in CONTRACTS:
                cfg = development[m, method, c][1]
                if cfg is not None:
                    unique.setdefault(canon(cfg), []).append(c)
            for cfg, roles in unique.items():
                for seed in SEEDS:
                    for rep in range(4):
                        key = ("reporting", m, seed, method, cfg, rep)
                        expected_keys.add(key)
                        assert rows[key]["selected_for_contracts"] == roles
            for c in CONTRACTS:
                dev_seconds, cfg = development[m, method, c]
                chosen = [rows[("reporting", m, s, method, canon(cfg), r)] for s in SEEDS for r in range(4)] if cfg is not None else []
                eligible = [r for r in chosen if r["repeat"] and r["passes_"+c+"_contract"]]
                warmups = [r for r in chosen if not r["repeat"] and r["passes_"+c+"_contract"]]
                stats = dict(median_seconds=float(np.median([r["seconds"] for r in eligible])),
                             median_peak_allocated_bytes=float(np.median([r["peak_allocated_bytes"] for r in eligible])),
                             max_peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in eligible),
                             median_incremental_peak_bytes=float(np.median([r["incremental_peak_bytes"] for r in eligible])),
                             by_seed={str(s): dict(qualified_repeats=len(part), median_seconds=float(np.median([r["seconds"] for r in part])),
                                max_peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in part)) for s in SEEDS if (part := [r for r in eligible if r["seed"] == s])}) if eligible else None
                sg = next(g for g in saved["benchmarks"][0]["groups"] if (g["M"], g["method"], g["contract"]) == (m, method, c))
                assert sg["qualified_nonwarmup"] == len(eligible) and sg["expected_nonwarmup"] == 9
                assert sg["qualified_rows_only"] == stats
                groups.append(dict(M=m, method=method, contract=c, configuration=cfg,
                    development_median_seconds=dev_seconds, passed=len(eligible), expected=9,
                    warmups_passed=len(warmups), warmups_expected=3, stats=stats))
    assert expected_keys == set(rows)
    for m in (8, 32):
        coverage = d["coverage"][str(m)]
        for c in CONTRACTS:
            gs = [g for g in groups if (g["M"], g["contract"]) == (m, c)]
            assert coverage["expected_nonwarmup_roles"][c] == 72
            assert coverage["retained_nonwarmup_roles"][c] == sum(9 for g in gs if g["configuration"] is not None)
            assert coverage["successful_nonwarmup_roles"][c] == sum(g["passed"] for g in gs)

    comparisons = []
    for m in (8, 32):
        for c in CONTRACTS:
            gs = [g for g in groups if (g["M"], g["contract"]) == (m, c)]
            baselines = [g for g in gs if not g["method"].startswith("direct_")]
            dev_fastest = min((g for g in baselines if g["configuration"] is not None), key=lambda g:g["development_median_seconds"])
            qualified = [g for g in baselines if g["passed"] == 9]
            b = min(qualified, key=lambda g:g["stats"]["median_seconds"])
            for a in (g for g in gs if g["method"].startswith("direct_")):
                item = dict(M=m, contract=c, direct=a["method"], direct_passed=a["passed"], expected=9,
                    fastest_development_baseline=dev_fastest["method"], fastest_development_baseline_reporting_passed=dev_fastest["passed"],
                    fastest_fully_qualified_reporting_baseline=b["method"], baseline_passed=b["passed"],
                    full_coverage_comparison=a["passed"] == 9)
                if a["passed"] == 9:
                    item.update(direct_seconds=a["stats"]["median_seconds"], baseline_seconds=b["stats"]["median_seconds"],
                        ratio_of_medians=b["stats"]["median_seconds"]/a["stats"]["median_seconds"],
                        paired_batch_ratios={str(s): b["stats"]["by_seed"][str(s)]["median_seconds"]/a["stats"]["by_seed"][str(s)]["median_seconds"] for s in SEEDS},
                        median_peak_ratio=b["stats"]["median_peak_allocated_bytes"]/a["stats"]["median_peak_allocated_bytes"])
                comparisons.append(item)

    fresh = json.loads(subprocess.check_output([sys.executable, str(ROOT / "analyze_language_baselines.py"), str(BASE / "pythia160m.json")], text=True))
    # Artifact path strings differ only if this repository has been relocated.
    assert fresh == saved, "Saved strict analyzer summary differs from fresh output"
    overhead = dict(native_reference_seconds=sum(r["seconds"] for r in d["native_references"].values()),
        weighted_backward_audit_seconds=sum(r["audit_weighted_seconds"] for r in rows.values()),
        dual_reference_audit_seconds=sum(r["audit_dual_seconds"] for r in rows.values()),
        recovery_seconds_all_retained_rows_including_warmups_and_gate_failures=sum(r["seconds"] for r in rows.values()))
    for k, v in overhead.items():
        near(saved["benchmarks"][0]["overhead"][k], v)
    overlaps = [{"seeds": [a, b], "overlapping_examples": len(set(by_seed[a]) & set(by_seed[b]))}
                for a, b in combinations(sorted(by_seed), 2)]
    output = dict(audit_complete=True, artifact_sha256=EXPECTED_SHA, analyzer_sha256=digest(ROOT / "analyze_language_baselines.py"),
        parity_sha256=digest(BASE / "audit_gpu_parity.json"), summary_reproduced_exactly=True,
        source_files_matched=len(manifest), executed_imports_matched=len(d["executed_imports"]),
        populations=dict(calibration_rows=len(d["calibration"]), calibration_warmups=sum(r["warmup"] for r in d["calibration"]),
            reporting_rows=len(d["rows"]), reporting_warmups=sum(r["warmup"] for r in d["rows"]),
            reporting_nonwarmup=sum(not r["warmup"] for r in d["rows"]), contract_roles=432,
            qualified_contract_roles=sum(g["passed"] for g in groups), failure_records=len(d["failures"]),
            runtime_oom_reference_audit_exceptions=0), failures_by_phase=dict(failure_counts),
        failed_numerical_gates=dict(gates), independent_reference_qps=refs, parity_checks=parity_rows,
        batch_identity_overlaps=overlaps, groups=groups, comparisons=comparisons, overhead=overhead,
        issues=[
            "No stored-artifact/source/summary inconsistency found in the performed checks.",
            "Both Direct variants have only 3/9 joint passes at M8 and M32; no unconditional joint speed comparison is supported.",
            "Development seed 5000 shares one sample with seed 5012 and one with 5013. Reporting batches are mutually disjoint; held-out seeds do not mean example-disjoint development/reporting sets.",
            "At M8 weighted/joint, the fastest development baseline is structured_auto, but it passes only 3/9. Ghost is only the descriptive fastest fully reporting-qualified baseline, not a preselected winner.",
        ], limitations=[
            "No native task-gradient tables, emitted directions, or weighted backward vectors were retained. Their vector errors/responses are source-attested and consistency-checked, not independently re-executed.",
            "Only three evaluation batches, one fixed model revision; three timing repetitions per batch do not make nine independent model/data seeds.",
            "Comparisons retain development-selected configurations and all failing rows. Fastest fully qualified reporting baseline is a descriptive comparison, not a reporting-selected configuration or inferential superiority claim.",
            "Setup, successful cleanup, data packing, full campaign wall time, host peak memory, audit peak memory, CUDA reserved/process memory, and optimizer updates are outside or absent from reported recovery accounting.",
            "All allocated GPU memory peaks include common 667744256-byte baseline; independent host task tables additionally occupy 5194334208 bytes (M8) or 20777336832 bytes (M32), outside recovery peaks.",
            "Source hash agreement authenticates retained bytes and internal provenance consistency, not remote execution/completeness. Dependency/runtime versions and tokenizer/pool hashes are recorded, not independently regenerated.",
            "Shared FP64 attention softmax changes forward/backward arithmetic for every arm. Ghost is a pinned task-column kernel adaptation with shared fallback, not an optimized all-pairs implementation.",
            "Positive stored Grams and numerical fidelity are not outward-rounded exact-real network-gradient or common-descent certificates.",
        ])
    rendered = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x") as f:
            f.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
