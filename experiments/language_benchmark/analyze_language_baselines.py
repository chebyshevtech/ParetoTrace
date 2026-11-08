"""Strict, read-only analysis of the frozen language baseline v5/v6 campaigns.

Run with the repository's NumPy environment; neither Torch nor Transformers is
imported. v3/v4 artifacts are deliberately rejected, never reclassified.
The sibling audit_gpu_parity.json is required unless --parity names its copy.
No reporting outcome is used to choose a method configuration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = ("direct_cached", "direct_online", "ghost_columns", "structured_auto",
           "structured_parameter", "structured_token", "native_serial", "native_batched")
CONTRACTS = ("table", "weighted", "joint")
DEV, HELDOUT, REPEATS = 5000, (5011, 5012, 5013), 3
MODEL_REVISIONS = {
    "EleutherAI/pythia-160m-deduped": "c54a0e0b28cc667b6f278803024438d57f847b5d",
    "EleutherAI/pythia-1b-deduped": "d1988145ec1cf1e76d786cd05af9f53e8b1c95cc",
}
OBJECTIVES = {"EleutherAI/pythia-160m-deduped": [8, 32],
              "EleutherAI/pythia-1b-deduped": [8]}
GHOST_COMMIT = "72ac352ae05436275e154b59334e5cb343b96339"
GHOST_SAMPLER = "d6c4c5b815266acbd6958fe36ac4bc0fc40c94b1c6df3cf74f2f05e12983b43c"
# Audited versions, not hashes learned from the artifact being evaluated.
SOURCE_HASHES = {
    "language_baseline_v5.py": "11ba3a3e91bb54b9fbfd2faea7a28018b8af520bbc2d8d10434662e1a1c222d8",
    "native_language_dual_reference_gpu.py": "6c9a4c1f11bb283517940bf56e2345d301da8c70870aa2099308b94c41fea694",
    "native_language_dual_reference.py": "f6f26170c755835b0f3313a0b2917005ccf8afceac1b26fcac864c0f648fe307",
    "language_tokenizer_contract.py": "1d59d9786f4a4ff753f42ec92cebbcd7bf02a9b0d5a3e13761ccfc1a7184c655",
    "language_common.py": "87f9720737656b2650ffcf463464cecc55871bf7403cd62efcfd1d4b83e2bff4",
    "structured_gram.py": "944abbb3809a51076143297d68e542651022394f87264303886e3a430cbe0e30",
    "ghost_adapter.py": "74f4e0706cc31d2721ea4cdeee8aa66495ccd8bb76680c1f42633c7a445f2e7f",
    "attention_precision.py": "a1b2c67ed471e1f0b41eecb1bcde2cb8dc98d7334804bca3d289e87331fbb83e",
    "simplex_qp.py": "512d1b501dc42f5ed26fca5cfad8eb65b38c6b64af58966ac0d30242f32fa8d3",
}
IMPORT_HASHES = {name: SOURCE_HASHES[name] for name in (
    "language_common.py", "native_language_dual_reference.py",
    "native_language_dual_reference_gpu.py", "language_tokenizer_contract.py")}
IMPORT_HASHES.update({
    "v46_factorized_jacobian.py": "4d98d256ab815800a4158ee3a15862bea4d36296ce8e51072d67d63f447668df",
    "v46_factorized_exact.py": "6984919555b4e58947f6701413204eb6d84ecc4f451f63b61e833a104b804305",
    "paretotrace_online_factor.py": "4b8e834d529b2feb9ed76eb81f1e2d530eb83b7453effb03b846c172240c04bd",
    "v46_supervised_head.py": "31abf06743c453c9c12848d7f98233a15738a58a97bd1209287145ccf432a4d8",
})
PARITY_HASH = "6a8bbe76b876ea9f64e79b7c1120de96824f991f1339c690a5550bc7fd4a6eca"
PROFILES = {
    "v5": {"benchmark": "language_baseline_v5.py", "audit": "native_language_dual_reference_gpu.py",
           "benchmark_hash": SOURCE_HASHES["language_baseline_v5.py"],
           "audit_hash": SOURCE_HASHES["native_language_dual_reference_gpu.py"],
           "parity": "language_audit_gpu_parity.py", "parity_hash": PARITY_HASH},
    "v6": {"benchmark": "language_baseline_v6.py", "audit": "native_language_dual_reference_gpu_v6.py",
           "benchmark_hash": "e97cc97e37ddaad4f2ad824e2f1858aec664241b37511f119ae9b8fe3841095f",
           "audit_hash": "d753acbca021be5037342ec92a5228b1aaf934ac44bcc0e99772a043561a1f48",
           "parity": "language_audit_gpu_parity_v6.py",
           "parity_hash": "2e57ff0f19602bca8992df1e93619f01dc5fc8575256da761aaad78b0b146ad9"},
}
ATTENTION_HASHES = (
    "6747c2be076a33cc7abc6f15abdb4208f2a1b6a82cbad70ad8fe54d5fa292520",
    "9d4af3acc11d24a74045d3571609ab3ce19ad1c2c5fa5ad9d0edf7b007289b06",
)
PHASE_TIMES = {"capture_seconds", "gram_seconds", "capture_and_gram_seconds",
               "transfer_and_qp_seconds", "reconstruction_seconds"}
AUDIT_NUMBERS = (
    "emitted_table_direction_relative_error", "emitted_table_direction_absolute_error",
    "emitted_cast_table_direction_relative_error", "emitted_weighted_backward_relative_error",
    "weighted_backward_cast_table_relative_error", "emitted_direction_norm", "table_direction_norm",
    "task_gradient_norm_max", "mixture_cancellation_condition", "cast_table_objective",
)
CONTRACT_TEXT = {
    "table": "Gram2e-5, solver-coefficient simplex1e-10, objective1.001 (additive near stationarity), emitted vs native-task-table direction2e-5; executed FP32 coefficients audited separately",
    "weighted": "same geometry gates, emitted vs separately evaluated weighted native backward2e-5",
    "joint": "both table and weighted contracts; not an either-or success rule",
}


def profile_hashes(version):
    profile = PROFILES[version]
    sources = {name: value for name, value in SOURCE_HASHES.items()
               if name not in ("language_baseline_v5.py", "native_language_dual_reference_gpu.py")}
    sources.update({profile["benchmark"]: profile["benchmark_hash"], profile["audit"]: profile["audit_hash"]})
    imports = {name: value for name, value in IMPORT_HASHES.items() if name != "native_language_dual_reference_gpu.py"}
    imports[profile["audit"]] = profile["audit_hash"]
    return sources, imports


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate JSON key: " + key)
            out[key] = value
        return out
    def constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    def floating(value):
        parsed = float(value)
        require(math.isfinite(parsed), "Nonfinite JSON number: " + value)
        return parsed
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)


def number(value, label, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), "Invalid number: " + label)
    require(not nonnegative or value >= 0, "Negative " + label)
    return float(value)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer: " + label)
    return value


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid digest: " + label)


def close(actual, expected, label, *, scale=1.0, rtol=2e-8):
    number(actual, label)
    require(math.isclose(actual, float(expected), rel_tol=rtol,
                         abs_tol=2e-12 * max(abs(float(scale)), 1e-100)),
            "Inconsistent " + label + f": {actual!r} != {float(expected)!r}")


def boolean(record, key, expected):
    require(type(record.get(key)) is bool and record[key] == bool(expected), "Inconsistent " + key)


def array(value, shape, label):
    a = np.asarray(value, dtype=np.float64)
    require(a.shape == shape and np.isfinite(a).all(), "Invalid shape/finiteness: " + label)
    return a


def configs(method):
    if method.startswith(("direct_", "structured_")):
        base = [dict(row_chunk=r, scratch_mib=s) for r, s in ((64, 32), (256, 32), (64, 128), (256, 128))]
    elif method == "native_batched":
        base = [dict(native_chunk=n, gram_precision=p) for p in ("fp64_tiled", "fp32") for n in (2, 4, 8)]
    elif method == "native_serial":
        base = [dict(gram_precision=p) for p in ("fp64_tiled", "fp32")]
    else:
        base = [{}]
    choices = ("weighted_native",) if method == "direct_online" else ("operator", "weighted_native")
    return [{**c, "reconstruction": r} for c in base for r in choices]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fit_reference(record, m):
    K = array(record["gram"], (m, m), "reference Gram")
    scale = max(float(np.abs(K).max()), 1e-100)
    require(np.max(np.abs(K - K.T)) <= scale * 1e-12, "Asymmetric reference Gram")
    # Verify the local diagnostic solver before importing it.
    require(sha(ROOT / "simplex_qp.py") == SOURCE_HASHES["simplex_qp.py"], "Local QP source changed")
    from simplex_qp import qp
    fit = qp(K, tolerance=2e-12, max_iterations=3000)
    require(fit["converged"], "Stored native-reference QP cannot be resolved")
    for key, field in (("lower", "lower"), ("objective", "objective"), ("qp_gap", "dual_gap")):
        close(record[key], fit[field], "reference " + key, scale=scale)
    number(record["seconds"], "reference seconds", True)
    size = integer(record["host_table_bytes"], "host table bytes", 1)
    require(size % (m * 4) == 0, "Reference host-table byte count is not M x P x FP32")
    return K, fit


def validate_audit(row, K, fit, w):
    """Recompute logged geometry/gates; vector norms themselves are source-attested."""
    m = len(w)
    for key in AUDIT_NUMBERS:
        number(row[key], key, key != "cast_table_objective")
    wc = w.astype(np.float32).astype(np.float64)
    require(np.array_equal(array(row["executed_coefficients"], (m,), "executed coefficients"), wc),
            "Executed coefficients differ from FP32 cast")
    close(row["executed_coefficient_sum"], wc.sum(), "executed coefficient sum")
    close(row["solver_coefficient_sum"], w.sum(), "solver coefficient sum")
    scale = max(float(np.abs(K).max()), 1e-100)
    close(row["cast_table_objective"], wc @ K @ wc, "cast objective", scale=scale)
    norm = row["table_direction_norm"]
    absolute = row["emitted_table_direction_absolute_error"]
    close(row["emitted_table_direction_relative_error"], absolute / max(norm, 1e-100), "table direction error")
    gradnorms = np.sqrt(np.maximum(np.diag(K), 0))
    close(row["task_gradient_norm_max"], gradnorms.max(), "maximum task norm", scale=gradnorms.max())
    close(row["mixture_cancellation_condition"], np.abs(w) @ gradnorms / max(norm, 1e-100), "cancellation condition")
    responses = array(row["executed_native_table_responses"], (m,), "executed responses")
    close(row["executed_native_table_min_response"], responses.min(), "minimum executed response", scale=scale)
    bound = array(row["table_response_error_bound_diagnostic"], (m,), "response diagnostic bound")
    require(np.allclose(bound, K @ w - gradnorms * absolute, rtol=2e-8, atol=2e-12 * scale),
            "Inconsistent response diagnostic bound")
    # Consistency checks allow FP64 contraction-order rounding; these are not
    # new fidelity gates or outward-rounded network-gradient certificates.
    require(abs(norm * norm - float(w @ K @ w)) <= 1e-10 * scale,
            "Table direction norm disagrees with native Gram")
    require(np.all(np.abs(responses - K @ w) <= gradnorms * absolute + 1e-10 * scale),
            "Executed responses violate their recorded direction-error diagnostic")
    require(abs(row["emitted_direction_norm"] - norm) <= absolute + 1e-10 * max(norm, 1e-100),
            "Emitted and table norms violate their recorded direction error")
    denominator = np.sqrt(row["emitted_direction_norm"] ** 2 * max(fit["objective"], 0))
    if denominator > 0:
        close(row["normalized_gain_to_reference_primal"], responses.min() / denominator, "normalized gain")
    else:
        require(row["normalized_gain_to_reference_primal"] is None, "Zero denominator needs null gain")
    table = row["emitted_table_direction_relative_error"] <= 2e-5
    weighted = row["emitted_weighted_backward_relative_error"] <= 2e-5
    boolean(row, "passes_table_direction", table)
    boolean(row, "passes_weighted_backward_direction", weighted)
    return table, weighted


def validate_row(row, K, fit):
    m = row["M"]
    require(row.get("timed_recovery_complete") is True, "Row lacks complete timed recovery")
    seconds = number(row["seconds"], "recovery seconds", True)
    require(seconds > 0, "Zero recovery duration")
    times = row["times"]
    expected = ({"capture_and_gram_seconds"} if row["method"] == "direct_online"
                else {"capture_seconds", "gram_seconds"}) | {"transfer_and_qp_seconds", "reconstruction_seconds"}
    require(set(times) == expected, "Recovery phase coverage mismatch")
    for key, value in times.items():
        number(value, key, True)
    require(sum(times.values()) <= seconds + 1e-7 * max(seconds, 1), "Phase times exceed recovery time")
    peak = integer(row["peak_allocated_bytes"], "peak allocated bytes", 1)
    increment = integer(row["incremental_peak_bytes"], "incremental peak")
    require(increment <= peak, "Incremental peak exceeds absolute peak")
    H = array(row["gram"], (m, m), "recovered Gram")
    w = array(row["weights"], (m,), "solver coefficients")
    u = H @ w
    q = float(w @ u)
    lower = max(0.0, min(q, 2 * float(u.min()) - q))
    scale_h = max(float(np.abs(H).max()), 1e-100)
    require(np.max(np.abs(H - H.T)) <= scale_h * 1e-12, "Asymmetric recovered Gram")
    close(row["objective"], q, "recovered objective", scale=scale_h)
    close(row["qp_gap"], q - lower, "recovered QP gap", scale=scale_h)
    require(q - lower <= 2e-12 * max(float(np.trace(H) / m), 1e-100, q) * 1.05 + 2e-14 * scale_h,
            "Timed recovery contains an unresolved QP")
    if "audit_failure" in row:
        require(isinstance(row["audit_failure"], str) and row["audit_failure"], "Missing audit exception")
        for contract in CONTRACTS:
            boolean(row, "passes_" + contract + "_contract", False)
        return
    for key in ("audit_weighted_seconds", "audit_dual_seconds"):
        number(row[key], key, True)
    td, wd = validate_audit(row, K, fit, w)
    native_u = K @ w
    native_q = float(w @ native_u)
    scale = max(float(np.abs(K).max()), 1e-100)
    error = float(np.linalg.norm(H - K) / max(np.linalg.norm(K), 1e-100))
    feasible = bool(w.min() >= -1e-12 and abs(w.sum() - 1) <= 1e-10)
    relative = fit["lower"] > scale * 1e-10
    excess = native_q - fit["lower"]
    factor = native_q / fit["lower"] if relative else None
    objective_pass = factor <= 1.001 if relative else excess <= 1e-7 * scale
    if relative:
        close(row["native_objective_factor_upper"], factor, "native objective factor")
        expected_scope = "relative native task-table objective"
    else:
        require(row["native_objective_factor_upper"] is None, "Near-stationary row advertises a relative factor")
        expected_scope = "additive gap to native lower bound near stationarity"
    require(row["objective_audit_scope"] == expected_scope, "Objective audit scope mismatch")
    for key, value in (("native_gram_relative_error", error), ("native_objective_excess_above_lower", excess),
                       ("native_reference_lower", fit["lower"]), ("native_reference_qp_gap", fit["dual_gap"]),
                       ("native_table_min_response", native_u.min())):
        close(row[key], value, key, scale=1.0 if key.endswith("relative_error") else scale)
    boolean(row, "passed_weight_feasibility_gate", feasible)
    boolean(row, "passed_native_gram_gate", error <= 2e-5)
    boolean(row, "passed_native_objective_gate", objective_pass)
    geometry = feasible and error <= 2e-5 and objective_pass
    boolean(row, "passes_table_contract", geometry and td)
    boolean(row, "passes_weighted_contract", geometry and wd)
    boolean(row, "passes_joint_contract", geometry and td and wd)
    require(row.get("direction_audit_backend") == "tiled GPU FP64, independent host task table, outside recovery timing",
            "Not the frozen GPU audit backend")


def identity(record):
    require(record.get("complete") is True and record.get("mode") == "reporting" and record.get("quick") is False,
            "Only completed, full reporting v5/v6 artifacts are eligible")
    require(not record.get("invalid") and not record.get("superseded"), "Invalid or superseded artifact")
    # Both source files can coexist in an upload directory. The actually
    # imported audit module, not merely presence in the directory, selects the
    # version profile. Mixed or ambiguous executed-import records are rejected.
    matches = [version for version, p in PROFILES.items() if p["audit"] in record.get("executed_imports", {})]
    require(len(matches) == 1, "Missing or ambiguous executed audit version")
    version = matches[0]
    sources, imports = profile_hashes(version)
    model = record["model"]["model"]
    require(model in MODEL_REVISIONS and record["model"]["revision"] == MODEL_REVISIONS[model], "Model/revision mismatch")
    require(record["objectives"] == OBJECTIVES[model], "Scheduled objective population changed")
    require(record["development_seed"] == DEV and record["reporting_seeds"] == list(HELDOUT), "Seed population changed")
    require(record["batch"] == 32 and record["model_dtype"] == "float32" and record["tf32"] is False, "Arithmetic/batch changed")
    require(record["data"] == read(ROOT / "language_data_manifest.json"), "Frozen data/tokenizer identity mismatch")
    for field, expected in (("sources", sources), ("executed_imports", imports)):
        require(isinstance(record[field], dict), "Missing source provenance")
        for name, value in expected.items():
            require(record[field].get(name) == value, "Source identity mismatch: " + name)
    require(set(record["executed_imports"]) == set(imports), "Unexpected executed-import population")
    require(record["ghost_commit"] == GHOST_COMMIT and record["ghost_sampler_sha256"] == GHOST_SAMPLER,
            "Ghost commit/kernel identity mismatch")
    attention = record["attention_precision"]
    require(attention["softmax_dtype"] == "float64" and
            (attention["original_source_sha256"], attention["executed_source_sha256"]) == ATTENTION_HASHES,
            "Shared attention arithmetic changed")
    integer(attention["calls"], "attention calls", 1)
    tok = record["tokenizer_contract"]
    require(tok["target_tokenizer_model"] == model and tok["target_revision"] == MODEL_REVISIONS[model] and
            tok["data_tokenizer_model"] == record["data"]["tokenizer_model"] and
            tok["data_tokenizer_revision"] == record["data"]["tokenizer_revision"] and tok["generation_used"] is False,
            "Tokenizer provenance mismatch")
    require(tok["data_padding_eos_id"] == 0, "Frozen data padding ID changed")
    digest(tok["encoding_backend_sha256"], "tokenizer backend")
    require(isinstance(record["gpu"], str) and "A100" in record["gpu"], "Not the scheduled A100 hardware")
    integer(record["gpu_total_bytes"], "GPU capacity", 1)
    for key in ("torch", "cuda", "transformers"):
        require(isinstance(record[key], str) and record[key], "Missing environment identity: " + key)
    require(record["contracts"] == CONTRACT_TEXT,
            "Wrong contract schema; historical results are not reclassified")
    return model, version


def validate_parity(record, version="v5"):
    require(record.get("complete") is True and record["seed"] == 4999, "Missing completed development-only GPU audit parity")
    profile = PROFILES[version]
    expected = {"native_language_dual_reference.py": SOURCE_HASHES["native_language_dual_reference.py"],
                profile["audit"]: profile["audit_hash"], profile["parity"]: profile["parity_hash"]}
    require(record["sources"] == expected, "Parity source identity mismatch")
    require([r["weighting"] for r in record["rows"]] == ["uniform", "qp", "onehot"], "Parity weighting coverage mismatch")
    K, fit = fit_reference(record["reference"], 8)
    for row in record["rows"]:
        require(row.get("complete") is True, "Incomplete parity row")
        for key in ("cpu_seconds", "gpu_seconds"):
            number(row[key], key, True)
        a, b = row["cpu"], row["gpu"]
        numeric = []
        for key, value in a.items():
            require(key in b, "Missing parity field: " + key)
            if type(value) is bool:
                boolean(b, key, value)
            elif isinstance(value, (int, float, list)) and value is not None:
                av = np.asarray(value, dtype=float)
                bv = np.asarray(b[key], dtype=float)
                require(av.shape == bv.shape and np.isfinite(av).all() and np.isfinite(bv).all() and
                        np.allclose(av, bv, rtol=1e-8, atol=1e-10), "CPU/GPU parity failed: " + key)
                numeric.append(key)
        require(row["numeric_fields"] == numeric, "Parity numeric-field coverage changed")
        w = {"uniform": np.full(8, 1 / 8), "qp": fit["weights"], "onehot": np.eye(8)[0]}[row["weighting"]]
        decisions = []
        for audit in (a, b):
            td, wd = validate_audit(audit, K, fit, w)
            scale = max(float(np.abs(K).max()), 1e-100)
            q = float(w @ K @ w)
            objective = q / fit["lower"] <= 1.001 if fit["lower"] > scale * 1e-10 else q - fit["lower"] <= 1e-7 * scale
            decisions.append({"passes_table_contract": bool(objective and td),
                              "passes_weighted_contract": bool(objective and wd),
                              "passes_joint_contract": bool(objective and td and wd)})
        require(decisions[0] == decisions[1] == row["contracts"], "Parity contract mismatch")
    return {"complete": True, "seed": 4999, "weightings": 3,
            "native_reference_seconds": record["reference"]["seconds"],
            "cpu_seconds": sum(r["cpu_seconds"] for r in record["rows"]),
            "gpu_seconds": sum(r["gpu_seconds"] for r in record["rows"]),
            "scope": "Separate prerequisite diagnostic; not recovery-method speed or setup time."}


def trial_key(row, phase, objectives):
    m, seed, repeat, method = (row[k] for k in ("M", "seed", "repeat", "method"))
    require(type(m) is int and m in objectives and method in METHODS, "Unknown objective/method")
    require(type(seed) is int and seed in ((DEV,) if phase == "calibration" else HELDOUT), "Trial uses wrong seed")
    require(type(repeat) is int and repeat in range(3 if phase == "calibration" else 4), "Repeat population changed")
    cfg = row["configuration"]
    require(cfg in configs(method), "Unknown candidate configuration")
    return phase, m, seed, method, canonical(cfg), repeat


def validate(record, parity):
    require(isinstance(record, dict) and isinstance(parity, dict), "Artifact roots must be JSON objects")
    model, version = identity(record)
    parity_summary = validate_parity(parity, version)
    require(record["gpu"] == parity["gpu"] and record["torch"] == parity["torch"], "Parity and reporting environment differ")
    objectives = record["objectives"]
    require(set(record["selected"]) == set(map(str, objectives)) and set(record["coverage"]) == set(map(str, objectives)),
            "Selection/coverage objective population mismatch")
    references = record["native_references"]
    require(set(references) == {f"{m}:{s}" for m in objectives for s in (DEV, *HELDOUT)}, "Reference-state coverage mismatch")
    fits, failed_references = {}, {}
    table_sizes = set()
    for name, ref in references.items():
        m, seed = map(int, name.split(":"))
        require(type(ref.get("complete")) is bool, "Missing reference terminal status")
        if ref["complete"]:
            fits[m, seed] = fit_reference(ref, m)
            table_sizes.add(ref["host_table_bytes"] // m)
        else:
            require(ref.get("phase") == "reference" and ref.get("M") == m and ref.get("seed") == seed and ref.get("error"),
                    "Malformed reference failure")
            failed_references[m, seed] = ref
    require(len(table_sizes) <= 1, "Reference parameter population changed between states")
    rows, runtime, gate_failures, ref_failures = {}, {}, {}, {}
    ids_by_seed = {}
    for phase, field in (("calibration", "calibration"), ("reporting", "rows")):
        for row in record[field]:
            key = trial_key(row, phase, objectives)
            require(key not in rows, "Duplicate timed row: " + str(key))
            boolean(row, "warmup", row["repeat"] == 0)
            ids = row["identities"]
            require(isinstance(ids, list) and len(ids) == 32 and len(set(ids)) == 32, "Batch identity coverage mismatch")
            for value in ids:
                digest(value, "sample identity")
            previous = ids_by_seed.setdefault(row["seed"], ids)
            require(ids == previous, "Methods/objective counts did not use the same seed's batch")
            require((row["M"], row["seed"]) in fits, "Timed row has no successful independent reference")
            validate_row(row, *fits[row["M"], row["seed"]])
            require(row["peak_allocated_bytes"] <= record["gpu_total_bytes"], "Allocated peak exceeds GPU capacity")
            rows[key] = row
    for failure in record["failures"]:
        phase = failure.get("phase")
        if phase == "reference":
            key = failure["M"], failure["seed"]
            require(key in failed_references and key not in ref_failures, "Unexpected/duplicate reference failure")
            require(failure == {k: v for k, v in failed_references[key].items() if k != "complete"}, "Reference failure was not preserved")
            ref_failures[key] = failure
            continue
        require(phase in ("calibration", "reporting"), "Fatal/setup failure in purported completed artifact")
        key = trial_key(failure, phase, objectives)
        kind = failure.get("kind")
        if kind == "contract_gate_failure":
            require(key in rows and key not in gate_failures, "Orphan/duplicate gate failure")
            row = rows[key]
            require(not row["passes_joint_contract"], "Passing row has a gate-failure record")
            boolean(failure, "table", row["passes_table_contract"])
            boolean(failure, "weighted", row["passes_weighted_contract"])
            gate_failures[key] = failure
        else:
            require(kind in ("CUDA_OOM", "runtime_failure") and failure.get("error"), "Unknown runtime failure")
            require(key not in rows and key not in runtime, "Duplicate/contradictory terminal trial")
            diag = failure["cost_diagnostics"]
            number(diag["attempt_wall_seconds_including_cleanup"], "failed attempt wall time", True)
            if diag.get("peak_reset"):
                require(diag.get("stage") in ("capture", "gram", "transfer_and_qp", "reconstruction", "independent_audit"), "Unknown failure stage")
                require(set(diag["times"]).issubset(PHASE_TIMES), "Unknown failure phase timing")
                for name, value in diag["times"].items():
                    number(value, name, True)
                require(sum(diag["times"].values()) <= diag["attempt_wall_seconds_including_cleanup"] + 1e-7,
                        "Completed phase times exceed failed-attempt wall time")
                peak = integer(diag["peak_allocated_bytes_until_failure"], "failed peak")
                base = integer(diag["base_allocated_bytes"], "failed baseline")
                require(peak >= base, "Failed peak below baseline allocation")
            runtime[key] = failure
    require(set(ref_failures) == set(failed_references), "Reference failure missing from failure log")
    require(set(gate_failures) == {k for k, r in rows.items() if not r["passes_joint_contract"]}, "Failed timed row missing from failure log")
    seen = set()
    selection = {}
    calibration_skipped = 0
    for m in objectives:
        selected = record["selected"][str(m)]
        require(set(selected) == set(METHODS), "Selected method coverage mismatch")
        for method in METHODS:
            require(set(selected[method]) == set(CONTRACTS), "Selected contract coverage mismatch")
            candidates = {c: [] for c in CONTRACTS}
            if (m, DEV) in fits:
                for cfg in configs(method):
                    trials = []
                    for repeat in range(3):
                        key = "calibration", m, DEV, method, canonical(cfg), repeat
                        require(key in rows or key in runtime, "Missing calibration terminal trial: " + str(key))
                        seen.add(key)
                        if key in runtime:
                            calibration_skipped += 2 - repeat
                            break
                        trials.append(rows[key])
                    if len(trials) == 3:
                        for contract in CONTRACTS:
                            if all(r["passes_" + contract + "_contract"] for r in trials):
                                candidates[contract].append((float(np.median([r["seconds"] for r in trials[1:]])), cfg))
            else:
                calibration_skipped += 3 * len(configs(method))
            for contract in CONTRACTS:
                expected = min(candidates[contract], key=lambda x: x[0])[1] if candidates[contract] else None
                require(selected[method][contract] == expected, "Selection differs from development-only rule: " + str((m, method, contract)))
                selection[m, method, contract] = expected
    roles = {}
    for m in objectives:
        retained = Counter({c: 0 for c in CONTRACTS})
        successful = Counter({c: 0 for c in CONTRACTS})
        for seed in HELDOUT:
            if (m, seed) not in fits:
                continue
            for method in METHODS:
                unique = {}
                for contract in CONTRACTS:
                    cfg = selection[m, method, contract]
                    if cfg is not None:
                        unique.setdefault(canonical(cfg), []).append(contract)
                for cfg, contracts in unique.items():
                    for repeat in range(REPEATS + 1):
                        key = "reporting", m, seed, method, cfg, repeat
                        require(key in rows or key in runtime, "Missing reporting terminal trial: " + str(key))
                        seen.add(key)
                        roles[key] = contracts
                        if key in rows:
                            row = rows[key]
                            require(row.get("selected_for_contracts") == contracts, "Reporting roles differ from development selection")
                            if repeat:
                                for contract in contracts:
                                    retained[contract] += 1
                                    successful[contract] += int(row["passes_" + contract + "_contract"])
        expected_coverage = {"expected_nonwarmup_roles": {c: len(METHODS) * len(HELDOUT) * REPEATS for c in CONTRACTS},
                             "retained_nonwarmup_roles": dict(retained), "successful_nonwarmup_roles": dict(successful)}
        require(record["coverage"][str(m)] == expected_coverage, "Stored reporting coverage does not match terminal outcomes")
    require(seen == set(rows) | set(runtime), "Unexpected trials, repetitions after calibration failure, or unselected reporting configuration")
    groups = []
    for m in objectives:
        for method in METHODS:
            for contract in CONTRACTS:
                cfg = selection[m, method, contract]
                eligible, states, warmups = [], Counter(), Counter()
                for seed in HELDOUT:
                    for repeat in range(REPEATS + 1):
                        tally = warmups if repeat == 0 else states
                        if cfg is None:
                            tally["no_development_selection"] += 1
                        elif (m, seed) not in fits:
                            tally["reference_unavailable"] += 1
                        else:
                            key = "reporting", m, seed, method, canonical(cfg), repeat
                            if key in runtime:
                                tally[runtime[key]["kind"]] += 1
                            elif rows[key]["passes_" + contract + "_contract"]:
                                tally["qualified"] += 1
                                if repeat:
                                    eligible.append(rows[key])
                            else:
                                tally["audit_exception" if "audit_failure" in rows[key] else "contract_failed"] += 1
                stats = None
                if eligible:
                    stats = {"median_seconds": float(np.median([r["seconds"] for r in eligible])),
                             "median_peak_allocated_bytes": float(np.median([r["peak_allocated_bytes"] for r in eligible])),
                             "max_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in eligible),
                             "median_incremental_peak_bytes": float(np.median([r["incremental_peak_bytes"] for r in eligible])),
                             "by_seed": {str(seed): {"qualified_repeats": len(part),
                                 "median_seconds": float(np.median([r["seconds"] for r in part])),
                                 "max_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in part)}
                                 for seed in HELDOUT if (part := [r for r in eligible if r["seed"] == seed])}}
                groups.append({"M": m, "method": method, "contract": contract, "development_selected_configuration": cfg,
                               "expected_nonwarmup": 9, "qualified_nonwarmup": len(eligible),
                               "failed_or_unavailable_nonwarmup": 9 - len(eligible), "nonwarmup_status_counts": dict(states),
                               "warmup_status_counts": dict(warmups), "complete_contract_coverage": len(eligible) == 9,
                               "qualified_rows_only": stats})
    failure_counts = Counter(f.get("kind", "reference_failure") for f in record["failures"])
    failure_phases = {}
    for failure in record["failures"]:
        phase = failure["phase"]
        if phase == "reporting":
            phase += "_warmup" if failure["repeat"] == 0 else "_nonwarmup"
        failure_phases.setdefault(phase, Counter())[failure.get("kind", "reference_failure")] += 1
    all_rows = list(rows.values())
    audits = [r for r in all_rows if "audit_weighted_seconds" in r and "audit_dual_seconds" in r]
    overhead = {"setup_seconds": None, "setup_status": "Successful setup duration is not logged; unavailable, not zero.",
                "native_reference_seconds": sum(ref["seconds"] for ref in references.values() if ref["complete"]),
                "native_reference_failures_without_duration": len(failed_references),
                "weighted_backward_audit_seconds": sum(r["audit_weighted_seconds"] for r in audits),
                "dual_reference_audit_seconds": sum(r["audit_dual_seconds"] for r in audits),
                "timed_rows_missing_complete_audit_cost": len(all_rows) - len(audits),
                "recovery_seconds_all_retained_rows_including_warmups_and_gate_failures": sum(r["seconds"] for r in all_rows),
                "failed_attempt_wall_seconds_until_exception_including_initial_cleanup":
                    sum(f["cost_diagnostics"]["attempt_wall_seconds_including_cleanup"] for f in runtime.values()),
                "scope": "Separate recorded components, not complete campaign wall time; successful cleanup/data preparation, failed reference/audit costs and setup are incompletely logged."}
    return {"validated": True, "benchmark_version": version, "model": record["model"], "objectives": objectives,
            "failure_counts": dict(failure_counts), "raw_failure_records": len(record["failures"]),
            "failure_counts_by_phase": {phase: dict(counts) for phase, counts in failure_phases.items()},
            "audit_exception_rows": sum("audit_failure" in r for r in all_rows),
            "all_contract_role_counts": {"expected": len(objectives) * len(METHODS) * len(CONTRACTS) * 9,
                "qualified": sum(g["qualified_nonwarmup"] for g in groups),
                "failed_or_unavailable": sum(g["failed_or_unavailable_nonwarmup"] for g in groups)},
            "trial_counts": {"calibration_rows": len(record["calibration"]), "reporting_rows": len(record["rows"]),
                "runtime_failure_records": len(runtime), "calibration_repeats_not_scheduled_after_terminal_failure_or_missing_reference": calibration_skipped,
                "reporting_warmup_rows": sum(r["warmup"] for r in record["rows"]),
                "reporting_nonwarmup_rows": sum(not r["warmup"] for r in record["rows"])},
            "environment": {k: record[k] for k in ("gpu", "gpu_total_bytes", "torch", "cuda", "transformers")},
            "validated_identity": {"sources": profile_hashes(version)[0], "executed_imports": profile_hashes(version)[1],
                "data_manifest_sha256": sha(ROOT / "language_data_manifest.json"),
                "dataset_revision": record["data"]["revision"], "pool_sha256": record["data"]["pool_sha256"],
                "tokenizer_backend_sha256": record["tokenizer_contract"]["encoding_backend_sha256"],
                "ghost_commit": record["ghost_commit"], "ghost_sampler_sha256": record["ghost_sampler_sha256"]},
            "parity_prerequisite": parity_summary, "groups": groups, "overhead": overhead,
            "negative_executed_response_rows": sum(r.get("executed_native_table_min_response", 0) < 0 for r in all_rows),
            "ghost_scope": "Pinned upstream Ghost kernels adapted to task-mean columns, with disclosed shared fallback; not an optimized all-pairs Ghost implementation or unmodified Neural Tangents.",
            "limitations": [
                "Timing/peak summaries condition on the named contract passing. Failures and unavailable roles remain in denominators; incomplete coverage cannot support an unconditional speed superiority claim.",
                "Three repeated timings on each of three batches are not nine independent data/model seeds. Per-seed summaries are included; no confidence interval or reporting-selected winner is computed.",
                "Stored Grams, QP geometry, coefficient casts, diagnostic consistency and gates are checked. Native gradient tables and emitted vectors were not retained, so their norms/responses cannot be independently re-executed from JSON.",
                "Provenance is checked against frozen hashes and revisions; hashes in a self-reported artifact do not prove remote execution or cryptographically establish completeness. Missing scheduled terminal records are rejected.",
                "Batch identities are checked for equality and uniqueness, but sampling membership cannot be reconstructed without the frozen dataset pool. Tokenizer backend digest is recorded, not independently regenerated.",
                "Fidelity pass does not certify common descent or exact-real network gradients. Negative executed native-table responses are retained.",
                "Original v3/v4 classifications are unchanged. v5 and v6 use separate frozen source/parity identities; partial or superseded development artifacts are ineligible.",
            ]}


def analyze(path, parity_path=None):
    path = Path(path)
    parity_path = Path(parity_path) if parity_path else path.parent / "audit_gpu_parity.json"
    summary = validate(read(path), read(parity_path))
    summary["artifact"] = {"path": str(path.resolve()), "sha256": sha(path)}
    summary["parity_artifact"] = {"path": str(parity_path.resolve()), "sha256": sha(parity_path)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--parity", type=Path, help="Completed matching-version CPU/GPU audit parity artifact; default: each input's sibling audit_gpu_parity.json")
    parser.add_argument("--output", type=Path, help="Optional new JSON summary; refuses to overwrite any existing file")
    args = parser.parse_args()
    try:
        if args.output:
            require(not args.output.exists(), "Refusing to overwrite summary: " + str(args.output))
        summaries = [analyze(path, args.parity) for path in args.artifacts]
        require(len({s["model"]["model"] for s in summaries}) == len(summaries), "Duplicate model artifacts")
        require(len({s["benchmark_version"] for s in summaries}) == 1, "Do not combine reporting artifacts from different versions")
        result = {"validated": True, "analyzer_sha256": sha(__file__), "benchmarks": summaries}
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as handle:
                handle.write(rendered)
        else:
            print(rendered, end="")
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, OSError, OverflowError) as error:
        print(json.dumps({"validated": False, "error": str(error)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
