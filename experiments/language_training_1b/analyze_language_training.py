"""Read-only analysis of a complete, frozen 36-run language training campaign.

Uses the audited campaign validator and frozen source hashes. Dataset loading
and checkpoint verification are CPU-only; no model is constructed. Retained
archives are fully verified. Selected bytes may be absent only under the frozen
final-only policy and its explicit verification receipt. No partial campaign
is eligible. Optional output must be a new JSON file.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import statistics
import sys

import language_campaign as campaign
from checkpoint_archive import sha

# Audited sources, never hashes inferred from the result being evaluated.
SOURCE_HASHES = {
    'attention_precision.py': 'a1b2c67ed471e1f0b41eecb1bcde2cb8dc98d7334804bca3d289e87331fbb83e',
    'checkpoint_archive.py': '168cdf882f2e97f06acb44503b4e9a18b15d0c7a383f1839d84a9d5320b3f56e',
    'drive_checkpoint_slots.py': '98f3e1e1cfde7a4bf4378caa53ba3e0f61cf005f1fd7107e6cb4f5197ca9e830',
    'ghost_adapter.py': '74f4e0706cc31d2721ea4cdeee8aa66495ccd8bb76680c1f42633c7a445f2e7f',
    'language_baseline_v6.py': 'e97cc97e37ddaad4f2ad824e2f1858aec664241b37511f119ae9b8fe3841095f',
    'language_campaign.py': 'fa94c89a0db3a58a6cbd96b79a9a233a1a58215898f84f2315cf17ac5428a424',
    'language_checkpoint_storage.py': '9eb4cf25d4d77c7453e9387fe546d82a308faef472929a536f6b6b87bd293239',
    'language_common.py': '87f9720737656b2650ffcf463464cecc55871bf7403cd62efcfd1d4b83e2bff4',
    'language_retained_manifest.json': '1ceeae4186e436a84a26febe1c11b2052d3a77a419839e6e0d7efc275508f853',
    'language_retained_v31.py': '1b76f16c1113d98fb7cb1ac90197015e08a2bb9ebed3906f59bb6be42da22992',
    'language_tokenizer_contract.py': '1d59d9786f4a4ff753f42ec92cebbcd7bf02a9b0d5a3e13761ccfc1a7184c655',
    'language_training_methods.py': 'd3c258f965aa336b537a38a9cc3dd583701864e59179dbfbf501dd5eeaaeaba6',
    'language_training_rebuttal.py': '8c43e059db44c0c17d319fd8c154f630e81e98fb85e7f974b03918bf5ceeca93',
    'native_language_dual_reference.py': 'f6f26170c755835b0f3313a0b2917005ccf8afceac1b26fcac864c0f648fe307',
    'native_language_dual_reference_gpu_v6.py': 'd753acbca021be5037342ec92a5228b1aaf934ac44bcc0e99772a043561a1f48',
    'paretotrace_online_factor.py': '4b8e834d529b2feb9ed76eb81f1e2d530eb83b7453effb03b846c172240c04bd',
    'retire_completed_language.py': '7883e53662b4939f3fd2906310c349f216d7c8a9b59b4d8d5a68fd0598bd1aed',
    'simplex_qp.py': '512d1b501dc42f5ed26fca5cfad8eb65b38c6b64af58966ac0d30242f32fa8d3',
    'structured_gram.py': '944abbb3809a51076143297d68e542651022394f87264303886e3a430cbe0e30',
    'v46_factorized_exact.py': '6984919555b4e58947f6701413204eb6d84ecc4f451f63b61e833a104b804305',
    'v46_factorized_jacobian.py': '4d98d256ab815800a4158ee3a15862bea4d36296ce8e51072d67d63f447668df',
    'v46_supervised_head.py': '31abf06743c453c9c12848d7f98233a15738a58a97bd1209287145ccf432a4d8',
}
METHODS = tuple(campaign.METHODS)
RATES = tuple(campaign.RATES)
SEEDS = tuple(campaign.SEEDS)
# For df=2, F(t)=1/2+t/(2*sqrt(t*t+2)); F(t)=.975 gives this value.
T95_DF2 = math.sqrt(2 * .95**2 / (1 - .95**2))
METRIC_KEYS = ("mean_ce", "worst_ce", "mean_answer_exact_match", "worst_answer_exact_match")
WALL_SCOPE = "main through latest recorded event, including loading, preparation, audits, evaluation and earlier checkpoint copies; excludes imports/final persistence; resumed uncheckpointed work may be missing"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    def floating(value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite JSON number: " + value)
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=reject, parse_float=floating)


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Invalid nonnegative number: " + label)
    return float(value)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer: " + label)
    return value


def describe(values):
    require(len(values) == 3 and all(math.isfinite(v) for v in values), "Three finite paired seeds required")
    return {"n": 3, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "minimum": min(values), "maximum": max(values)}


def process_cost(root, label, trainer_attempts):
    path = root / "process_attempts" / (label + ".json")
    if not path.exists():
        return {"artifact_sha256": None, "attempts": None, "known_wall_seconds_sum": 0.,
                "wall_seconds_sum": None, "unknown_cost_reasons": ["Process-attempt log missing"],
                "recorded_attempt_count": 0, "untimed_recorded_attempt_count": None}
    attempts = read(path)
    require(isinstance(attempts, list) and attempts, "Empty process-attempt log: " + label)
    known, untimed, reasons = 0., 0, []
    for index, row in enumerate(attempts):
        number(row["start_utc_seconds"], "attempt start")
        require(type(row["complete"]) is bool and type(row["resume"]) is bool, "Invalid attempt state")
        if row["complete"]:
            require(type(row["returncode"]) is int, "Invalid process return code")
            known += number(row["subprocess_wall_seconds"], "subprocess wall")
            number(row["end_utc_seconds"], "attempt end")
        else:
            require(not any(k in row for k in ("returncode", "subprocess_wall_seconds", "end_utc_seconds")), "Inconsistent unfinished attempt")
            untimed += 1
            reasons.append(f"Attempt {index + 1} has no completed monotonic wall measurement")
    if len(attempts) < trainer_attempts:
        reasons.append("Fewer process records than checkpointed trainer attempts")
    return {"artifact_sha256": sha(path), "attempts": attempts, "known_wall_seconds_sum": known,
            "wall_seconds_sum": None if reasons else known, "unknown_cost_reasons": reasons,
            "recorded_attempt_count": len(attempts), "untimed_recorded_attempt_count": untimed}


def run_cost(root, label, result):
    update = number(result["update_seconds"], "update seconds")
    prep = number(result["preparation_seconds"], "preparation seconds")
    wall = number(result["recorded_wall_seconds"], "trainer recorded wall")
    attempts = integer(result["attempts"], "trainer attempts", 1)
    require(result["wall_scope"] == WALL_SCOPE, "Changed trainer wall scope")
    require(wall + 1e-6 >= update + prep, "Trainer wall below recorded update/preparation cost")
    uninterrupted = result["uninterrupted_training_evaluation_seconds"]
    if attempts == 1:
        require(uninterrupted == wall, "Inconsistent uninterrupted trainer wall")
    else:
        require(uninterrupted is None, "Resumed run labeled uninterrupted")
    events = []
    last_wall, last_update = -1., -1.
    for row in result["history"]:
        event_wall = number(row["recorded_wall_seconds"], "validation event wall")
        event_update = number(row["update_seconds"], "validation event update seconds")
        require(last_wall <= event_wall <= wall + 1e-6 and last_update <= event_update <= update + 1e-6,
                "Nonmonotonic or excessive validation-event cost")
        events.append({k: row[k] for k in ("after_updates", "recorded_wall_seconds", "update_seconds")})
        last_wall, last_update = event_wall, event_update
    require(math.isclose(last_update, update, rel_tol=1e-10, abs_tol=1e-7), "Last validation update cost differs")
    process = process_cost(root, label, attempts)
    if process["wall_seconds_sum"] is not None:
        require(process["wall_seconds_sum"] + 1e-6 >= wall, "Process wall below trainer recorded wall")
    return {"update_seconds": update, "preparation_seconds": prep, "update_and_preparation_seconds": update + prep,
            "trainer_recorded_wall_seconds": wall, "trainer_attempts": attempts,
            "uninterrupted_training_evaluation_seconds": uninterrupted,
            "validation_checkpoint_events": events, "process_attempts": process,
            "retirement_wall_seconds": None, "retirement_wall_status": "Not recorded by the frozen retirement helper; excluded from subprocess wall",
            "peak_update_allocated_bytes": integer(result["peak_update_allocated_bytes"], "update allocated peak"),
            "periodic_scheduled_opportunities": result["validated_periodic_scheduled_opportunities"],
            "periodic_refinements_applied": result["periodic_refinements"],
            "gram_captures": result["gram_captures"], "post_update_forwards": result["post_update_forwards"]}


def summarize_cost(rows):
    costs = [r["costs"] for r in rows]
    keys = ("update_seconds", "preparation_seconds", "update_and_preparation_seconds", "trainer_recorded_wall_seconds",
            "periodic_scheduled_opportunities", "periodic_refinements_applied", "gram_captures", "post_update_forwards")
    result = {key + "_sum": sum(c[key] for c in costs) for key in keys}
    result["runs"] = len(rows)
    result["peak_update_allocated_bytes_max"] = max(c["peak_update_allocated_bytes"] for c in costs)
    result["known_process_attempt_wall_seconds_sum"] = sum(c["process_attempts"]["known_wall_seconds_sum"] for c in costs)
    result["runs_with_unknown_process_costs"] = [r["label"] for r in rows if r["costs"]["process_attempts"]["wall_seconds_sum"] is None]
    result["process_attempt_wall_seconds_sum"] = None if result["runs_with_unknown_process_costs"] else result["known_process_attempt_wall_seconds_sum"]
    result["retirement_wall_seconds_sum"] = None
    return result


def validate_sources(root):
    sources = read(root / "campaign_sources.json")
    require(isinstance(sources, dict), "Invalid frozen campaign source map")
    expected_python = {n for n in SOURCE_HASHES if n.endswith(".py")}
    require(expected_python <= set(sources), "Incomplete campaign source coverage")
    for name, expected in SOURCE_HASHES.items():
        require(sha(root / name) == expected, "Unaudited source: " + name)
        if name.endswith(".py"):
            require(sources[name] == expected, "Changed frozen source identity: " + name)
    for name, expected in sources.items():
        require(Path(name).name == name and name.endswith(".py"), "Unsafe campaign source reference")
        require(sha(root / name) == expected, "Frozen additional source changed: " + name)
    # The validator's CPU import chain must be the same audited implementation.
    for name in ("language_campaign", "checkpoint_archive", "retire_completed_language", "language_checkpoint_storage", "drive_checkpoint_slots", "language_common", "v46_supervised_head", "simplex_qp"):
        module = importlib.import_module(name)
        require(sha(module.__file__) == SOURCE_HASHES[name + ".py"], "Loaded validator dependency differs: " + name)
    return sources


def validate_protocol(root, protocol, pilot, manifest):
    fixed = {"version": "language_training_v1", "model": "EleutherAI/pythia-1b-deduped", "arms": list(METHODS),
             "development_seed": 49100, "reporting_seeds": list(SEEDS), "learning_rates": list(RATES),
             "batch_size": 32, "validation_every": 100}
    for key, value in fixed.items():
        require(protocol[key] == value, "Changed frozen protocol field: " + key)
    for key in ("development_steps", "reporting_steps"):
        require(integer(protocol[key], key, 100) % 100 == 0, "Budget is not a multiple of 100")
    require(protocol["completed_checkpoint_policy"] in ("keep", "vm", "final-only"), "Unknown checkpoint policy")
    require(protocol["checkpoint_backup"] in ("slots", "metadata"), "Unknown checkpoint backup policy")
    if protocol["checkpoint_backup"] == "slots":
        integer(protocol["checkpoint_slot_budget_bytes"], "slot byte budget", 1)
    else:
        require(protocol["checkpoint_slot_budget_bytes"] is None, "Metadata policy cannot claim a model-slot budget")
    require(protocol["pilot_sha256"] == sha(root / "training_preflight.json"), "Pilot hash differs")
    require(pilot["complete"] is True and pilot["selected"] == protocol["systems"], "Pilot selection differs")
    require(set(protocol["systems"]) == set(METHODS), "Pilot arm coverage")
    require(pilot["seed"] == 49100 and pilot["batch"] == 32 and pilot["warmup_updates"] == pilot["measured_updates"] == 10,
            "Pilot development identity")
    require(pilot["data"] == manifest, "Training and pilot data differ")
    require(pilot["model"] == read(root / "data/pythia-1b-deduped_manifest.json"), "Training and pilot model differ")
    for method in METHODS:
        choice = protocol["systems"][method]
        require(choice["method"] == method, "Pilot selected another method")
        options = choice.get("method_options", {})
        require(not options or (method == "periodic" and set(options) == {"periodic_backend"} and options["periodic_backend"] in ("native", "factor")),
                "Pilot changed fixed method defaults")
        eligible = [r for r in pilot["rows"] if r["configuration"]["method"] == method and r["complete"] is True
                    and integer(r["peak_allocated_bytes"], "pilot peak") <= 34 * 2**30]
        require(eligible, "No eligible pilot configuration: " + method)
        winner = min(eligible, key=lambda r: number(r["mean_update_and_preparation_seconds"], "pilot update/preparation mean"))
        require(choice == winner["configuration"], "Pilot did not select its fastest eligible configuration")


def validate_marker(root, phase, runs):
    marker = read(root / (phase + "_complete.json"))
    require(marker["complete"] is True and marker["runs"] == runs and len(runs) == 18, "Incomplete or stale " + phase + " completion map")
    for key, name in (("protocol_sha256", "protocol.json"), ("sources_sha256", "campaign_sources.json"), ("selection_sha256", "selection.json")):
        require(marker[key] == sha(root / name), "Changed " + phase + " marker " + key)


def analyze(root):
    root = Path(root)
    sources = validate_sources(root)
    protocol, pilot, selection = (read(root / name) for name in ("protocol.json", "training_preflight.json", "selection.json"))
    _, manifest = campaign.data_contract(root)
    validate_protocol(root, protocol, pilot, manifest)
    protocol_sha = sha(root / "protocol.json")
    require(selection["protocol_sha256"] == protocol_sha and set(selection["rates"]) == set(selection["candidates"]) == set(METHODS), "Selection identity/coverage")
    stages, maps, pairing, common_identity = {p: [] for p in ("development", "confirmatory")}, {}, {}, None

    def validate_run(method, seed, lr, phase):
        nonlocal common_identity
        label = campaign.run_label(method, seed, lr, phase)
        cfg = campaign.configuration(protocol, protocol_sha, method, seed, lr, phase)
        require(read(root / "configs" / (label + ".json")) == cfg, "Run configuration differs: " + label)
        out = root / "runs" / label
        # Reject ambiguous/nonfinite JSON even where the underlying validator uses json.loads.
        raw = read(out / "metrics.json")
        for block in raw["update_blocks"]:
            read(campaign.relative_file(out, block["file"], "updates"))
        for audit in raw["native_audits"]:
            read(out / "audits" / f'step{audit["after_updates"]:08d}.json')
        result = campaign.validate_result(root, out, cfg, verify_archives=True)
        retention_path = out / "completed_checkpoint_retention.json"
        retention = read(retention_path) if retention_path.exists() else None
        if protocol["completed_checkpoint_policy"] == "final-only":
            require(retention is not None and retention.get("storage_format") == "language-final-only-retention-v1",
                    "Final-only campaign requires its completed verification receipt")
            require(set(retention["local_checkpoint_archives"]) == {"latest.pt"} and
                    retention["selected_model"]["model_bytes_retained"] is False, "Final-only model retention differs")
        identity = {k: v for k, v in result["identity"].items() if k != "config_sha256"}
        if common_identity is None:
            common_identity = identity
        require(identity == common_identity, "Unpaired model/data/tokenizer/attention/source identity")
        require(identity["tokenizer_contract"] == pilot["tokenizer_contract"], "Pilot/training tokenizer differs")
        require(identity["attention"] == {k: v for k, v in pilot["attention_precision"].items() if k != "calls"}, "Pilot/training attention precision differs")
        batch_sha = result["validated_batch_sequence_sha256"]
        pair_key = (phase, seed)
        require(pair_key not in pairing or pairing[pair_key] == batch_sha, "Unpaired training batches")
        pairing[pair_key] = batch_sha
        maps.setdefault(phase, {})[label] = {"metrics_sha256": sha(out / "metrics.json"),
            "config_sha256": sha(root / "configs" / (label + ".json")), "batch_sequence_sha256": batch_sha}
        record = {"label": label, "method": method, "seed": seed, "lr": lr, "steps": result["steps"],
                  "validated_artifacts": maps[phase][label], "costs": run_cost(root, label, result),
                  "selected_checkpoint": result["selected_checkpoint"], "validation_history": result["history"],
                  "native_audits": result["native_audits"],
                  "completed_model_retention": {"policy": protocol["completed_checkpoint_policy"],
                      "receipt_sha256": sha(retention_path) if retention is not None else None,
                      "selected_model_bytes_intentionally_not_retained": protocol["completed_checkpoint_policy"] == "final-only"}}
        if phase == "confirmatory":
            record.update(final_test=result["final_test"], validation_selected_test=result["validation_selected_test"])
        stages[phase].append(record)
        return result

    for method in METHODS:
        choices = selection["candidates"][method]
        require(len(choices) == 3 and sorted(c["lr"] for c in choices) == list(RATES), "Incomplete development learning rates")
        for choice in choices:
            require(choice["label"] == campaign.run_label(method, 49100, choice["lr"], "development"), "Development selection label differs")
            number(choice["score"], "development score")
            result = validate_run(method, 49100, choice["lr"], "development")
            require(choice["score"] == result["selected_checkpoint"]["worst_ce"], "Development selection score differs")
        require(selection["rates"][method] == min(choices, key=lambda x: (x["score"], x["lr"]))["lr"], "Wrong selected learning rate")
    validate_marker(root, "development", maps["development"])
    for seed in SEEDS:
        for method in METHODS:
            validate_run(method, seed, selection["rates"][method], "confirmatory")
    validate_marker(root, "confirmatory", maps["confirmatory"])
    expected_labels = set(maps["development"]) | set(maps["confirmatory"])
    require({p.name for p in (root / "runs").iterdir() if p.is_dir()} == expected_labels, "Unexpected or missing campaign run directories")

    reporting = {(r["method"], r["seed"]): r for r in stages["confirmatory"]}
    contrasts = []
    for first, second in (("direct", "uniform"), ("periodic", "fa")):
        rows = [{"seed": seed, "first": reporting[first, seed]["final_test"]["worst_ce"],
                 "second": reporting[second, seed]["final_test"]["worst_ce"]} for seed in SEEDS]
        for row in rows:
            row["difference"] = row["first"] - row["second"]
        summary = describe([r["difference"] for r in rows])
        half = T95_DF2 * summary["sample_sd"] / math.sqrt(3)
        contrasts.append({"contrast": first + "-minus-" + second, "endpoint": "final heldout worst-task answer CE",
                          "per_seed": rows, **summary, "df": 2, "t95_critical": T95_DF2,
                          "descriptive_t95_interval": [summary["mean"] - half, summary["mean"] + half],
                          "direction": "Negative differences mean lower CE for the first method"})
    summaries = []
    for method in METHODS:
        rows = [reporting[method, seed] for seed in SEEDS]
        endpoints = {}
        for endpoint in ("final_test", "validation_selected_test"):
            endpoints[endpoint] = {key: describe([r[endpoint][key] for r in rows]) for key in METRIC_KEYS}
            endpoints[endpoint]["tasks"] = [{"task": task, "examples_per_seed": manifest["counts"]["test"][task],
                **{key: describe([r[endpoint]["tasks"][i][key] for r in rows]) for key in ("answer_ce", "teacher_forced_answer_exact_match")}}
                for i, task in enumerate(manifest["tasks"])]
        summaries.append({"method": method, "selected_lr": selection["rates"][method], "seeds": list(SEEDS),
                          **endpoints, "costs": summarize_cost(rows)})
    return {"validated": True, "complete": True, "analyzer_sha256": sha(__file__), "root": str(root.resolve()),
            "protocol": protocol, "selection": selection, "coverage": {"development_runs": 18, "confirmatory_runs": 18,
                "paired_reporting_seeds": list(SEEDS), "tasks": manifest["tasks"], "evaluation_counts": manifest["counts"],
                "paired_batch_sequence_sha256": {phase + ":" + str(seed): value for (phase, seed), value in pairing.items()}},
            "validated_identity": common_identity, "campaign_sources": sources,
            "input_sha256": {name: sha(root / name) for name in ("protocol.json", "selection.json", "campaign_sources.json", "training_preflight.json", "development_complete.json", "confirmatory_complete.json")},
            "primary_contrasts": contrasts, "method_summaries": summaries, "runs": stages,
            "cost_totals": {phase: summarize_cost(rows) for phase, rows in stages.items()},
            "scope": {
                "uncertainty": "Descriptive Student-t 95% intervals across exactly three paired reporting seeds, df=2. No power, significance, equivalence, convergence, or population guarantee is asserted; no p-values or outcome-selected comparisons are computed.",
                "accuracy": "Teacher-forced answer-token exact match; not free-generation accuracy.",
                "update_and_preparation": "Separate sums over the retained training trajectory. Includes every recorded update and preparation; discarded work before the last recovered checkpoint can be absent.",
                "trainer_wall": WALL_SCOPE,
                "validation_checkpoint_wall": "Cumulative trainer wall logged at each validation event, before that event's checkpoint persistence; events are not additive durations.",
                "process_wall": "Sum of recorded subprocess wall durations, including imports/setup, loading, updates, audits, evaluation and checkpoint persistence. Failed completed attempts are included. Unfinished/missing attempts make the total unknown; known durations remain reported. Conditional on the retained process-attempt log.",
                "retirement": "Campaign validation, parent-side backup restoration or completion of interrupted publication, and completed-model archival/retirement occur outside the timed training subprocess. No retirement timer exists in this frozen campaign; no campaign-total wall estimate is formed.",
                "memory": "Maximum torch CUDA allocated bytes sampled within training updates after per-update reset, including resident allocations. Excludes independent native reference/audit, loading, preparation and evaluation peaks; not reserved memory, total device usage, or global run peak. Raw independent audit records are retained separately.",
                "periodic": "Scheduled five-update opportunities and actually applied refinements are separate counts; neither is a per-update fidelity certificate.",
                "completed_models": "Under an explicitly frozen final-only policy, final checkpoint bytes are fully verified; selected-model absence is accepted only with the completion verification receipt binding its source/archive hashes and recorded evaluation metrics. The selected model was verified before intentional retirement and cannot subsequently be re-evaluated without an independent copy. Other unexplained model absence is rejected.",
                "provenance": "Checks artifact hashes, audited sources, cardinalities, deterministic batch sampling and complete declared coverage. These checks do not independently prove remote execution or cloud synchronization."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New JSON report; existing files are never overwritten")
    args = parser.parse_args()
    try:
        if args.output:
            require(not args.output.exists(), "Refusing to overwrite output: " + str(args.output))
        result = analyze(args.root)
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
