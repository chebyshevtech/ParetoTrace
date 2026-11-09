"""Bounded, verified checkpoint storage without replacing Drive model files.

Integration contract (no trainer changes are made by this module):

    with SlotWriter(backup, owner, vm_temp=local_tmp) as writer:
        writer.publish(local_run)  # after its ordinary checkpoint/metrics save

Hold that context for the ENTIRE active training process. Retirement and restore
take the same exclusive lease and therefore fail while a writer is active.
All users of this store must honor that lease; flock is not a distributed lock
against unrelated Drive clients. A restarted VM can acquire a released lease.

There are exactly four owned .gz paths. Model files are created once and then
overwritten/truncated IN PLACE, never renamed or unlinked. Only small JSON
metadata uses atomic replacement. Compression and full round-trip verification
use the existing VM gzip helpers in an explicitly separate local directory.

This describes local filesystem operations, not verified Google Drive revision,
quota, upload completion, or remote durability behavior. fsync and read-back
checks do not establish that a disconnected Drive client has uploaded bytes.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile

from checkpoint_archive import archive_file, restore_file, sha, fsync_path
from retire_completed_language import verify_archive


FORMAT = "drive-fixed-slots-v1"
RETENTION_FORMAT = "drive-fixed-slots-retention-v1"
COMMIT_FILE = "backup_commit.json"
RETENTION_FILE = "slot_retention.json"
SLOTS = tuple(f"slots/{role}.{slot}.gz" for role in ("latest", "selected") for slot in (0, 1))
CHUNK = 8 * 2**20
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


class SlotError(ValueError):
    pass


class ActiveRunError(SlotError):
    pass


class RetiredRunError(SlotError):
    pass


def _need(test, message):
    if not test:
        raise SlotError(message)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _read(path):
    _regular(path)
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "Duplicate metadata key")
            result[key] = value
        return result
    def bad(value):
        raise SlotError("Nonfinite metadata: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad)


def _root(path, create=False):
    path = Path(path).absolute()
    _need(".." not in path.parts and not path.is_symlink(), "Unsafe root path")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    _need(path.is_dir(), "Directory is missing: " + str(path))
    # Canonicalize OS aliases such as /tmp -> /private/tmp before comparisons.
    return path.resolve(strict=True)


def _disjoint(*roots):
    for i, a in enumerate(roots):
        for b in roots[i + 1:]:
            _need(a != b and a not in b.parents and b not in a.parents, "Storage roots must be disjoint")


def _relative(value, *, model=False):
    _need(isinstance(value, str) and bool(value) and "\\" not in value, "Invalid relative path")
    path = PurePosixPath(value)
    _need(not path.is_absolute() and ".." not in path.parts and str(path) == value,
          "Unsafe relative path: " + value)
    if model:
        _need(value == "latest.pt" or (path.parent == PurePosixPath("selected") and
              re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.pt", path.name)), "Invalid model reference")
    return Path(*path.parts)


def _inside(root, value, *, model=False):
    relative = _relative(value, model=model)
    target = root / relative
    for part in (target, *target.parents):
        if part == root:
            break
        _need(not part.is_symlink(), "Symlink inside owned storage")
    _need(target.resolve(strict=False).is_relative_to(root), "Path escapes owned storage")
    return target


def _regular(path):
    path = Path(path)
    info = path.lstat()
    _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Expected a private regular file: " + str(path))
    return info


def _hash(value, label):
    _need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid hash: " + label)


def _integer(value, label, minimum=0):
    _need(type(value) is int and value >= minimum, "Invalid integer: " + label)


def _fault(callback, event, **details):
    if callback is not None:
        callback(event, details)


def _atomic_json(path, value):
    """Atomic SMALL metadata only; never call this for a model slot."""
    path = Path(path)
    _need(path.suffix == ".json" and not path.is_symlink(), "Only JSON metadata may be replaced")
    if path.exists():
        _regular(path)
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_path(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()  # small JSON temp, never a model file


def _seal(record):
    return {**record, "commit_id": _digest(record)}


def _check_seal(record):
    _need(isinstance(record, dict) and "commit_id" in record, "Missing commit identity")
    _need(record["commit_id"] == _digest({k: v for k, v in record.items() if k != "commit_id"}), "Commit metadata changed")


def _owner_value(owner):
    _need(isinstance(owner, dict) and set(owner) == {"config", "identity"} and
          isinstance(owner["config"], dict) and isinstance(owner["identity"], dict), "Invalid run owner")
    # JSON round-trip also rejects unserializable owner state and nonfinite data.
    return json.loads(_json_bytes(owner))


def _ownership(backup, owner):
    identity = _inside(backup, "run_identity.json")
    _need(_read(identity) == owner, "Backup belongs to another run")
    expected = {"storage_format": FORMAT, "owner_identity_sha256": _digest(owner),
                "run_identity_sha256": sha(identity), "slot_files": list(SLOTS)}
    _need(_read(_inside(backup, "slots/ownership.json")) == expected, "Slot ownership differs")
    return expected


def _inventory(backup, *, allow_missing=False):
    for path in backup.rglob("*"):
        _need(not path.is_symlink(), "Symlink in backup storage")
        if path.is_dir():
            continue
        _regular(path)
        if path.suffix in (".gz", ".pt"):
            _need(path.relative_to(backup).as_posix() in SLOTS, "Foreign/legacy model file in slot store")
    for relative in SLOTS:
        path = _inside(backup, relative)
        if path.exists():
            _regular(path)
        else:
            _need(allow_missing, "Missing owned slot: " + relative)


@contextmanager
def _lease(backup):
    path = _inside(backup, ".checkpoint_slots.lock")
    if path.exists():
        _regular(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Invalid lease file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ActiveRunError("An active writer/restore/retirement holds this run's lease") from error
        yield
    finally:
        os.close(fd)


@contextmanager
def run_lease(backup):
    """Public exclusive run lease, also usable by explicit metadata-only adapters.

    This creates no owner/slot schema and performs no implicit format migration.
    Hold it for the entire active run, not just individual backup calls.
    """
    backup = _root(backup, create=True)
    with _lease(backup):
        yield backup


def _initialize(backup, owner):
    commit_path = _inside(backup, COMMIT_FILE)
    if commit_path.exists():
        _need(_read(commit_path).get("storage_format") == FORMAT, "Explicit migration required for legacy backups")
    identity = _inside(backup, "run_identity.json")
    if not identity.exists():
        _need(not any(p.name != ".checkpoint_slots.lock" for p in backup.iterdir()), "Nonempty unowned backup")
        _atomic_json(identity, owner)
    _need(_read(identity) == owner, "Backup belongs to another run")
    _inventory(backup, allow_missing=True)
    slot_dir = _inside(backup, "slots")
    slot_dir.mkdir(exist_ok=True)
    ownership = slot_dir / "ownership.json"
    expected = {"storage_format": FORMAT, "owner_identity_sha256": _digest(owner),
                "run_identity_sha256": sha(identity), "slot_files": list(SLOTS)}
    if ownership.exists():
        _need(_read(ownership) == expected, "Slot ownership differs")
    else:
        _need(not any(slot_dir.iterdir()) and not commit_path.exists(), "Unowned pre-existing slots")
        _atomic_json(ownership, expected)
    # Ownership is durable before even the empty model slots are created.
    for relative in SLOTS:
        path = _inside(backup, relative)
        if not path.exists():
            _need(not commit_path.exists(), "Refuse to recreate a missing committed store slot")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o600)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    fsync_path(slot_dir)
    fsync_path(backup)


def _dependencies(backup, commit, owner):
    _need(commit["owner_identity_sha256"] == _digest(owner) and
          commit["run_identity_sha256"] == sha(_inside(backup, "run_identity.json")), "Committed owner differs")
    relative = _relative(commit["metrics_file"])
    _need(relative.parent == Path("generations") and relative.suffix == ".json", "Invalid metrics generation")
    path = _inside(backup, commit["metrics_file"])
    _need(sha(path) == commit["metrics_sha256"], "Committed metrics bytes differ")
    metrics = _read(path)
    _integer(metrics["steps"], "metrics steps")
    _need("complete" not in metrics or type(metrics["complete"]) is bool, "Invalid metrics completion flag")
    _need({"config": metrics["config"], "identity": metrics["identity"]} == owner and
          metrics["steps"] == commit["steps"], "Committed metrics identity/step differs")
    _need((metrics.get("complete") is True) == commit["run_complete"], "Completed-state identity differs")
    _need(metrics.get("selected_checkpoint") == commit["selected_checkpoint"], "Selected checkpoint differs from metrics")
    seen = set()
    for block in metrics["update_blocks"]:
        relative = _relative(block["file"])
        _need(relative.parent == Path("updates") and relative.suffix == ".json" and str(relative) not in seen,
              "Invalid/duplicate update-block reference")
        seen.add(str(relative))
        _hash(block["sha256"], "update block")
        path = _inside(backup, str(relative))
        _regular(path)
        _need(sha(path) == block["sha256"], "Committed update block differs")
    expected_audits = {}
    for audit in metrics.get("native_audits", []):
        _integer(audit["after_updates"], "audit step")
        relative = f"audits/step{audit['after_updates']:08d}.json"
        _need(relative not in expected_audits, "Duplicate native audit")
        path = _inside(backup, relative)
        _need(_read(path) == audit, "Committed native audit differs")
        expected_audits[relative] = sha(path)
    _need(commit.get("auxiliary_files", {}) == expected_audits, "Auxiliary audit coverage differs")
    return metrics


def _archive_record(entry, role, relative, owner_hash):
    _need(isinstance(entry, dict) and entry.get("format") == "gzip-lossless-v1" and
          entry.get("storage_format") == FORMAT and entry.get("role") == role and
          type(entry.get("slot")) is int and entry["slot"] in (0, 1), "Invalid slot archive record")
    expected = f"slots/{role}.{entry['slot']}.gz"
    _need(entry.get("slot_file") == expected and entry.get("archive_file") == Path(expected).name and
          entry.get("source_relpath") == relative and entry.get("owner_identity_sha256") == owner_hash,
          "Slot role/path/owner identity differs")
    for field in ("source_sha256", "archive_sha256"):
        _hash(entry[field], field)
    for field in ("source_bytes", "archive_bytes"):
        _integer(entry[field], field, 1)


def _active_schema(commit, owner):
    _check_seal(commit)
    _need(commit.get("storage_format") == FORMAT and commit.get("state") == "active", "Not an active slot commit")
    _integer(commit["generation"], "generation", 1)
    _integer(commit["steps"], "steps")
    _need(type(commit["run_complete"]) is bool, "Invalid completion status")
    _need(commit["owner_identity_sha256"] == _digest(owner), "Commit belongs to another run")
    selected = commit["selected_checkpoint"]
    refs = commit["checkpoint_archives"]
    expected = {"latest.pt"}
    if selected is not None:
        _need(isinstance(selected, dict), "Invalid selected checkpoint")
        rel = selected["file"]
        _relative(rel, model=True)
        _need(rel != "latest.pt", "Selected checkpoint aliases latest")
        expected.add(rel)
        _hash(selected["sha256"], "selected source")
    _need(set(refs) == expected, "Committed checkpoint role coverage differs")
    for relative, entry in refs.items():
        role = "latest" if relative == "latest.pt" else "selected"
        _archive_record(entry, role, relative, _digest(owner))
    _need(refs["latest.pt"]["source_sha256"] == commit["latest_sha256"], "Latest identity differs")
    if selected is not None:
        _need(refs[selected["file"]]["source_sha256"] == selected["sha256"], "Selected archive identity differs")


def _validate_unlocked(backup, owner, full=True):
    _ownership(backup, owner)
    _inventory(backup)
    commit = _read(_inside(backup, COMMIT_FILE))
    _check_seal(commit)
    _need(commit.get("storage_format") == FORMAT, "Unsupported slot storage format")
    _dependencies(backup, commit, owner)
    if commit.get("state") == "retired":
        _need(commit["checkpoint_archives"] == {}, "Retired commit still references Drive models")
        previous = commit["retired_from_commit"]
        _active_schema(previous, owner)
        _need(previous["run_complete"] and commit["previous_commit_id"] == previous["commit_id"] and
              commit["generation"] == previous["generation"] + 1, "Invalid retirement transition")
        for key in ("steps", "run_complete", "selected_checkpoint", "latest_sha256", "metrics_file", "metrics_sha256", "auxiliary_files"):
            _need(commit.get(key) == previous.get(key), "Retirement changed committed run state")
        proof = _read(_inside(backup, RETENTION_FILE))
        _need(_digest(proof) == commit["retention_identity_sha256"], "Retirement proof changed")
        _retention_schema(proof, previous, owner)
        return commit
    _active_schema(commit, owner)
    for entry in commit["checkpoint_archives"].values():
        path = _inside(backup, entry["slot_file"])
        _regular(path)
        verify_archive(path, entry, full=full)
    return commit


def validate_commit(backup, owner, *, full=True):
    """Read/verify the canonical commit. A retired commit requires VM retention to restore."""
    backup, owner = _root(backup), _owner_value(owner)
    with _lease(backup):
        return _validate_unlocked(backup, owner, full)


def _publish_dependency(source, destination, expected, *, replace_unreferenced=False):
    """Publish small JSON, replacing divergent abandoned attempts only.

    The caller derives replace_unreferenced from the previous verified commit.
    This permits replay after an interrupted publication to record new timings
    at the same logical step filename without mutating a committed dependency.
    """
    _regular(source)
    _need(source.suffix == destination.suffix == ".json", "Only numerical JSON is a dependency")
    _need(sha(source) == expected, "Local metadata changed")
    if destination.exists():
        _regular(destination)
        if sha(destination) == expected:
            return
        _need(replace_unreferenced, "Committed/immutable metadata conflict")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
            dst.flush()
            os.fsync(dst.fileno())
        _need(sha(temporary) == expected and sha(source) == expected, "Copied metadata changed")
        os.replace(temporary, destination)
        fsync_path(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _overwrite_slot(source, destination, entry, callback):
    before = _regular(destination)
    fd = os.open(destination, os.O_WRONLY | NOFOLLOW)
    try:
        _need((os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (before.st_dev, before.st_ino), "Slot changed before opening")
        count = 0
        with source.open("rb") as src:
            for block in iter(lambda: src.read(CHUNK), b""):
                view = memoryview(block)
                while view:
                    written = os.write(fd, view)
                    _need(written > 0, "Short slot write")
                    view = view[written:]
                    count += written
                _fault(callback, "after_slot_chunk", role=entry["role"], slot=entry["slot"], bytes=count)
        _need(count == entry["archive_bytes"], "Local archive length changed")
        os.ftruncate(fd, count)
        os.fsync(fd)
    finally:
        os.close(fd)
    after = _regular(destination)
    _need((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino), "Slot inode was replaced")
    _need(after.st_size == entry["archive_bytes"] and sha(destination) == entry["archive_sha256"], "Persisted slot verification failed")
    fsync_path(destination.parent)
    _fault(callback, "after_slot_verified", role=entry["role"], slot=entry["slot"])


class SlotWriter:
    """Single active-run lease plus four-slot publisher.

    vm_temp must be a local VM directory, disjoint from both the backup and run.
    slot_budget_bytes optionally bounds the aggregate local size of the four
    slots; it is NOT an account-wide or retained-revision quota guarantee.
    fault(event, details) is an optional crash-test injection hook.
    """
    def __init__(self, backup, owner, *, vm_temp, slot_budget_bytes=None, fault=None):
        self.backup = _root(backup, create=True)
        self.vm_temp = _root(vm_temp, create=True)
        _disjoint(self.backup, self.vm_temp)
        self.owner = _owner_value(owner)
        if slot_budget_bytes is not None:
            _integer(slot_budget_bytes, "slot budget", 1)
        self.budget, self.fault = slot_budget_bytes, fault
        self._guard = None

    def __enter__(self):
        _need(self._guard is None, "Writer context already entered")
        guard = _lease(self.backup)
        guard.__enter__()
        try:
            _initialize(self.backup, self.owner)
            if (self.backup / COMMIT_FILE).exists():
                commit = _validate_unlocked(self.backup, self.owner, full=False)
                _need(commit["state"] == "active" and not commit["run_complete"], "Completed/retired store cannot resume publication")
        except BaseException:
            guard.__exit__(*__import__('sys').exc_info())
            raise
        self._guard = guard
        return self

    def __exit__(self, *args):
        guard, self._guard = self._guard, None
        if guard is not None:
            return guard.__exit__(*args)

    def publish(self, run_dir):
        _need(self._guard is not None, "Publish requires an active writer context")
        run = _root(run_dir)
        _disjoint(run, self.backup, self.vm_temp)
        _need(_read(_inside(run, "run_identity.json")) == self.owner, "Local run belongs to another owner")
        _need(not (run / "failure.json").exists() and not (self.backup / "failure.json").exists(), "Failed runs require diagnosis, not publication")
        previous = (_validate_unlocked(self.backup, self.owner, full=False)
                    if (self.backup / COMMIT_FILE).exists() else None)
        if previous:
            _need(previous["state"] == "active" and not previous["run_complete"], "Completed/retired store is terminal")
        source_metrics = _inside(run, "metrics.json")
        _regular(source_metrics)
        metrics_digest = sha(source_metrics)
        metrics = _read(source_metrics)
        _need("complete" not in metrics or type(metrics["complete"]) is bool, "Invalid completion flag")
        _need({"config": metrics["config"], "identity": metrics["identity"]} == self.owner, "Local metrics identity differs")
        _integer(metrics["steps"], "steps")
        _need(previous is None or metrics["steps"] >= previous["steps"], "Checkpoint step regressed")
        selected = metrics.get("selected_checkpoint")
        sources = {"latest.pt": _inside(run, "latest.pt", model=True)}
        if selected is not None:
            _need(isinstance(selected, dict), "Invalid selected metadata")
            _relative(selected["file"], model=True)
            _need(selected["file"] != "latest.pt", "Selected checkpoint aliases latest")
            _hash(selected["sha256"], "selected checkpoint")
            sources[selected["file"]] = _inside(run, selected["file"], model=True)
        _need(not metrics.get("complete") or selected is not None, "Completed run must retain its selected model")
        source_digests, source_inodes = {}, set()
        for relative, path in sources.items():
            info = _regular(path)
            _need(info.st_size > 0 and (info.st_dev, info.st_ino) not in source_inodes, "Empty/aliased source checkpoints")
            source_inodes.add((info.st_dev, info.st_ino))
            source_digests[relative] = sha(path)
        if selected is not None:
            _need(source_digests[selected["file"]] == selected["sha256"], "Local selected source differs")
        refs, prepared = {}, {}
        old_refs = previous["checkpoint_archives"] if previous else {}
        old_by_role = {entry["role"]: entry for entry in old_refs.values()}
        protected_dependencies = set()
        if previous:
            old_metrics = _dependencies(self.backup, previous, self.owner)
            protected_dependencies = {previous["metrics_file"], "run_identity.json",
                *(block["file"] for block in old_metrics["update_blocks"]), *previous.get("auxiliary_files", {})}
        with tempfile.TemporaryDirectory(prefix="checkpoint-slots-", dir=self.vm_temp) as name:
            stage = Path(name)
            for relative, source in sources.items():
                role = "latest" if relative == "latest.pt" else "selected"
                old = old_by_role.get(role)
                if old and old["source_sha256"] == source_digests[relative] and old["source_relpath"] == relative:
                    refs[relative] = old.copy()  # unchanged selected archive is not recompressed or overwritten
                    continue
                record = archive_file(source, stage / (role + ".pt.gz"))
                archived = stage / record["archive_file"]
                verify_archive(archived, record, full=True)
                _need(record["source_sha256"] == source_digests[relative], "Source changed during preparation")
                slot = 1 - old["slot"] if old else 0
                slot_file = f"slots/{role}.{slot}.gz"
                entry = {**record, "storage_format": FORMAT, "role": role, "slot": slot,
                         "archive_file": Path(slot_file).name, "slot_file": slot_file,
                         "source_relpath": relative, "owner_identity_sha256": _digest(self.owner)}
                refs[relative] = entry
                prepared[relative] = archived
            _need(sha(source_metrics) == metrics_digest and all(sha(p) == source_digests[r] for r, p in sources.items()),
                  "Checkpoint/metrics changed before publication")
            _fault(self.fault, "after_prepare", roles=[refs[r]["role"] for r in prepared])
            # Evaluate peak logical slot size before ANY Drive model write. A
            # shorter replacement retains the old tail until final ftruncate.
            sizes = {rel: _regular(_inside(self.backup, rel)).st_size for rel in SLOTS}
            projected_peak = sum(sizes.values())
            for relative in prepared:
                entry = refs[relative]
                path = entry["slot_file"]
                projected_peak = max(projected_peak, sum(sizes.values()) - sizes[path] + max(sizes[path], entry["archive_bytes"]))
                sizes[path] = entry["archive_bytes"]
            _need(self.budget is None or projected_peak <= self.budget, "Prepared archives exceed fixed-slot byte budget")
            metrics_relative = f"generations/metrics_{metrics_digest}.json"
            _publish_dependency(source_metrics, _inside(self.backup, metrics_relative), metrics_digest)
            for block in metrics["update_blocks"]:
                relative = _relative(block["file"])
                _need(relative.parent == Path("updates") and relative.suffix == ".json", "Unsafe event block")
                _hash(block["sha256"], "event block")
                _publish_dependency(_inside(run, str(relative)), _inside(self.backup, str(relative)), block["sha256"],
                    replace_unreferenced=str(relative) not in protected_dependencies)
            auxiliary = {}
            for audit in metrics.get("native_audits", []):
                _integer(audit["after_updates"], "audit step")
                relative = f"audits/step{audit['after_updates']:08d}.json"
                _need(relative not in auxiliary, "Duplicate native audit")
                source = _inside(run, relative)
                _need(_read(source) == audit, "Local native audit differs from metrics")
                auxiliary[relative] = sha(source)
                _publish_dependency(source, _inside(self.backup, relative), auxiliary[relative],
                    replace_unreferenced=relative not in protected_dependencies)
            protected = {entry["slot_file"] for entry in old_refs.values()}
            for relative, archived in prepared.items():
                entry = refs[relative]
                _need(entry["slot_file"] not in protected, "Refuse to overwrite a currently committed slot")
                _overwrite_slot(archived, _inside(self.backup, entry["slot_file"]), entry, self.fault)
            commit = _seal({"storage_format": FORMAT, "state": "active",
                "generation": previous["generation"] + 1 if previous else 1,
                "previous_commit_id": previous["commit_id"] if previous else None,
                "owner_identity_sha256": _digest(self.owner),
                "run_identity_sha256": sha(_inside(self.backup, "run_identity.json")),
                "steps": metrics["steps"], "run_complete": metrics.get("complete") is True,
                "selected_checkpoint": selected, "checkpoint_archives": refs,
                "latest_sha256": refs["latest.pt"]["source_sha256"],
                "metrics_file": metrics_relative, "metrics_sha256": metrics_digest, "auxiliary_files": auxiliary})
            _active_schema(commit, self.owner)
            _dependencies(self.backup, commit, self.owner)
            for entry in refs.values():
                verify_archive(_inside(self.backup, entry["slot_file"]), entry, full=False)
            _fault(self.fault, "before_commit", commit_id=commit["commit_id"])
            _atomic_json(self.backup / COMMIT_FILE, commit)
            _fault(self.fault, "after_commit", commit_id=commit["commit_id"])
            return commit


def _retention_schema(proof, previous, owner):
    _need(proof.get("storage_format") == RETENTION_FORMAT and proof.get("complete") is True and
          proof.get("owner_identity_sha256") == _digest(owner) and
          proof.get("active_commit_id") == previous["commit_id"] and
          proof.get("previous_backup_commit") == previous and
          proof.get("metrics_sha256") == previous["metrics_sha256"] and
          proof.get("config") == owner["config"], "Invalid completed-model retention proof")
    refs = proof["local_checkpoint_archives"]
    _need(set(refs) == set(previous["checkpoint_archives"]), "VM retention role coverage differs")
    for relative, entry in refs.items():
        _relative(relative, model=True)
        name = entry["archive_file"]
        _need(isinstance(name, str) and Path(name).name == name and re.fullmatch(r"[A-Za-z0-9_.-]+\.gz", name), "Unsafe VM archive filename")
        prior = previous["checkpoint_archives"][relative]
        _need(entry["format"] == "gzip-lossless-v1" and entry["source_sha256"] == prior["source_sha256"] and
              entry["source_bytes"] == prior["source_bytes"], "VM retention source differs")


def _verify_retention(root, proof, previous, owner):
    _retention_schema(proof, previous, owner)
    _need(_read(_inside(root, "retention_owner.json")) == {
        "owner_identity_sha256": _digest(owner), "active_commit_id": previous["commit_id"]}, "VM retention directory belongs to another run/commit")
    for relative, entry in proof["local_checkpoint_archives"].items():
        path = _inside(root, (Path(relative).parent / entry["archive_file"]).as_posix())
        _regular(path)
        verify_archive(path, entry, full=True)


def validate_retention_manifest(proof, owner):
    """Validate historical retention metadata without reading any model files.

    This checks the sealed completed slot commit and its retention schema only.
    It does not establish that archived bytes still exist or permit restoration
    or deletion; those operations require their own byte-verification policy.
    """
    owner = _owner_value(owner)
    previous = proof["previous_backup_commit"]
    _active_schema(previous, owner)
    _need(previous["run_complete"] is True and previous["selected_checkpoint"] is not None,
          "VM retention requires a completed selected-model commit")
    _retention_schema(proof, previous, owner)
    for entry in proof["local_checkpoint_archives"].values():
        _hash(entry["archive_sha256"], "VM archive hash")
        _integer(entry["archive_bytes"], "VM archive bytes", 1)
    return proof


def validate_retention(vm_root, proof, owner):
    """Fully verify this format's completed VM retention, without modifying it.

    Metadata-only recovery policies require their own explicit format/proof; a
    proof from that policy must not be relabeled as a former fixed-slot commit.
    """
    root, owner = _root(vm_root), _owner_value(owner)
    previous = proof["previous_backup_commit"]
    _active_schema(previous, owner)
    _need(previous["run_complete"] is True, "VM retention requires a completed commit")
    _verify_retention(root, proof, previous, owner)
    return proof


def _truncate_owned_slot(path):
    before = _regular(path)
    fd = os.open(path, os.O_WRONLY | NOFOLLOW)
    try:
        _need((os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (before.st_dev, before.st_ino), "Slot changed before truncation")
        os.ftruncate(fd, 0)
        os.fsync(fd)
    finally:
        os.close(fd)
    after = _regular(path)
    _need(after.st_size == 0 and (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino), "Slot was replaced during retirement")


def retire_completed(backup, run_dir, owner, *, vm_archive_root, fault=None):
    """Verify completed VM retention, retire commit, then truncate four owned slots.

    Never deletes raw VM checkpoints or immutable VM archives. A crash after
    the retired commit can be resumed by calling this function again. Completed
    VM archives can still be lost on VM reset; the manifest states this limit.
    """
    backup, run, vm = _root(backup), _root(run_dir), _root(vm_archive_root, create=True)
    _disjoint(backup, run, vm)
    owner = _owner_value(owner)
    with _lease(backup):
        _need(not (run / "failure.json").exists() and not (backup / "failure.json").exists(), "Failed run cannot retire checkpoints")
        _need(_read(_inside(run, "run_identity.json")) == owner, "Local run ownership differs")
        current = _validate_unlocked(backup, owner, full=True)
        previous = current["retired_from_commit"] if current["state"] == "retired" else current
        _need(previous["run_complete"] is True and previous["selected_checkpoint"] is not None, "Active/incomplete run cannot retire")
        metrics = _read(_inside(run, "metrics.json"))
        _need(metrics.get("complete") is True and sha(run / "metrics.json") == previous["metrics_sha256"] and
              {"config": metrics["config"], "identity": metrics["identity"]} == owner,
              "Only the exact committed completed run may retire")
        local_owner = _inside(vm, "retention_owner.json")
        expected_owner = {"owner_identity_sha256": _digest(owner), "active_commit_id": previous["commit_id"]}
        if local_owner.exists():
            _need(_read(local_owner) == expected_owner, "VM retention root belongs to another run")
        else:
            _need(not any(vm.iterdir()), "Nonempty unowned VM retention directory")
            _atomic_json(local_owner, expected_owner)
        proof_path = _inside(backup, RETENTION_FILE)
        if proof_path.exists():
            proof = _read(proof_path)
            _verify_retention(vm, proof, previous, owner)
        else:
            _need(current["state"] == "active", "Retired commit has no retention proof")
            archives = {}
            for relative, entry in previous["checkpoint_archives"].items():
                source = _inside(run, relative, model=True)
                _regular(source)
                _need(sha(source) == entry["source_sha256"], "Local completed source differs from committed checkpoint")
                target = _inside(vm, relative + ".gz")
                record = archive_file(source, target)
                path = target.parent / record["archive_file"]
                verify_archive(path, record, full=True)
                archives[relative] = record
            proof = {"storage_format": RETENTION_FORMAT, "complete": True,
                "owner_identity_sha256": _digest(owner), "active_commit_id": previous["commit_id"],
                "config": owner["config"], "metrics_sha256": previous["metrics_sha256"],
                "previous_backup_commit": previous, "local_checkpoint_archives": archives,
                "vm_archive_root_at_retirement": str(vm),
                "policy": "Verified lossless VM model archives; numerical records and retention proof remain in backup; owned model slots are truncated in-place.",
                "limitation": "Completed model archives may be lost on VM reset until an independent retained copy is verified. Local fsync is not proof of Drive upload or cloud quota behavior."}
            _verify_retention(vm, proof, previous, owner)
            _atomic_json(vm / RETENTION_FILE, proof)
            _atomic_json(proof_path, proof)
        _fault(fault, "after_retention_proof", commit_id=previous["commit_id"])
        if current["state"] != "retired":
            retired = {k: v for k, v in previous.items() if k != "commit_id"}
            retired.update(state="retired", generation=previous["generation"] + 1,
                previous_commit_id=previous["commit_id"], checkpoint_archives={},
                retired_from_commit=previous, retention_file=RETENTION_FILE,
                retention_identity_sha256=_digest(proof))
            current = _seal(retired)
            _atomic_json(backup / COMMIT_FILE, current)
        _fault(fault, "after_retired_commit", commit_id=current["commit_id"])
        # No canonical commit references these paths now; the previously
        # referenced current roles have fully verified VM retention proof.
        _verify_retention(vm, proof, previous, owner)
        for relative in SLOTS:
            _ownership(backup, owner)
            _fault(fault, "before_truncate", slot_file=relative)
            _truncate_owned_slot(_inside(backup, relative))
            _fault(fault, "after_truncate", slot_file=relative)
        fsync_path(backup / "slots")
        finished = {"storage_format": FORMAT, "state": "retired", "complete": True,
            "owner_identity_sha256": _digest(owner), "retired_commit_id": current["commit_id"],
            "retention_identity_sha256": _digest(proof), "slot_bytes": {name: 0 for name in SLOTS},
            "models_unlinked_or_renamed": False}
        _atomic_json(backup / "slot_retirement_complete.json", finished)
        _fault(fault, "after_retirement_complete", commit_id=current["commit_id"])
        return finished


def restore_run(backup, output, owner, *, retention_root=None):
    """Restore only the canonical committed run, into an absent/empty LOCAL directory.

    For a retired store an explicit VM/downloaded retention_root is mandatory.
    It may be relocated; its owner/commit manifest and every archive are verified.
    A stale mutable slot filename alone is never a valid restore reference.
    """
    backup, owner = _root(backup), _owner_value(owner)
    output = Path(output).absolute()
    _need(".." not in output.parts and not output.is_symlink(), "Unsafe restore destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.parent.resolve(strict=True) / output.name
    _disjoint(backup, output)
    _need(not output.exists() or (output.is_dir() and not any(output.iterdir())), "Restore destination must be absent or empty")
    with _lease(backup):
        commit = _validate_unlocked(backup, owner, full=True)
        if commit["state"] == "retired":
            if retention_root is None:
                raise RetiredRunError("Completed Drive model slots are retired; verified VM retention is required")
            retained = _root(retention_root)
            _disjoint(backup, output, retained)
            proof = _read(backup / RETENTION_FILE)
            previous = commit["retired_from_commit"]
            _verify_retention(retained, proof, previous, owner)
            sources = {relative: (retained / Path(relative).parent / entry["archive_file"], entry)
                       for relative, entry in proof["local_checkpoint_archives"].items()}
        else:
            sources = {relative: (_inside(backup, entry["slot_file"]), entry)
                       for relative, entry in commit["checkpoint_archives"].items()}
        stage = Path(tempfile.mkdtemp(prefix=output.name + ".restore-", dir=output.parent))
        try:
            for relative, (archive, entry) in sources.items():
                restore_file(archive, _inside(stage, relative, model=True), entry)
            metadata = _read(_inside(backup, commit["metrics_file"]))
            shutil.copyfile(backup / "run_identity.json", stage / "run_identity.json")
            shutil.copyfile(backup / commit["metrics_file"], stage / "metrics.json")
            for block in metadata["update_blocks"]:
                target = _inside(stage, block["file"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_inside(backup, block["file"]), target)
                _need(sha(target) == block["sha256"], "Restored event block differs")
            for relative, digest in commit.get("auxiliary_files", {}).items():
                target = _inside(stage, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_inside(backup, relative), target)
                _need(sha(target) == digest, "Restored audit artifact differs")
            _need(sha(stage / "metrics.json") == commit["metrics_sha256"], "Restored metrics differ")
            _need(sha(stage / "latest.pt") == commit["latest_sha256"], "Restored latest source differs")
            _atomic_json(stage / "restored_slot_commit.json", commit)
            for path in stage.rglob("*"):
                if path.is_file():
                    fsync_path(path)
            for path in sorted((p for p in stage.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                fsync_path(path)
            fsync_path(stage)
            if output.exists():
                _need(not any(output.iterdir()), "Restore destination changed")
                output.rmdir()  # empty LOCAL directory only
            os.replace(stage, output)
            fsync_path(output.parent)
        finally:
            if stage.exists():
                shutil.rmtree(stage)  # LOCAL restore stage only
        return commit
