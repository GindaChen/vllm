# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-entry, content-addressed store of FlashInfer autotune winners.

FlashInfer's JSON config cache (``AutoTuner.load_configs``) does not record
the measurement policy an entry was tuned under. While tuning, it therefore
ignores every persisted entry whose op uses a non-default policy (for example
``use_cold_l2_cache=True``, which all fused-MoE ops use) and profiles those ops
again on every start, even when the cache file holds their winners.

This store records one winner per file with its policy, and serves it back to
the tuner during the warmup tuning pass. An entry is only served when all of
these match the current process:

* the FlashInfer environment (versions of FlashInfer, CUDA, cuBLAS, cuDNN,
  CuTe-DSL and the GPU name), which names the store directory;
* the tuner's own lookup key (op, runner class, shape bucket, extras);
* the measurement policy;
* a digest of the source files that define the runner class, so editing a
  runner retunes only that runner's entries.

Entries hold no absolute paths or process state, so a store directory can be
copied or shared. ``VLLM_FLASHINFER_AUTOTUNE_STORE`` lists store roots
separated by ``os.pathsep``: the first is written, the others are read-only
layers consulted in order. It defaults to
``$VLLM_CACHE_ROOT/flashinfer_autotune_store``; an empty value disables the
store.
"""

import contextlib
import functools
import hashlib
import inspect
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_SCHEMA = "rsii-v1"


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=repr)


@functools.cache
def runner_digest(cls: type) -> str:
    """Hash the source files of ``cls`` and its tunable base classes."""
    digest = hashlib.sha256()
    for klass in cls.__mro__:
        if klass.__module__ in ("builtins", "abc"):
            continue
        try:
            path = inspect.getsourcefile(klass)
        except TypeError:
            path = None
        digest.update(f"{klass.__module__}.{klass.__qualname__}\0".encode())
        if path and os.path.isfile(path):
            digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def environment() -> dict[str, str]:
    """FlashInfer's own compatibility metadata plus the device capability."""
    import torch
    from flashinfer.autotuner import _collect_metadata

    meta = dict(_collect_metadata())
    major, minor = torch.cuda.get_device_capability()
    meta["compute_capability"] = f"{major}.{minor}"
    meta["schema"] = _SCHEMA
    return meta


class WinnerStore:
    """Layered on-disk winners: ``<root>/<env hash>/<entry hash>.json``."""

    def __init__(self, roots: list[Path], env: dict[str, str]):
        self.env = env
        self.env_hash = _sha256(_canonical(env))[:32]
        self.dirs = [Path(root) / self.env_hash for root in roots]
        self.hits = 0
        self.misses = 0  # lookups, not distinct entries
        self.published = 0

    @staticmethod
    def entry_key(file_key: str, policy: tuple, digest: str) -> str:
        return _sha256(_canonical([file_key, list(policy), digest]))

    def lookup(self, file_key: str, policy: tuple, digest: str):
        name = self.entry_key(file_key, policy, digest) + ".json"
        for directory in self.dirs:
            try:
                entry = json.loads((directory / name).read_text())
            except (OSError, ValueError):
                continue
            # Guard against hash collisions and hand-edited files.
            if (
                entry.get("file_key") == file_key
                and entry.get("policy") == list(policy)
                and entry.get("runner_digest") == digest
            ):
                return entry
        return None

    def publish(self, file_key, policy, digest, runner_name, tactic) -> None:
        directory = self.dirs[0]
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "environment.json"
        if not manifest.exists():
            _atomic_write(manifest, _canonical(self.env))
        entry = {
            "file_key": file_key,
            "policy": list(policy),
            "runner_digest": digest,
            "runner": runner_name,
            "tactic": tactic,
        }
        name = self.entry_key(file_key, policy, digest) + ".json"
        _atomic_write(directory / name, _canonical(entry))
        self.published += 1


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def store_roots() -> list[Path]:
    """Store roots from the environment; empty when the store is disabled."""
    import vllm.envs as envs

    value = envs.VLLM_FLASHINFER_AUTOTUNE_STORE
    if value is None:
        return [Path(envs.VLLM_CACHE_ROOT) / "flashinfer_autotune_store"]
    return [Path(p).expanduser() for p in value.split(os.pathsep) if p]


@contextlib.contextmanager
def serve_persisted_winners(store: WinnerStore) -> Iterator[WinnerStore]:
    """Serve and record winners through ``store`` while the tuner tunes.

    Patches ``search_cache`` on the tuner instance only for the duration of
    the context. A miss in tuning mode is looked up in the store; a hit is
    promoted into the tuner's in-memory winners with its policy, exactly as
    a freshly profiled winner would be, so profiling for it is skipped. On
    exit, every in-memory winner whose runner class was seen is published.
    """
    from flashinfer.autotuner import AutoTuner
    from flashinfer.autotuner.autotuner import _json_to_tactic, _tactic_to_json

    tuner = AutoTuner.get()
    original = tuner.search_cache
    runner_classes: dict[str, type] = {}
    served: set[Any] = set()

    def search_cache(custom_op, runners, input_shapes, tuning_config, inputs=None):
        result = original(
            custom_op, runners, input_shapes, tuning_config, inputs=inputs
        )
        for runner in runners:
            runner_classes.setdefault(type(runner).__name__, type(runner))
        if result[0] or not tuner.is_tuning_mode:
            return result
        policy = tuner._profiling_policy(tuning_config)
        for r_id, runner in enumerate(runners):
            extras = runner.get_cache_key_extras(inputs) if inputs is not None else ()
            key = AutoTuner._get_cache_key(
                custom_op, runner, input_shapes, tuning_config, extras
            )
            entry = store.lookup(key.file_key, policy, runner_digest(type(runner)))
            if entry is None or entry["runner"] != type(runner).__name__:
                continue
            tactic = _json_to_tactic(entry["tactic"])
            if not tuner._tactic_still_valid(
                runner, inputs, tactic, custom_op, "vllm winner store"
            ):
                continue
            with tuner._lock:
                tuner._winner_cache()[key] = (tactic, None)
                tuner._profiling_cache_policies[key] = policy
            served.add(key)
            store.hits += 1
            return True, r_id, tactic, None
        store.misses += 1
        return result

    tuner.search_cache = search_cache
    try:
        yield store
    finally:
        del tuner.search_cache
        default_policy_of = getattr(tuner, "_profiling_cache_policies", {})
        for key, (tactic, _) in list(tuner._winner_cache().items()):
            cls = runner_classes.get(key.runner_class_name)
            policy = default_policy_of.get(key)
            if key in served or cls is None or policy is None:
                continue
            try:
                store.publish(
                    key.file_key,
                    policy,
                    runner_digest(cls),
                    key.runner_class_name,
                    _tactic_to_json(tactic),
                )
            except OSError as exc:
                logger.warning("Could not persist FlashInfer winner: %s", exc)
                break
        logger.info(
            "FlashInfer winner store %s: %d served, %d missed, %d published.",
            store.dirs[0],
            store.hits,
            store.misses,
            store.published,
        )
