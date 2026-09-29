"""siq Model Registry — provenance, archiving, and library management.

Every champion model produced by a training run is registered here with full
metadata: architecture, factor, stage, iteration, val_metrics, git commit, and
the paths where copies are stored in ``model_archive/``.

The registry lives at ``<repo_root>/model_registry.json``.  The immutable
model archive lives at ``<repo_root>/model_archive/<entry_id>/``.

API
---
    register_model(keras_path, config, ...)  -> entry_id : str
    archive_model(keras_path, config, tag)   -> archive_dir : str
    list_models(filter_dict)                 -> list[dict]
    get_best_model(metric, factor, stage)    -> dict
    summarize_registry()                     -> str (printable table)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

__all__ = [
    "register_model",
    "archive_model",
    "list_models",
    "get_best_model",
    "summarize_registry",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the siq repo root (parent of the ``siq/`` package directory)."""
    return Path(__file__).resolve().parent.parent


def _registry_path(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or _repo_root()
    return root / "model_registry.json"


def _archive_root(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or _repo_root()
    arc = root / "model_archive"
    arc.mkdir(exist_ok=True)
    return arc


def _git_sha(repo_root: Optional[Path] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root or _repo_root()),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _load_registry(path: Path) -> list:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _save_registry(entries: list, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def _make_entry_id(config: dict, ts: str) -> str:
    """Deterministic entry ID: <model_type>_<factor_str>_<stage_tag>_<iter>_<ts>."""
    mtype = config.get("model_type", "model")
    factor = config.get("upsample_factor", [])
    factor_str = "x".join(str(f) for f in factor) if factor else "unknown"
    stage = config.get("stage", "").replace(" ", "_").lower()
    itr = config.get("iteration", 0)
    # truncate timestamp to seconds
    ts_short = ts.replace(":", "").replace("-", "").replace("T", "_")[:15]
    return f"{mtype}_{factor_str}_{stage}_{itr:05d}_{ts_short}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_model(
    keras_path: str,
    config: dict,
    tag: str = "",
    notes: str = "",
    repo_root: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Register a saved model in the siq model registry.

    Parameters
    ----------
    keras_path : str
        Path to the ``.keras`` file that was already saved to disk.
    config : dict
        Provenance config dict (as from ``save_siq_model``).
    tag : str, optional
        Human-readable label, e.g. ``"v0.10.14_cqs_champion"``.
    notes : str, optional
        Free-form notes about this checkpoint.
    repo_root : str, optional
        Override the repo root (defaults to the package parent).
    verbose : bool
        Print registry confirmation.

    Returns
    -------
    str
        Unique ``entry_id`` string for this registry entry.
    """
    root = Path(repo_root) if repo_root else _repo_root()
    reg_path = _registry_path(root)
    entries = _load_registry(reg_path)

    ts = datetime.now(timezone.utc).isoformat()
    entry_id = _make_entry_id(config, ts)

    entry = {
        "entry_id": entry_id,
        "tag": tag,
        "notes": notes,
        "registered_at": ts,
        "git_sha": _git_sha(root),
        "keras_path": str(keras_path),
        "config_path": str(keras_path).replace(".keras", "_config.json"),
        "model_type": config.get("model_type", ""),
        "upsample_factor": config.get("upsample_factor", []),
        "stage": config.get("stage", ""),
        "iteration": config.get("iteration", 0),
        "siq_version": config.get("siq_version", ""),
        "val_metrics": config.get("val_metrics", {}),
        "loss_weights": config.get("loss_weights", {}),
        "archive_dir": "",  # filled in by archive_model if called
    }

    # Avoid duplicate entries (same keras_path + same iteration).
    existing_ids = [
        e["entry_id"]
        for e in entries
        if e.get("keras_path") == entry["keras_path"]
        and e.get("iteration") == entry["iteration"]
    ]
    if existing_ids:
        if verbose:
            print(f"[registry] Entry already exists for {keras_path} iter={entry['iteration']} — skipping duplicate.")
        return existing_ids[0]

    entries.append(entry)
    _save_registry(entries, reg_path)

    if verbose:
        m = entry.get("val_metrics", {})
        cqs = m.get("val_cqs", float("nan"))
        print(f"[registry] Registered: {entry_id}  CQS={cqs:.4f}  tag='{tag}'")

    return entry_id


def archive_model(
    keras_path: str,
    config: dict,
    tag: str = "",
    notes: str = "",
    repo_root: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Copy a model + its config into the immutable model archive and register it.

    The archive is a write-once directory; existing archives are never
    overwritten (the function is idempotent for the same iteration).

    Parameters
    ----------
    keras_path : str
        Source ``.keras`` file path.
    config : dict
        Provenance config dict.
    tag : str
        Human-readable label.
    notes : str
        Free-form notes.
    repo_root : str, optional
        Override the repo root.
    verbose : bool
        Print archive destination.

    Returns
    -------
    str
        Path to the archive directory where the model was stored.
    """
    root = Path(repo_root) if repo_root else _repo_root()
    arc_root = _archive_root(root)

    ts = datetime.now(timezone.utc).isoformat()
    entry_id = _make_entry_id(config, ts)
    arc_dir = arc_root / entry_id

    # Idempotency: if this specific keras_path + iteration is already archived,
    # find the existing dir and return it.
    reg_path = _registry_path(root)
    entries = _load_registry(reg_path)
    for e in entries:
        if (
            e.get("keras_path") == str(keras_path)
            and e.get("iteration") == config.get("iteration")
            and e.get("archive_dir")
        ):
            if verbose:
                print(f"[archive] Already archived at {e['archive_dir']}")
            return e["archive_dir"]

    arc_dir.mkdir(parents=True, exist_ok=True)

    # Copy .keras
    src_keras = Path(keras_path)
    dst_keras = arc_dir / src_keras.name
    shutil.copy2(src_keras, dst_keras)

    # Copy companion _config.json
    src_cfg = Path(str(keras_path).replace(".keras", "_config.json"))
    if src_cfg.exists():
        shutil.copy2(src_cfg, arc_dir / src_cfg.name)

    # Write an archive README
    readme_path = arc_dir / "README.md"
    m = config.get("val_metrics", {})
    with open(readme_path, "w") as f:
        f.write(f"# siq Model Archive: {entry_id}\n\n")
        f.write(f"- **Tag**: {tag}\n")
        f.write(f"- **Notes**: {notes}\n")
        f.write(f"- **Archived at**: {ts}\n")
        f.write(f"- **Git SHA**: {_git_sha(root)}\n")
        f.write(f"- **Model type**: {config.get('model_type', '')}\n")
        f.write(f"- **Factor**: {config.get('upsample_factor', [])}\n")
        f.write(f"- **Stage**: {config.get('stage', '')}\n")
        f.write(f"- **Iteration**: {config.get('iteration', 0)}\n")
        f.write(f"- **CQS**: {m.get('val_cqs', float('nan')):.6f}\n")
        f.write(f"- **PSNR**: {m.get('val_psnr', float('nan')):.4f} dB\n")
        f.write(f"- **SSIM**: {m.get('val_ssim', float('nan')):.6f}\n")
        f.write(f"- **GMSD**: {m.get('val_gmsd', float('nan')):.6f}\n")
        f.write(f"- **CBI**: {m.get('val_cbi', float('nan')):.6f}\n")
        f.write(f"\n## Loss Weights\n```json\n")
        f.write(json.dumps(config.get("loss_weights", {}), indent=2))
        f.write("\n```\n")

    if verbose:
        print(f"[archive] Archived to: {arc_dir}")

    # Register (updates archive_dir in registry entry)
    reg_entries = _load_registry(reg_path)
    entry_id_registered = register_model(
        keras_path=str(dst_keras),
        config=config,
        tag=tag,
        notes=notes,
        repo_root=str(root),
        verbose=False,
    )
    # Patch archive_dir into the entry
    reg_entries = _load_registry(reg_path)
    for e in reg_entries:
        if e.get("entry_id") == entry_id_registered:
            e["archive_dir"] = str(arc_dir)
            e["keras_path"] = str(dst_keras)
            e["config_path"] = str(arc_dir / src_cfg.name) if src_cfg.exists() else ""
            break
    _save_registry(reg_entries, reg_path)

    return str(arc_dir)


def list_models(
    filter_dict: Optional[dict] = None,
    repo_root: Optional[str] = None,
) -> list:
    """Return registry entries, optionally filtered.

    Parameters
    ----------
    filter_dict : dict, optional
        Key-value pairs that must match the entry (e.g.
        ``{"upsample_factor": [2,2,2], "stage": "Stage 3 Refinement"}``).
        Nested ``val_metrics`` keys are also supported
        (e.g. ``{"val_metrics.val_cqs": 0.74}`` is NOT yet supported;
        filter on top-level fields only).
    repo_root : str, optional
        Override repo root.

    Returns
    -------
    list[dict]
        Matching registry entries, sorted by ``iteration`` descending.
    """
    root = Path(repo_root) if repo_root else _repo_root()
    entries = _load_registry(_registry_path(root))
    if filter_dict:
        filtered = []
        for e in entries:
            match = True
            for k, v in filter_dict.items():
                if e.get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(e)
        entries = filtered
    entries.sort(key=lambda e: e.get("iteration", 0), reverse=True)
    return entries


def get_best_model(
    metric: str = "val_cqs",
    factor: Optional[list] = None,
    stage: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Optional[dict]:
    """Return the registry entry with the best value for ``metric``.

    Parameters
    ----------
    metric : str
        Key inside ``val_metrics`` to maximise (default: ``"val_cqs"``).
        Use ``"val_gmsd"`` or ``"val_cbi"`` if you want to minimise — pass
        a negated key like ``"-val_gmsd"`` is not yet supported; caller
        should use :func:`list_models` and filter manually for minimisation.
    factor : list, optional
        Filter by ``upsample_factor``, e.g. ``[2, 2, 2]``.
    stage : str, optional
        Filter by ``stage``, e.g. ``"Stage 3 Refinement"``.
    repo_root : str, optional
        Override repo root.

    Returns
    -------
    dict or None
        Best matching registry entry, or ``None`` if the registry is empty
        or has no matching entries.
    """
    flt: dict = {}
    if factor is not None:
        flt["upsample_factor"] = factor
    if stage is not None:
        flt["stage"] = stage
    entries = list_models(filter_dict=flt or None, repo_root=repo_root)
    if not entries:
        return None
    return max(
        entries,
        key=lambda e: e.get("val_metrics", {}).get(metric, float("-inf")),
    )


def summarize_registry(repo_root: Optional[str] = None) -> str:
    """Return a human-readable table of the model registry.

    Returns
    -------
    str
        Formatted table with one row per registry entry.
    """
    root = Path(repo_root) if repo_root else _repo_root()
    entries = _load_registry(_registry_path(root))
    if not entries:
        return "(model registry is empty)"

    header = (
        f"{'entry_id':<60}  {'iter':>5}  {'CQS':>7}  {'PSNR':>7}  "
        f"{'SSIM':>7}  {'CBI':>7}  tag"
    )
    lines = [header, "-" * len(header)]
    for e in sorted(entries, key=lambda e: e.get("iteration", 0)):
        m = e.get("val_metrics", {})
        lines.append(
            f"{e['entry_id']:<60}  "
            f"{e.get('iteration', 0):>5}  "
            f"{m.get('val_cqs', float('nan')):>7.4f}  "
            f"{m.get('val_psnr', float('nan')):>7.3f}  "
            f"{m.get('val_ssim', float('nan')):>7.4f}  "
            f"{m.get('val_cbi', float('nan')):>7.5f}  "
            f"{e.get('tag', '')}"
        )
    return "\n".join(lines)
