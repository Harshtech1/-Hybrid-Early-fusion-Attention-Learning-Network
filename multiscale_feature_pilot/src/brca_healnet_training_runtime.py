"""Authorized-only HEALNet training runtime for frozen BRCA compact features.

This module deliberately imports only the standard library at import time.
Torch, HEALNet and project tensor loaders are imported inside
``execute_authorized_training`` after the stdlib launcher has passed its
authorization and source-identity gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
from typing import Any


class TrainingRuntimeError(RuntimeError):
    """Raised for any runtime identity, data, CUDA or optimization drift."""


@dataclass(frozen=True)
class TrainingRuntimePaths:
    project_root: Path
    official_healnet_root: Path
    omic_archive: Path
    split_manifest: Path
    cutpoints: Path
    feature_registry: Path
    checkpoint_root: Path
    result_root: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingRuntimeError(message)


def _read_secure(path: Path, maximum_bytes: int = 64_000_000) -> bytes:
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and not path.is_symlink(), f"regular non-symlink file required: {path}")
    _require(0 < before.st_size <= maximum_bytes, f"bound file size invalid: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        token = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        _require(token == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns), "bound file changed before open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            _require(bool(chunk), "unexpected bound-file EOF")
            chunks.append(chunk)
            remaining -= len(chunk)
        _require(os.read(descriptor, 1) == b"", "bound file grew during read")
        final = os.fstat(descriptor)
        after = path.lstat()
        final_token = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns)
        path_token = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        _require(token == final_token == path_token, "bound file identity changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256(path: Path, maximum_bytes: int = 64_000_000) -> str:
    return hashlib.sha256(_read_secure(path, maximum_bytes)).hexdigest()


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
    )
    _require(result.returncode == 0, "could not resolve Git HEAD")
    return result.stdout.strip()


def _load_module(name: str, path: Path, payload: bytes) -> Any:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(payload, str(path), "exec"), module.__dict__)
    return module


def _torch_bytes(torch: Any, value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _load_compact_patient(torch: Any, compact_module: Any, entry: Any, device: Any) -> Any:
    directory = Path(entry.compact_directory)
    before_directory = directory.lstat()
    _require(stat.S_ISDIR(before_directory.st_mode) and not directory.is_symlink(), "compact directory identity invalid")
    manifest = compact_module.validate_compact_feature_artifacts(
        directory, expected_manifest_sha256=entry.compact_manifest_sha256,
    )
    _require(manifest["identity"]["patient_id"] == entry.patient_id, "compact patient mismatch")
    _require(manifest["identity"]["slide_id"] == entry.slide_id, "compact slide mismatch")
    _require(manifest["identity"]["gdc_file_uuid"] == entry.gdc_uuid, "compact UUID mismatch")
    tensor_info = manifest["tensor"]
    _require(tensor_info["file_sha256"] == entry.combined_tensor_file_sha256, "compact file hash mismatch")
    _require(tensor_info["content_sha256"] == entry.combined_tensor_content_sha256, "compact content hash mismatch")
    _require(manifest["provenance"]["file_sha256"] == entry.row_provenance_file_sha256, "row provenance hash mismatch")
    _require(tensor_info["shape"] == [entry.rows, 2048], "feature registry row/shape mismatch")
    _require(hashlib.sha256(_read_secure(directory / "row_provenance.csv", 2_000_000_000)).hexdigest() == entry.row_provenance_file_sha256, "secure row provenance hash mismatch")
    tensor_payload = _read_secure(directory / "combined_features.pt", 2_000_000_000)
    _require(hashlib.sha256(tensor_payload).hexdigest() == entry.combined_tensor_file_sha256, "secure tensor file hash mismatch")
    tensor = torch.load(io.BytesIO(tensor_payload), map_location="cpu", weights_only=True)
    after_directory = directory.lstat()
    _require((before_directory.st_dev, before_directory.st_ino) == (after_directory.st_dev, after_directory.st_ino), "compact directory changed during load")
    _require(tuple(tensor.shape) == (entry.rows, 2048), "loaded WSI tensor shape drift")
    _require(tensor.dtype == torch.float32 and tensor.device.type == "cpu", "loaded WSI tensor dtype/device drift")
    _require(tensor.is_contiguous() and bool(torch.isfinite(tensor).all()), "loaded WSI tensor must be finite contiguous")
    return tensor.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=False)


def _evaluate_partition(torch: Any, model: Any, loss_function: Any, entries: tuple[Any, ...],
                        records: dict[int, Any], omics: dict[int, Any], compact_module: Any,
                        device: Any, *, include_bootstrap_ci: bool = False) -> dict[str, Any]:
    losses: list[float] = []
    events: list[int] = []
    times: list[float] = []
    risks: list[float] = []
    model.eval()
    with torch.no_grad():
        for entry in entries:
            record = records[entry.cohort_index]
            omic = omics[entry.cohort_index]
            wsi = _load_compact_patient(torch, compact_module, entry, device)
            inputs = [
                wsi,
                omic.rna.to(device=device, dtype=torch.float32),
                omic.mutation.to(device=device, dtype=torch.float32),
                omic.cnv.to(device=device, dtype=torch.float32),
            ]
            logits = model(inputs)
            _require(tuple(logits.shape) == (1, 4) and bool(torch.isfinite(logits).all()), "HEALNet output drift")
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            label = torch.tensor([[record.discrete_time_bin]], dtype=torch.int64, device=device)
            censorship = torch.tensor([[record.censorship]], dtype=torch.float32, device=device)
            loss = loss_function(hazards, survival, label, censorship, weights=None, alpha=0.4, eps=1e-7)
            _require(bool(torch.isfinite(loss)), "partition NLL is nonfinite")
            losses.append(float(loss.detach().cpu()))
            events.append(record.event_observed)
            times.append(record.survival_months)
            risks.append(float((-torch.sum(survival, dim=1)).detach().cpu()[0]))
            del inputs, wsi, logits, hazards, survival, loss
    from .brca_healnet_training_contract import harrell_concordance_index, patient_bootstrap_c_index
    concordance = harrell_concordance_index(events, times, risks)
    result = {
        "patients": len(entries),
        "mean_nll": sum(losses) / len(losses),
        "harrell_c_index": concordance.concordance_index,
        "comparable_pairs": concordance.comparable_pairs,
    }
    if include_bootstrap_ci:
        interval = patient_bootstrap_c_index(events, times, risks, replicates=2000, seed=20_260_820)
        result["patient_bootstrap_95_percent_ci"] = [interval.lower, interval.upper]
        result["bootstrap_valid_replicates"] = interval.valid_replicates
        result["bootstrap_requested_replicates"] = interval.requested_replicates
    return result


def execute_authorized_training(
    paths: TrainingRuntimePaths,
    *,
    identity: Any,
    expected_project_commit: str,
) -> dict[str, Any]:
    """Run the frozen protocol. The caller must pass the locked stdlib gate first."""

    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    _require(_git_head(paths.project_root) == expected_project_commit, "project source commit drift")
    _require(_git_head(paths.official_healnet_root) == "28ba5da6ab99fd8069972c22e986d83edb658dd4", "official HEALNet commit drift")
    split_payload = _read_secure(paths.split_manifest)
    cutpoint_payload = _read_secure(paths.cutpoints)
    registry_payload = _read_secure(paths.feature_registry, 16_000_000)
    _require(hashlib.sha256(split_payload).hexdigest() == identity.split_manifest_sha256, "split manifest hash drift")
    _require(hashlib.sha256(cutpoint_payload).hexdigest() == identity.cutpoints_sha256, "cutpoints hash drift")
    _require(hashlib.sha256(registry_payload).hexdigest() == identity.feature_registry_sha256, "feature registry hash drift")

    import torch
    from .brca_compact_feature_artifacts import validate_compact_feature_artifacts
    from . import brca_compact_feature_artifacts as compact_module
    from .brca_healnet_training_contract import (
        TrainingPolicy, parse_feature_registry, validate_training_policy,
        deterministic_epoch_order,
    )
    from .brca_omic import BRCA_RELEASE_ARCHIVE_SHA256, load_brca_patient_omics
    from .brca_survival_protocol import parse_and_validate_split_artifacts
    from .brca_training_checkpoint import EpochState, plan_recovery, publish_checkpoint
    from .brca_training_result import (
        RESULT_DIRECTORY, START_MARKER, begin_locked_test,
        publish_training_result, validate_training_result,
    )

    del validate_compact_feature_artifacts  # imported explicitly to preserve reviewed dependency closure
    policy = TrainingPolicy()
    validate_training_policy(policy)
    split = parse_and_validate_split_artifacts(split_payload, cutpoint_payload)
    registry = parse_feature_registry(registry_payload, split)
    records = {record.cohort_index: record for record in split.records}
    _require(_sha256(paths.omic_archive, 16_000_000) == BRCA_RELEASE_ARCHIVE_SHA256, "Omic archive hash drift")

    _require(torch.cuda.is_available(), "CUDA is required; CPU fallback is prohibited")
    _require(torch.cuda.device_count() == 1, "exactly one visible CUDA device is required")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(policy.seed)
    torch.cuda.manual_seed_all(policy.seed)

    official_model = paths.official_healnet_root / "healnet/models/healnet.py"
    official_loss = paths.official_healnet_root / "healnet/models/survival_loss.py"
    model_payload = _read_secure(official_model)
    loss_payload = _read_secure(official_loss)
    _require(hashlib.sha256(model_payload).hexdigest() == "78e9c2b10455f6ad43a88ed4e0a646cbf6304bf52cac06a5140f49dbc4796820", "official model source drift")
    _require(hashlib.sha256(loss_payload).hexdigest() == "63dcd4c83169dcddea68a3c623ba3486d4263956e14d1a3db9480304470fdd2f", "official loss source drift")
    model_module = _load_module("brca_pinned_healnet_model", official_model, model_payload)
    loss_module = _load_module("brca_pinned_healnet_loss", official_loss, loss_payload)
    model = model_module.HealNet(
        n_modalities=4, channel_dims=[2048, 1558, 21, 1333], num_spatial_axes=[1, 1, 1, 1],
        out_dims=4, depth=2, num_freq_bands=2, max_freq=2.0, l_c=17, l_d=126,
        x_heads=1, l_heads=8, cross_dim_head=63, latent_dim_head=20,
        attn_dropout=0.45526926537716805, ff_dropout=0.364741344399059,
        weight_tie_layers=False, fourier_encode_data=True, self_per_cross_attn=0, snn=True,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=policy.learning_rate)
    steps_per_epoch = math.ceil(660 / policy.gradient_accumulation_patients)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=policy.maximum_learning_rate,
        epochs=policy.maximum_epochs, steps_per_epoch=steps_per_epoch,
    )

    # The Omic archive is small; cache the exact 894 independently matched rows
    # in CPU memory once. WSI bags remain patient-streamed and are never pooled.
    omics: dict[int, Any] = {}
    for entry in registry:
        value = load_brca_patient_omics(
            paths.omic_archive, case_id=entry.patient_id, slide_id=entry.slide_id,
            expected_archive_sha256=None,
        )
        _require(value.source_row_index == str(records[entry.cohort_index].omic_source_row_id), "Omic source-row drift")
        omics[entry.cohort_index] = value

    training = tuple(entry for entry in registry if entry.partition == "training")
    validation = tuple(entry for entry in registry if entry.partition == "validation")
    locked_test = tuple(entry for entry in registry if entry.partition == "locked_test")
    recovery = plan_recovery(paths.checkpoint_root, identity)
    if recovery.action == "TRAINING_EPOCHS_COMPLETE" and os.path.lexists(paths.result_root / RESULT_DIRECTORY):
        return dict(validate_training_result(paths.result_root, identity)["summary"])
    history: list[dict[str, Any]] = []
    best_validation_nll = math.inf
    best_epoch = 0
    best_model_state: dict[str, Any] | None = None
    optimizer_steps = 0
    epochs_without_improvement = 0

    if recovery.latest is not None:
        latest = recovery.latest.directory
        model_payload = torch.load(latest / "model_state.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(model_payload["current"])
        best_model_state = model_payload["best"]
        optimizer.load_state_dict(torch.load(latest / "optimizer_state.pt", map_location=device, weights_only=True))
        scheduler.load_state_dict(torch.load(latest / "scheduler_state.pt", map_location=device, weights_only=True))
        rng = torch.load(latest / "rng_state.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        history = json.loads((latest / "epoch_history.json").read_text(encoding="utf-8"))["epochs"]
        best_validation_nll = recovery.latest.state.best_validation_nll
        best_epoch = recovery.latest.state.best_epoch
        optimizer_steps = recovery.latest.state.optimizer_steps
        epochs_without_improvement = recovery.latest.state.epochs_without_improvement

    for epoch in range(recovery.next_epoch, policy.maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        group_count = 0
        order = deterministic_epoch_order(training, epoch=epoch, seed=policy.seed)
        by_index = {entry.cohort_index: entry for entry in training}
        for position, cohort_index in enumerate(order, start=1):
            entry = by_index[cohort_index]
            record = records[cohort_index]
            omic = omics[cohort_index]
            wsi = _load_compact_patient(torch, compact_module, entry, device)
            inputs = [wsi, omic.rna.to(device), omic.mutation.to(device), omic.cnv.to(device)]
            logits = model(inputs)
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            label = torch.tensor([[record.discrete_time_bin]], dtype=torch.int64, device=device)
            censorship = torch.tensor([[record.censorship]], dtype=torch.float32, device=device)
            nll = loss_module.nll_loss(hazards, survival, label, censorship, weights=None, alpha=0.4, eps=1e-7)
            l1 = policy.l1_coefficient * sum(parameter.abs().sum() for parameter in model.parameters())
            loss = nll + l1
            _require(bool(torch.isfinite(loss)), "training loss is nonfinite")
            loss.backward()
            losses.append(float(loss.detach().cpu()))
            group_count += 1
            if group_count == policy.gradient_accumulation_patients or position == len(order):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_count)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                group_count = 0
            del inputs, wsi, logits, hazards, survival, nll, l1, loss

        validation_result = _evaluate_partition(
            torch, model, loss_module.nll_loss, validation, records, omics, compact_module, device,
        )
        current_validation = float(validation_result["mean_nll"])
        improved = current_validation < best_validation_nll
        if improved:
            best_validation_nll = current_validation
            best_epoch = epoch
            best_model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        early_stop = epochs_without_improvement >= policy.early_stopping_patience
        history.append({
            "epoch": epoch, "training_mean_objective": sum(losses) / len(losses),
            "validation_nll": current_validation,
            "validation_c_index": validation_result["harrell_c_index"],
            "optimizer_steps": optimizer_steps, "best_epoch": best_epoch,
            "best_validation_nll": best_validation_nll, "early_stop_reached": early_stop,
        })
        _require(best_model_state is not None and best_epoch >= 1, "best model state was not established")
        payloads = {
            "model_state.pt": _torch_bytes(torch, {"current": model.state_dict(), "best": best_model_state}),
            "optimizer_state.pt": _torch_bytes(torch, optimizer.state_dict()),
            "scheduler_state.pt": _torch_bytes(torch, scheduler.state_dict()),
            "rng_state.pt": _torch_bytes(torch, {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}),
            "epoch_history.json": (json.dumps({"epochs": history}, sort_keys=True, indent=2) + "\n").encode(),
        }
        publish_checkpoint(
            paths.checkpoint_root, identity=identity,
            state=EpochState(epoch, optimizer_steps, best_epoch, best_validation_nll, epochs_without_improvement, early_stop),
            payloads=payloads,
        )
        if early_stop:
            break

    _require(best_model_state is not None, "best validation checkpoint is absent")
    _require(not os.path.lexists(paths.result_root / START_MARKER), "locked test has already started without a valid result; manual review required")
    begin_locked_test(paths.result_root, identity)
    model.load_state_dict(best_model_state)
    locked_result = _evaluate_partition(
        torch, model, loss_module.nll_loss, locked_test, records, omics, compact_module, device,
        include_bootstrap_ci=True,
    )
    summary = {
        "protocol_id": policy.protocol_id,
        "best_epoch": best_epoch,
        "best_validation_nll": best_validation_nll,
        "locked_test": locked_result,
        "locked_test_evaluations_this_run": 1,
        "training_complete": True,
    }
    publish_training_result(paths.result_root, identity, summary)
    return summary


__all__ = ["TrainingRuntimeError", "TrainingRuntimePaths", "execute_authorized_training"]
