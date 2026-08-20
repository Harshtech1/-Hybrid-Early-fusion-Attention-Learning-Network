"""Fail-closed P0001 recovery-v2 prefix bootstrap.

This CPU-only module validates frozen evidence, derives the exact historical
five-stage/ten-event prefix, builds it in a sibling staging directory with the
existing append-only ledger primitive, and publishes the complete directory
with Linux RENAME_NOREPLACE.  It has no WSI, OpenSlide, pixel, coordinate,
Torch/CUDA, feature, model, network, Drive, deletion, or training surface.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
from pathlib import Path
import stat
import uuid
from typing import Mapping

from .brca_p0001_feature_package import (
    COHORT_INDEX,
    GDC_UUID,
    PATIENT_ID,
    SLIDE_ID,
    TRANSACTION_ID,
    rehearse_p0001_feature_transaction,
)
from .brca_singleton_streaming_policy import PatientStage
from .brca_streaming_recovery_v2 import (
    EventType,
    ReplayAction,
    append_event,
    load_events,
    replay_events,
)


LEDGER_PATH = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/BRCA_PRODUCTION_P0001.recovery_v2"
)
AUTHORIZATION_PATH = Path(
    "multiscale_feature_pilot/config/brca_p0001_recovery_prefix_bootstrap_authorization.yaml"
)
AUTHORIZATION_STATEMENT_SHA256 = (
    "3ada92434dfc6f78bbf29e7d127fe2dd199e8fa11a7ea9356bbcea92e85f88fc"
)
BOOTSTRAP_RECORDED_AT_UTC = "2026-08-20T23:40:00Z"
EXPECTED_STAGES = (
    PatientStage.PLANNED,
    PatientStage.ACQUISITION_AUTHORIZED,
    PatientStage.RAW_VERIFIED,
    PatientStage.HEADER_POLICY_VERIFIED,
    PatientStage.COORDINATES_VERIFIED,
)
EVIDENCE_HASHES = {
    "multiscale_feature_pilot/config/brca_first_eight_production_execution_request.yaml": "04da2f211bfae390ec94ec2dd082ed032204d9c3b81424c6273e33679b14134b",
    "reports/brca_first_eight_canary_proposal.tsv": "940b8fd1f7d194c2c9b7c69ddae58ffff3c55196b6841ac859c60cbc01095dfd",
    "multiscale_feature_pilot/provenance/brca_first_eight_production_manifests/P0001_93b26333-5723-4fa4-a4de-6124c04ab243.REQUEST_ONLY.gdc.tsv": "1c05dc64b6af54604648b52e688e02f611f8e42aaf82cf911c81035b3fc385f2",
    "multiscale_feature_pilot/config/brca_p0001_acquisition_header_authorization.yaml": "691a8d536c9a614a326d26f971f586b7e5ab441a5efea189e347f2c4d664d61a",
    "multiscale_feature_pilot/provenance/brca_p0001_header_metadata_result/result.yaml": "6c7faa0c4e80f4649d784140b907280c1f5b889f5153b4550da9e3e2f198efb3",
    "multiscale_feature_pilot/config/brca_p0001_scale_coordinate_policy.yaml": "d5ddfdf38f98a921a0876e71ed57fbf196407cf8bbf234433f3c0c0a46513cd4",
    "multiscale_feature_pilot/provenance/brca_p0001_scale_coordinate_policy_review.yaml": "aefd0e5de9b41ce726e880b14d5002d4012acbffaed7196ba301b28c323da77c",
    "multiscale_feature_pilot/config/brca_p0001_coordinate_execution_authorization.yaml": "dcf916daf81a25c4a412f4a6aa43fe22bd616ec5755c68f94a7d2f2f2f6f5baa",
    "multiscale_feature_pilot/provenance/brca_p0001_coordinate_execution_result.yaml": "687cc3d7515f102666846310ef00fbad4e1df6b77ee3fd1f4689e659fa0f7ef2",
}
EXTERNAL_EVIDENCE_HASHES = {
    Path("/teamspace/studios/this_studio/brca_pilot_data/BRCA_PRODUCTION_P0001.coordinates/coordinate_manifest.json"):
        "f1825acf7d0b96c92bfb66038d60af6e1572ea945490e2245d5e3ec222677d6e",
}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class PrefixBootstrapError(RuntimeError):
    """Raised before any publication when evidence or state drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrefixBootstrapError(message)


def _sha256_regular_nofollow(path: Path) -> str:
    _validate_no_symlink_ancestors(path)
    _require(os.path.lexists(path), f"required evidence absent: {path}")
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), f"regular non-symlink evidence required: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        _require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "evidence identity changed during secure open")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1_048_576)
            if not chunk:
                break
            digest.update(chunk)
        final = os.fstat(descriptor)
        after = path.lstat()
        token = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        _require(token == (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns), "evidence changed while hashing")
        _require(token == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "evidence pathname changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _validate_no_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate):
            _require(not stat.S_ISLNK(os.lstat(candidate).st_mode), f"symlink path ancestry forbidden: {candidate}")


def _validate_publication_parent(destination: Path) -> None:
    _require(destination.is_absolute(), "ledger destination must be absolute")
    _validate_no_symlink_ancestors(destination.parent)
    _require(os.path.lexists(destination.parent), "ledger parent must already exist")
    details = destination.parent.lstat()
    _require(stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode), "ledger parent must be a regular non-symlink directory")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_frozen_evidence(root: Path, expected: Mapping[str, str] = EVIDENCE_HASHES) -> None:
    for relative, digest in expected.items():
        _require(_sha256_regular_nofollow(root / relative) == digest, f"frozen evidence SHA256 drift: {relative}")
    for path, digest in EXTERNAL_EVIDENCE_HASHES.items():
        _require(_sha256_regular_nofollow(path) == digest, f"external frozen evidence SHA256 drift: {path}")


def derive_prefix(root: Path):
    validation = rehearse_p0001_feature_transaction(
        root / "reports/brca_row_level_alignment.csv",
        through_stage=PatientStage.COORDINATES_VERIFIED,
        event_timestamps=(BOOTSTRAP_RECORDED_AT_UTC,) * 16,
    )
    events = validation.events
    _require(len(events) == 10, "prefix must contain exactly ten events")
    _require([event.sequence for event in events] == list(range(1, 11)), "event sequence drift")
    _require([event.stage for event in events[::2]] == list(EXPECTED_STAGES), "stage order drift")
    _require(all(event.event_type is EventType.STAGE_STARTED for event in events[::2]), "stage-start event drift")
    _require(all(event.event_type is EventType.STAGE_SUCCEEDED for event in events[1::2]), "stage-success event drift")
    _require(all(event.attempt_number == 1 for event in events), "bootstrap cannot synthesize retries")
    identity = events[0].identity
    _require((identity.cohort_index, identity.patient_id, identity.slide_id, identity.gdc_uuid, identity.transaction_id) == (COHORT_INDEX, PATIENT_ID, SLIDE_ID, GDC_UUID, TRANSACTION_ID), "P0001 transaction identity drift")
    replay = replay_events(events)
    _require(replay.action is ReplayAction.ADVANCE_STAGE, "prefix replay is not ready to advance")
    _require(replay.last_durable_stage is PatientStage.COORDINATES_VERIFIED, "prefix durable stage drift")
    _require(replay.target_stage is PatientStage.GPU_AUTHORIZED, "prefix target stage drift")
    return events


def _rename_noreplace(source: Path, destination: Path) -> None:
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    _require(function is not None, "atomic RENAME_NOREPLACE is unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(_AT_FDCWD, os.fsencode(source), _AT_FDCWD, os.fsencode(destination), _RENAME_NOREPLACE)
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise PrefixBootstrapError("destination appeared; no-overwrite publication refused")
        raise PrefixBootstrapError(f"atomic RENAME_NOREPLACE failed: errno={number}")


def publish_prefix(root: Path, destination: Path = LEDGER_PATH) -> tuple[object, ...]:
    """Publish the complete absent prefix once; never delete or overwrite."""

    _validate_publication_parent(destination)
    _require(not os.path.lexists(destination), "destination recovery ledger must be absent")
    validate_frozen_evidence(root)
    events = derive_prefix(root)
    staging = destination.parent / f".{destination.name}.bootstrap-{uuid.uuid4()}"
    _require(not os.path.lexists(staging), "runner staging collision")
    staging.mkdir(mode=0o750, parents=False)
    for event in events:
        append_event(staging, event)
    staged = load_events(staging)
    _require(staged == events, "staged prefix round-trip drift")
    _fsync_directory(staging)
    _require(not os.path.lexists(destination), "destination appeared before publication")
    _rename_noreplace(staging, destination)
    _fsync_directory(destination.parent)
    final = load_events(destination)
    _require(final == events, "published prefix round-trip drift")
    replay = replay_events(final)
    _require(replay.action is ReplayAction.ADVANCE_STAGE and replay.target_stage is PatientStage.GPU_AUTHORIZED, "published replay tip drift")
    return final


__all__ = [
    "AUTHORIZATION_PATH",
    "AUTHORIZATION_STATEMENT_SHA256",
    "BOOTSTRAP_RECORDED_AT_UTC",
    "EVIDENCE_HASHES",
    "EXPECTED_STAGES",
    "LEDGER_PATH",
    "PrefixBootstrapError",
    "derive_prefix",
    "publish_prefix",
    "validate_frozen_evidence",
]
