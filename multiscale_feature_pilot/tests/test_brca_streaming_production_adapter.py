from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path
from uuid import UUID

import pytest

from multiscale_feature_pilot.src.brca_singleton_streaming_policy import PatientStage
from multiscale_feature_pilot.src.brca_streaming_executor_v2 import (
    STAGE_CONTRACTS,
    resume_pending_plan,
    start_event_from_plan,
    validate_next_patient_gate,
)
from multiscale_feature_pilot.src.brca_streaming_production_adapter import (
    ALIGNMENT_SHA256,
    COHORT_ORDER_SHA256,
    COHORT_RAW_BYTES,
    COMPACT_FILES,
    SOURCE_POLICY_HASHES,
    CompactArtifactValidationEvidence,
    FORBIDDEN_OPERATION_SURFACES,
    ProductionAdapterError,
    ValidatedStageOutcome,
    append_control_event,
    exact_manifest_bytes,
    load_frozen_cohort_order,
    plan_bound_stage,
    transaction_identity,
    validate_compact_artifact_evidence,
    validate_exact_manifest_bytes,
    validate_exact_omic_rematch,
    validated_success_event,
)
from multiscale_feature_pilot.src.brca_streaming_recovery_v2 import (
    ReplayAction,
    load_events,
    replay_events,
)


ROOT = Path(__file__).resolve().parents[2]
ALIGNMENT = ROOT / "reports/brca_row_level_alignment.csv"
SOURCE = ROOT / "multiscale_feature_pilot/src/brca_streaming_production_adapter.py"
CANARY_TSV = ROOT / "reports/brca_first_eight_canary_proposal.tsv"
CANARY_TSV_SHA256 = "940b8fd1f7d194c2c9b7c69ddae58ffff3c55196b6841ac859c60cbc01095dfd"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _uuid(number: int) -> str:
    return str(UUID(int=number + 10_000))


def _compact(binding, plan=None, rows: int = 12, split: int = 9) -> CompactArtifactValidationEvidence:
    manifest = _hash("compact manifest")
    source_hashes = tuple(SOURCE_POLICY_HASHES.items()) if plan is None else plan.source_policy_hashes
    input_hashes = (("checkpoint_identity", _hash("checkpoint")),) if plan is None else plan.input_hashes
    return CompactArtifactValidationEvidence(
        patient_id=binding.patient_id,
        slide_id=binding.slide_id,
        gdc_uuid=binding.gdc_uuid,
        omic_source_row_id=binding.omic_source_row_id,
        bound_source_policy_hashes=source_hashes,
        bound_input_hashes=input_hashes,
        exact_files=COMPACT_FILES,
        manifest_sha256=manifest,
        sidecar_manifest_sha256=manifest,
        file_hashes=tuple((name, manifest if name == "compact_manifest.json" else _hash(name)) for name in COMPACT_FILES),
        tensor_shape=(rows, 2048),
        tensor_dtype="float32",
        tensor_device="cpu",
        tensor_contiguous=True,
        tensor_finite=True,
        tensor_requires_grad=False,
        scale_2x_row_range=(0, split),
        scale_4x_row_range=(split, rows),
        row_provenance_count=rows,
    )


def _inputs(stage: PatientStage, binding) -> dict[str, str]:
    values = {label: _hash(f"{binding.cohort_index}:{stage.value}:{label}") for label in STAGE_CONTRACTS[stage].required_input_labels}
    if stage is PatientStage.PLANNED:
        values = {"cohort_alignment": ALIGNMENT_SHA256, "singleton_identity": binding.identity_sha256}
    elif stage is PatientStage.ACQUISITION_AUTHORIZED:
        values = {"one_row_manifest": binding.manifest_sha256}
    elif stage is PatientStage.RAW_VERIFIED:
        values = {"raw_identity_declaration": binding.raw_identity_declaration_sha256}
    return values


def _run_full_transaction(binding, transaction_number: int, ledger: Path, *, compact_transform=None):
    identity = transaction_identity(binding, _uuid(transaction_number))
    events = []
    for number, stage in enumerate(PatientStage, start=1):
        assert replay_events(events).target_stage is stage
        authorization = _hash(f"authorization:{binding.cohort_index}:{stage.value}")
        plan = plan_bound_stage(
            events,
            binding=binding,
            identity=identity,
            run_id=_uuid(transaction_number * 100 + number),
            authorization_sha256=authorization,
            retry_authorization_sha256=None,
            stage_input_hashes=_inputs(stage, binding),
        )
        start = start_event_from_plan(events, plan, recorded_at_utc=f"2026-08-20T10:{number * 2:02d}:00Z")
        append_control_event(ledger, start)
        events.append(start)
        pending = resume_pending_plan(events)
        output_label = pending.required_success_output_labels[0]
        compact = _compact(binding, pending) if stage is PatientStage.FEATURES_VERIFIED else None
        if compact is not None and compact_transform is not None:
            compact = compact_transform(compact)
        output_hash = compact.manifest_sha256 if compact is not None else _hash(f"output:{binding.cohort_index}:{stage.value}")
        outcome = ValidatedStageOutcome(
            stage=stage,
            authorization_sha256=authorization,
            output_hashes=((output_label, output_hash),),
            validation_record_sha256=output_hash,
            compact_artifact=compact,
        )
        success = validated_success_event(
            events,
            pending,
            outcome,
            existing_output_hashes=None,
            recorded_at_utc=f"2026-08-20T10:{number * 2 + 1:02d}:00Z",
        )
        append_control_event(ledger, success)
        events.append(success)
    assert load_events(ledger) == tuple(events)
    assert replay_events(events).action is ReplayAction.STOP_TERMINAL
    return events


def test_secure_loader_reproduces_exact_894_order_and_first_eight() -> None:
    cohort = load_frozen_cohort_order(ALIGNMENT)
    assert len(cohort) == 894
    assert sum(item.size_bytes for item in cohort) == COHORT_RAW_BYTES
    assert cohort[0].patient_id == "TCGA-3C-AALK"
    assert cohort[7].patient_id == "TCGA-A1-A0SE"
    assert sum(item.size_bytes for item in cohort[:8]) == 8_297_129_620


def test_loader_rejects_hash_drift_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(ProductionAdapterError, match="alignment SHA256"):
        load_frozen_cohort_order(ALIGNMENT, expected_alignment_sha256="0" * 64)
    linked = tmp_path / "alignment.csv"
    linked.symlink_to(ALIGNMENT)
    with pytest.raises(ProductionAdapterError, match="symlink"):
        load_frozen_cohort_order(linked)


def test_exact_manifest_and_omic_rematch_bind_patient_slide_uuid_and_row() -> None:
    binding = load_frozen_cohort_order(ALIGNMENT)[0]
    payload = exact_manifest_bytes(binding)
    assert validate_exact_manifest_bytes(binding, payload) == binding.manifest_sha256
    with pytest.raises(ProductionAdapterError, match="manifest"):
        validate_exact_manifest_bytes(binding, payload + b"\n")
    digest = validate_exact_omic_rematch(
        binding,
        patient_id=binding.patient_id,
        slide_id=binding.slide_id,
        source_row_id=binding.omic_source_row_id,
    )
    assert digest == binding.omic_identity_sha256
    with pytest.raises(ProductionAdapterError, match="source-row"):
        validate_exact_omic_rematch(
            binding,
            patient_id=binding.patient_id,
            slide_id=binding.slide_id,
            source_row_id="9999",
        )


def test_compact_evidence_is_strict_and_has_no_tensor_or_operation_surface() -> None:
    binding = load_frozen_cohort_order(ALIGNMENT)[0]
    evidence = _compact(binding)
    assert validate_compact_artifact_evidence(evidence) == evidence.manifest_sha256
    with pytest.raises(ProductionAdapterError, match="row provenance"):
        validate_compact_artifact_evidence(replace(evidence, row_provenance_count=11))
    with pytest.raises(ProductionAdapterError, match="contiguous"):
        validate_compact_artifact_evidence(replace(evidence, scale_4x_row_range=(10, 12)))
    assert FORBIDDEN_OPERATION_SURFACES and not any(FORBIDDEN_OPERATION_SURFACES.values())


def test_special_stage_bindings_fail_closed() -> None:
    binding = load_frozen_cohort_order(ALIGNMENT)[0]
    identity = transaction_identity(binding, _uuid(1))
    values = _inputs(PatientStage.PLANNED, binding)
    values["singleton_identity"] = "0" * 64
    with pytest.raises(ProductionAdapterError, match="singleton identity"):
        plan_bound_stage(
            [], binding=binding, identity=identity, run_id=_uuid(2),
            authorization_sha256=_hash("auth"), retry_authorization_sha256=None,
            stage_input_hashes=values,
        )
    with pytest.raises(ProductionAdapterError, match="transaction UUID"):
        plan_bound_stage(
            [], binding=binding, identity=replace(identity, gdc_uuid=_uuid(3)), run_id=_uuid(4),
            authorization_sha256=_hash("auth"), retry_authorization_sha256=None,
            stage_input_hashes=_inputs(PatientStage.PLANNED, binding),
        )


def test_first_eight_synthetic_end_to_end_transactions_are_serial(tmp_path: Path) -> None:
    cohort = load_frozen_cohort_order(ALIGNMENT)
    previous = []
    for index, binding in enumerate(cohort[:8], start=1):
        identity = transaction_identity(binding, _uuid(index))
        validate_next_patient_gate(previous, next_identity=identity)
        ledger = tmp_path / f"patient-{index:04d}"
        previous = _run_full_transaction(binding, index, ledger)
        assert len(previous) == 16


def test_feature_success_requires_strict_compact_evidence(tmp_path: Path) -> None:
    binding = load_frozen_cohort_order(ALIGNMENT)[0]
    events = _run_full_transaction(binding, 1, tmp_path / "complete")
    assert replay_events(events).action is ReplayAction.STOP_TERMINAL
    bad = replace(_compact(binding), tensor_shape=(12, 1024))
    with pytest.raises(ProductionAdapterError, match="2048"):
        validate_compact_artifact_evidence(bad)
    with pytest.raises(ProductionAdapterError, match="patient differs"):
        _run_full_transaction(
            binding,
            2,
            tmp_path / "wrong-patient",
            compact_transform=lambda evidence: replace(evidence, patient_id="TCGA-00-0000"),
        )


def test_success_rejects_validation_record_not_present_in_ledger_outputs() -> None:
    binding = load_frozen_cohort_order(ALIGNMENT)[0]
    identity = transaction_identity(binding, _uuid(200))
    authorization = _hash("planned auth")
    plan = plan_bound_stage(
        [], binding=binding, identity=identity, run_id=_uuid(201),
        authorization_sha256=authorization, retry_authorization_sha256=None,
        stage_input_hashes=_inputs(PatientStage.PLANNED, binding),
    )
    start = start_event_from_plan([], plan, recorded_at_utc="2026-08-20T10:00:00Z")
    pending = resume_pending_plan([start])
    outcome = ValidatedStageOutcome(
        stage=PatientStage.PLANNED,
        authorization_sha256=authorization,
        output_hashes=(("patient_plan", _hash("patient plan")),),
        validation_record_sha256=_hash("unrecorded validation"),
    )
    with pytest.raises(ProductionAdapterError, match="ledger-recorded"):
        validated_success_event(
            [start], pending, outcome, existing_output_hashes=None,
            recorded_at_utc="2026-08-20T10:01:00Z",
        )


def test_frozen_constants_are_exact() -> None:
    assert COHORT_ORDER_SHA256 == "1c97fa4f8305185f2da191f5ebaed603db7d2bdd11c89a580e784ef46655af5a"
    assert ALIGNMENT_SHA256 == "13b1e8e58b28d4669d8015f759e7d6df3f3296a16f77920b6a83a099999c19fe"


def test_source_has_no_real_operation_import_or_call_surface() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    forbidden_imports = {"openslide", "torch", "torchvision", "requests", "urllib", "socket", "subprocess"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(forbidden_imports)
    forbidden_calls = {
        "read_region", "get_thumbnail", "unlink", "rmdir", "remove", "rename",
        "replace", "run", "Popen", "system", "urlopen", "cuda",
    }
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert called.isdisjoint(forbidden_calls)


def test_first_eight_proposal_tsv_is_exactly_derived_from_frozen_order() -> None:
    cohort = load_frozen_cohort_order(ALIGNMENT)
    header = (
        "cohort_index\tpatient_id\tslide_id\tgdc_uuid\tomic_source_row_id\t"
        "filename\tmd5\tsize_bytes\tstate\tdisposition\n"
    )
    rows = [
        "\t".join(
            (
                str(item.cohort_index), item.patient_id, item.slide_id, item.gdc_uuid,
                item.omic_source_row_id, item.filename, item.md5, str(item.size_bytes),
                item.state, "PROPOSED_NOT_AUTHORIZED",
            )
        )
        + "\n"
        for item in cohort[:8]
    ]
    expected = (header + "".join(rows)).encode("utf-8")
    actual = CANARY_TSV.read_bytes()
    assert actual == expected
    assert hashlib.sha256(actual).hexdigest() == CANARY_TSV_SHA256


def test_source_policy_drift_is_rejected_before_planning() -> None:
    binding = load_frozen_cohort_order(ALIGNMENT)[0]
    policies = dict(SOURCE_POLICY_HASHES)
    policies["compact_artifact_policy"] = "0" * 64
    with pytest.raises(ProductionAdapterError, match="source policy"):
        plan_bound_stage(
            [], binding=binding, identity=transaction_identity(binding, _uuid(100)),
            run_id=_uuid(101), authorization_sha256=_hash("auth"),
            retry_authorization_sha256=None,
            stage_input_hashes=_inputs(PatientStage.PLANNED, binding),
            source_policy_hashes=policies,
        )
