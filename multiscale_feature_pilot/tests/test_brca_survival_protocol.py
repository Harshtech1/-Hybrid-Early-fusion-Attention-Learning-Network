from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "multiscale_feature_pilot/src/brca_survival_protocol.py"
ALIGNMENT = ROOT / "reports/brca_row_level_alignment.csv"
OMIC = ROOT.parent / "Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip"


def _module():
    spec = importlib.util.spec_from_file_location("brca_survival_protocol_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _split():
    protocol = _module()
    return protocol, protocol.build_frozen_split(protocol.load_endpoint_records(ALIGNMENT, OMIC))


def test_exact_endpoint_population_and_split() -> None:
    _, split = _split()
    counts = {
        partition: sum(record.partition == partition for record in split.records)
        for partition in ("training", "validation", "locked_test")
    }
    events = {
        partition: sum(record.partition == partition and record.event_observed for record in split.records)
        for partition in counts
    }
    assert counts == {"training": 660, "validation": 116, "locked_test": 118}
    assert events == {"training": 79, "validation": 14, "locked_test": 33}
    assert split.cutpoints_months == tuple(map(_module().Decimal, ("22.505", "44.84", "67.940")))
    assert split.train_validation_ks <= _module().MAX_SURVIVAL_KS


def test_training_event_bins_are_nonempty_and_held_out_bins_use_frozen_cutpoints() -> None:
    _, split = _split()
    assert [
        sum(record.partition == "training" and record.event_observed and record.discrete_time_bin == index for record in split.records)
        for index in range(4)
    ] == [20, 19, 20, 20]
    assert all(0 <= record.discrete_time_bin <= 3 for record in split.records)


def test_serialized_artifacts_round_trip_and_bind_each_other() -> None:
    protocol, split = _split()
    manifest = protocol.split_manifest_bytes(split.records)
    cutpoints = protocol.cutpoints_bytes(split)
    parsed = protocol.parse_and_validate_split_artifacts(manifest, cutpoints)
    assert parsed == split
    assert hashlib.sha256(manifest).hexdigest() == split.manifest_sha256
    assert hashlib.sha256(cutpoints).hexdigest() == split.cutpoints_sha256
    with pytest.raises(protocol.SurvivalProtocolError, match="bind split"):
        protocol.parse_and_validate_split_artifacts(manifest + b"\n", cutpoints)


def test_alignment_hash_and_symlinks_fail_closed(tmp_path: Path) -> None:
    protocol = _module()
    linked = tmp_path / "alignment.csv"
    linked.symlink_to(ALIGNMENT)
    with pytest.raises(protocol.SurvivalProtocolError, match="symlink"):
        protocol.load_endpoint_records(linked, OMIC)
    altered = tmp_path / "alignment-copy.csv"
    altered.write_bytes(ALIGNMENT.read_bytes() + b"\n")
    with pytest.raises(protocol.SurvivalProtocolError, match="SHA256"):
        protocol.load_endpoint_records(altered, OMIC)


def test_split_is_independent_of_input_order() -> None:
    protocol = _module()
    records = protocol.load_endpoint_records(ALIGNMENT, OMIC)
    forward = protocol.build_frozen_split(records)
    reverse = protocol.build_frozen_split(tuple(reversed(records)))
    assert protocol.split_manifest_bytes(forward.records) == protocol.split_manifest_bytes(reverse.records)


def test_writer_is_exclusive_and_round_trips(tmp_path: Path) -> None:
    protocol, split = _split()
    manifest = tmp_path / "split.tsv"
    cutpoints = tmp_path / "cutpoints.json"
    protocol.write_split_artifacts(split, manifest, cutpoints)
    assert protocol.parse_and_validate_split_artifacts(manifest.read_bytes(), cutpoints.read_bytes()) == split
    with pytest.raises(protocol.SurvivalProtocolError, match="overwrite"):
        protocol.write_split_artifacts(split, manifest, cutpoints)
