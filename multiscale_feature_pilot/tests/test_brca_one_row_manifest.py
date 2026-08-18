from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from multiscale_feature_pilot.src.brca_one_row_manifest import (
    AUTHORIZATION_RECORD_BASENAME,
    AUTHORIZATION_STATUS,
    DEFAULT_ALIGNMENT,
    DEFAULT_PROPOSAL,
    DEFAULT_SOURCE_MANIFEST,
    EXPECTED_SELECTIONS,
    GDC_FIELDS,
    OneRowManifestError,
    build_phase2_manifest_set,
    load_reviewed_selections,
    validate_phase2_manifest_set,
)


def _copy(path: Path, destination: Path) -> Path:
    destination.write_bytes(path.read_bytes())
    return destination


def test_builds_exactly_three_distinct_standard_one_row_manifests(
    tmp_path: Path,
) -> None:
    output = tmp_path / "guarded"
    artifacts = build_phase2_manifest_set(output_directory=output)

    assert len(artifacts) == 3
    assert len({item.selection.patient_id for item in artifacts}) == 3
    assert len({item.selection.gdc_file_uuid for item in artifacts}) == 3
    assert {item.path.name for item in artifacts} == {
        item.selection.guarded_basename for item in artifacts
    }
    assert all(AUTHORIZATION_STATUS in item.path.name for item in artifacts)
    assert {path.name for path in output.iterdir()} == {
        *(item.path.name for item in artifacts),
        AUTHORIZATION_RECORD_BASENAME,
    }

    for artifact in artifacts:
        with artifact.path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            rows = list(reader)
        assert tuple(reader.fieldnames or ()) == GDC_FIELDS
        assert rows == [artifact.selection.as_gdc_row()]

    validated = validate_phase2_manifest_set(output_directory=output)
    assert [item.sha256 for item in validated] == [item.sha256 for item in artifacts]


def test_directory_record_is_explicitly_unauthorized_and_has_no_combined_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "guarded"
    build_phase2_manifest_set(output_directory=output)
    record = yaml.safe_load(
        (output / AUTHORIZATION_RECORD_BASENAME).read_text(encoding="utf-8")
    )

    assert record["status"] == AUTHORIZATION_STATUS
    assert record["download_authorized"] is False
    assert record["metadata_only"] is True
    assert record["manifest_count"] == 3
    assert record["combined_manifest_present"] is False
    assert len(record["entries"]) == 3
    assert all(entry["status"] == AUTHORIZATION_STATUS for entry in record["entries"])


def test_reviewed_selection_identities_are_exact() -> None:
    selections = load_reviewed_selections()
    actual = tuple(
        (
            item.label,
            item.patient_id,
            item.gdc_file_uuid,
            item.filename,
            item.md5,
            item.size_bytes,
            item.state,
        )
        for item in selections
    )
    assert actual == EXPECTED_SELECTIONS


def test_proposal_hash_drift_fails_closed(tmp_path: Path) -> None:
    proposal = _copy(DEFAULT_PROPOSAL, tmp_path / "proposal.yaml")
    proposal.write_text(proposal.read_text() + "\n", encoding="utf-8")
    with pytest.raises(OneRowManifestError, match="proposal SHA256 drift"):
        build_phase2_manifest_set(
            output_directory=tmp_path / "output", proposal_path=proposal
        )


def test_alignment_hash_drift_fails_closed(tmp_path: Path) -> None:
    alignment = _copy(DEFAULT_ALIGNMENT, tmp_path / "alignment.csv")
    alignment.write_text(alignment.read_text() + "\n", encoding="utf-8")
    with pytest.raises(OneRowManifestError, match="alignment SHA256 drift"):
        build_phase2_manifest_set(
            output_directory=tmp_path / "output", alignment_path=alignment
        )


def test_source_manifest_hash_drift_fails_closed(tmp_path: Path) -> None:
    source = _copy(DEFAULT_SOURCE_MANIFEST, tmp_path / "source.tsv")
    source.write_text(source.read_text() + "\n", encoding="utf-8")
    with pytest.raises(OneRowManifestError, match="manifest SHA256 drift"):
        build_phase2_manifest_set(
            output_directory=tmp_path / "output", source_manifest_path=source
        )


@pytest.mark.parametrize("mutation", ["duplicate", "extra"])
def test_duplicate_or_extra_proposal_row_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    payload = yaml.safe_load(DEFAULT_PROPOSAL.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        payload["selected_patients"][1] = dict(payload["selected_patients"][0])
    else:
        payload["selected_patients"].append(dict(payload["selected_patients"][0]))
    proposal = tmp_path / "proposal.yaml"
    proposal.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    expected = "duplicate proposal" if mutation == "duplicate" else "exactly 3 rows"
    with pytest.raises(OneRowManifestError, match=expected):
        load_reviewed_selections(proposal, verify_hash=False)


def test_extra_output_file_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "guarded"
    build_phase2_manifest_set(output_directory=output)
    (output / "combined.NOT_AUTHORIZED.gdc.tsv").write_text(
        "\t".join(GDC_FIELDS) + "\n", encoding="utf-8"
    )
    with pytest.raises(OneRowManifestError, match="output file set drift"):
        validate_phase2_manifest_set(output_directory=output)


def test_nested_bulk_directory_fails_validation_and_build_precheck(
    tmp_path: Path,
) -> None:
    output = tmp_path / "guarded"
    build_phase2_manifest_set(output_directory=output)
    nested = output / "bulk"
    nested.mkdir()
    (nested / "combined.gdc.tsv").write_text(
        "\t".join(GDC_FIELDS) + "\n", encoding="utf-8"
    )

    with pytest.raises(OneRowManifestError, match="output file set drift"):
        validate_phase2_manifest_set(output_directory=output)
    with pytest.raises(OneRowManifestError, match="unexpected output entries"):
        build_phase2_manifest_set(output_directory=output)


def test_expected_manifest_symlink_fails_validation_and_build_precheck(
    tmp_path: Path,
) -> None:
    output = tmp_path / "guarded"
    artifacts = build_phase2_manifest_set(output_directory=output)
    target = artifacts[0].path
    referent = tmp_path / "referent.tsv"
    referent.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(referent)

    with pytest.raises(OneRowManifestError, match="must not be a symlink"):
        validate_phase2_manifest_set(output_directory=output)
    with pytest.raises(OneRowManifestError, match="must not be a symlink"):
        build_phase2_manifest_set(output_directory=output)


def test_output_directory_symlink_fails_build_and_validation(tmp_path: Path) -> None:
    target = tmp_path / "target"
    build_phase2_manifest_set(output_directory=target)
    output = tmp_path / "guarded"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(OneRowManifestError, match="directory must not be a symlink"):
        validate_phase2_manifest_set(output_directory=output)
    with pytest.raises(OneRowManifestError, match="directory must not be a symlink"):
        build_phase2_manifest_set(output_directory=output)


def test_extra_data_row_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "guarded"
    artifacts = build_phase2_manifest_set(output_directory=output)
    target = artifacts[0].path
    lines = target.read_text(encoding="utf-8").splitlines()
    target.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(OneRowManifestError, match="exactly one data row"):
        validate_phase2_manifest_set(output_directory=output)


def test_mismatching_selected_md5_fails_closed(tmp_path: Path) -> None:
    source = _copy(DEFAULT_SOURCE_MANIFEST, tmp_path / "source.tsv")
    text = source.read_text(encoding="utf-8")
    expected_md5 = EXPECTED_SELECTIONS[0][4]
    source.write_text(text.replace(expected_md5, "0" * 32, 1), encoding="utf-8")
    with pytest.raises(OneRowManifestError, match="source manifest md5 drift"):
        build_phase2_manifest_set(
            output_directory=tmp_path / "output",
            source_manifest_path=source,
            verify_source_hashes=False,
        )


def test_manifest_basename_guard_is_required(tmp_path: Path) -> None:
    output = tmp_path / "guarded"
    artifacts = build_phase2_manifest_set(output_directory=output)
    target = artifacts[0].path
    unguarded = output / target.name.replace(".NOT_AUTHORIZED", "")
    target.rename(unguarded)
    with pytest.raises(OneRowManifestError, match="output file set drift"):
        validate_phase2_manifest_set(output_directory=output)
