from __future__ import annotations

from scripts.build_brca_three_patient_proposal import (
    EXPECTED_PROPOSAL,
    EXPECTED_PROPOSAL_TOTAL_BYTES,
    SingletonCandidate,
    nearest_observed_index,
    select_quantile_representatives,
    validate_official_proposal,
)


def _candidate(index: int, size: int) -> SingletonCandidate:
    patient = f"TCGA-AA-{index:04d}"
    return SingletonCandidate(
        patient_id=patient,
        omic_source_index=index,
        omic_csv_line=index + 2,
        slide_id=f"{patient}-01Z-00-DX1.UUID{index}.svs",
        wsi_uuid=f"00000000-0000-0000-0000-{index:012d}",
        wsi_size_bytes=size,
        wsi_md5=f"{index:032x}",
        gdc_state="released",
    )


def test_nearest_observed_inclusive_quantile_ranks_for_894() -> None:
    assert nearest_observed_index(894, 0.25) + 1 == 224
    assert nearest_observed_index(894, 0.50) + 1 == 448
    assert nearest_observed_index(894, 0.75) + 1 == 671


def test_selection_sorts_by_size_then_patient_then_uuid() -> None:
    candidates = [_candidate(index, 1000 - index) for index in range(1, 9)]
    selections = select_quantile_representatives(candidates)

    assert [item.one_based_rank for item in selections] == [3, 5, 6]
    ordered = sorted(
        candidates,
        key=lambda item: (item.wsi_size_bytes, item.patient_id, item.wsi_uuid),
    )
    assert [item.candidate for item in selections] == [
        ordered[2],
        ordered[4],
        ordered[5],
    ]


def test_selection_keeps_three_distinct_patients() -> None:
    candidates = [_candidate(index, index * 100) for index in range(1, 10)]
    selections = select_quantile_representatives(candidates)

    assert len({item.candidate.patient_id for item in selections}) == 3


def test_reviewed_official_proposal_identity_is_frozen() -> None:
    candidates = []
    for position, (_, _, patient, wsi_uuid, size, md5) in enumerate(
        EXPECTED_PROPOSAL
    ):
        candidates.append(
            SingletonCandidate(
                patient_id=patient,
                omic_source_index=position,
                omic_csv_line=position + 2,
                slide_id=f"{patient}-01Z-00-DX1.REVIEWED{position}.svs",
                wsi_uuid=wsi_uuid,
                wsi_size_bytes=size,
                wsi_md5=md5,
                gdc_state="released",
            )
        )

    # Validation consumes quantile selections, so construct the reviewed ranks
    # explicitly; the live CLI independently derives them from all 894 rows.
    from scripts.build_brca_three_patient_proposal import QuantileSelection

    selections = [
        QuantileSelection(
            label=label,
            quantile=quantile,
            one_based_rank=rank,
            candidate=candidate,
        )
        for (label, rank, *_), quantile, candidate in zip(
            EXPECTED_PROPOSAL, (0.25, 0.50, 0.75), candidates, strict=True
        )
    ]
    validate_official_proposal(selections)
    assert sum(item.candidate.wsi_size_bytes for item in selections) == (
        EXPECTED_PROPOSAL_TOTAL_BYTES
    )
