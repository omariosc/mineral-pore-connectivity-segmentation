import hashlib
import json
from pathlib import Path

import pytest

import src.training.screen_selection as selection


def _cell(index, *, campaign="700", failed=False):
    candidate = selection.SCREEN_CANDIDATE_ORDER[index // 3]
    seed = selection.SCREEN_SEEDS[index % 3]
    base_score = 0.4 + 0.1 * (index // 3)
    common = {
        "array_index": index,
        "campaign_id": campaign,
        "candidate": candidate,
        "seed": seed,
        "result_artifact_repo_relative_identifier": f"cell-{index}.json",
        "result_artifact_sha256": f"{index + 1:064x}",
        "source_code_sha256": {"source.py": "a" * 64},
        "smoke_preflight_manifest": {
            "repo_relative_identifier": "smoke.json",
            "sha256": "b" * 64,
            "campaign_id": "600",
        },
    }
    if failed:
        return {
            **common,
            "outcome_status": "failed",
            "failure": {"exit_code": 1},
            "selected_checkpoint_repo_relative_identifier": None,
            "selected_checkpoint_sha256": None,
            "selection_metrics": None,
            "scientific_execution_contract": None,
            "split_manifest_sha256": None,
            "target_mask_aggregate_sha256": None,
            "input_image_aggregate_sha256": None,
            "training_image_aggregate_sha256": None,
        }
    return {
        **common,
        "outcome_status": "success",
        "selected_checkpoint_repo_relative_identifier": f"checkpoint-{index}.pth",
        "selected_checkpoint_sha256": "c" * 64,
        "selection_metrics": {
            "score": base_score,
            "c0_iou": base_score,
            "c1_iou": base_score,
            "pore_union_iou": base_score + 0.01,
            "validation_loss": 1.0 - base_score,
        },
        "scientific_execution_contract": {
            "loss": {
                "training_class_statistics": {
                    "training_mask_aggregate_sha256": "d" * 64,
                }
            }
        },
        "split_manifest_sha256": "e" * 64,
        "target_mask_aggregate_sha256": "f" * 64,
        "input_image_aggregate_sha256": "1" * 64,
        "training_image_aggregate_sha256": "2" * 64,
        "resolved_data_split": {"manifest_sha256": "e" * 64},
        "input_provenance": {"image_aggregate_sha256": "1" * 64},
        "target_provenance": {"mask_aggregate_sha256": "f" * 64},
    }


def _build_with_cells(monkeypatch, cells):
    by_name = {str(index): cell for index, cell in enumerate(cells)}
    monkeypatch.setattr(
        selection,
        "_load_screen_cell",
        lambda path, _root, **_kwargs: by_name[Path(path).name],
    )
    return selection.build_selected_method_lock_document(
        [Path(str(index)) for index in range(15)], Path("unused")
    )


def test_alternative_failed_seed_is_ineligible_without_blocking_lock(monkeypatch):
    cells = [_cell(index) for index in range(15)]
    cells[14] = _cell(14, failed=True)

    lock = _build_with_cells(monkeypatch, cells)

    aggregates = lock["screen_selection_provenance"]["candidate_aggregates"]
    assert aggregates["C2-FP"]["eligible"] is False
    assert aggregates["C2-FP"]["failed_seeds"] == [2025]
    assert aggregates["C2-FP"]["eligibility_reason"] == (
        "ineligible_due_to_one_or_more_failed_seeds"
    )
    assert lock["selected_method"] == "C2-F"
    assert lock["screen_selection_provenance"]["successful_cell_count"] == 14


def test_any_reference_failure_invalidates_the_campaign(monkeypatch):
    cells = [_cell(index) for index in range(15)]
    cells[1] = _cell(1, failed=True)
    with pytest.raises(ValueError, match="R3 has a failed seed"):
        _build_with_cells(monkeypatch, cells)


def test_cells_cannot_be_mixed_across_campaigns_or_source_snapshots(monkeypatch):
    cells = [_cell(index) for index in range(15)]
    cells[8] = _cell(8, campaign="701")
    with pytest.raises(ValueError, match="one immutable array campaign"):
        _build_with_cells(monkeypatch, cells)

    cells = [_cell(index) for index in range(15)]
    cells[8]["source_code_sha256"] = {"source.py": "9" * 64}
    with pytest.raises(ValueError, match="source-code hashes"):
        _build_with_cells(monkeypatch, cells)


def test_checkpoint_authentication_rejects_missing_tampered_and_noncanonical(tmp_path):
    campaign = "700"
    canonical = selection._canonical_cell_identifier(
        "validation_screen_cell", campaign, 0, "checkpoints/best_model.pth"
    )
    path = tmp_path / canonical
    path.parent.mkdir(parents=True)
    path.write_bytes(b"trusted checkpoint")
    digest = selection.sha256_file(path)

    assert selection._resolve_authenticated_cell_checkpoint(
        tmp_path,
        role="validation_screen_cell",
        campaign_id=campaign,
        cell_index=0,
        checkpoint_identifier=canonical,
        expected_sha256=digest,
    ) == path.resolve()
    with pytest.raises(ValueError, match="not canonical"):
        selection._resolve_authenticated_cell_checkpoint(
            tmp_path,
            role="validation_screen_cell",
            campaign_id=campaign,
            cell_index=0,
            checkpoint_identifier="../escape.pth",
            expected_sha256=digest,
        )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        selection._resolve_authenticated_cell_checkpoint(
            tmp_path,
            role="validation_screen_cell",
            campaign_id=campaign,
            cell_index=0,
            checkpoint_identifier=canonical,
            expected_sha256="0" * 64,
        )
    path.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        selection._resolve_authenticated_cell_checkpoint(
            tmp_path,
            role="validation_screen_cell",
            campaign_id=campaign,
            cell_index=0,
            checkpoint_identifier=canonical,
            expected_sha256=digest,
        )


def test_checkpoint_symlink_escape_is_rejected(tmp_path):
    campaign = "700"
    canonical = selection._canonical_cell_identifier(
        "validation_screen_cell", campaign, 0, "checkpoints/best_model.pth"
    )
    outside = tmp_path.parent / "outside-checkpoint.pth"
    outside.write_bytes(b"outside")
    path = tmp_path / canonical
    path.parent.mkdir(parents=True)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic-link component"):
        selection._resolve_authenticated_cell_checkpoint(
            tmp_path,
            role="validation_screen_cell",
            campaign_id=campaign,
            cell_index=0,
            checkpoint_identifier=canonical,
            expected_sha256=selection.sha256_file(outside),
        )


def test_smoke_manifest_requires_one_common_three_path_campaign(monkeypatch):
    cells = []
    for index, candidate in enumerate(selection.SMOKE_CANDIDATE_ORDER):
        cells.append(
            {
                "array_index": index,
                "campaign_id": "600",
                "candidate": candidate,
                "seed": 42,
                "result_artifact_repo_relative_identifier": f"smoke-{index}.json",
                "result_artifact_sha256": f"{index + 1:064x}",
                "selected_checkpoint_repo_relative_identifier": f"smoke-{index}.pth",
                "selected_checkpoint_sha256": "a" * 64,
                "source_code_sha256": {"source.py": "b" * 64},
                "split_manifest_sha256": "c" * 64,
                "target_mask_aggregate_sha256": "d" * 64,
                "input_image_aggregate_sha256": "e" * 64,
                "training_image_aggregate_sha256": "f" * 64,
                "resolved_data_split": {"manifest_sha256": "c" * 64},
                "input_provenance": {"image_aggregate_sha256": "e" * 64},
                "target_provenance": {"mask_aggregate_sha256": "d" * 64},
                "scientific_execution_contract": {"epochs_planned": 1},
                "execution_environment": {"gpu_name": "NVIDIA L40S"},
            }
        )
    by_name = {str(index): cell for index, cell in enumerate(cells)}
    monkeypatch.setattr(
        selection,
        "_load_smoke_cell",
        lambda path, _root, **_kwargs: by_name[Path(path).name],
    )
    document = selection.build_smoke_preflight_manifest_document(
        [Path(str(index)) for index in range(3)], Path("unused")
    )
    assert document["smoke_campaign_provenance"]["candidate_order"] == [
        "R3",
        "C2-F",
        "C2-FP",
    ]
    assert document["smoke_campaign_provenance"][
        "scientific_result_eligible"
    ] is False

    cells[2]["campaign_id"] = "601"
    with pytest.raises(ValueError, match="one array campaign"):
        selection.build_smoke_preflight_manifest_document(
            [Path(str(index)) for index in range(3)], Path("unused")
        )


def test_data_attestation_authenticates_exact_manifest_ids_and_files(
    tmp_path, monkeypatch
):
    annotation_path = (
        tmp_path / "results/step3_coco_dataset/pore_annotations.json"
    )
    manifest_path = tmp_path / "config/confirmatory_splits.json"
    annotation_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    images = [
        {
            "id": image_id,
            "file_name": f"tile_{image_id:03d}.png",
            "height": 1,
            "width": 1,
        }
        for image_id in range(1, 101)
    ]
    annotation_path.write_text(
        json.dumps({"images": images, "annotations": [], "categories": []}),
        encoding="utf-8",
    )
    manifest = {
        "_provenance": {
            "train_series": ["train-group"],
            "validation_series": ["val-group"],
            "test_series": ["test-group"],
        },
        "train": list(range(1, 75)),
        "val": list(range(75, 80)),
        "test": list(range(80, 101)),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    file_by_id = {image["id"]: image["file_name"] for image in images}
    partitions = {
        name: {
            "image_ids": list(manifest[name]),
            "image_files": [file_by_id[value] for value in manifest[name]],
            "image_count": len(manifest[name]),
        }
        for name in ("train", "val", "test")
    }
    assignment_hash = hashlib.sha256(
        json.dumps(
            partitions, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    data_split = {
        "manifest_source": "explicit_manifest",
        "manifest_repo_relative_identifier": "config/confirmatory_splits.json",
        "manifest_sha256": selection.sha256_file(manifest_path),
        "annotation_index_repo_relative_identifier": (
            "results/step3_coco_dataset/pore_annotations.json"
        ),
        "annotation_index_sha256": selection.sha256_file(annotation_path),
        "partition_assignment_sha256": assignment_hash,
        "allocation_unit": "leading_source_identifier_group",
        "observation_unit": "2048x2048_tile",
        "group_membership_map": {
            "train": ["train-group"],
            "val": ["val-group"],
            "test": ["test-group"],
        },
        "specimen_independence_confirmation": "pending_data_owner_confirmation",
        "group_semantics": (
            "filename_derived_acquisition_series_kept_wholly_within_one_partition"
        ),
        "partitions": partitions,
        "validation_only": True,
        "held_out_dataset_constructed": False,
        "held_out_evaluation_count": 0,
    }
    target = {
        "mask_count": 79,
        "mask_aggregate_sha256": "a" * 64,
        "held_out_dataset_constructed": False,
    }
    monkeypatch.setattr(
        selection, "CONFIRMATORY_SPLIT_MANIFEST_SHA256",
        data_split["manifest_sha256"],
    )
    monkeypatch.setattr(
        selection, "CONFIRMATORY_ANNOTATION_SHA256",
        data_split["annotation_index_sha256"],
    )
    monkeypatch.setattr(
        selection,
        "CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS",
        {
            "train_plus_validation": {
                "mask_count": 79,
                "mask_aggregate_sha256": "a" * 64,
            }
        },
    )
    monkeypatch.setattr(
        selection,
        "aggregate_indexed_file_bytes",
        lambda *_args, **_kwargs: {
            "image_count": 79,
            "image_aggregate_sha256": "a" * 64,
        },
    )

    selection._validate_data_attestations(
        data_split, target, tmp_path, "valid-cell"
    )

    tampered = json.loads(json.dumps(data_split))
    tampered["partitions"]["train"]["image_ids"][0] = 75
    with pytest.raises(ValueError, match="partition IDs/files"):
        selection._validate_data_attestations(
            tampered, target, tmp_path, "tampered-cell"
        )


def _corpus_free_attestations():
    train_files = [f"pdo1_12_segment_{index}.png" for index in range(71)] + [
        "pdo1_7_segment_0.png",
        "pdo4_1_140721_segment_0.png",
        "pdo4_2_151020_segment_0.png",
    ]
    val_files = [f"pdo8_21_segment_{index}.png" for index in range(5)]
    test_files = [f"pdo2_24_segment_{index}.png" for index in range(21)]
    files = {"train": train_files, "val": val_files, "test": test_files}
    ids = {
        "train": list(range(1, 75)),
        "val": list(range(75, 80)),
        "test": list(range(80, 101)),
    }
    partitions = {
        name: {
            "image_ids": ids[name],
            "image_files": files[name],
            "image_count": len(files[name]),
        }
        for name in ("train", "val", "test")
    }
    assignment_sha256 = hashlib.sha256(
        json.dumps(partitions, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    data_split = {
        "manifest_source": "explicit_manifest",
        "manifest_sha256": selection.CONFIRMATORY_SPLIT_MANIFEST_SHA256,
        "manifest_repo_relative_identifier": "config/confirmatory_splits.json",
        "annotation_index_sha256": selection.CONFIRMATORY_ANNOTATION_SHA256,
        "annotation_index_repo_relative_identifier": (
            "results/step3_coco_dataset/pore_annotations.json"
        ),
        "partition_assignment_sha256": assignment_sha256,
        "partitions": partitions,
        "validation_only": True,
        "held_out_dataset_constructed": False,
        "held_out_evaluation_count": 0,
        "allocation_unit": "leading_source_identifier_group",
        "observation_unit": "2048x2048_tile",
        "specimen_independence_confirmation": "pending_data_owner_confirmation",
        "group_semantics": (
            "filename_derived_acquisition_series_kept_wholly_within_one_partition"
        ),
        "group_membership_map": selection.CANONICAL_GROUP_MEMBERSHIP_MAP,
    }
    target = {
        "mask_count": 79,
        "mask_aggregate_sha256": selection.CONFIRMATORY_DEVELOPMENT_TARGET_ATTESTATIONS[
            "train_plus_validation"
        ]["mask_aggregate_sha256"],
        "held_out_dataset_constructed": False,
    }
    algorithm = (
        "sha256 over lexicographically sorted UTF-8 relative filename, "
        "NUL, raw file bytes, NUL"
    )

    def name_sha(names):
        return hashlib.sha256(
            json.dumps(sorted(names), separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    input_provenance = {
        "input_source": "indexed_source_images",
        "scope": "development_train_plus_validation",
        "split_names": ["train", "val"],
        "held_out_bytes_read": 0,
        "image_count": 79,
        "image_aggregate_sha256": selection.CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS[
            "train_plus_validation"
        ]["image_aggregate_sha256"],
        "image_aggregate_sha256_algorithm": algorithm,
        "file_name_list_sha256": name_sha(train_files + val_files),
        "training_subset": {
            "scope": "training_only",
            "split_names": ["train"],
            "image_count": 74,
            "image_aggregate_sha256": selection.CONFIRMATORY_DEVELOPMENT_IMAGE_ATTESTATIONS[
                "train"
            ]["image_aggregate_sha256"],
            "image_aggregate_sha256_algorithm": algorithm,
            "file_name_list_sha256": name_sha(train_files),
        },
    }
    return data_split, target, input_provenance


def test_corpus_free_attestation_verifies_recorded_contract_without_data_files(
    tmp_path, monkeypatch
):
    data_split, target, input_provenance = _corpus_free_attestations()

    def forbidden_byte_read(*_args, **_kwargs):
        raise AssertionError("corpus bytes must not be opened")

    monkeypatch.setattr(
        selection, "aggregate_indexed_file_bytes", forbidden_byte_read
    )
    selection._validate_data_attestations(
        data_split,
        target,
        tmp_path,
        "freeze",
        verify_live_corpus_bytes=False,
    )
    selection._validate_input_provenance(
        input_provenance,
        data_split,
        tmp_path,
        "freeze",
        verify_live_corpus_bytes=False,
    )
    assert not (tmp_path / "results").exists()

    with pytest.raises(ValueError, match="split/annotation files changed"):
        selection._validate_data_attestations(
            data_split, target, tmp_path, "strict-default"
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda split, _inputs: split["partitions"]["test"]["image_ids"].__setitem__(
                0, 1
            ),
            "isolation/group allocation",
        ),
        (
            lambda split, _inputs: split["partitions"]["val"]["image_files"].__setitem__(
                0, "pdo2_24_segment_wrong.png"
            ),
            "filename/group mapping",
        ),
        (
            lambda _split, inputs: inputs.__setitem__(
                "file_name_list_sha256", "0" * 64
            ),
            "filename provenance",
        ),
    ],
)
def test_corpus_free_attestation_rejects_recorded_partition_or_name_drift(
    tmp_path, mutator, message
):
    data_split, target, input_provenance = _corpus_free_attestations()
    mutator(data_split, input_provenance)
    with pytest.raises(ValueError, match=message):
        selection._validate_data_attestations(
            data_split,
            target,
            tmp_path,
            "freeze",
            verify_live_corpus_bytes=False,
        )
        selection._validate_input_provenance(
            input_provenance,
            data_split,
            tmp_path,
            "freeze",
            verify_live_corpus_bytes=False,
        )
