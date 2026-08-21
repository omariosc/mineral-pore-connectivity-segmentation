import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aire_confirmatory.slurm"
LOCKED_EVALUATION_SCRIPT = ROOT / "scripts" / "aire_locked_evaluation.slurm"


def _write_executable(path, body):
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run(tmp_path, mode="screen", task=0, python_status=0, **overrides):
    bin_dir = tmp_path / "bin"
    project_dir = tmp_path / "project"
    conda_base = tmp_path / "conda"
    profile_dir = conda_base / "etc" / "profile.d"
    bin_dir.mkdir(parents=True)
    project_dir.mkdir()
    profile_dir.mkdir(parents=True)
    _write_executable(bin_dir / "module", "#!/bin/bash\nexit 0\n")
    _write_executable(
        bin_dir / "conda",
        "#!/bin/bash\n"
        "if [[ \"$1\" == info && \"$2\" == --base ]]; then\n"
        "  printf '%s\\n' \"$MOCK_CONDA_BASE\"\n"
        "fi\n",
    )
    (profile_dir / "conda.sh").write_text(
        "conda() { return 0; }\n", encoding="utf-8"
    )
    _write_executable(
        bin_dir / "nvidia-smi",
        "#!/bin/bash\nprintf '%s\\n' \"${MOCK_GPU_NAME:-NVIDIA L40S}\"\n",
    )
    capture = tmp_path / "python_calls.txt"
    _write_executable(
        bin_dir / "python3",
        "#!/bin/bash\n"
        "printf '%s\\n' __CALL__ \"$@\" >> \"$PORE_ARGUMENT_CAPTURE\"\n"
        "if [[ \"${1:-}\" == scripts/train_patches.py ]]; then\n"
        "  exit \"$MOCK_TRAIN_STATUS\"\n"
        "fi\n"
        "exit 0\n",
    )
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("BASH_FUNC_") or key.startswith("PORE_") or key in {
            "QUICK_TEST",
            "SINGLE_BATCH",
            "DISABLE_AMP",
            "DISABLE_TRANSFORMS",
            "BOOTSTRAP_FACTOR",
        }:
            environment.pop(key)
    environment.update(
        {
            "PATH": f"{bin_dir}:{environment['PATH']}",
            "MOCK_CONDA_BASE": str(conda_base),
            "MOCK_TRAIN_STATUS": str(python_status),
            "PORE_ARGUMENT_CAPTURE": str(capture),
            "PORE_PROJECT_DIR": str(project_dir),
            "SLURM_JOB_ID": "900_0",
            "SLURM_ARRAY_JOB_ID": "900",
            "SLURM_ARRAY_TASK_ID": str(task),
            "PORE_RUN_MODE": mode,
            "PORE_LAUNCH_WRAPPER": mode,
        }
    )
    if mode in {"screen", "smoke"}:
        environment["PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE"] = "1"
    if mode == "screen":
        environment["PORE_SMOKE_PREFLIGHT_MANIFEST"] = (
            "results/patch_training/protocol_runs/smoke_manifest.json"
        )
    environment.update({key: str(value) for key, value in overrides.items()})
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    calls = []
    if capture.exists():
        current = None
        for line in capture.read_text(encoding="utf-8").splitlines():
            if line == "__CALL__":
                current = []
                calls.append(current)
            else:
                current.append(line)
    return result, calls


def _value(arguments, option):
    return arguments[arguments.index(option) + 1]


def _run_locked_evaluation(
    tmp_path,
    *,
    task=0,
    architecture_role="primary_multiscale",
    freeze_id="neural-freeze-aaaaaaaaaaaaaaaa",
    **overrides,
):
    bin_dir = tmp_path / "bin"
    project_dir = tmp_path / "project"
    conda_base = tmp_path / "conda"
    profile_dir = conda_base / "etc" / "profile.d"
    bin_dir.mkdir(parents=True)
    project_dir.mkdir()
    profile_dir.mkdir(parents=True)
    _write_executable(bin_dir / "module", "#!/bin/bash\nexit 0\n")
    _write_executable(
        bin_dir / "conda",
        "#!/bin/bash\n"
        "if [[ \"$1\" == info && \"$2\" == --base ]]; then\n"
        "  printf '%s\\n' \"$MOCK_CONDA_BASE\"\n"
        "fi\n",
    )
    (profile_dir / "conda.sh").write_text(
        "conda() { return 0; }\n", encoding="utf-8"
    )
    _write_executable(
        bin_dir / "nvidia-smi",
        "#!/bin/bash\nprintf '%s\\n' \"${MOCK_GPU_NAME:-NVIDIA L40S}\"\n",
    )
    capture = tmp_path / "python_calls.txt"
    _write_executable(
        bin_dir / "python3",
        "#!/bin/bash\n"
        "printf '%s\\n' __CALL__ \"$@\" >> \"$PORE_ARGUMENT_CAPTURE\"\n"
        "if [[ \"${1:-}\" == - && \"${MOCK_RUN_REAL_PREFLIGHT:-0}\" == 1 ]]; then\n"
        "  exec \"$MOCK_REAL_PYTHON\" \"$@\"\n"
        "fi\n"
        "exit 0\n",
    )
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("BASH_FUNC_") or key.startswith("PORE_") or key in {
            "QUICK_TEST",
            "SINGLE_BATCH",
            "DISABLE_AMP",
        }:
            environment.pop(key)
    environment.update(
        {
            "PATH": f"{bin_dir}:{environment['PATH']}",
            "MOCK_CONDA_BASE": str(conda_base),
            "PORE_ARGUMENT_CAPTURE": str(capture),
            "MOCK_REAL_PYTHON": sys.executable,
            "MOCK_RUN_REAL_PREFLIGHT": "0",
            "SLURM_SUBMIT_DIR": str(project_dir),
            "SLURM_JOB_ID": "901_0",
            "SLURM_ARRAY_JOB_ID": "901",
            "SLURM_ARRAY_TASK_ID": str(task),
            "PORE_NEURAL_FREEZE_ID": freeze_id,
            "PORE_SELECTED_ARCHITECTURE_ROLE": architecture_role,
        }
    )
    environment.update({key: str(value) for key, value in overrides.items()})
    result = subprocess.run(
        ["/bin/bash", str(LOCKED_EVALUATION_SCRIPT)],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    calls = []
    if capture.exists():
        current = None
        for line in capture.read_text(encoding="utf-8").splitlines():
            if line == "__CALL__":
                current = []
                calls.append(current)
            else:
                current.append(line)
    return result, calls


@pytest.mark.parametrize(
    ("index", "candidate", "seed", "model", "loss", "patch", "batch"),
    [
        (0, "R3", "42", "multiscale_attention_unet", "focal_dice", "683", "4"),
        (2, "R3", "2025", "multiscale_attention_unet", "focal_dice", "683", "4"),
        (3, "H3", "42", "multiscale_attention_unet", "hierarchical_pore_connectivity", "683", "4"),
        (6, "C2-P", "42", "multiscale_attention_unet", "conditional_pore_focal_dice", "683", "4"),
        (9, "C2-F", "42", "multiscale_attention_unet", "conditional_pore_focal_dice", "2048", "1"),
        (14, "C2-FP", "2025", "multiscale_attention_unet_pyramid", "conditional_pore_focal_dice", "2048", "1"),
    ],
)
def test_screen_array_mapping_is_candidate_major_and_validation_only(
    tmp_path, index, candidate, seed, model, loss, patch, batch
):
    result, calls = _run(tmp_path, mode="screen", task=index)
    assert result.returncode == 0, result.stderr
    arguments = calls[0]
    assert _value(arguments, "--protocol-candidate-key") == candidate
    assert _value(arguments, "--seed") == seed
    assert _value(arguments, "--model-type") == model
    assert _value(arguments, "--loss-type") == loss
    assert _value(arguments, "--patch-size") == patch
    assert _value(arguments, "--batch-size") == batch
    assert _value(arguments, "--epochs") == "30"
    assert _value(arguments, "--workers") == "8"
    assert _value(arguments, "--protocol-run-role") == "validation_screen_cell"
    assert _value(arguments, "--protocol-campaign-id") == "900"
    assert _value(arguments, "--protocol-cell-index") == str(index)
    assert "--validation-only" in arguments
    assert "--max-batches" not in arguments
    assert _value(arguments, "--smoke-preflight-manifest").endswith(
        "smoke_manifest.json"
    )
    assert _value(arguments, "--evaluation-patch-size") == "2048"
    assert _value(arguments, "--evaluation-batch-size") == "1"
    if candidate == "R3":
        start = arguments.index("--class-weights") + 1
        assert arguments[start : start + 3] == ["3", "2", "1"]
    else:
        assert "--class-weights" not in arguments
    if candidate.startswith("C2"):
        assert _value(arguments, "--conditional-pore-threshold") == "100"
        assert "--acknowledge-recovered-threshold-rule" in arguments
    if candidate == "C2-FP":
        assert _value(arguments, "--dropout") == "0.0"


@pytest.mark.parametrize(
    ("index", "candidate", "patch", "model"),
    [
        (0, "R3", "683", "multiscale_attention_unet"),
        (1, "C2-F", "2048", "multiscale_attention_unet"),
        (2, "C2-FP", "2048", "multiscale_attention_unet_pyramid"),
    ],
)
def test_smoke_maps_three_distinct_paths_and_is_never_scientific(
    tmp_path, index, candidate, patch, model
):
    result, calls = _run(tmp_path, mode="smoke", task=index)
    assert result.returncode == 0, result.stderr
    arguments = calls[0]
    assert _value(arguments, "--protocol-candidate-key") == candidate
    assert _value(arguments, "--seed") == "42"
    assert _value(arguments, "--epochs") == "1"
    assert _value(arguments, "--max-batches") == "2"
    assert _value(arguments, "--patch-size") == patch
    assert _value(arguments, "--model-type") == model
    assert _value(arguments, "--protocol-run-role") == "validation_smoke_cell"
    assert _value(arguments, "--protocol-campaign-id") == "900"
    assert _value(arguments, "--protocol-cell-index") == str(index)
    assert "--validation-only" in arguments


@pytest.mark.parametrize("candidate", ["R3", "H3", "C2-P", "C2-F", "C2-FP"])
@pytest.mark.parametrize(
    ("architecture_role", "expected_model"),
    [("primary_multiscale", None), ("plain_unet_comparator", "plain_unet")],
)
def test_selected_retraining_is_lock_bound_validation_only_and_immutable(
    tmp_path, candidate, architecture_role, expected_model
):
    result, calls = _run(
        tmp_path,
        mode="selected_retrain",
        task=1,
        PORE_SELECTED_METHOD_KEY=candidate,
        PORE_SELECTED_METHOD_LOCK="results/locks/winner.json",
        PORE_SELECTED_ARCHITECTURE_ROLE=architecture_role,
        PORE_ACKNOWLEDGE_RECOVERED_THRESHOLD_RULE="1",
    )
    assert result.returncode == 0, result.stderr
    arguments = calls[0]
    assert _value(arguments, "--protocol-run-role") == "selected_winner_retraining"
    assert _value(arguments, "--protocol-campaign-id") == "900"
    assert _value(arguments, "--protocol-cell-index") == "1"
    assert _value(arguments, "--selected-method-lock") == "results/locks/winner.json"
    assert _value(arguments, "--selected-architecture-role") == architecture_role
    assert _value(arguments, "--seed") == "123"
    assert "--validation-only" in arguments
    if expected_model is not None:
        assert _value(arguments, "--model-type") == expected_model
    if candidate == "C2-FP":
        assert _value(arguments, "--dropout") == "0.0"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"PORE_MAX_BATCHES": "1"}, "scientific screen forbids"),
        ({"QUICK_TEST": "1"}, "scientific screen forbids"),
        ({"PORE_EPOCHS": "29"}, "fixes PORE_EPOCHS=30"),
        ({"DISABLE_AMP": "1"}, "requires AMP"),
        ({"PORE_MODEL_TYPE": "plain_unet"}, "scientific screen forbids"),
    ],
)
def test_screen_rejects_scientific_overrides(tmp_path, overrides, message):
    result, calls = _run(tmp_path, mode="screen", **overrides)
    assert result.returncode == 2
    assert message in result.stderr
    assert calls == []


def test_direct_or_non_l40s_execution_fails_before_training(tmp_path):
    direct, calls = _run(tmp_path, mode="screen", PORE_LAUNCH_WRAPPER="smoke")
    assert direct.returncode == 2
    assert "direct main-script launch is forbidden" in direct.stderr
    assert calls == []

    wrong_gpu, calls = _run(tmp_path / "gpu", mode="screen", MOCK_GPU_NAME="A100")
    assert wrong_gpu.returncode == 2
    assert "NVIDIA L40S" in wrong_gpu.stderr
    assert calls == []

    no_job, calls = _run(tmp_path / "job", mode="screen", SLURM_JOB_ID="")
    assert no_job.returncode == 2
    assert "active Slurm array" in no_job.stderr
    assert calls == []


def test_failed_screen_invokes_immutable_failure_recorder(tmp_path):
    result, calls = _run(tmp_path, mode="screen", task=4, python_status=7)
    assert result.returncode == 7
    assert len(calls) == 2
    failure = calls[1]
    assert failure[0] == "scripts/record_protocol_cell_failure.py"
    assert _value(failure, "--run-role") == "validation_screen_cell"
    assert _value(failure, "--campaign-id") == "900"
    assert _value(failure, "--cell-index") == "4"
    assert _value(failure, "--candidate") == "H3"
    assert _value(failure, "--seed") == "123"
    assert _value(failure, "--exit-code") == "7"
    assert "--smoke-preflight-manifest" in failure


def test_mode_wrappers_freeze_array_concurrency_and_time_limits():
    screen = (ROOT / "scripts/aire_validation_screen.slurm").read_text()
    smoke = (ROOT / "scripts/aire_validation_smoke.slurm").read_text()
    selected = (ROOT / "scripts/aire_selected_retrain.slurm").read_text()
    locked_evaluation = LOCKED_EVALUATION_SCRIPT.read_text()
    assert "#SBATCH --array=0-14%1" in screen
    assert "#SBATCH --time=01:15:00" in screen
    assert "#SBATCH --array=0-2%1" in smoke
    assert "#SBATCH --time=00:15:00" in smoke
    assert "#SBATCH --array=0-2%1" in selected
    assert "#SBATCH --time=01:15:00" in selected
    assert "#SBATCH --array=0-2%1" in locked_evaluation
    assert "#SBATCH --time=01:15:00" in locked_evaluation


@pytest.mark.parametrize("task", [0, 1, 2])
@pytest.mark.parametrize(
    "architecture_role",
    ["primary_multiscale", "plain_unet_comparator"],
)
def test_locked_evaluation_derives_canonical_checkpoint_and_fixed_inputs(
    tmp_path, task, architecture_role
):
    result, calls = _run_locked_evaluation(
        tmp_path,
        task=task,
        architecture_role=architecture_role,
    )
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    preflight = calls[0]
    assert preflight == [
        "-",
        "neural-freeze-aaaaaaaaaaaaaaaa",
        architecture_role,
        str(task),
    ]

    evaluation = calls[1]
    assert evaluation[0] == "scripts/evaluate_confirmatory_checkpoint.py"
    assert _value(evaluation, "--neural-freeze-id") == (
        "neural-freeze-aaaaaaaaaaaaaaaa"
    )
    assert _value(evaluation, "--architecture-role") == architecture_role
    assert _value(evaluation, "--cell-index") == str(task)
    assert _value(evaluation, "--annotations") == (
        "results/step3_coco_dataset/pore_annotations.json"
    )
    assert _value(evaluation, "--image-dir") == (
        "results/step3_coco_dataset/images"
    )
    assert _value(evaluation, "--mask-dir") == (
        "results/step2_pore_classification/pore_classifications"
    )
    assert _value(evaluation, "--split-manifest") == (
        "config/confirmatory_splits.json"
    )
    assert "--output-dir" not in evaluation
    assert "--checkpoint" not in evaluation
    assert "--selected-method-lock" not in evaluation
    assert "--model-type" not in evaluation
    assert _value(evaluation, "--device") == "cuda"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"SLURM_JOB_ID": ""}, "active Slurm array"),
        ({"SLURM_ARRAY_TASK_ID": "3"}, "must be 0, 1, or 2"),
        ({"MOCK_GPU_NAME": "A100"}, "NVIDIA L40S"),
        ({"PORE_NEURAL_FREEZE_ID": "../escape"}, "canonical neural-freeze"),
        ({"PORE_SELECTED_ARCHITECTURE_ROLE": "anything"}, "PORE_SELECTED_ARCHITECTURE_ROLE"),
        ({"PORE_SELECTED_RETRAIN_CAMPAIGN_ID": "selected900"}, "forbids PORE_SELECTED_RETRAIN_CAMPAIGN_ID"),
        ({"PORE_SELECTED_METHOD_LOCK": "lock.json"}, "forbids PORE_SELECTED_METHOD_LOCK"),
        ({"PORE_CHECKPOINT": "arbitrary.pth"}, "forbids PORE_CHECKPOINT"),
        ({"PORE_OUTPUT_DIR": "elsewhere"}, "forbids PORE_OUTPUT_DIR"),
        ({"PORE_IMAGE_DIR": "elsewhere"}, "forbids PORE_IMAGE_DIR"),
    ],
)
def test_locked_evaluation_fails_closed_before_checkpoint_preflight(
    tmp_path, overrides, message
):
    result, calls = _run_locked_evaluation(tmp_path, **overrides)
    assert result.returncode == 2
    assert message in result.stderr
    assert calls == []


def test_locked_evaluation_invokes_freeze_preflight_before_evaluator(tmp_path):
    result, calls = _run_locked_evaluation(tmp_path, task=2)
    assert result.returncode == 0, result.stderr
    assert [call[0] for call in calls] == ["-", "scripts/evaluate_confirmatory_checkpoint.py"]
