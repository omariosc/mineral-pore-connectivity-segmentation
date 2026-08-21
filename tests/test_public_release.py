import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.audit_public_snapshot import (
    PROJECT_ROOT,
    audit_materialized_boundary,
    audit_paths,
    audit_snapshot_content,
    audit_tracked_boundary,
    expand_allowlist,
    read_allowlist,
    snapshot_tree_sha256,
)
from scripts.export_public_snapshot import export_public_snapshot


class PublicReleaseBoundaryTests(unittest.TestCase):
    def make_source(self, parent: Path) -> tuple[Path, Path]:
        source = parent / "source"
        (source / "config").mkdir(parents=True)
        (source / "README.md").write_text("safe public fixture\n", encoding="utf-8")
        allowlist = source / "config" / "public_release_allowlist.txt"
        allowlist.write_text(
            "README.md\nconfig/public_release_allowlist.txt\n",
            encoding="utf-8",
        )
        return source, allowlist

    def make_asset_source(self, parent: Path) -> tuple[Path, Path, str]:
        source, allowlist = self.make_source(parent)
        relative = "paper_assets/figures/approved.svg"
        asset = source / relative
        asset.parent.mkdir(parents=True)
        asset.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>\n',
            encoding="utf-8",
        )
        allowlist.write_text(
            allowlist.read_text(encoding="utf-8") + relative + "\n",
            encoding="utf-8",
        )
        return source, allowlist, relative

    def write_approval_manifest(
        self,
        path: Path,
        approved_paths: list[str],
        *,
        template_only: bool = False,
        approval_status: str = "approved",
    ) -> Path:
        document = {
            "schema_version": 1,
            "template_only": template_only,
            "assets": [
                {
                    "paths": approved_paths,
                    "asset_kind": "generated_figure",
                    "source_description": "Synthetic test fixture",
                    "contains_source_image_pixels": False,
                    "contains_annotation_or_mask_pixels": False,
                    "copyright_owner": "Test fixture owner",
                    "licence_identifier": "CC0-1.0",
                    "written_approval_reference": "fixture-approval-001",
                    "approved_by": "Release test",
                    "approval_date": "2026-08-21",
                    "approval_status": approval_status,
                }
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def test_unversioned_release_audit_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source, allowlist = self.make_source(Path(temp_dir))
            paths, missing = expand_allowlist(read_allowlist(allowlist), source)
            self.assertEqual(missing, [])
            self.assertEqual(audit_paths(paths, source), [])
            self.assertRegex(
                audit_tracked_boundary(paths, source)[0],
                "release boundary is unavailable",
            )

    def test_export_materializes_exact_byte_identical_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            source, allowlist = self.make_source(parent)
            output = parent / "public"
            copied = export_public_snapshot(source, output, allowlist)

            self.assertEqual(
                copied,
                ["README.md", "config/public_release_allowlist.txt"],
            )
            paths, missing = expand_allowlist(read_allowlist(allowlist), output)
            self.assertEqual(missing, [])
            self.assertEqual(audit_materialized_boundary(paths, output), [])
            source_paths, _ = expand_allowlist(read_allowlist(allowlist), source)
            self.assertEqual(
                audit_snapshot_content(source_paths, source, output), []
            )
            self.assertEqual(
                snapshot_tree_sha256(source_paths, source),
                snapshot_tree_sha256(paths, output),
            )

    def test_tree_hash_is_order_independent_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            source, allowlist = self.make_source(parent)
            paths, missing = expand_allowlist(read_allowlist(allowlist), source)
            self.assertEqual(missing, [])

            original = snapshot_tree_sha256(paths, source)
            self.assertRegex(original, r"^[0-9a-f]{64}$")
            self.assertEqual(
                original,
                snapshot_tree_sha256(reversed(paths), source),
            )

            (source / "README.md").write_text(
                "safe public fixture changed\n",
                encoding="utf-8",
            )
            self.assertNotEqual(original, snapshot_tree_sha256(paths, source))

    def test_materialized_audit_rejects_extra_and_modified_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            source, allowlist = self.make_source(parent)
            output = parent / "public"
            export_public_snapshot(source, output, allowlist)
            (output / "private-note.txt").write_text("must not ship\n")
            (output / "README.md").write_text("modified after review\n")

            patterns = read_allowlist(allowlist)
            source_paths, _ = expand_allowlist(patterns, source)
            snapshot_paths, _ = expand_allowlist(patterns, output)
            self.assertIn(
                "materialized path is outside public allowlist: private-note.txt",
                audit_materialized_boundary(snapshot_paths, output),
            )
            self.assertIn(
                "materialized file differs from reviewed source: README.md",
                audit_snapshot_content(source_paths, source, output),
            )

    def test_export_refuses_existing_or_in_checkout_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            source, allowlist = self.make_source(parent)
            existing = parent / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_public_snapshot(source, existing, allowlist)
            with self.assertRaisesRegex(ValueError, "outside the mixed research checkout"):
                export_public_snapshot(source, source / "public", allowlist)

    def test_asset_export_default_discovery_fails_without_completed_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            source, allowlist, _relative = self.make_asset_source(parent)
            output = parent / "public"

            with self.assertRaisesRegex(
                ValueError,
                "selected public assets require a completed approval manifest",
            ):
                export_public_snapshot(source, output, allowlist)

            self.assertFalse(output.exists())

    def test_asset_export_rejects_template_pending_and_wrong_path_records(self):
        cases = (
            (True, "approved", "paper_assets/figures/approved.svg", "template"),
            (False, "pending", "paper_assets/figures/approved.svg", "approval_status"),
            (False, "approved", "paper_assets/figures/other.svg", "no complete approved entry"),
            (False, "approved", "paper_assets/../private.svg", "path is unsafe"),
        )
        for template_only, status, approved_path, expected in cases:
            with self.subTest(
                template_only=template_only,
                status=status,
                approved_path=approved_path,
            ), tempfile.TemporaryDirectory() as temp_dir:
                parent = Path(temp_dir)
                source, allowlist, _relative = self.make_asset_source(parent)
                manifest = self.write_approval_manifest(
                    parent / "approval.yml",
                    [approved_path],
                    template_only=template_only,
                    approval_status=status,
                )
                with self.assertRaisesRegex(ValueError, expected):
                    export_public_snapshot(
                        source,
                        parent / "public",
                        allowlist,
                        approval_manifest_path=manifest,
                    )

    def test_asset_export_accepts_complete_exact_external_approval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            source, allowlist, relative = self.make_asset_source(parent)
            manifest = self.write_approval_manifest(
                parent / "controlled" / "approval.yml",
                [relative],
            )
            output = parent / "public"

            copied = export_public_snapshot(
                source,
                output,
                allowlist,
                approval_manifest_path=manifest,
            )

            self.assertIn(relative, copied)
            self.assertEqual(
                (source / relative).read_bytes(),
                (output / relative).read_bytes(),
            )
            self.assertFalse((output / "controlled" / "approval.yml").exists())

    def test_asset_export_accepts_completed_default_private_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            source, allowlist, relative = self.make_asset_source(parent)
            manifest = self.write_approval_manifest(
                source / "config" / "public_asset_approvals.yml",
                [relative],
            )
            output = parent / "public"

            copied = export_public_snapshot(source, output, allowlist)

            self.assertIn(relative, copied)
            self.assertTrue(manifest.is_file())
            self.assertFalse(
                (output / "config" / "public_asset_approvals.yml").exists()
            )

    @unittest.skipIf(os.name == "nt", "symbolic-link semantics differ on Windows")
    def test_source_selection_rejects_symlinked_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            source.mkdir()
            (source / "real.txt").write_text("safe\n")
            (source / "link.txt").symlink_to(source / "real.txt")
            allowlist = source / "allowlist.txt"
            allowlist.write_text("link.txt\n", encoding="utf-8")
            paths, missing = expand_allowlist(read_allowlist(allowlist), source)
            self.assertEqual(missing, [])
            self.assertIn(
                "link.txt: symbolic links are not allowed",
                audit_paths(paths, source),
            )

    def test_repository_script_allowlist_matches_gitignore_exceptions(self):
        patterns = read_allowlist(
            PROJECT_ROOT / "config" / "public_release_allowlist.txt"
        )
        ignored_scripts = {
            pattern
            for pattern in patterns
            if pattern.startswith("scripts/")
            and not any(character in pattern for character in "*?[")
        }
        exceptions = {
            line[1:]
            for line in (PROJECT_ROOT / ".gitignore").read_text().splitlines()
            if line.startswith("!scripts/") and line != "!scripts/"
        }
        self.assertEqual(ignored_scripts, exceptions)
        self.assertIn("scripts/build_neural_freeze_manifest.py", patterns)
        self.assertNotIn(
            "scripts/traditional_algorithms_comparison.py", patterns
        )

    def test_community_files_are_required_and_exported_byte_identically(self):
        canonical_allowlist = (
            PROJECT_ROOT / "config" / "public_release_allowlist.txt"
        )
        patterns = read_allowlist(canonical_allowlist)
        community_files = {"CONTRIBUTING.md", "SECURITY.md"}
        self.assertTrue(community_files.issubset(patterns))

        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            allowlist = parent / "code-only-allowlist.txt"
            allowlist.write_text(
                "CONTRIBUTING.md\nSECURITY.md\n",
                encoding="utf-8",
            )
            output = parent / "public"
            copied = set(
                export_public_snapshot(PROJECT_ROOT, output, allowlist)
            )
            self.assertTrue(community_files.issubset(copied))
            for relative in community_files:
                self.assertEqual(
                    (PROJECT_ROOT / relative).read_bytes(),
                    (output / relative).read_bytes(),
                )

    def test_canonical_export_can_be_staged_as_the_exact_tracked_tree(self):
        canonical_allowlist = (
            PROJECT_ROOT / "config" / "public_release_allowlist.txt"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "public"
            copied = export_public_snapshot(
                PROJECT_ROOT,
                output,
                canonical_allowlist,
            )
            subprocess.run(
                ["git", "init", "--initial-branch=main"],
                cwd=output,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "add", "--all"],
                cwd=output,
                check=True,
                capture_output=True,
            )
            paths, missing = expand_allowlist(
                read_allowlist(output / "config" / "public_release_allowlist.txt"),
                output,
            )
            self.assertEqual(missing, [])
            self.assertEqual(len(paths), len(copied))
            self.assertEqual(audit_tracked_boundary(paths, output), [])

    def test_canonical_boundary_is_exact_and_excludes_all_paper_assets(self):
        patterns = read_allowlist(
            PROJECT_ROOT / "config" / "public_release_allowlist.txt"
        )
        self.assertFalse(
            any(
                any(character in pattern for character in "*?[")
                for pattern in patterns
            )
        )
        allowlisted_assets = {
            pattern for pattern in patterns if pattern.startswith("paper_assets/")
        }
        asset_exceptions = {
            line[1:]
            for line in (PROJECT_ROOT / ".gitignore").read_text().splitlines()
            if line.startswith("!paper_assets/") and not line.endswith("/")
        }
        self.assertEqual(allowlisted_assets, set())
        self.assertEqual(asset_exceptions, set())
        self.assertIn(
            "paper_assets/",
            (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines(),
        )

        resolved, missing = expand_allowlist(patterns, PROJECT_ROOT)
        self.assertEqual(missing, [])
        self.assertTrue(resolved)
        self.assertTrue(
            all(
                not path.relative_to(PROJECT_ROOT)
                .as_posix()
                .startswith("paper_assets/")
                for path in resolved
            )
        )

        approval_template = (
            PROJECT_ROOT / "config" / "public_asset_approvals.template.yml"
        ).read_text(encoding="utf-8")
        template_document = json.loads(approval_template)
        self.assertIs(template_document["template_only"], True)
        self.assertEqual(
            template_document["assets"][0]["approval_status"], "pending"
        )

    def test_cli_secrets_and_ci_dependencies_are_not_mutable_inputs(self):
        trainer_source = (PROJECT_ROOT / "scripts" / "train_patches.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("--wandb-api-key", trainer_source)
        self.assertNotIn("wandb_api_key", trainer_source)
        self.assertIn(
            "requires WANDB_API_KEY in the process environment",
            trainer_source,
        )
        rejected_secret_flag = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_patches.py"),
                "--wandb-api-key",
                "dummy-not-a-secret",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(rejected_secret_flag.returncode, 2)
        self.assertIn(
            "unrecognized arguments: --wandb-api-key",
            rejected_secret_flag.stderr,
        )

        dino_source = (PROJECT_ROOT / "src" / "models" / "dinov2_unet.py").read_text(
            encoding="utf-8"
        )
        factory_source = (
            PROJECT_ROOT / "src" / "training" / "model_factory.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("torch.hub.load(", dino_source)
        self.assertIn("raise RuntimeError(DINOV2_DISABLED_MESSAGE)", dino_source)
        self.assertIn("raise RuntimeError(DINOV2_DISABLED_MESSAGE)", factory_source)

        workflow = (
            PROJECT_ROOT / ".github" / "workflows" / "public-smoke.yml"
        ).read_text(encoding="utf-8")
        action_refs = re.findall(r"^\s*uses:\s*([^\s]+)\s*$", workflow, re.MULTILINE)
        self.assertTrue(action_refs)
        for action_ref in action_refs:
            self.assertRegex(action_ref, r"^[^@]+@[0-9a-f]{40}$")
        self.assertNotRegex(workflow, r"uses:\s*[^\s]+@v\d")
        self.assertNotIn("pip install --disable-pip-version-check \"", workflow)
        self.assertGreaterEqual(workflow.count("--require-hashes"), 3)

        lock_paths = (
            PROJECT_ROOT
            / ".github"
            / "requirements"
            / "public-smoke-linux-py311.lock",
            PROJECT_ROOT
            / ".github"
            / "requirements"
            / "ml-unit-tests-linux-py311.lock",
            PROJECT_ROOT
            / ".github"
            / "requirements"
            / "torch-cpu-linux-py311.lock",
        )
        requirement_pattern = re.compile(
            r"^[A-Za-z0-9_.+-]+==[^\s]+ --hash=sha256:[0-9a-f]{64}$"
        )
        for lock_path in lock_paths:
            lines = [
                line
                for line in lock_path.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")
            ]
            self.assertTrue(lines)
            self.assertTrue(
                all(requirement_pattern.fullmatch(line) for line in lines)
            )


if __name__ == "__main__":
    unittest.main()
