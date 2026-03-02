from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_repo"
SPACE_DIR = ROOT / "leaderboard_space"


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _print_plan(results_repo: str, space_repo: str) -> None:
    print("Upload plan:")
    print(f"  - Dataset repo  : {results_repo}")
    print(f"  - Source folder : {RESULTS_DIR}")
    print(f"  - Space repo    : {space_repo}")
    print(f"  - Source folder : {SPACE_DIR}")


def _upload_dataset(results_repo: str, *, private: bool, token: str | None) -> None:
    create_repo(
        repo_id=results_repo,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )
    upload_folder(
        repo_id=results_repo,
        repo_type="dataset",
        folder_path=str(RESULTS_DIR),
        path_in_repo=".",
        commit_message="Initial BTZSC Phase 1 results dataset",
        token=token,
    )


def _upload_space(space_repo: str, *, private: bool, token: str | None) -> None:
    create_repo(
        repo_id=space_repo,
        repo_type="space",
        space_sdk="gradio",
        private=private,
        exist_ok=True,
        token=token,
    )
    upload_folder(
        repo_id=space_repo,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        path_in_repo=".",
        commit_message="Initial BTZSC Phase 1 leaderboard space",
        token=token,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish BTZSC Phase 1 artifacts to Hugging Face Hub")
    parser.add_argument(
        "--org",
        default="btzsc",
        help="HF username or org name that owns the repos (default: btzsc)",
    )
    parser.add_argument(
        "--results-repo",
        default="btzsc-results",
        help="Dataset repo name (default: btzsc-results)",
    )
    parser.add_argument(
        "--space-repo",
        default="btzsc-leaderboard",
        help="Space repo name (default: btzsc-leaderboard)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repos as private (default: public)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be uploaded",
    )
    parser.add_argument(
        "--skip-results",
        action="store_true",
        help="Skip uploading dataset repo",
    )
    parser.add_argument(
        "--skip-space",
        action="store_true",
        help="Skip uploading space repo",
    )
    args = parser.parse_args()

    _ensure_exists(RESULTS_DIR)
    _ensure_exists(SPACE_DIR)

    results_repo_id = f"{args.org}/{args.results_repo}"
    space_repo_id = f"{args.org}/{args.space_repo}"

    _print_plan(results_repo_id, space_repo_id)

    if args.dry_run:
        print("\nDry run complete. No upload performed.")
        return

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    HfApi(token=token).whoami()

    if not args.skip_results:
        print("\nUploading dataset repo...")
        _upload_dataset(results_repo_id, private=args.private, token=token)
        print("Dataset upload done.")

    if not args.skip_space:
        print("\nUploading space repo...")
        _upload_space(space_repo_id, private=args.private, token=token)
        print("Space upload done.")

    print("\nAll requested uploads completed.")


if __name__ == "__main__":
    main()
