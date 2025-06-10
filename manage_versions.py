import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List


class VersionManager:
    def __init__(self, docs_dir: Path | None = None):
        self.docs_dir = docs_dir or Path(__file__).parent
        self.versions_file = self.docs_dir / "versions.json"
        self.site_dir = self.docs_dir / "_site"

    def load_versions(self) -> List[Dict]:
        """Load versions from versions.json."""
        if not self.versions_file.exists():
            return []

        with open(self.versions_file, "r") as f:
            return json.load(f)

    def save_versions(self, versions: List[Dict]):
        """Save versions to versions.json."""
        with open(self.versions_file, "w") as f:
            json.dump(versions, f, indent=2)
        print(f"✅ Updated {self.versions_file}")

    def version_exists(self, version: str) -> bool:
        """Check if a version already exists."""
        versions = self.load_versions()
        return any(v.get("text") == version for v in versions)

    def add_version(
        self,
        version: str,
        from_version: str = "latest",
    ):
        """Add a new version."""
        if self.version_exists(version):
            print(f"❌ Version {version} already exists")
            return False

        # Determine source directory
        if from_version == "latest":
            source_dir = self.site_dir
        else:
            source_dir = self.docs_dir / from_version

        if not source_dir.exists():
            print(f"❌ Source version directory {source_dir} does not exist")
            return False

        # Create new version directory
        version_dir = self.docs_dir / version
        if version_dir.exists():
            print(f"❌ Directory {version_dir} already exists")
            return False

        print(f"📁 Creating version directory: {version_dir}")

        # Copy content
        if from_version == "latest":
            # If copying from latest, we need to build it first or copy from _site
            if self.site_dir.exists():
                shutil.copytree(self.site_dir, version_dir)
                print(f"✅ Copied content from {source_dir} to {version_dir}")
            else:
                print(
                    "❌ _site directory not found. Please build the documentation first."
                )
                return False
        else:
            shutil.copytree(source_dir, version_dir)
            print(f"✅ Copied content from {source_dir} to {version_dir}")

        # Update versions.json
        versions = self.load_versions()

        # Determine href path
        href = f"/{version}/"

        new_version = {
            "text": version,
            "href": href,
        }

        if version != "latest":
            versions.insert(2, new_version)
        else:
            versions.insert(0, new_version)

        self.save_versions(versions)
        print(f"✅ Added version {version} to versions.json")

        return True

    def remove_version(self, version: str, confirm: bool = False):
        """Remove a version."""
        if version == "latest":
            print("❌ Cannot remove 'latest' version")
            return False

        if not confirm:
            response = input(
                f"⚠️  Are you sure you want to remove version {version}? (y/N): "
            )
            if response.lower() != "y":
                print("Cancelled")
                return False

        # Remove from versions.json
        versions = self.load_versions()
        original_count = len(versions)
        versions = [
            v
            for v in versions
            if v.get("text") != version and v.get("version") != version
        ]

        if len(versions) == original_count:
            print(f"❌ Version {version} not found in versions.json")
            return False

        self.save_versions(versions)

        # Remove directory
        version_dir = self.docs_dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
            print(f"✅ Removed directory {version_dir}")

        print(f"✅ Removed version {version}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Manage documentation versions")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add version command
    add_parser = subparsers.add_parser("add", help="Add a new version")
    add_parser.add_argument(
        "--version", required=True, help="Version to add (e.g., v1.0.0)"
    )
    add_parser.add_argument(
        "--from",
        dest="from_version",
        default="latest",
        help="Version to copy from (default: latest)",
    )

    # Remove version command
    remove_parser = subparsers.add_parser("remove", help="Remove a version")
    remove_parser.add_argument("--version", required=True, help="Version to remove")
    remove_parser.add_argument("--yes", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = VersionManager()

    if args.command == "add":
        success = manager.add_version(args.version, args.from_version)
        sys.exit(0 if success else 1)

    elif args.command == "remove":
        success = manager.remove_version(args.version, args.yes)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
