"""Extract bundled frozen inputs; no network or scientific computation."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path(__file__).with_name("inputs.tar.gz"))
    args = parser.parse_args()
    root = args.output_root.resolve()
    with tarfile.open(args.archive, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not member.isfile() or not target.is_relative_to(root):
                raise ValueError("Archive contains an unsafe/non-file member")
            if target.exists():
                raise FileExistsError("Use a new input root; existing files are never overwritten")
        for member in members:
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    print(f"Extracted {len(members)} files to {root}")


if __name__ == "__main__":
    main()
