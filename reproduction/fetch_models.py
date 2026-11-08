"""Explicit opt-in acquisition from pinned original model sources; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", choices=("chemberta", "esm2", "all"), required=True)
    args = parser.parse_args()
    directories = []
    if args.model in ("chemberta", "all"):
        directories.extend(("chemberta_mtr_model", "chemberta_mtr_tokenizer"))
    if args.model in ("esm2", "all"):
        directories.append("esm2_8m")
    for directory in directories:
        asset = args.root.resolve() / "assets" / directory
        provenance = json.loads((asset / "provenance.json").read_text(encoding="utf-8"))
        for record in provenance["files"]:
            name = record.get("relative_file", record.get("file"))
            # This third-party source is linked for provenance, not needed by the
            # first-party wrapper and not redistributed under an invented license.
            if name == "roberta_regression_official.py":
                continue
            source = record.get("source_url", record.get("source"))
            source = source.replace("https://hf-mirror.com/", "https://huggingface.co/")
            destination = asset / name
            expected = record.get("sha256")
            if destination.exists():
                if expected and hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Existing model file has wrong SHA256: {destination}")
                print(f"Keep existing {directory}/{name}")
                continue
            size = record.get("size_bytes", record.get("size"))
            with urllib.request.urlopen(source, timeout=90) as response:
                data = response.read()
            if size is not None and len(data) != size:
                raise ValueError(f"Source byte count changed: {source}")
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Source SHA256 changed: {source}")
            with destination.open("xb") as output:
                output.write(data)
            print(f"Acquired {directory}/{name}")


if __name__ == "__main__":
    main()
