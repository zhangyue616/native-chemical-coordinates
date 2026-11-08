"""Inspect/unpack completed lineB results or explicitly obtain an external asset.

The default smoke is offline and uses the Python standard library. It reads saved
results; it never imports audit runners, model code, or statistical estimators.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def npy_vector(raw):
    """Read numeric one-dimensional NPY vectors without importing NumPy."""
    checked(raw[:6] == b"\x93NUMPY", "Not an NPY array")
    major = raw[6]
    if major == 1:
        length = struct.unpack("<H", raw[8:10])[0]
        offset = 10
    else:
        length = struct.unpack("<I", raw[8:12])[0]
        offset = 12
    header = ast.literal_eval(raw[offset:offset + length].decode("latin1"))
    checked(header["shape"] == (173,), "Expected a 173-concept vector")
    dtype = header["descr"]
    checked(dtype in ("<f8", "<f4", "<i8", "<i4"), "Unexpected NPY dtype")
    code = {"<f8": "d", "<f4": "f", "<i8": "q", "<i4": "i"}[dtype]
    data = raw[offset + length:]
    checked(len(data) == 173 * struct.calcsize(code), "Truncated NPY payload")
    values = struct.unpack("<" + code * 173, data)
    checked(all(math.isfinite(x) for x in values), "Nonfinite saved array value")
    return values


def smoke():
    result_root = ROOT / "frozen_results/lineB"
    v5 = read(result_root / "v5/gates_v5.json")
    v6 = read(result_root / "v6/gates_v6.json")
    concepts = read(ROOT / "data/derived/lineB/concepts/concepts_v1.json")
    expected_ids = [c["id"] for c in concepts if c["partition"] == "readable"]
    checked(len(concepts) == 200 and len(expected_ids) == 173, "Concept dictionary count")
    checked(v5["n_models"] == len(v5["models"]) == 8, "v5 model count")
    checked(v5["supporting_familywise"] == ["chemberta_mlm", "chemberta_mtr"], "Saved v5 decisions")
    checked(all((r["K"], r["R"], r["n_concepts"]) == (199, 9, 173)
                for r in v5["models"].values()), "v5 protocol sizes")
    checked(v6["primary"]["K"] == 199 and v6["primary"]["n_concepts"] == 173, "v6 protocol sizes")
    checked(v6["primary"]["layer"] == "L9" and v6["primary"]["J1"]["p"] == .005
            and v6["primary"]["J2"]["p"] == .005, "Saved v6 primary fields")
    checked(set(v6["primary"]["T1_by_layer_target"]) == {"L" + str(i) for i in range(12)}, "v6 12-layer fields")
    with (ROOT / "data/derived/lineB/corpus/H4_used.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    checked(len(rows) == len({r["cid"] for r in rows}) == 10000, "H4 molecule/CID count")
    index = read(result_root / "VECTOR_INDEX.json")
    reports = {}
    for version in ("v5", "v6"):
        description = index[version]
        with zipfile.ZipFile(ROOT / description["archive"]) as archive:
            actual = set(archive.namelist())
            expected = {x["path"] for x in description["members"]}
            checked(actual == expected, version + " archive coverage")
            sampled = []
            if version == "v5":
                groups = {"chemeleon": ["chemeleon"], "molclr": ["molclr_gin"],
                          "hu": ["hu_masking", "hu_contextpred", "hu_edgepred", "hu_infomax"],
                          "chemberta": ["chemberta_mlm", "chemberta_mtr"]}
                for group, models in groups.items():
                    ids = npy_vector(archive.read(group + "/readable_concept_ids.npy"))
                    checked(list(ids) == expected_ids, group + " concept identities")
                    for model in models:
                        required = {group + "/" + model + "/S_native.npy"}
                        required |= {group + "/" + model + "/S_rot_" + str(r) + ".npy" for r in range(9)}
                        checked(required <= actual, model + " native/rotation coverage")
                        sampled.append(group + "/" + model + "/S_native.npy")
                    twins = {group + "/twins/S_native_" + str(k) + ".npy" for k in range(199)}
                    twins |= {group + "/twins/S_rot_" + str(k) + "_" + str(r) + ".npy"
                              for k in range(199) for r in range(9)}
                    checked(twins <= actual, group + " 199-twin/9-rotation coverage")
                    sampled += [group + "/twins/S_native_0.npy", group + "/twins/S_rot_198_8.npy"]
            else:
                ids = npy_vector(archive.read("molformer/readable_concept_ids.npy"))
                checked(list(ids) == expected_ids, "v6 concept identities")
                for layer in range(12):
                    prefix = "molformer/"
                    name = "L" + str(layer)
                    required = {prefix + "pretrained/" + name + "/S_native.npy"}
                    required |= {prefix + "pretrained/" + name + "/S_rot_" + str(r) + ".npy" for r in range(9)}
                    required |= {prefix + "twins/" + name + "/S_native_" + str(k) + ".npy" for k in range(199)}
                    required |= {prefix + "twins/" + name + "/S_rot_" + str(k) + "_" + str(r) + ".npy"
                                 for k in range(199) for r in range(9)}
                    required |= {prefix + "twins_hf/" + name + "/S_native_" + str(k) + ".npy" for k in range(39)}
                    required |= {prefix + "pretrained_orf/" + name + "/S_native_" + str(k) + ".npy" for k in range(9)}
                    checked(required <= actual, "v6 " + name + " saved network coverage")
                    sampled += [prefix + "pretrained/" + name + "/S_native.npy",
                                prefix + "twins/" + name + "/S_rot_198_8.npy"]
            for name in sampled:
                values = npy_vector(archive.read(name))
                checked(all(x > 0 for x in values), "Saved lift must be positive: " + name)
            reports[version] = {"frozen_vectors": len(actual), "sampled_numeric_vectors": len(sampled),
                                "all_expected_network_rotation_paths_present": True}
    return {"status": "PASS", "scope": "Offline saved-results integrity and scientific-object coverage",
            "concepts": len(concepts), "readable_concepts": len(expected_ids), "H4_molecules": len(rows),
            "checks": reports, "model_calls": 0, "statistical_recomputations": 0,
            "limitations": "No model execution, clean-machine installation or historical confirmation rerun."}


def unpack(version, output):
    output = Path(output).resolve()
    checked(not output.exists(), "Choose a new output directory; existing outputs are never overwritten")
    archive_path = ROOT / "frozen_results/lineB" / version / "S_vectors.zip"
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            resolved = (output / name).resolve()
            checked(resolved.is_relative_to(output), "Unsafe archive path")
        output.mkdir(parents=True)
        archive.extractall(output)
    if version == "v6":
        for name in ("run_record.json", "fingerprints.json"):
            shutil.copyfile(ROOT / "frozen_results/lineB/v6/molformer" / name,
                            output / "molformer" / name)
    return {"status": "UNPACKED", "version": version, "output": str(output)}


def fetch_asset(asset_id, asset_root):
    assets = read(ROOT / "acquisition/EXTERNAL_ASSETS.json")["assets"]
    asset = next((a for a in assets if a["id"] == asset_id), None)
    checked(asset is not None, "Unknown external asset id; inspect acquisition/EXTERNAL_ASSETS.json")
    root = Path(asset_root).resolve()
    target = (root / asset["id"]).resolve()
    checked(target.is_relative_to(root), "Unsafe asset path")
    checked(not target.exists(), "Existing asset is not overwritten")
    partial = target.with_name(target.name + ".partial")
    checked(not partial.exists(), "Existing partial download is not overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "lineB-review-reproduction/1"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            count += len(block)
            checked(count <= asset["bytes"], "Downloaded asset exceeds frozen byte count")
            digest.update(block)
            stream.write(block)
    checked(count == asset["bytes"] and digest.hexdigest() == asset["sha256"],
            "Publisher bytes differ from historical snapshot; partial file retained, not accepted")
    partial.rename(target)
    return {"status": "FETCHED_EXACT_SNAPSHOT", "asset": asset["id"], "bytes": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke", help="Offline inspection of frozen JSON and S-vector coverage; no recomputation")
    unpack_parser = sub.add_parser("unpack-results", help="Unpack exact frozen vectors into a new explicit directory")
    unpack_parser.add_argument("version", choices=["v5", "v6"])
    unpack_parser.add_argument("--output", required=True)
    fetch_parser = sub.add_parser("fetch-asset", help="Explicit network download of one publisher asset, with frozen byte/hash check")
    fetch_parser.add_argument("--id", required=True)
    fetch_parser.add_argument("--asset-root", required=True, help="External root containing models/<model>/...")
    args = parser.parse_args()
    if args.command == "smoke":
        result = smoke()
    elif args.command == "unpack-results":
        result = unpack(args.version, args.output)
    else:
        result = fetch_asset(args.id, args.asset_root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
