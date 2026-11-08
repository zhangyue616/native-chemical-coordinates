"""Freeze the pre-specified finite input tables and anonymous mechanism objects on D.

Uses the already accepted D run and never opens sealed inputs. No model forward.
This entry point runs only after the precision gate and final D winners exist.
"""
from __future__ import annotations
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from chem_input_reference import input_keys, fit_one
from chem_mechanism_plan import choose_plan
from chem_run import ROOT, DOCS, BASE, D_DIR, ACT, OUT, Budget, dump, read_json, sha, require_lock, utc


def run():
    binding = require_lock(DOCS / "D_RUN_LOCK.json")
    precision = read_json(OUT / "D_PRECISION.json")
    handoff = read_json(OUT / "D_RUN_HANDOFF.json")
    if not precision["precision_gate"] or handoff["binding"] != binding:
        raise ValueError("Final accepted D run and passing precision gate required")
    destination = OUT / "secondary"
    receipt_path = destination / "D_SECONDARY_FREEZE.json"
    if receipt_path.exists():
        raise ValueError("D secondary freeze already exists; inspect it instead of refitting")
    destination.mkdir(exist_ok=True)
    winner_path = OUT / "D_FINAL_WINNERS.npz"
    winner_receipt = read_json(winner_path.with_suffix(".json"))
    if winner_receipt["binding"] != binding or winner_receipt["sha256"] != sha(winner_path):
        raise ValueError("D final winner identity differs")
    with np.load(winner_path, allow_pickle=False) as values:
        concept_ids = values["concept_ids"].tolist()
        candidates = values["candidate_indices"][0, 0].copy()
        if values["networks"].tolist()[0] != "trained" or values["bases"].tolist()[0] != "native":
            raise ValueError("Wrong trained/native winner axes")
    panel = read_json(D_DIR / "panel.json")
    if panel["C_ids"] != concept_ids or len(panel["panel_ids"]) > 12:
        raise ValueError("Panel and frozen winner concept identities differ")
    with (D_DIR / "D.tsv").open(encoding="utf-8", newline="") as stream:
        molecules = list(csv.DictReader(stream, delimiter="\t"))
    with gzip.open(D_DIR / "anchors_D.jsonl.gz", "rt", encoding="utf-8") as stream:
        anchors = [json.loads(line) for line in stream]
    with np.load(D_DIR / "labels_D.npz", allow_pickle=False) as source:
        labels, label_ids = source["labels"], source["concept_ids"].tolist()
    if len(molecules) != 5000 or len(anchors) != 5000:
        raise ValueError("Expected the complete D5000")
    token_ids = []
    for index, (mol, anchor) in enumerate(zip(molecules, anchors)):
        if int(mol["rank"]) != index or anchor["rank"] != index or int(mol["cid"]) != anchor["cid"]:
            raise ValueError("D molecule/anchor identity differs")
        token_ids.append([anchor["input_ids"][a["anchor_token_index"]] for a in anchor["atoms"]])
    trained_dir = ACT / "trained"
    cache = read_json(trained_dir / "CACHE_RECEIPT.json")
    if cache["binding"] != binding or cache["status"] != "PASS_D_EXTRACTION":
        raise ValueError("Trained cache belongs to another D run")
    for name, identity in cache["files"].items():
        path = trained_dir / name
        if path.stat().st_size != identity["bytes"] or path.stat().st_mtime_ns != identity["mtime_ns"]:
            raise ValueError("Previously accepted D cache changed")
    native = np.load(trained_dir / "atom_a.npy", mmap_mode="r", allow_pickle=False)
    outputs = np.load(trained_dir / "normalized_head.npy", mmap_mode="r", allow_pickle=False)
    if native.shape != (len(labels), 464):
        raise ValueError("D atom cache/label shape differs")
    budget = Budget(binding)
    with budget.track("statistics", "D_finite_input_tables_and_anonymous_mechanism_freeze"):
        rows = input_keys([m["smiles"] for m in molecules], token_ids)
        tables = []
        for cid in panel["panel_ids"]:
            budget.check()
            candidate = int(candidates[concept_ids.index(cid)])
            if not 0 <= candidate < 928:
                raise ValueError("Invalid D selected candidate")
            scalar = native[:, candidate % 464] * (1.0 if candidate < 464 else -1.0)
            table = fit_one(rows, labels[:, label_ids.index(cid)], scalar, cid)
            table.update(D_selected_candidate=candidate, D_binding=binding,
                         D_final_winners_sha256=winner_receipt["sha256"])
            path = destination / f"input_reference_{cid}.json"
            dump(path, table)
            tables.append({"concept_id": cid, "candidate": candidate, "file": str(path.relative_to(ROOT)),
                           "sha256": sha(path), "category_counts": table["category_counts"],
                           "observed_bin_count": table["observed_bin_count"]})
        with np.load(D_DIR / "occurrence_counts_D.npz", allow_pickle=False) as occurrence:
            ids = occurrence["concept_ids"].tolist()
            mech_ids = panel["mechanism_candidate_ids"]
            counts = occurrence["counts"][:, [ids.index(c) for c in mech_ids]]
        with np.load(trained_dir / "D_moments.npz", allow_pickle=False) as moments:
            plan = choose_plan(mech_ids, [candidates[concept_ids.index(c)] for c in mech_ids], counts, outputs,
                moments["atom_mean"], moments["atom_mean_abs"], moments["atom_sd"], moments["cls_mean"],
                moments["output_dense_column_norm"])
        plan.update(binding=binding, D_final_winners_sha256=winner_receipt["sha256"])
        plan_path = destination / "MECHANISM_D_PLAN.json"
        dump(plan_path, plan)
    files = [Path(__file__), Path(__file__).with_name("chem_input_reference.py"),
             Path(__file__).with_name("chem_mechanism_plan.py")]
    receipt = {"status": "D_SECONDARY_FROZEN_AWAIT_ROOT_D_OUTPUT_LOCK", "created_utc": utc(), "binding": binding,
        "D_final_winners_sha256": winner_receipt["sha256"], "panel_ids": panel["panel_ids"],
        "readout_tables": tables, "mechanism_plan": {"file": str(plan_path.relative_to(ROOT)), "sha256": sha(plan_path)},
        "implementation": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "mechanism_object_statuses": [{"concept_id": o["concept_id"], "status": o["status"],
                                        "failure_reasons": o["failure_reasons"]} for o in plan["objects"]],
        "T_reads": 0, "model_forwards": 0, "T_authorized": False,
        "CPU_ledger": str(budget.path.relative_to(ROOT))}
    dump(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "panel": panel["panel_ids"],
                      "mechanism_statuses": receipt["mechanism_object_statuses"],
                      "receipt_sha256": sha(receipt_path)}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
