"""Post-hoc structure-frozen scaffold slice; old science and ledgers are read-only."""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/revision_20261017"))
from chem_statistics import (CachedScalar, average_precision, cluster_weights, estimands,
    fixed_winner_ap, molecule_label_counts, weighted_prevalence)
from chem_secondary_statistics import panel_brier

OUT = ROOT / "results/strengthening_20261025/scaffold_subset"
BASE = ROOT / "data/revision_20261017/chemistry"
DOUT = ROOT / "results/revision_20261017/chemistry_D"
TROOT = ROOT / "results/revision_20261017/chemistry_T"
LINEAGE = ROOT / "results/strengthening_20261025/lineage"
PROTOCOL = ROOT / "docs/strengthening_20261025/A_SUBSET_PROTOCOL.md"
NETWORKS = ["trained", "init21000", "init21001", "init21002"]
BASES = ["native"] + [f"Q{s}" for s in range(30000, 30009)]
REPS, SEED, CPU_CAP = 2000, 20261025, 9000.0


def utc():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def memory():
    if os.name != "nt":
        return {"rss_bytes": None, "peak_rss_bytes": None}
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    c = Counters(); c.cb = ctypes.sizeof(c)
    current = ctypes.windll.kernel32.GetCurrentProcess; current.restype = ctypes.c_void_p
    get = ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    get.restype = ctypes.c_int
    if not get(current(), ctypes.byref(c), c.cb):
        raise RuntimeError("cannot measure process memory")
    return {"rss_bytes": int(c.WorkingSetSize), "peak_rss_bytes": int(c.PeakWorkingSetSize)}


class Ledger:
    """Serial per-process additive ledger, separate from all old ledgers."""
    def __init__(self, stage):
        self.path = OUT / "CPU_LEDGER.json"
        self.record = read(self.path) if self.path.exists() else {
            "schema": "A3-independent-process-cpu/1", "cpu_cap_seconds": CPU_CAP, "sessions": []}
        self.previous = sum(s["cpu_seconds"] for s in self.record["sessions"])
        self.item = {"stage": stage, "pid": os.getpid(), "started_utc": utc(), "cpu_seconds": 0.0,
            "status": "RUNNING", "threads": 4}
        self.record["sessions"].append(self.item)
        self.last = 0.0
        self.wall_start = time.monotonic()

    def check(self, force=False):
        used = time.process_time()
        if force or used - self.last >= 4:
            self.item.update(cpu_seconds=used, wall_seconds=time.monotonic() - self.wall_start,
                updated_utc=utc(), memory=memory())
            self.record["cumulative_cpu_seconds"] = self.previous + used
            write(self.path, self.record)
            self.last = used
        if self.previous + used >= CPU_CAP:
            raise TimeoutError("A3 cumulative 9000 process-CPU seconds exhausted; partial retained")
        # RSS target is recorded, with a conservative hard stop at the stated target.
        if memory()["rss_bytes"] is not None and memory()["rss_bytes"] > 2 * 1024**3:
            raise MemoryError("A3 2GiB process working-set target exceeded; partial retained")

    def finish(self, status, error=None):
        self.item.update(status=status, error=error)
        self.check(force=True)


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(clean(rows))


def engineering():
    scores = np.array([0, 0, 1, 2, 2, 3], np.float32)
    y = np.array([1, 0, 1, 0, 1, 0], bool)
    mol = np.array([0, 0, 1, 2, 2, 2], np.int32)
    w = np.array([[1, 1, 1], [2, 0, 1], [0, 2, 1]], np.int32)
    score_matrix = np.zeros((6, 464), np.float32); score_matrix[:, 5] = scores
    actual = fixed_winner_ap(score_matrix, y[:, None], mol, np.array([5]), w)[:, 0]
    expected = []
    for row in w:
        indices = np.concatenate([np.tile(np.flatnonzero(mol == m), row[m]) for m in range(3)])
        s, labels = scores[indices], y[indices]
        positive_total = labels.sum(); numerator = 0.0
        for threshold in np.unique(s)[::-1]:
            group_positive = labels[s == threshold].sum()
            numerator += group_positive * labels[s >= threshold].sum() / (s >= threshold).sum()
        expected.append(numerator / positive_total)
    assert np.allclose(actual, expected, atol=1e-15, rtol=0)
    generated = cluster_weights(3, 8, SEED)
    assert generated.dtype == np.int32 and np.all(generated.sum(1) == 3)
    labels = np.c_[y, ~y]
    p = np.zeros((6, 2, 4), np.float64) + .4
    p[..., 3] = .6
    panel = panel_brier(labels, p, mol, w)
    direct = []
    for row in w:
        atomw = row[mol]
        difference = (labels - p[..., 2])**2 - (labels - p[..., 3])**2
        direct.append(((difference * atomw[:, None]).sum(0) / atomw.sum()).mean())
    assert np.allclose(panel["replicated_panel"], direct, atol=1e-15, rtol=0)
    return {"status": "PASS_ONE_MINIMAL_IMPLEMENTATION_FIXTURE", "exact_tie_weighted_AP_matches_literal_resampling": True,
        "fixed_m_cluster_weights": True, "shared_weight_Brier_matches_direct": True}


def prepare(ledger):
    path = OUT / "PREPARED.json"
    if path.exists():
        saved = read(path)
        for name, target in (("script_sha256", Path(__file__)), ("protocol_sha256", PROTOCOL),
            ("mask_sha256", LINEAGE / "T_D_UNSEEN_SCAFFOLD_MASK.npy"), ("freeze_sha256", LINEAGE / "FREEZE.json")):
            if saved[name] != sha(target):
                raise ValueError(f"prepared identity drift: {name}")
        return saved
    write(OUT / "ENGINEERING_CHECK.json", engineering())
    freeze = read(LINEAGE / "FREEZE.json")
    mask_path = LINEAGE / "T_D_UNSEEN_SCAFFOLD_MASK.npy"
    mask = np.load(mask_path, allow_pickle=False)
    if freeze["freeze_status"] != "STRUCTURE_ONLY_POST_HOC_SLICE_FROZEN" or sha(mask_path) != freeze["mask_sha256"]:
        raise ValueError("lineage structure-only freeze missing/drifted")
    if mask.dtype != bool or mask.shape != (10000,) or int(mask.sum()) != freeze["n_selected"]:
        raise ValueError("mask schema differs")
    if sha(BASE / "sealed/T.tsv") != freeze["inputs"]["T"]["sha256"]:
        raise ValueError("frozen T row identity differs")
    with np.load(DOUT / "D_FINAL_WINNERS.npz", allow_pickle=False) as z:
        C, winners = z["concept_ids"], z["candidate_indices"]
        assert z["networks"].tolist() == NETWORKS and z["bases"].tolist() == BASES
    if len(C) != 71 or winners.shape != (4, 10, 71) or np.any((winners < 0) | (winners >= 928)):
        raise ValueError("full frozen 71/network/basis winner schema differs")
    with np.load(BASE / "test/labels_T.npz", allow_pickle=False) as z:
        columns = [int(np.flatnonzero(z["concept_ids"] == c)[0]) for c in C]
        labels_all = np.ascontiguousarray(z["labels"][:, columns], dtype=bool)
    with np.load(BASE / "test/atoms_T.npz", allow_pickle=False) as z:
        mol_all, atom_indices, offsets = z["mol_idx"], z["atom_idx"], z["mol_offsets"]
    if len(offsets) != 10001 or offsets[-1] != len(labels_all) or not np.array_equal(mol_all, np.repeat(np.arange(10000), np.diff(offsets))):
        raise ValueError("original atom/molecule order differs")
    selected = np.flatnonzero(mask).astype(np.int32)
    atom_rows = np.flatnonzero(mask[mol_all]).astype(np.int32)
    labels = np.ascontiguousarray(labels_all[atom_rows])
    molecules = np.searchsorted(selected, mol_all[atom_rows]).astype(np.int32)
    m = len(selected)
    positives, counts = molecule_label_counts(labels, molecules, m)
    weights = cluster_weights(m, REPS, SEED)
    assert weights.dtype == np.int32 and np.all(weights.sum(1) == m)
    np.save(OUT / "SLICE_BOOTSTRAP_WEIGHTS.npy", weights)
    pi = weighted_prevalence(positives, counts, np.vstack((np.ones((1, m), np.int32), weights)))
    np.save(OUT / "SLICE_PREVALENCE.npy", pi)
    support = []
    for ci, cid in enumerate(C):
        posmol, posatom = int((positives[:, ci] > 0).sum()), int(positives[:, ci].sum())
        reasons = []
        if posmol < 20: reasons.append("positive_molecules_lt_20")
        if m - posmol < 100: reasons.append("all_negative_molecules_lt_100")
        if posatom < 50: reasons.append("positive_atoms_lt_50")
        support.append({"concept_id": int(cid), "positive_molecules": posmol, "all_negative_molecules": m-posmol,
            "positive_atoms": posatom, "negative_atoms": len(labels)-posatom, "prevalence": float(pi[0, ci]),
            "historical_support_reference_met": not reasons, "historical_support_shortfalls": ";".join(reasons),
            "has_positive_atoms": posatom > 0, "has_negative_atoms": posatom < len(labels),
            "bootstrap_zero_positive_replicates": int((pi[1:, ci] == 0).sum()),
            "bootstrap_zero_negative_replicates": int((pi[1:, ci] == 1).sum())})
    np.savez(OUT / "SLICE_INPUT.npz", labels=labels, molecules=molecules, atom_rows=atom_rows,
        selected_molecule_rows=selected, concept_ids=C, candidates=winners)
    csv_write(OUT / "CONCEPT_SUPPORT.csv", support)
    write(OUT / "CONCEPT_SUPPORT.json", {"concepts": support, "historical_reference_only_not_filter": True,
        "concepts_deleted": False, "support_reference_shortfall_ids": [v["concept_id"] for v in support if not v["historical_support_reference_met"]]})
    saved = {"status": "A3_STRUCTURE_SLICE_PREPARED", "created_utc": utc(), "n_molecules": m,
        "n_atoms": len(labels), "original_n_atoms": len(labels_all), "concept_ids": C.tolist(),
        "networks": NETWORKS, "bases": BASES, "replicates": REPS, "seed": SEED,
        "all_point_prevalences_positive": bool(np.all(pi[0] > 0)), "mask_sha256": sha(mask_path),
        "freeze_sha256": sha(LINEAGE / "FREEZE.json"), "script_sha256": sha(__file__),
        "protocol_sha256": sha(PROTOCOL), "weights_sha256": sha(OUT / "SLICE_BOOTSTRAP_WEIGHTS.npy"),
        "D_final_winners_sha256": sha(DOUT / "D_FINAL_WINNERS.npz"),
        "pure_statistics_sha256": sha(ROOT / "scripts/revision_20261017/chem_statistics.py"),
        "pure_secondary_statistics_sha256": sha(ROOT / "scripts/revision_20261017/chem_secondary_statistics.py"),
        "read_only_input_files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in [BASE / "test/labels_T.npz", BASE / "test/atoms_T.npz", DOUT / "D_FINAL_WINNERS.npz",
                TROOT / "secondary/input/predictions_T.npy", *[BASE / f"activations_T/{n}/atom_a.npy" for n in NETWORKS]]},
        "no_model_forwards": True, "no_D_reselection": True, "no_T_oracle": True, "old_scientific_writes": 0}
    write(path, saved); ledger.check(force=True)
    print(json.dumps({"phase": "prepared", "n_molecules": m, "n_atoms": len(labels), "positive_pi_all71": saved["all_point_prevalences_positive"]}), flush=True)
    return saved


def verify_prepared():
    saved = read(OUT / "PREPARED.json")
    if saved["script_sha256"] != sha(__file__) or saved["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("implementation/protocol changed after preparation")
    for rel, ident in saved["read_only_input_files"].items():
        p = ROOT / rel
        if p.stat().st_size != ident["bytes"] or p.stat().st_mtime_ns != ident["mtime_ns"]:
            raise ValueError(f"read-only input changed: {rel}")
    return saved


def unit(network, basis, ledger):
    saved = verify_prepared()
    path = OUT / "fixed" / f"{network}_{basis}.npz"
    if path.exists() and path.with_suffix(".json").exists():
        receipt = read(path.with_suffix(".json"))
        if receipt["prepared_sha256"] != sha(OUT / "PREPARED.json") or receipt["sha256"] != sha(path):
            raise ValueError("existing deterministic unit checkpoint differs")
        return
    z = np.load(OUT / "SLICE_INPUT.npz", allow_pickle=False)
    labels, molecules, rows = z["labels"], z["molecules"], z["atom_rows"]
    candidates = z["candidates"][NETWORKS.index(network), BASES.index(basis)]
    native = np.load(BASE / f"activations_T/{network}/atom_a.npy", mmap_mode="r", allow_pickle=False)
    if native.dtype != np.float32 or native.shape != (saved["original_n_atoms"], 464):
        raise ValueError("original T atom cache shape/dtype differs")
    scratch_path = OUT / "scores_scratch.npy"
    scores = np.lib.format.open_memmap(scratch_path, mode="w+", dtype=np.float32, shape=(len(rows), 464))
    if basis != "native":
        qpath = DOUT / f"rotations/{basis}.npy"
        qreceipt = read(qpath.with_suffix(".json"))
        if qreceipt["sha256"] != sha(qpath):
            raise ValueError("original Q changed")
        Q = np.load(qpath, allow_pickle=False)
        if Q.shape != (464,464) or Q.dtype != np.float64:
            raise ValueError("Q float64 shape differs")
        q32 = Q.astype(np.float32)
    for begin in range(0, len(native), 2048):
        ledger.check()
        end = min(begin+2048, len(native))
        lo, hi = np.searchsorted(rows, (begin, end))
        if lo == hi: continue
        block = native[begin:end] if basis == "native" else native[begin:end] @ q32
        scores[lo:hi] = block[rows[lo:hi] - begin]
    scores.flush()
    weights = np.load(OUT / "SLICE_BOOTSTRAP_WEIGHTS.npy", mmap_mode="r", allow_pickle=False)
    pi = np.load(OUT / "SLICE_PREVALENCE.npy", allow_pickle=False)
    # Undefined fixed-set point does not acquire a survivor-only bootstrap.
    use_weights = np.vstack((np.ones((1, saved["n_molecules"]), np.int32), weights)) if saved["all_point_prevalences_positive"] else np.ones((1, saved["n_molecules"]), np.int32)
    ap = fixed_winner_ap(scores, labels, molecules, candidates, use_weights, budget_check=ledger.check)
    with np.errstate(divide="ignore", invalid="ignore"):
        ell = np.log(ap / pi[:len(ap)])
    path.parent.mkdir(exist_ok=True)
    np.savez(path, AP=ap, ell=ell)
    ledger.check(force=True)
    receipt = {"prepared_sha256": sha(OUT / "PREPARED.json"), "sha256": sha(path),
        "network": network, "basis": basis, "cpu_seconds": time.process_time(), "memory": memory(),
        "replicates": len(ap)-1, "fixed_D_winners": True, "score_dtype": "float32", "score_ties": "exact equality"}
    write(path.with_suffix(".json"), receipt)
    print(json.dumps({"phase": "unit_complete", **receipt}), flush=True)


def finalize(ledger):
    saved = verify_prepared()
    z = np.load(OUT / "SLICE_INPUT.npz", allow_pickle=False)
    labels, molecules, rows, C = z["labels"], z["molecules"], z["atom_rows"], z["concept_ids"]
    nrep = REPS+1 if saved["all_point_prevalences_positive"] else 1
    ap = np.empty((nrep, 4, 10, 71), np.float64)
    ell = np.empty_like(ap)
    for mi, network in enumerate(NETWORKS):
        for bi, basis in enumerate(BASES):
            path = OUT / "fixed" / f"{network}_{basis}.npz"
            with np.load(path, allow_pickle=False) as unit_data:
                ap[:, mi, bi], ell[:, mi, bi] = unit_data["AP"], unit_data["ell"]
    point, reps = estimands(ell[0]), estimands(ell[1:]) if nrep > 1 else None
    valid = np.isfinite(ell[1:]).all(axis=(1,2,3))
    point_valid = bool(np.isfinite(ell[0]).all())
    summary = {"status": "POSTHOC_SCAFFOLD_SENSITIVITY_COMPLETE" if point_valid else "NOT_ESTIMABLE",
        "n_molecules": saved["n_molecules"], "n_atoms": saved["n_atoms"], "fixed_concepts": 71,
        "concept_ids": C, "all_fixed_concepts_point_estimable": point_valid,
        "point_nonestimable_concept_ids": C[~np.isfinite(ell[0]).all(axis=(0,1))],
        "defined_replicates": int(valid.sum()), "undefined_replicates": int((~valid).sum()),
        "undefined_fraction": float((~valid).mean()) if len(valid) else None,
        "bootstrap_replicates_executed": len(valid), "bootstrap_replicates_planned": REPS,
        "interval_stability_reference_undefined_le_1pct": bool((~valid).sum()*100 <= len(valid)) if len(valid) else None,
        "conditional_interval_after_undefined_exclusion": bool((~valid).any()), "quantile_method": "linear",
        "no_new_confirmatory_test": True, "no_new_p_values": True, "confirmation_PASS_claimed": False,
        "interpretation": "post-hoc sensitivity conditional on fixed checkpoint,3 initialization references,9 Q,D-selected winners and structure-only slice; not an independent scaffold-split replication",
        "mask_sha256": saved["mask_sha256"], "weights_sha256": saved["weights_sha256"]}
    for name in ("Theta", "G_init"):
        values = reps[name][valid] if reps is not None else np.array([])
        summary[name] = {"point": point[name], "ci95": np.quantile(values, [.025,.975], method="linear") if len(values) else [None,None]}
    concepts = []
    for ci, cid in enumerate(C):
        defined = np.isfinite(ell[1:,:,:,ci]).all(axis=(1,2))
        concepts.append({"concept_id": int(cid), "Theta_point": point["Theta_per_concept"][ci],
            "G_init_point": point["G_init_per_concept"][ci], "point_estimable": bool(np.isfinite(ell[0,:,:,ci]).all()),
            "undefined_replicates": int((~defined).sum()), "undefined_fraction": float((~defined).mean()) if len(defined) else None})
    csv_write(OUT / "CONCEPT_ESTIMATES.csv", concepts)
    np.savez(OUT / "SLICE_FIXED_REPLICATES.npz", AP=ap, ell=ell, concept_ids=C,
        Theta=point["Theta"], G_init=point["G_init"],
        Theta_per_concept=point["Theta_per_concept"], G_init_per_concept=point["G_init_per_concept"],
        replicated_Theta=reps["Theta"] if reps is not None else np.array([]),
        replicated_G_init=reps["G_init"] if reps is not None else np.array([]), joint_defined=valid)
    write(OUT / "SLICE_FIXED_SUMMARY.json", summary)
    # Same structural slice, old frozen predictions, same newly sampled molecule weights.
    panel_ids = read(DOUT / "secondary/D_SECONDARY_FREEZE.json")["panel_ids"]
    panel_columns = [int(np.flatnonzero(C == c)[0]) for c in panel_ids]
    panel_labels = labels[:, panel_columns]
    predictions = np.load(TROOT / "secondary/input/predictions_T.npy", mmap_mode="r", allow_pickle=False)[rows]
    weights = np.load(OUT / "SLICE_BOOTSTRAP_WEIGHTS.npy", mmap_mode="r", allow_pickle=False)
    brier = panel_brier(panel_labels, predictions, molecules, weights)
    np.save(OUT / "PANEL_BRIER_BOOTSTRAP.npy", brier.pop("replicated_panel"))
    panel_rows = []
    for ci, cid in enumerate(panel_ids):
        panel_rows.append({"concept_id": cid, "prevalence": float(panel_labels[:,ci].mean()),
            "positive_atoms": int(panel_labels[:,ci].sum()), "negative_atoms": int((~panel_labels[:,ci]).sum()),
            "Brier_L": brier["Brier_per_concept_reference"][ci][2],
            "Brier_Lj": brier["Brier_per_concept_reference"][ci][3], "Delta_Brier": brier["per_concept"][ci]})
    csv_write(OUT / "PANEL_BRIER.csv", panel_rows)
    brier.update(panel_ids=panel_ids, concepts=panel_rows, shared_weights_sha256=saved["weights_sha256"],
        fitted_new_tables=False, modified_bins_or_fallback=False, posthoc_sensitivity=True,
        no_new_p_values=True, zero_label_Brier_mathematically_defined=True)
    write(OUT / "PANEL_BRIER_SUMMARY.json", brier)
    ledger.check(force=True)
    write(OUT / "A3_HANDOFF.json", {"status": summary["status"], "summary": summary,
        "panel_DeltaBrier": brier["Delta_L_panel"], "panel_ci95": brier["panel_ci95"],
        "cpu_ledger": str(ledger.path), "process_memory": memory(), "new_model_forwards": 0,
        "old_scientific_writes": 0, "all_outputs_root": str(OUT)})
    print(json.dumps(clean({"phase": "final", "summary": summary, "panel_DeltaBrier": brier["Delta_L_panel"], "panel_ci95": brier["panel_ci95"]})), flush=True)


def run():
    # Child processes release BLAS/Numba/sorted-label memory at each basis boundary.
    commands = [["prepare"]] + [["unit", "--network", n, "--basis", b] for n in NETWORKS for b in BASES] + [["final"]]
    for args in commands:
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *args], cwd=str(ROOT))
        if completed.returncode:
            raise RuntimeError(f"A3 child failed ({completed.returncode}): {args}")
    # Parent's own CPU is additive too; read the final child ledger only now.
    ledger = Ledger("orchestrator"); ledger.finish("COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "prepare", "unit", "final"])
    parser.add_argument("--network", choices=NETWORKS)
    parser.add_argument("--basis", choices=BASES)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    set_num_threads(4)
    with threadpool_limits(limits=4):
        if args.stage == "run":
            run(); return
        ledger = Ledger(f"{args.stage}:{args.network or ''}:{args.basis or ''}")
        try:
            if args.stage == "prepare": prepare(ledger)
            elif args.stage == "unit":
                if not args.network or not args.basis: raise ValueError("unit needs network and basis")
                unit(args.network, args.basis, ledger)
            else: finalize(ledger)
            ledger.finish("COMPLETE")
        except BaseException as error:
            # Keep every failure, without clearing earlier failures on recovery.
            path = OUT / f"FAILURE_{os.getpid()}_{time.time_ns()}.json"
            write(path, {"utc": utc(), "stage": vars(args), "error_type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(),
                "resume_rule": "same mask,all71,4x10,2000 reps,seed; preserve old science and partials"})
            try: ledger.finish("FAILED", f"{type(error).__name__}: {error}")
            except BaseException: pass
            raise


if __name__ == "__main__":
    main()
