"""Locked chemistry T extraction, fixed-winner confirmation and symmetric oracle.

The ready command reads implementation receipts only. Scientific commands need
the root D_OUTPUT_LOCK, accepted T data receipt, and the unchanged D runner.
No command builds labels or changes D selection, references or mechanisms.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from numba import set_num_threads
import torch
from threadpoolctl import threadpool_limits, threadpool_info

from chem_model import ChembertaMTR, PRIMARY_LAYER, REPOSITORY, REVISION, TOKENIZER_SPECIAL_IDS, batch_from_audits
from chem_statistics import (bootstrap_summary, cluster_weights, estimands, fixed_winner_ap,
                             molecule_label_counts, select_winners, weighted_prevalence)
from chem_run import (ROOT, DOCS, BASE, NETWORKS, BASES, CONFIG, Budget, array_file,
                      check_memory, dump, memory_usage, read_json, safe_json, sha, utc,
                      require_lock as require_D_lock)

OUT = ROOT / "results/revision_20261017/chemistry_T"
ACT = BASE / "activations_T"
DATA = BASE / "test"
DOUT = ROOT / "results/revision_20261017/chemistry_D"
READY = OUT / "T_RUNNER_READY.json"
T_DATA_READY = ROOT / "results/revision_20261017/chemistry_T_data_engineering/T_DATA_READY.json"
T_RECEIPT = DATA / "T_DATA_RECEIPT.json"
DEFAULT_LOCK = DOCS / "D_OUTPUT_LOCK.json"
T_CONFIG = {
    "schema": "chemistry-T-runner/1", "molecules": 10000,
    "networks": list(NETWORKS), "bases": list(BASES), "candidates_per_basis": 928,
    "fixed_winners": "only D_FINAL_WINNERS.npz, no reselection for confirmation",
    "oracle": "same T/all928 for all4 networks/all10 bases, point estimates only",
    "bootstrap_replicates": 2000, "bootstrap_seed": 42017,
    "bootstrap": "whole-molecule shared integer weights; each replicate recomputes weighted AP,prevalence,medians,means",
    "native_and_rotated_scores": "float32; rotations use the D saved float64 Q cast to float32",
    "score_ties": "exact float32 equality", "AP": "float64 distinct-score-threshold AP",
    "winner_tie_tolerance": float(8 * np.finfo(np.float64).eps),
    "winner_tie_order": "unit ascending then positive pole first",
    "quantile": "NumPy linear", "event_gate": {"positive_molecules": 20, "all_negative_molecules": 100, "positive_atoms": 50},
    "threads": 4, "batch_size": 4,
    "shared_CPU_ledger": str(DOUT / "CPU_LEDGER.json"),
    "encoder_and_intervention_total_CPU_seconds": 14400,
    "statistics_D_T_oracle_all_bootstrap_total_CPU_seconds": 43200,
    "finite_input_and_mechanism": "separate root-owned runner, same shared CPU ledger",
}


def own_identity():
    files = [Path(__file__), DOUT / "D_RUNNER_READY.json", OUT / "T_RUNNER_ENGINEERING.json"]
    return {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files}


def write_ready():
    if read_json(OUT / "T_RUNNER_ENGINEERING.json").get("status") != "PASS_TARGETED_T_RUNNER_GATES":
        raise ValueError("T runner synthetic gates missing")
    value = {"status": "READY_T_RUNNER_AWAIT_D_OUTPUT_LOCK", "created_utc": utc(),
             "config": T_CONFIG, "files": own_identity(), "T_scientific_reads": 0,
             "T_model_forwards": 0, "T_label_calls": 0, "python": sys.executable,
             "lock_contract": {"status": "D_OUTPUT_LOCKED", "T_scientific_processing_authorized": True,
                "required_hashes": ["data_build_receipt_sha256", "D_final_winners_sha256", "T_data_ready_sha256", "T_runner_ready_sha256"]},
             "cache_schema": {"root": str(ACT), "atom_a.npy": "float32[T atoms,464]",
                "cls_a.npy": "float32[10000,464]", "normalized_head.npy": "float32[10000,199]",
                "row_order": "T.tsv molecule order, then exact canonical graph atom index",
                "receipt": "CACHE_RECEIPT.json for each network, same files/identity schema as D"},
             "shared_weights": {"path": str(OUT / "T_BOOTSTRAP_WEIGHTS.npy"), "shape": [2000,10000],
                "dtype": "int32", "seed": 42017, "created_only_after_D_OUTPUT_LOCK": True},
             "commands": [f".venv_chem/Scripts/python.exe -B scripts/revision_20261017/chem_T_run.py {stage}"
                          for stage in ("extract-T", "fixed-T", "oracle-T", "final-T")]}
    dump(READY, value)
    return {"status": value["status"], "path": str(READY), "T_runner_ready_sha256": sha(READY)}


def require_T_lock(path):
    lock = read_json(path)
    if lock.get("status") != "D_OUTPUT_LOCKED" or lock.get("T_scientific_processing_authorized") is not True:
        raise ValueError("T scientific processing needs the root D_OUTPUT_LOCK")
    binding = require_D_lock(DOCS / "D_RUN_LOCK.json")
    checks = {"data_build_receipt_sha256": BASE / "corpus/BUILD_RECEIPT.json",
              "D_final_winners_sha256": DOUT / "D_FINAL_WINNERS.npz",
              "T_data_ready_sha256": T_DATA_READY, "T_runner_ready_sha256": READY}
    for key, target in checks.items():
        if lock.get(key) != sha(target):
            raise ValueError(f"T lock identity mismatch: {key}")
    ready = read_json(READY)
    if ready["config"] != T_CONFIG or ready["files"] != own_identity():
        raise ValueError("T runner drift after READY")
    if not read_json(DOUT / "D_PRECISION.json").get("precision_gate"):
        raise ValueError("T prohibited after failed D precision gate")
    return binding, {"D_OUTPUT_LOCK_sha256": sha(path), **binding}


def load_T(binding, T_binding):
    receipt = read_json(T_RECEIPT)
    if receipt.get("status") != "PASS_T_DATA_BUILD" or receipt.get("D_output_lock_sha256") != T_binding["D_OUTPUT_LOCK_sha256"]:
        raise ValueError("accepted T data build receipt missing")
    for name in ("INPUT_POINTER.json", "labels_T.npz", "atoms_T.npz", "T_EVENT_GATE.json"):
        path = DATA / name
        identity = receipt["outputs"][path.relative_to(ROOT).as_posix()]
        if path.stat().st_size != identity["bytes"] or sha(path) != identity["sha256"]:
            raise ValueError(f"T data identity drift: {name}")
    pointer = read_json(DATA / "INPUT_POINTER.json")
    expected_inputs = read_json(T_DATA_READY)["T_inputs_from_build_manifest"]
    if pointer["files"] != expected_inputs or receipt["T_inputs"] != expected_inputs:
        raise ValueError("T input pointer differs from locked data metadata")
    for relative, identity in expected_inputs.items():
        path = ROOT / relative
        if path.stat().st_size != identity["bytes"] or sha(path) != identity["sha256"]:
            raise ValueError("T input/anchor identity drift after label build")
    with (BASE / "sealed/T.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    with gzip.open(BASE / "sealed/anchors_T.jsonl.gz", "rt", encoding="utf-8") as stream:
        anchors = [json.loads(line) for line in stream]
    with np.load(DATA / "labels_T.npz", allow_pickle=False) as z:
        labels_all, ids = z["labels"], z["concept_ids"]
    with np.load(DATA / "atoms_T.npz", allow_pickle=False) as z:
        molecules, atom_indices, offsets = z["mol_idx"], z["atom_idx"], z["mol_offsets"]
    with np.load(DOUT / "D_FINAL_WINNERS.npz", allow_pickle=False) as z:
        C, winners = z["concept_ids"], z["candidate_indices"]
        if z["networks"].tolist() != list(NETWORKS) or z["bases"].tolist() != list(BASES):
            raise ValueError("D winner network/basis order differs")
    if receipt["C_ids"] != C.tolist() or read_json(DATA / "T_EVENT_GATE.json")["C_ids"] != C.tolist():
        raise ValueError("T data must retain the entire D-frozen concept set")
    if winners.shape != (4,10,len(C)) or np.any((winners < 0) | (winners >= 928)):
        raise ValueError("D frozen winner candidate shape/range differs")
    columns = [int(np.flatnonzero(ids == c)[0]) for c in C]
    labels = np.ascontiguousarray(labels_all[:, columns], dtype=np.bool_)
    if len(rows) != 10000 or len(anchors) != 10000 or len(offsets) != 10001 or offsets[-1] != len(labels):
        raise ValueError("T molecule/atom/anchor dimension mismatch")
    if not np.array_equal(molecules, np.repeat(np.arange(10000), np.diff(offsets))):
        raise ValueError("T molecule/atom order differs")
    if not np.array_equal(atom_indices, np.concatenate([np.arange(n) for n in np.diff(offsets)])):
        raise ValueError("T graph atom index order differs")
    audits = []
    for mi, (row, anchor) in enumerate(zip(rows, anchors)):
        if int(row["rank"]) != 5000 + mi or anchor["rank"] != 5000 + mi or int(row["cid"]) != anchor["cid"]:
            raise ValueError("T rank/CID identity order differs")
        if hashlib.sha256(row["smiles"].encode("utf-8")).hexdigest() != row["canonical_smiles_sha256"]:
            raise ValueError("T exact canonical identity differs")
        if anchor["canonical_smiles_sha256"] != row["canonical_smiles_sha256"]:
            raise ValueError("T anchor canonical identity differs")
        if hashlib.sha256(json.dumps(anchor["input_ids"],separators=(",",":")).encode("ascii")).hexdigest() != anchor["input_ids_sha256"]:
            raise ValueError("T anchor input IDs differ")
        if [a["graph_atom_id"] for a in anchor["atoms"]] != list(range(int(offsets[mi+1]-offsets[mi]))):
            raise ValueError("T anchor atom order differs")
        audits.append({**anchor, "accepted": anchor["gate"] == "PASS_POSITIONAL_ELEMENT_ANCHOR",
                       "tokenizer": {"repository": REPOSITORY, "revision": REVISION},
                       "special_tokens_mask": [int(i in TOKENIZER_SPECIAL_IDS) for i in anchor["input_ids"]],
                       "graph_atom_count": int(offsets[mi+1]-offsets[mi])})
    dump(OUT / "T_INPUT_ACCEPTANCE.json", {"status": "PASS_T_INPUT_IDENTITY", "binding": T_binding,
         "n_molecules":10000, "n_atoms":len(labels), "concept_ids":C.tolist(),
         "max_tokens":max(len(a["input_ids"]) for a in audits), "T_data_receipt_sha256":sha(T_RECEIPT)})
    return {"rows":rows,"audits":audits,"labels":labels,"molecules":molecules,"offsets":offsets,
            "concept_ids":C,"winners":winners,"binding":binding,"T_binding":T_binding,
            "coverage":read_json(DATA / "T_EVENT_GATE.json")}


def extract_T(data, budget):
    ACT.mkdir(parents=True,exist_ok=True)
    for name in NETWORKS:
        destination=ACT/name;destination.mkdir(exist_ok=True)
        progress_path=destination/"PROGRESS.json";receipt_path=destination/"CACHE_RECEIPT.json"
        if receipt_path.exists():
            receipt=read_json(receipt_path)
            if receipt.get("status")!="PASS_T_EXTRACTION" or receipt["binding"]!=data["binding"] or receipt["T_binding"]!=data["T_binding"]:
                raise ValueError("T cache binding differs")
            continue
        progress=read_json(progress_path) if progress_path.exists() else {"binding":data["binding"],"T_binding":data["T_binding"],"completed_molecules":0}
        if progress["binding"]!=data["binding"] or progress["T_binding"]!=data["T_binding"]:
            raise ValueError("partial T cache binding differs")
        resume=progress_path.exists()
        atoms=array_file(destination/"atom_a.npy",(len(data["labels"]),464),np.float32,resume=resume)
        cls=array_file(destination/"cls_a.npy",(10000,464),np.float32,resume=resume)
        outputs=array_file(destination/"normalized_head.npy",(10000,199),np.float32,resume=resume)
        dump(progress_path,progress)
        with budget.track("encoder",f"extract_T_{name}"):
            model=ChembertaMTR.from_official_checkpoint() if name=="trained" else ChembertaMTR.initialized_from_config(int(name[4:]))
            for begin in range(progress["completed_molecules"],10000,4):
                budget.check();end=min(begin+4,10000)
                result=model(batch_from_audits(data["audits"][begin:end]))
                for local,mi in enumerate(range(begin,end)):
                    positions=[a["anchor_token_index"] for a in data["audits"][mi]["atoms"]]
                    atoms[data["offsets"][mi]:data["offsets"][mi+1]]=result.activations[local,positions].numpy()
                cls[begin:end]=result.activations[:,0].numpy();outputs[begin:end]=result.normalized_predictions.numpy()
                budget.record["model_forward_molecule_evaluations"]+=end-begin
                atoms.flush();cls.flush();outputs.flush()
                progress.update(completed_molecules=end,updated_utc=utc());dump(progress_path,progress)
                if end%200==0:print(json.dumps({"phase":"extract_T","network":name,"completed_molecules":end}),flush=True)
            files=[destination/n for n in ("atom_a.npy","cls_a.npy","normalized_head.npy")]
            dump(receipt_path,{"status":"PASS_T_EXTRACTION","binding":data["binding"],"T_binding":data["T_binding"],
                "network":name,"n_molecules":10000,"n_atoms":len(atoms),"dtype":"float32","layer_zero_based":PRIMARY_LAYER,
                "atom_shape":list(atoms.shape),"cls_shape":list(cls.shape),"head_shape":list(outputs.shape),
                "normalization":"normalized anonymous head","completed_utc":utc(),
                "files":{p.name:{"bytes":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns,"sha256":sha(p)} for p in files}})
            del model,atoms,cls,outputs;gc.collect()


def require_cache(data,name):
    destination=ACT/name;receipt=read_json(destination/"CACHE_RECEIPT.json")
    if receipt.get("status")!="PASS_T_EXTRACTION" or receipt["binding"]!=data["binding"] or receipt["T_binding"]!=data["T_binding"]:
        raise ValueError("T extraction cache missing/mismatched")
    path=destination/"atom_a.npy";identity=receipt["files"][path.name]
    if path.stat().st_size!=identity["bytes"] or path.stat().st_mtime_ns!=identity["mtime_ns"]:
        raise ValueError("T activation cache changed")
    value=np.load(path,mmap_mode="r",allow_pickle=False)
    if value.shape!=(len(data["labels"]),464) or value.dtype!=np.float32:raise ValueError("T atom shape/dtype differs")
    return value


def basis_matrix(native,basis,budget):
    if basis=="native":return native
    path=DOUT/"rotations"/f"{basis}.npy";receipt=read_json(path.with_suffix(".json"))
    if sha(path)!=receipt["sha256"]:raise ValueError("D saved shared Q changed")
    Q=np.load(path,allow_pickle=False)
    if Q.dtype!=np.float64 or Q.shape!=(464,464):raise ValueError("D Q shape/dtype differs")
    q32=Q.astype(np.float32)
    scores=np.lib.format.open_memmap(ACT/"rotation_scratch.npy",mode="w+",dtype=np.float32,shape=native.shape)
    for begin in range(0,len(native),2048):
        budget.check();scores[begin:begin+2048]=native[begin:begin+2048]@q32
    scores.flush();return scores


def event_gate(data):
    positives,counts=molecule_label_counts(data["labels"],data["molecules"],10000)
    events=[]
    for ci,c in enumerate(data["concept_ids"]):
        positive_molecules=int((positives[:,ci]>0).sum());negative_molecules=10000-positive_molecules
        positive_atoms=int(positives[:,ci].sum())
        passed=positive_molecules>=20 and negative_molecules>=100 and positive_atoms>=50
        events.append({"concept_id":int(c),"positive_molecules":positive_molecules,
            "all_negative_molecules":negative_molecules,"positive_atoms":positive_atoms,
            "max_single_molecule_positive_atom_contribution":int(positives[:,ci].max()),"event_gate":bool(passed)})
    return positives,counts,{"joint_all_frozen_C_event_gate":all(v["event_gate"] for v in events),"concepts":events,
        "failed_concept_ids":[v["concept_id"] for v in events if not v["event_gate"]],"C_deleted_or_reselected":False}


def shared_weights(data,budget):
    path=OUT/"T_BOOTSTRAP_WEIGHTS.npy";receipt_path=path.with_suffix(".json")
    if path.exists() and receipt_path.exists():
        receipt=read_json(receipt_path)
        if receipt["binding"]!=data["T_binding"] or receipt["sha256"]!=sha(path):raise ValueError("T resample weights changed")
        weights=np.load(path,allow_pickle=False)
    else:
        with budget.track("statistics","T_shared_2000_molecule_weights"):
            weights=cluster_weights(10000,2000,42017);np.save(path,weights)
            dump(receipt_path,{"binding":data["T_binding"],"sha256":sha(path),"shape":[2000,10000],
                "dtype":"int32","seed":42017,"all_atoms_and_labels_and_scalars_carried":True})
    if weights.shape!=(2000,10000) or weights.dtype!=np.int32 or not np.all(weights.sum(1)==10000):
        raise ValueError("T resampling schema differs")
    return np.vstack((np.ones((1,10000),np.int32),weights))


def fixed_T(data,budget):
    weights=shared_weights(data,budget)
    with budget.track("statistics","T_prevalence_and_frozen_concept_events"):
        positives,counts,events=event_gate(data);pi=weighted_prevalence(positives,counts,weights)
        dump(OUT/"T_EVENTS.json",events)
    ell=np.full((2001,4,10,len(data["concept_ids"])),np.nan,np.float64)
    point_AP=np.full((4,10,len(data["concept_ids"])),np.nan,np.float64)
    destination=OUT/"fixed";destination.mkdir(exist_ok=True)
    for mi,network in enumerate(NETWORKS):
        native=require_cache(data,network)
        for bi,basis in enumerate(BASES):
            path=destination/f"{network}_{basis}.npz";receipt_path=path.with_suffix(".json")
            if path.exists() and receipt_path.exists():
                receipt=read_json(receipt_path)
                if receipt["binding"]!=data["T_binding"] or receipt["sha256"]!=sha(path):raise ValueError("fixed T partial cache differs")
                with np.load(path,allow_pickle=False) as z:
                    ell[:,mi,bi]=z["ell"];point_AP[mi,bi]=z["AP"][0]
                continue
            with budget.track("statistics",f"Tfixed_{network}_{basis}_rotation"):
                scores=basis_matrix(native,basis,budget)
            with budget.track("statistics",f"Tfixed_{network}_{basis}_2000_cluster_bootstrap"):
                ap=fixed_winner_ap(scores,data["labels"],data["molecules"],data["winners"][mi,bi],weights,budget_check=budget.check)
                with np.errstate(divide="ignore",invalid="ignore"):values=np.log(ap/pi)
                ell[:,mi,bi]=values;point_AP[mi,bi]=ap[0];np.savez(path,ell=values,AP=ap)
                dump(receipt_path,{"binding":data["T_binding"],"sha256":sha(path),"replicates":2000,"D_winners_reselected":False})
            del scores;print(json.dumps({"phase":"fixed_T","network":network,"basis":basis,"status":"complete"}),flush=True)
        del native
    with budget.track("statistics","Tfixed_estimands_and_joint_intersection_union"):
        summary=bootstrap_summary(ell[0],ell[1:])
        summary["statistical_IUT_lower_bounds_positive"]=summary["joint_IUT_supported"]
        summary["event_gate"]=events["joint_all_frozen_C_event_gate"]
        summary["joint_IUT_supported"]=bool(summary["joint_IUT_supported"] and summary["event_gate"])
        summary.update({"status":"FIXED_T_CONFIRMATION_COMPLETE","binding":data["T_binding"],
            "concept_ids":data["concept_ids"].tolist(),"fixed_D_winners":True,
            "interpretation":"conditional on one trained checkpoint,3 initialization references,9 fixed Q and selected D; no general training causal claim"})
        np.savez(OUT/"T_FIXED_REPLICATES.npz",point_ell=ell[0],replicated_ell=ell[1:],
                 point_AP=point_AP,point_prevalence=pi[0],concept_ids=data["concept_ids"])
        dump(OUT/"T_FIXED_SUMMARY.json",safe_json(summary))
    return safe_json(summary)


def oracle_T(data,budget):
    with budget.track("statistics","Toracle_point_prevalence"):
        positives,counts,_=event_gate(data);pi=weighted_prevalence(positives,counts,np.ones((1,10000),np.int32))[0]
    candidates=np.full_like(data["winners"],-1);aps=np.full(candidates.shape,np.nan,np.float64)
    destination=OUT/"oracle";destination.mkdir(exist_ok=True)
    for mi,network in enumerate(NETWORKS):
        native=require_cache(data,network)
        for bi,basis in enumerate(BASES):
            path=destination/f"{network}_{basis}.npz";receipt_path=path.with_suffix(".json")
            if path.exists() and receipt_path.exists():
                receipt=read_json(receipt_path)
                if receipt["binding"]!=data["T_binding"] or receipt["sha256"]!=sha(path):raise ValueError("oracle partial cache differs")
                with np.load(path,allow_pickle=False) as z:candidates[mi,bi],aps[mi,bi]=z["candidate_indices"],z["AP"]
                continue
            with budget.track("statistics",f"Toracle_{network}_{basis}_rotation"):
                scores=basis_matrix(native,basis,budget)
            with budget.track("statistics",f"Toracle_{network}_{basis}_all928_AP"):
                candidate,ap=select_winners(scores,data["labels"],progress=lambda unit:budget.check())
                candidates[mi,bi],aps[mi,bi]=candidate,ap;np.savez(path,candidate_indices=candidate,AP=ap)
                dump(receipt_path,{"binding":data["T_binding"],"sha256":sha(path),"candidates_per_concept":928,
                    "selection_on_T":"descriptive oracle only,never replaces D fixed winners"})
            del scores;print(json.dumps({"phase":"oracle_T","network":network,"basis":basis,"status":"complete"}),flush=True)
        del native
    with budget.track("statistics","Toracle_estimands_and_oracle_minus_fixed"):
        with np.errstate(divide="ignore",invalid="ignore"):ell=np.log(aps/pi)
        result=estimands(ell)
        with np.load(OUT/"T_FIXED_REPLICATES.npz",allow_pickle=False) as z:
            fixed_ell=z["point_ell"];fixed_ap=z["point_AP"]
        fixed=estimands(fixed_ell)
        # Numerical tolerance is only the locked float64 AP winner tolerance.
        violation=np.isfinite(aps)&np.isfinite(fixed_ap)&(aps+8*np.finfo(np.float64).eps<fixed_ap)
        if violation.any():raise AssertionError("oracle AP below included fixed candidate beyond floating-point tolerance")
        np.savez(OUT/"T_ORACLE.npz",candidate_indices=candidates,AP=aps,ell=ell,
                 oracle_minus_fixed_ell=ell-fixed_ell,AP_fixed=fixed_ap,
                 oracle_minus_fixed_AP=aps-fixed_ap,concept_ids=data["concept_ids"])
        summary={"status":"DESCRIPTIVE_SYMMETRIC_T_ORACLE_COMPLETE","binding":data["T_binding"],
            "networks":list(NETWORKS),"bases":list(BASES),"candidates_per_concept":928,"uncertainty":"point estimates only",
            "Theta":float(result["Theta"]),"G_init":float(result["G_init"]),
            "oracle_minus_fixed_Theta":float(result["Theta"]-fixed["Theta"]),
            "oracle_minus_fixed_G_init":float(result["G_init"]-fixed["G_init"]),
            "oracle_minus_fixed_Theta_nonnegative_not_required":True,"fixed_winners_replaced":False}
        dump(OUT/"T_ORACLE_SUMMARY.json",safe_json(summary))
    return safe_json(summary)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("ready","extract-T","fixed-T","oracle-T","final-T"))
    parser.add_argument("--lock",type=Path,default=DEFAULT_LOCK);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);set_num_threads(4)
    with threadpool_limits(limits=4):
        if args.stage=="ready":result=write_ready()
        else:
            binding,T_binding=require_T_lock(args.lock);data=load_T(binding,T_binding);budget=Budget(binding)
            budget.record["T_scientific_reads"] += 1
            dump(OUT/"RUNTIME.json",{"started_utc":utc(),"threads":4,"BLAS":threadpool_info(),"memory":memory_usage(),"config":T_CONFIG})
            try:
                if args.stage=="extract-T":extract_T(data,budget);result={"status":"PASS_T_EXTRACTION","root":str(ACT)}
                elif args.stage=="fixed-T":result=fixed_T(data,budget)
                elif args.stage=="oracle-T":result=oracle_T(data,budget)
                else:
                    result={"status":"T_PRIMARY_AND_ORACLE_HANDOFF","binding":T_binding,
                        "fixed_summary":read_json(OUT/"T_FIXED_SUMMARY.json"),"oracle_summary":read_json(OUT/"T_ORACLE_SUMMARY.json"),
                        "activation_root":str(ACT),"bootstrap_weights":str(OUT/"T_BOOTSTRAP_WEIGHTS.npy"),
                        "CPU_ledger":budget.record,"finite_input_and_mechanism":"root-owned separate outputs"}
                    dump(OUT/"T_RUN_HANDOFF.json",result)
            except BaseException as error:
                dump(OUT/"T_FAILURE.json",{"stage":args.stage,"binding":T_binding,"time_utc":utc(),
                    "error_type":type(error).__name__,"message":str(error),
                    "resume_rule":"same frozen objects only; retain failed evidence, no C/winner/control/sample changes"})
                raise
        print(json.dumps(safe_json(result),indent=2),flush=True)


if __name__=="__main__":main()
