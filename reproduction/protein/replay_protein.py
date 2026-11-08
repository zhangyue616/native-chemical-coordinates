"""Offline replay of the completed protein confirmation, never a new confirmation.

Only explicit input/output roots are used for scientific files. Frozen project
modules supply the graph, P5 selector and saved KRR prediction functions. No
network, PLM forward, fitting, bootstrap, p value or CI calculation is allowed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

META = "data/revision_20261017/protein_metadata/"
FRESH = "data/revision_20261017/protein_fresh/"
DEV = "results/revision_20261017/protein_D/"
TEST = "results/revision_20261017/protein_T/"
TOLERANCE = 1e-10


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def normalize(relative):
    path = Path(str(relative).replace("\\", "/"))
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ValueError("Input manifest needs a project-relative path, not a machine absolute path")
    return path.as_posix()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def system_metadata():
    value = {"sys_platform": sys.platform, "os_name": os.name}
    if hasattr(sys, "getwindowsversion"):
        version = sys.getwindowsversion()
        value["windows_version"] = {key: getattr(version,key) for key in ("major","minor","build","platform","service_pack")}
    return value


def manifest(root):
    """Small explicit protein-only dependency set, no PubChem/model/old activations."""
    names = {META + name for name in ("OLD_POOL_GRAPH.json", "OLD_SELECTED_UNION_GROUPS.json", "http_provenance.json", "metadata_projection.jsonl")}
    names.update(FRESH + name for name in ("http_provenance.json", "OLD_EXACT_SEQUENCE_INDEX.json", "OLD_SELECTED_FINAL_GROUPS.json",
        "components.jsonl", "raw_group_join.jsonl", "FRESH_DATA_RESULT.json", "FRESH_DATA_ACCEPTANCE.json", "SEALED_FRESH_CASES.json", "proteins.jsonl"))
    names.update(DEV + name for name in ("FINAL_MODELS.npz", "D_DEVELOPMENT.json", "PRECISION_PLAN.json"))
    names.update(TEST + name for name in ("T_NATIVE.npz", "T_PREDICTIONS.npz", "T_ARTIFACT_CHECK.json", "T_RUNNER_READY.json"))
    old = read_json(root / (META + "OLD_POOL_GRAPH.json"))
    names.update(normalize(item["path"]) for item in old["sources"] if item["path"] == "data/proteins.jsonl" or "/raw/" in item["path"])
    names.update(f"data/{pilot}/proteins.jsonl" for pilot in ("pilot3a", "pilot4a", "pilot5a"))
    for prefix in (META, FRESH):
        receipt = read_json(root / (prefix + "http_provenance.json"))
        names.update(normalize(page["raw_path"]) for page in receipt["pages"])
    return sorted(names)


class Replay:
    def __init__(self, args):
        self.args = args
        self.project, self.inputs, self.output = (p.resolve() for p in (args.project_root, args.input_root, args.output_dir))
        self.cpu_start, self.wall_start = time.process_time(), time.perf_counter()
        self.child_cpu = 0.0
        self.prior_cpu = 0.0
        self.phase = "initialization"
        self.done = threading.Event()
        self.reads = Counter()
        self.receipt = {"status": "RUNNING_OFFLINE_COMPLETED_DATA_REPLAY", "started_utc": utc(),
            "cwd": str(Path.cwd().resolve()), "project_root_readonly_code": str(self.project),
            "input_root": str(self.inputs), "output_dir": str(self.output),
            "different_cwd_from_project": Path.cwd().resolve() != self.project,
            "different_input_root_from_project": self.inputs != self.project,
            "scientific_threads": 1, "maximum_cumulative_process_CPU_seconds": 600,
            "network_attempts": 0, "network_connections": 0, "PLM_loads": 0, "PLM_forwards": 0,
            "fits": 0, "bootstrap_calls": 0, "p_or_CI_calls": 0,
            "original_scientific_reads_during_replay": [], "phases": [], "checks": {},
            "interpretation": "Replay of completed exposed data; not a new result-blind confirmation"}
        if args.prior_engineering_receipt:
            prior = read_json(args.prior_engineering_receipt)
            if (prior["status"] != "OFFLINE_REPLAY_STOP_NO_AUTOMATIC_RETRY" or prior.get("checks")
                    or prior.get("cohort_replay") or prior.get("readout_replay")):
                raise ValueError("Only a pre-replay engineering stop can be charged as a prior attempt")
            self.prior_cpu = float(prior["cumulative_process_CPU_seconds"])
            self.receipt["prior_engineering_stop"] = {"path": str(args.prior_engineering_receipt.resolve()),
                "CPU_seconds": self.prior_cpu, "error_type": prior["error_type"], "error": prior["error"]}

    def audit(self, event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
            self.receipt["network_attempts"] += 1
            raise RuntimeError("Network is disabled in this replay")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        if writing:
            allowed = path.is_relative_to(self.output) or (self.phase == "stage_inputs" and path.is_relative_to(self.inputs))
            if not allowed:
                raise RuntimeError("Replay attempted to write outside explicit new output/staging roots")
        elif self.phase not in ("initialization", "stage_inputs"):
            if path.is_relative_to(self.inputs):
                self.reads[path.relative_to(self.inputs).as_posix()] += 1
            if self.inputs != self.project and any(path.is_relative_to(self.project / folder) for folder in ("data", "results")):
                self.receipt["original_scientific_reads_during_replay"].append(str(path))
                raise RuntimeError("Implicit read from the original project's scientific directory")

    def save(self):
        self.receipt.update(finished_utc=utc(), process_CPU_seconds=time.process_time()-self.cpu_start,
            child_metadata_CPU_seconds=self.child_cpu, prior_engineering_CPU_seconds=self.prior_cpu, cumulative_process_CPU_seconds=self.elapsed(),
            wall_seconds=time.perf_counter()-self.wall_start)
        (self.output / "REPLAY_RECEIPT.json").write_bytes(json_bytes(self.receipt))

    def elapsed(self):
        return time.process_time()-self.cpu_start+self.child_cpu+self.prior_cpu

    def check(self, name, value):
        self.receipt["checks"][name] = bool(value)
        if not value:
            raise RuntimeError("Replay mismatch: " + name)

    def phase_start(self, name):
        self.phase = name
        self.phase_cpu = time.process_time()

    def phase_end(self):
        elapsed = time.process_time()-self.phase_cpu
        self.receipt["phases"].append({"name": self.phase, "CPU_seconds": elapsed})
        print(json.dumps({"phase": self.phase, "CPU_seconds": elapsed, "cumulative_CPU_seconds": self.elapsed()}), flush=True)
        if self.elapsed() >= 600:
            raise RuntimeError("600 CPU-second replay cap reached")

    def start_guard(self):
        sys.addaudithook(self.audit)
        def watch():
            while not self.done.wait(.1):
                if self.elapsed() >= 600:
                    self.receipt["status"] = "CPU_CAP_STOP_NO_AUTOMATIC_RETRY"
                    try:
                        self.save()
                    finally:
                        os._exit(124)
        self.watchdog = threading.Thread(target=watch, name="offline-replay-CPU-cap", daemon=True)
        self.watchdog.start()

    def load(self, relative, expected_sha=None, expected_bytes=None):
        data = (self.inputs / normalize(relative)).read_bytes()
        value = digest(data)
        if expected_sha and value != expected_sha:
            raise RuntimeError("Required input snapshot SHA mismatch: " + relative)
        if expected_bytes is not None and len(data) != expected_bytes:
            raise RuntimeError("Required input snapshot byte-count mismatch: " + relative)
        return data

    def stage(self):
        self.phase_start("stage_inputs")
        source = self.args.stage_inputs_from.resolve() if self.args.stage_inputs_from else self.inputs
        files = manifest(source)
        missing = [name for name in files if not (source / name).is_file()]
        if missing:
            self.receipt["missing_inputs"] = missing
            raise FileNotFoundError("Required saved protein inputs are missing; no network fallback")
        if self.args.stage_inputs_from:
            if source == self.inputs:
                raise ValueError("Staging source and independent input root must differ")
            for name in files:
                destination = self.inputs / name
                if destination.exists():
                    raise FileExistsError("Refuse to overwrite a staged input")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, destination)
        self.receipt["input_manifest"] = [{"relative_path": name, "bytes": (self.inputs / name).stat().st_size} for name in files]
        self.receipt["input_manifest_total_bytes"] = sum(r["bytes"] for r in self.receipt["input_manifest"])
        self.receipt["staged_from"] = str(source) if self.args.stage_inputs_from else None
        self.phase_end()

    def modules(self):
        self.phase_start("frozen_functions_and_one_thread")
        from threadpoolctl import threadpool_limits, threadpool_info
        self.pool = threadpool_limits(limits=1)
        def module(name, relative):
            path = self.project / relative
            spec = importlib.util.spec_from_file_location(name, path)
            value = importlib.util.module_from_spec(spec)
            sys.modules[name] = value
            spec.loader.exec_module(value)
            return value
        self.metadata = module("_offline_protein_metadata", "scripts/revision_20261017/protein_metadata_gate.py")
        self.fresh = module("_offline_protein_fresh", "scripts/revision_20261017/protein_fresh_data.py")
        self.krr = module("_offline_protein_krr", "scripts/revision_20261017/protein_krr.py")
        self.np = self.krr.np
        self.pool = threadpool_limits(limits=1)
        self.receipt["BLAS_threadpools"] = [{k: item.get(k) for k in ("internal_api", "num_threads", "version")} for item in threadpool_info()]
        self.check("all_discovered_BLAS_threadpools_one_thread", all(p["num_threads"] == 1 for p in self.receipt["BLAS_threadpools"]))
        self.fresh.ROOT = self.inputs
        self.fresh.META = self.inputs / META
        self.fresh.OUT = self.output / "in_memory_writer_sink"
        self.fresh.PRECISION = self.inputs / (DEV + "PRECISION_PLAN.json")
        def forbidden(*_, **__):
            raise RuntimeError("A network/model/fit/inference top-level entry is forbidden in replay")
        for mod, names in ((self.metadata, ("fetch", "fetch_page", "prepare")),
                           (self.fresh, ("run", "fetch_page", "prepare", "old_sequence_index")),
                           (self.krr, ("fit_krr", "future_primary_summary", "precision_plan", "develop"))):
            for name in names:
                setattr(mod, name, forbidden)
        ready = read_json(self.inputs / (TEST + "T_RUNNER_READY.json"))
        dependencies = {"frozen_KRR": "scripts/revision_20261017/protein_krr.py"}
        self.receipt["frozen_code"] = {}
        for name in ("scripts/revision_20261017/protein_metadata_gate.py", "scripts/revision_20261017/protein_fresh_data.py", "scripts/prepare_pilot5a_data.py", "scripts/revision_20261017/protein_krr.py"):
            self.receipt["frozen_code"][name] = digest((self.project / name).read_bytes())
        for key, name in dependencies.items():
            self.check("frozen_D_predictor_code_identity", self.receipt["frozen_code"][name] == ready["bindings"][key]["sha256"])
        self.check("frozen_long_selector_salt_and_target", self.fresh.SALT == "protein-long-transfer-20261017-v1" and self.fresh.TARGET == 1304 and self.fresh.P5.SEED == self.fresh.SALT)
        self.phase_end()

    def rebuild_old(self):
        self.phase_start("all_16896_old_raw_graph")
        accepted = read_json(self.inputs / (META + "OLD_POOL_GRAPH.json"))
        nodes, origins = [], Counter()
        for item in accepted["sources"]:
            name = item["path"]
            if name != "data/proteins.jsonl" and "/raw/" not in name:
                continue
            data = self.load(name, item["existing_declared_sha256"], item["current_bytes"])
            legacy = name == "data/proteins.jsonl"
            rows = [json.loads(s) for s in data.decode("utf-8-sig").splitlines() if s.strip()] if legacy else json.loads(data)["results"]
            if len(rows) != item["records"]:
                raise RuntimeError("Old raw count changed")
            nodes.extend(self.metadata.old_node(row, legacy=legacy) for row in rows)
            origins["pilot1" if legacy else normalize(name).split("/")[1]] += len(rows)
        old = self.metadata.compress_old(nodes)
        self.check("old_16896_raw_all_origins", dict(origins) == {"pilot1": 512, "pilot2": 4096, "pilot3a": 4096, "pilot4a": 4096, "pilot5a": 4096})
        self.check("old_graph_rebuilt_from_raw_exact", all(old[key] == accepted[key] for key in old))
        by_id = {accession: i for i,c in enumerate(old["components"]) for accession in c["accessions"]}
        seq_index = {}
        for node in nodes:
            key, group = digest(node["sequence"].encode("utf-8")), by_id[node["id"]]
            if key in seq_index and seq_index[key] != group:
                raise RuntimeError("Exact-sequence key crosses old supernodes")
            seq_index[key] = group
        expected_index = read_json(self.inputs / (FRESH + "OLD_EXACT_SEQUENCE_INDEX.json"))["sequence_sha256_to_old_component"]
        self.check("old_exact_sequence_index_rebuilt_exact", seq_index == expected_index)
        selected = []
        for pilot, count in (("pilot3a",176),("pilot4a",128),("pilot5a",128)):
            rows = read_rows(self.inputs / f"data/{pilot}/proteins.jsonl")
            self.check(pilot + "_D_pair_count", len(rows) == len({r["id"] for r in rows}) == count)
            for row in rows:
                index = by_id[row["id"]]
                selected.append({"origin": pilot, "id": row["id"], "prior_component_key": row["component_key"],
                    "old_component_index": index, "old_union_group": f"old-union-{index+1:06d}"})
        self.check("old_432_D_pair_join_rebuilt_exact", selected == accepted["selected_pair_groups"])
        old["selected_pair_groups"] = selected
        self.receipt["old_raw_count"] = len(nodes)
        self.receipt["old_supernodes"] = len(old["components"])
        self.receipt["old_exact_sequence_keys"] = len(seq_index)
        self.phase_end()
        return old, seq_index

    def saved_source(self, prefix, expected_ids):
        receipt = read_json(self.inputs / (prefix + "http_provenance.json"))
        if not receipt["complete"] or len(receipt["pages"]) != 28:
            raise RuntimeError("Saved source is incomplete, not a valid insufficiency result")
        rows = []
        for number, page in enumerate(receipt["pages"], 1):
            payload = self.load(normalize(page["raw_path"]), page.get("sha256") or page["raw_sha256"], page["bytes"])
            records = json.loads(payload)["results"]
            self.fresh.check_page(records, page["headers"], expected_ids, len(rows))
            if page["page"] != number:
                raise RuntimeError("Saved page order changed")
            rows.extend(records)
        self.check("source_complete_" + ("metadata" if prefix == META else "full"), len(rows) == 13658)
        return rows, receipt

    def rebuild_new(self, old, sequence_index):
        self.phase_start("metadata_and_full_source_offline_graph")
        expected_projection = read_rows(self.inputs / (META + "metadata_projection.jsonl"))
        expected_ids = [r["id"] for r in expected_projection]
        metadata, receipt = self.saved_source(META, expected_ids)
        projection, unexpected, offset = [], Counter(), 0
        for page in receipt["pages"]:
            for entry in metadata[offset:offset+page["records"]]:
                projection.append(self.metadata.project_new(entry, page["page"], len(projection)+1, unexpected))
            offset += page["records"]
        self.check("metadata_projection_rebuilt_exact", projection == expected_projection)
        del metadata
        entries, _ = self.saved_source(FRESH, expected_ids)
        self.receipt["full_raw_field_presence"] = {"missing_sequence_value": sum(not isinstance(e.get("sequence"),dict) or not e["sequence"].get("value") for e in entries),
            "no_feature_field": sum("features" not in e for e in entries), "no_Pfam": sum(not self.fresh.project(e)["pfam"] for e in entries)}
        nodes = [self.fresh.project(e) for e in entries]
        components, join, mapping = self.fresh.full_union(old, sequence_index, nodes)
        baseline = read_json(self.inputs / (META + "OLD_SELECTED_UNION_GROUPS.json"))["selected_pair_group_mapping"]
        final = read_json(self.inputs / (FRESH + "OLD_SELECTED_FINAL_GROUPS.json"))["selected_pair_group_mapping"]
        self.check("D_group_identity_421_unchanged", self.fresh.groups_unchanged(baseline, mapping) and mapping == final and len({r["current_known_union_group"] for r in mapping}) == 421)
        self.check("full_new_components_rebuilt_exact", components == read_rows(self.inputs / (FRESH + "components.jsonl")))
        actual_join = [{"raw_index":i,"id":n["id"],"known_group_id":join[i]["known_group_id"],
            "connected_to_old_pool":join[i]["connected_to_old_pool"],"component_key":join[i]["component_key"]} for i,n in enumerate(nodes)]
        self.check("all_13658_raw_group_joins_exact", actual_join == read_rows(self.inputs / (FRESH + "raw_group_join.jsonl")))
        self.phase_end()
        self.phase_start("frozen_P5_eligibility_and_pair_selection")
        captured, serialized = {}, {}
        def capture_jsonl(path, rows):
            name = Path(path).name
            if name == "proteins.jsonl":
                captured[name] = rows
                serialized[name] = "".join(json.dumps(r,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n" for r in rows).encode("utf-8")
        def capture_json(path, value, exclusive=False):
            captured[Path(path).name] = value
            serialized[Path(path).name] = json_bytes(value)
        def captured_sha(path):
            return digest(serialized[Path(path).name])
        self.fresh.write_jsonl, self.fresh.write_json, self.fresh.file_sha = capture_jsonl, capture_json, captured_sha
        result = self.fresh.qualify(entries, nodes, components, join)
        accepted = read_json(self.inputs / (FRESH + "FRESH_DATA_RESULT.json"))
        fields = ("stage_survivors","record_exclusion_counts_nonexclusive","feature_exclusion_counts_nonexclusive",
                  "site_exclusion_counts_nonexclusive","evidence_exclusion_counts_nonexclusive","candidate_legal_pair_combinations",
                  "eligible_final_fresh_components","selected_components","unused_eligible_components","raw_sequence_lengths","selected_sequence_lengths")
        self.check("complete_eligibility_and_exclusion_funnel_exact", all(result[k] == accepted[k] for k in fields))
        rebuilt, saved = captured["SEALED_FRESH_CASES.json"], read_json(self.inputs / (FRESH + "SEALED_FRESH_CASES.json"))
        self.check("sealed_all_case_identity_position_sequence_label_group_exact", rebuilt["cases"] == saved["cases"])
        self.check("sealed_all_1304_canonical_pairs_exact", rebuilt["pairs"] == saved["pairs"])
        self.check("selected_1304_protein_records_exact", captured["proteins.jsonl"] == read_rows(self.inputs / (FRESH + "proteins.jsonl")))
        self.check("selected_protein_serialization_identity_exact", digest(serialized["proteins.jsonl"]) == saved["selected_proteins_sha256"])
        self.receipt["cohort_replay"] = {"raw_records":13658,"old_connected_excluded":5949,
            "eligible_components":result["eligible_final_fresh_components"],"selected_pairs":len(rebuilt["pairs"]),
            "cases":len(rebuilt["cases"]),"unused_eligible_components":result["unused_eligible_components"],
            "funnel":result["stage_survivors"],"scientific_case_pair_sha256":digest(json_bytes({k:rebuilt[k] for k in ("cases","pairs")})),
            "generated_timestamp_excluded_from_comparison":True,"frozen_writer_adapter":"in-memory only; no original output writer called"}
        self.phase_end()
        return rebuilt

    def readout(self, sealed):
        self.phase_start("saved_native_KRR_readout_replay")
        np, krr = self.np, self.krr
        acceptance = read_json(self.inputs / (TEST + "T_ARTIFACT_CHECK.json"))
        if acceptance["status"] != "PROTEIN_LONG_T_ARTIFACT_PASS":
            raise RuntimeError("Expected the completed accepted protein result")
        with np.load(self.inputs / (TEST + "T_NATIVE.npz"), allow_pickle=False) as saved:
            z, ids = saved["z"], saved["case_ids"]
        self.check("native_cache_case_order_exact", np.array_equal(ids,np.asarray([c["case_id"] for c in sealed["cases"]])) and z.shape==(2608,1280))
        j = z[:,1246].astype(np.float64)
        data = krr.pair_data(sealed["cases"],sealed["pairs"],j)
        y = np.asarray([p["y_a"] for p in sealed["pairs"]],dtype=np.float64)
        groups = np.asarray([p["group"] for p in sealed["pairs"]])
        losses, comparisons = {}, {}
        with np.load(self.inputs / (TEST + "T_PREDICTIONS.npz"),allow_pickle=False) as expected:
            self.check("saved_prediction_identity_targets_groups_j", np.array_equal(expected["case_ids"],ids) and np.array_equal(expected["y_a"],y) and np.array_equal(expected["groups"],groups) and np.array_equal(expected["j1246"],j))
            for model in krr.MODELS:
                score, probability = krr.predict_saved(data,model,self.inputs/(DEV+"FINAL_MODELS.npz"))
                losses[model] = (probability-y)**2
                comparisons[model] = {}
                for kind, value in (("score",score),("p_a",probability),("loss",losses[model])):
                    reference = expected[f"{model}_{kind}"]
                    error = float(np.max(np.abs(value-reference)))
                    comparisons[model][kind+"_max_absolute_error"] = error
                    self.check(model+"_"+kind+"_within_tolerance", bool(np.isfinite(value).all() and np.allclose(value,reference,atol=TOLERANCE,rtol=TOLERANCE)))
            contribution = krr.paired_contributions(losses)
            self.check("deterministic_four_contributions_within_tolerance", np.allclose(contribution,expected["pair_contributions"],atol=TOLERANCE,rtol=TOLERANCE))
        self.receipt["readout_replay"] = {"models":list(krr.MODELS),"absolute_relative_tolerance":TOLERANCE,
            "comparisons":comparisons,"primary_summary_recomputed":False,"p_CI_bootstrap_helpers_called":False,
            "D_source":"FINAL_MODELS.npz saved alpha/trace/train_dx/pa/pb/dj; no D refit or old PLM replay"}
        self.phase_end()

    def chemical_metadata(self):
        self.phase_start("chemical_environment_pip_metadata_only")
        code = "import importlib.metadata as m,json,sys,time;d={p.metadata['Name']:p.version for p in m.distributions()};print(json.dumps({'python':sys.version,'platform':sys.platform,'executable':sys.executable,'packages':dict(sorted(d.items(),key=lambda x:x[0].lower())),'metadata_CPU_seconds':time.process_time()}))"
        child = subprocess.run([str(self.args.chemistry_python.resolve()),"-B","-c",code],capture_output=True,text=True,check=True,timeout=60,env=os.environ.copy())
        result = json.loads(child.stdout)
        self.child_cpu = float(result.pop("metadata_CPU_seconds"))
        result.update(installed_distribution_metadata_only=True,installations=0,models_or_scientific_inputs_read=0)
        self.receipt["chemical_environment"] = result
        self.phase_end()

    def run(self):
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError("Use a new empty independent replay output directory")
        if self.inputs == self.output or self.output.is_relative_to(self.inputs / "data") or self.output.is_relative_to(self.inputs / "results"):
            raise ValueError("Replay output must be independent of scientific inputs")
        self.output.mkdir(parents=True,exist_ok=True)
        temporary = self.output / "runtime_tmp"
        temporary.mkdir()
        tempfile.tempdir = str(temporary)
        for variable in ("TMPDIR", "TEMP", "TMP"):
            os.environ[variable] = str(temporary)
        self.start_guard()
        try:
            self.stage()
            self.modules()
            old, index = self.rebuild_old()
            sealed = self.rebuild_new(old,index)
            self.readout(sealed)
            self.chemical_metadata()
            self.check("zero_original_scientific_reads_in_core_replay", not self.receipt["original_scientific_reads_during_replay"])
            self.check("zero_network_attempts", self.receipt["network_attempts"] == self.receipt["network_connections"] == 0)
            self.check("CPU_within_600_seconds", self.elapsed() <= 600)
            self.receipt.update(status="OFFLINE_PROTEIN_REPLAY_PASS",read_input_files=sorted(self.reads),
                runtime={"python":sys.version,"platform":system_metadata(),"executable":sys.executable,"numpy":self.np.__version__},
                remaining_limits=["Same Windows machine; copied protein-only input root and foreign cwd validated, not a clean OS/container installation",
                    "Frozen source modules remain required at explicit project-root","No model or chemical scientific replay; chemical environment metadata only",
                    "Historical exposed-data replay, not new result-blind confirmation or public archive"])
        except BaseException as exc:
            self.receipt.update(status="OFFLINE_REPLAY_STOP_NO_AUTOMATIC_RETRY",error_type=type(exc).__name__,error=str(exc))
            raise
        finally:
            self.done.set()
            self.watchdog.join()
            self.save()
        print(json.dumps({k:self.receipt[k] for k in ("status","cwd","input_root","output_dir","cumulative_process_CPU_seconds","network_attempts","PLM_forwards","fits","p_or_CI_calls")}))


def finalize_metadata(args):
    """Repair only receipt metadata after all saved data/readout checks passed."""
    started = time.process_time()
    source = args.finalize_receipt.resolve()
    payload = source.read_bytes()
    result = json.loads(payload)
    required = {"all_16896_old_raw_graph", "metadata_and_full_source_offline_graph",
        "frozen_P5_eligibility_and_pair_selection", "saved_native_KRR_readout_replay", "chemical_environment_pip_metadata_only"}
    if (result["status"] != "OFFLINE_REPLAY_STOP_NO_AUTOMATIC_RETRY"
            or not required <= {p["name"] for p in result["phases"]}
            or not result["checks"] or not all(result["checks"].values())
            or result["cohort_replay"]["selected_pairs"] != 1304
            or result["cohort_replay"]["cases"] != 2608
            or result["original_scientific_reads_during_replay"]
            or any(result[key] != 0 for key in ("network_attempts","PLM_forwards","fits","p_or_CI_calls"))):
        raise RuntimeError("Metadata-only finalization requires all saved replay stages/checks already complete")
    destination = args.output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Use a new empty final receipt directory; preserve original stop evidence")
    destination.mkdir(parents=True,exist_ok=True)
    result["status"] = "OFFLINE_PROTEIN_REPLAY_PASS_METADATA_FINALIZED"
    result["data_replay_output_dir"] = result["output_dir"]
    result["final_receipt_output_dir"] = str(destination)
    result["metadata_only_repair"] = {"source_stop_receipt":str(source),"source_stop_receipt_sha256":digest(payload),
        "source_process_exit_code":1,"source_error_type":result.pop("error_type"),"source_error":result.pop("error"),
        "earlier_import_stop_receipt":result["prior_engineering_stop"]["path"],
        "full_data_and_readout_executions":1,"data_graph_selector_readout_repeated":False,
        "repaired_runtime":system_metadata(),"finalizer_python":sys.executable,
        "repair":"Replace platform.platform subprocess/NUL probe with sys.getwindowsversion built-in metadata"}
    result["finalized_utc"] = utc()
    cpu = time.process_time()-started
    result["metadata_only_repair"]["CPU_seconds"] = cpu
    result["cumulative_process_CPU_seconds"] += cpu
    result["metadata_only_finalizer_exit_code"] = 0
    if result["cumulative_process_CPU_seconds"] >= 600:
        raise RuntimeError("Cumulative CPU cap exceeded")
    (destination/"REPLAY_RECEIPT.json").write_bytes(json_bytes(result))
    print(json.dumps({"status":result["status"],"checks_passed":len(result["checks"]),
        "cumulative_process_CPU_seconds":result["cumulative_process_CPU_seconds"],"full_data_replay_executions":1,
        "network_attempts":0,"PLM_forwards":0,"fits":0,"p_CI_bootstrap_calls":0,
        "final_receipt":str(destination/"REPLAY_RECEIPT.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,required=True,help="Read-only frozen scripts root")
    parser.add_argument("--input-root",type=Path,required=True,help="Root containing the required data/ and results/ protein snapshots")
    parser.add_argument("--output-dir",type=Path,required=True,help="New empty independent receipt output")
    parser.add_argument("--stage-inputs-from",type=Path,help="Optional source root for one necessary-input-only copy; never overwrites")
    parser.add_argument("--chemistry-python",type=Path,required=True,help="Existing chemical environment interpreter, metadata read only")
    parser.add_argument("--prior-engineering-receipt",type=Path,help="Charge a preserved pre-replay engineering stop to the same CPU cap; not a scientific resume")
    parser.add_argument("--finalize-receipt",type=Path,help="Metadata-only repair of a preserved completed-stage receipt; never reruns data/readouts")
    args = parser.parse_args()
    if args.finalize_receipt:
        finalize_metadata(args)
    else:
        Replay(args).run()


if __name__ == "__main__":
    main()
