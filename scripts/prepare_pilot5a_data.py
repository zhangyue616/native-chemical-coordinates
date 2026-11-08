"""Bounded, source-only Pilot 5-A data preparation (Python standard library).

No candidate request is possible without an explicit protocol path and SHA256.
Raw pages and final data are immutable; the HTTP receipt may only grow while a
bounded download is incomplete. No model library or model artifact is used.
"""

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pilot5a"
RAW = OUT / "raw"
SEED = 2026101605
MAX_RECORDS = 4096
MAX_PAGES = 9
MAX_ATTEMPTS = 3
EXPECTED_RELEASE = ("2026_03", "02-September-2026")
EXPECTED_OLD_LAST_ACCESSION = "P92533"
EXPECTED_START_CURSOR = "nnn1hsuzifwzrv8k2rhir4e7i2"
PREVIOUS_SAME_QUERY_RAW_RECORDS = 8192
PRIOR_SOURCE_TOTAL_RESULTS = 13942
REQUIRED_COMPONENTS = 128
SPLIT_SIZES = (("T_confirm", 128),)
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
QUERY = "(reviewed:true) AND (length:[64 TO 256]) AND (database:pdb)"
FIELDS = "accession,sequence,length,xref_pfam,xref_pdb,ft_helix,ft_strand"
SORT = "accession asc"
SOURCE = "https://rest.uniprot.org/uniprotkb/search"
METHODS = frozenset(("X-ray", "NMR", "EM", "Neutron", "Electron diffraction"))
FEATURE_TYPES = {"Helix": "HELIX", "Beta strand": "STRAND"}
SAFE_HEADERS = {
    "content-type", "content-length", "date", "etag", "last-modified",
    "x-uniprot-release", "x-uniprot-release-date", "x-api-deployment-date",
    "x-total-results", "link", "cache-control",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hash_text(value):
    return sha(value.encode("ascii"))


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def jsonl(rows):
    return b"".join((json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for x in rows)


def save_frozen(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Frozen output differs; refusing overwrite: {path.relative_to(ROOT)}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_receipt(kind, value):
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    path = OUT / "run_receipts" / (name + "_" + kind + ".json")
    save_frozen(path, canonical({"recorded_utc": utc_now(), "model_forward_count": 0, **value}))
    return path.relative_to(ROOT).as_posix()


def config():
    return {"query": QUERY, "fields": FIELDS, "sort": SORT,
            "maximum_raw_response_records": MAX_RECORDS, "maximum_successful_pages": MAX_PAGES,
            "maximum_attempts_per_page": MAX_ATTEMPTS, "expected_release": list(EXPECTED_RELEASE),
            "seed": SEED, "accepted_method_strings": sorted(METHODS),
            "required_components": REQUIRED_COMPONENTS, "split_sizes": dict(SPLIT_SIZES)}


class DataFailure(RuntimeError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status


def file_binding(path, records=None):
    path = path.resolve()
    row = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}
    if records is not None:
        row["records"] = records
    return row


def continuation_binding():
    receipt_path = ROOT / "data/pilot4a/http_provenance.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (not receipt.get("complete") or len(receipt["pages"]) != 9
            or receipt.get("total_raw_records") != 4096 or receipt["pages"][-1]["records"] != 96):
        raise RuntimeError("Old Pilot 4-A continuation receipt differs from the frozen 9-page/4096 pool")
    last = receipt["pages"][-1]
    content = (ROOT / last["path"]).read_bytes()
    if sha(content) != last["sha256"]:
        raise RuntimeError("Old Pilot 4-A final raw page SHA mismatch")
    rows = json.loads(content)["results"]
    last_accession = rows[-1]["primaryAccession"]
    official_next = next_url(last)
    if (len(rows) != 96 or last_accession != EXPECTED_OLD_LAST_ACCESSION or official_next is None
            or url_params(official_next).get("cursor") != EXPECTED_START_CURSOR):
        raise RuntimeError("Old Pilot 4-A last accession or official continuation cursor changed")
    if any((p["headers"].get("x-uniprot-release"), p["headers"].get("x-uniprot-release-date")) != EXPECTED_RELEASE
           for p in receipt["pages"]):
        raise RuntimeError("Old Pilot 4-A release is not the frozen 2026_03 release")
    return {"old_receipt_path": receipt_path.relative_to(ROOT).as_posix(),
            "old_receipt_sha256": sha(receipt_path.read_bytes()), "old_raw_records": 4096,
            "old_final_page_path": last["path"], "old_final_page_sha256": last["sha256"],
            "old_final_page_records": len(rows), "old_last_accession": last_accession,
            "official_next_url": official_next, "start_request_url": page_size(official_next, 500),
            "cursor": EXPECTED_START_CURSOR, "expected_release": list(EXPECTED_RELEASE),
            "previous_same_query_raw_records": PREVIOUS_SAME_QUERY_RAW_RECORDS,
            "prior_reported_source_total_results": PRIOR_SOURCE_TOTAL_RESULTS}


def rule_inputs(path):
    path = path.resolve()
    _, old_sources = old_pool()
    script = Path(__file__).resolve()
    return {"schema_version": 1, "pilot": "pilot5a", "protocol_path": path.as_posix(),
            "protocol_sha256": sha(path.read_bytes()), "script_path": script.as_posix(),
            "script_sha256": sha(script.read_bytes()), "configuration": config(),
            "dependencies": [file_binding(ROOT / "scripts/prepare_pilot4a_data.py"),
                             file_binding(ROOT / "docs/PILOT4A_DATA_PROTOCOL.md"),
                             file_binding(ROOT / "scripts/prepare_pilot3a_data.py"),
                             file_binding(ROOT / "docs/PILOT3A_DATA_PROTOCOL.md")],
            "old_pool_sources": old_sources, "continuation": continuation_binding()}


def lock_protocol(path, expected_sha):
    path = path.resolve()
    content = path.read_bytes()
    if sha(content) != expected_sha.lower():
        raise RuntimeError("DATA protocol SHA256 mismatch; no request sent")
    rule_path = OUT / "DATA_RULE_LOCK.json"
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    expected = rule_inputs(path)
    for key, value in expected.items():
        if rule.get(key) != value:
            raise RuntimeError(f"Root DATA_RULE_LOCK binding mismatch: {key}; no request sent")
    if not rule.get("frozen_utc"):
        raise RuntimeError("Root DATA_RULE_LOCK lacks freeze time; no request sent")
    frozen_time = datetime.fromisoformat(rule["frozen_utc"].replace("Z", "+00:00"))
    if frozen_time.tzinfo is None or frozen_time > datetime.now(timezone.utc):
        raise RuntimeError("Root freeze timestamp is timezone-naive or in the future; no request sent")
    lock = {**expected,
            "data_rule_lock_path": rule_path.relative_to(ROOT).as_posix(), "data_rule_lock_sha256": sha(rule_path.read_bytes()),
            "frozen_utc": rule["frozen_utc"]}
    save_frozen(OUT / "protocol_lock.json", canonical(lock))
    return lock


def url_params(url):
    return dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))


def public_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "rest.uniprot.org" or parsed.fragment
            or parsed.username or parsed.password or parsed.path != "/uniprotkb/search"):
        raise ValueError("Refusing non-UniProt public search endpoint")
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    params = dict(pairs)
    allowed = {"query", "fields", "sort", "format", "size", "cursor"}
    if set(params) != allowed or len(pairs) != len(params) or not params["cursor"]:
        raise ValueError("Source link changed fields or lacks its official continuation cursor")
    if any(params.get(k) != v for k, v in {"query": QUERY, "fields": FIELDS, "sort": SORT, "format": "json"}.items()):
        raise ValueError("Source link changed the frozen scientific query/fields/order")
    if not params["size"].isdigit() or not 1 <= int(params["size"]) <= 500:
        raise ValueError("Invalid source page size")
    return url


def page_size(url, size):
    if type(size) is not int or not 1 <= size <= 500:
        raise ValueError("Invalid bounded page size")
    parsed = urlsplit(public_url(url))
    params = dict(parse_qsl(parsed.query))
    params["size"] = str(size)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(params), ""))


def next_url(item):
    header = item["headers"].get("link", "")
    matches = re.findall(r'<([^>]+)>;\s*rel="next"', header)
    if len(matches) > 1 or (not matches and re.search(r'\bnext\b', header)):
        raise ValueError("Malformed or multiple official next links")
    return public_url(matches[0]) if matches else None


class PublicSearchRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        if url_params(newurl) != url_params(req.full_url):
            raise DataFailure("DATA_SOURCE_BLOCKED", "Redirect changed the frozen public request")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def source_page(body, item, request_url, total, previous_accession):
    """Validate a saved response; all successful raw records count before filtering."""
    if item.get("http_status") != 200:
        raise DataFailure("DATA_SOURCE_BLOCKED", "Unexpected successful source HTTP status")
    if url_params(public_url(item["response_url"])) != url_params(request_url):
        raise DataFailure("DATA_SOURCE_BLOCKED", "Response URL changed the frozen request")
    release = (item["headers"].get("x-uniprot-release"), item["headers"].get("x-uniprot-release-date"))
    if release != EXPECTED_RELEASE:
        raise DataFailure("DATA_SOURCE_BLOCKED", "Source release differs from frozen 2026_03")
    if not item["headers"].get("content-type", "").lower().startswith("application/json"):
        raise DataFailure("DATA_SOURCE_BLOCKED", "Source did not return JSON content type")
    try:
        parsed = json.loads(body)
        rows = parsed["results"]
    except (ValueError, KeyError, TypeError):
        raise DataFailure("DATA_INTEGRITY_ERROR", "Saved successful response is not a results JSON object") from None
    if not isinstance(rows, list) or item.get("records") != len(rows):
        raise DataFailure("DATA_INTEGRITY_ERROR", "Raw record count cannot be reconciled with the receipt")
    requested_size = min(500, MAX_RECORDS - total)
    if item.get("requested_size") != requested_size or len(rows) > requested_size:
        raise DataFailure("DATA_SOURCE_BLOCKED", "Server exceeded requested size/raw record cap; all received rows retained, none used")
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("primaryAccession"), str)
                or not row["primaryAccession"] or not isinstance(row.get("sequence"), dict)
                or not isinstance(row["sequence"].get("value"), str)
                or type(row["sequence"].get("length")) is not int):
            raise DataFailure("DATA_INTEGRITY_ERROR", "Successful raw response lacks required accession/sequence fields")
    ids = [r["primaryAccession"] for r in rows]
    if ids and (ids[0] <= previous_accession or any(a >= b for a, b in zip(ids, ids[1:]))):
        raise DataFailure("DATA_SOURCE_BLOCKED", "Accession order is not strictly increasing from the old Pilot 4-A last accession")
    following = next_url(item)
    if not rows and following:
        raise DataFailure("DATA_SOURCE_BLOCKED", "Empty source page still advertises a next cursor")
    if following and url_params(following)["cursor"] == url_params(request_url)["cursor"]:
        raise DataFailure("DATA_SOURCE_BLOCKED", "Pagination cursor did not advance")
    return rows, release, following


def receipt_pages(receipt, lock):
    if receipt["configuration"] != config():
        raise RuntimeError("HTTP receipt differs from the frozen configuration")
    if (receipt.get("protocol_sha256") != lock["protocol_sha256"]
            or receipt.get("data_rule_lock_sha256") != lock["data_rule_lock_sha256"]
            or receipt.get("continuation") != lock["continuation"]):
        raise RuntimeError("HTTP receipt differs from the frozen protocol/continuation lock")
    if len(receipt["pages"]) > MAX_PAGES:
        raise RuntimeError("Saved receipt exceeds the nine successful page cap")
    records, releases = [], set()
    following = lock["continuation"]["start_request_url"]
    previous = lock["continuation"]["old_last_accession"]
    frozen = datetime.fromisoformat(lock["frozen_utc"].replace("Z", "+00:00"))
    for index, page in enumerate(receipt["pages"], 1):
        if not following or len(records) >= MAX_RECORDS:
            raise RuntimeError("Saved pages extend past the bounded source chain")
        expected_url = page_size(following, min(500, MAX_RECORDS - len(records)))
        expected_path = (RAW / f"uniprot_reviewed_pdb_len64_256_asc_page{index:02d}.json").relative_to(ROOT).as_posix()
        if page["request_url"] != expected_url or page["path"] != expected_path:
            raise RuntimeError("Saved pagination chain or raw page path mismatch")
        path = ROOT / page["path"]
        if path.resolve().parent != RAW.resolve():
            raise RuntimeError("Raw page path escaped Pilot 5-A raw directory")
        body = path.read_bytes()
        if sha(body) != page["sha256"] or len(body) != page["bytes"]:
            raise RuntimeError("Saved raw page SHA256 mismatch")
        attempts = page.get("attempts", [])
        if not 1 <= len(attempts) <= MAX_ATTEMPTS or any(
                datetime.fromisoformat(a["request_started_utc"].replace("Z", "+00:00")) <= frozen for a in attempts):
            raise RuntimeError("Request attempts lack a valid post-freeze UTC or exceed the cap")
        rows, release, following = source_page(body, page, expected_url, len(records), previous)
        records.extend(rows)
        if rows:
            previous = rows[-1]["primaryAccession"]
        releases.add(release)
    if "total_raw_records" in receipt and receipt["total_raw_records"] != len(records):
        raise RuntimeError("HTTP receipt total_raw_records differs from all saved raw rows")
    return records, releases, following


def download(lock):
    receipt_path = OUT / "http_provenance.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {
        "configuration": config(), "protocol_sha256": lock["protocol_sha256"],
        "data_rule_lock_sha256": lock["data_rule_lock_sha256"], "continuation": lock["continuation"],
        "pages": [], "complete": False, "model_forward_count": 0,
    }
    if receipt.get("terminal_failure"):
        raise DataFailure(receipt["terminal_failure"]["status"], "Prior terminal source failure; no further request permitted")
    if receipt.get("pending_request"):
        raise RuntimeError("Unresolved prior HTTP attempt; refusing any additional request")
    rows, _, url = receipt_pages(receipt, lock)
    if receipt["complete"]:
        return receipt
    total = len(rows)
    previous = rows[-1]["primaryAccession"] if rows else lock["continuation"]["old_last_accession"]
    opener = build_opener(PublicSearchRedirects())
    while url and total < MAX_RECORDS and len(receipt["pages"]) < MAX_PAGES:
        requested_size = min(500, MAX_RECORDS - total)
        url = page_size(url, requested_size)
        index = len(receipt["pages"]) + 1
        raw_path = RAW / f"uniprot_reviewed_pdb_len64_256_asc_page{index:02d}.json"
        if raw_path.exists():
            raise RuntimeError("Raw page lacks provenance receipt; refusing replacement")
        request = Request(public_url(url), headers={"User-Agent": "Pilot5A-bounded-UniProt-continuation/1.0", "Accept": "application/json", "Accept-Encoding": "identity"})
        attempts = []
        for attempt in range(MAX_ATTEMPTS):
            started = time.perf_counter()
            request_started_utc = utc_now()
            attempt_row = {"attempt": attempt + 1, "request_started_utc": request_started_utc, "outcome": "started"}
            attempts.append(attempt_row)
            receipt["pending_request"] = {"page_index": index, "request_url": url, "requested_size": requested_size, "attempts": attempts}
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_bytes(canonical(receipt))
            try:
                with opener.open(request, timeout=90) as response:
                    body = response.read()
                    item = {"request_url": url, "response_url": response.geturl(),
                        "http_status": response.status, "request_started_utc": request_started_utc, "retrieved_utc": utc_now(),
                        "headers": {k.lower(): v for k, v in response.headers.items() if k.lower() in SAFE_HEADERS},
                        "path": raw_path.relative_to(ROOT).as_posix(), "bytes": len(body), "sha256": sha(body)}
                attempt_row.update({"outcome": "received", "walltime_seconds": round(time.perf_counter() - started, 6), "status": item["http_status"]})
                break
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                attempt_row.update({"outcome": "failed", "walltime_seconds": round(time.perf_counter() - started, 6), "error_type": type(exc).__name__})
                if isinstance(exc, HTTPError):
                    error_body = exc.read()
                    error_path = OUT / "http_errors" / f"page{index:02d}_attempt{attempt + 1}.body"
                    save_frozen(error_path, error_body)
                    attempt_row.update({"status": exc.code, "error_body_path": error_path.relative_to(ROOT).as_posix(), "error_body_sha256": sha(error_body)})
                receipt_path.write_bytes(canonical(receipt))
                retryable = not isinstance(exc, HTTPError) or exc.code >= 500
                if not retryable or attempt + 1 == MAX_ATTEMPTS:
                    reason = f"UniProt HTTP {exc.code}" if isinstance(exc, HTTPError) else "UniProt transport failure after bounded same-URL attempts"
                    receipt["terminal_failure"] = {"status": "DATA_SOURCE_BLOCKED", "reason": reason, "recorded_utc": utc_now()}
                    receipt.pop("pending_request", None)
                    receipt_path.write_bytes(canonical(receipt))
                    raise DataFailure("DATA_SOURCE_BLOCKED", reason) from None
                time.sleep(2 * (attempt + 1))
        # Persist every successful raw response before parsing or validating it.
        save_frozen(raw_path, body)
        try:
            parsed_rows = json.loads(body)["results"]
            observed_records = len(parsed_rows) if isinstance(parsed_rows, list) else None
        except (ValueError, KeyError, TypeError):
            observed_records = None
        item.update({"records": observed_records, "requested_size": requested_size, "attempts": attempts,
            "request_walltime_seconds": round(sum(x["walltime_seconds"] for x in attempts), 6)})
        receipt["pages"].append(item)
        receipt.pop("pending_request", None)
        receipt["total_raw_records"] = sum(x["records"] for x in receipt["pages"]) if all(type(x["records"]) is int for x in receipt["pages"]) else None
        receipt_path.write_bytes(canonical(receipt))
        try:
            received, _, following = source_page(body, item, url, total, previous)
        except Exception as exc:
            receipt["terminal_failure"] = {"status": getattr(exc, "status", "DATA_INTEGRITY_ERROR"), "reason": str(exc), "recorded_utc": utc_now()}
            receipt_path.write_bytes(canonical(receipt))
            raise
        total += len(received)
        if received:
            previous = received[-1]["primaryAccession"]
        url = following
        print(f"Saved page {index}: {item['records']} raw records; total {total}/{MAX_RECORDS}", flush=True)
    receipt_pages(receipt, lock)
    receipt.update({"complete": True, "total_raw_records": total,
                    "stop_reason": "raw_record_cap" if total == MAX_RECORDS else "successful_page_cap" if len(receipt["pages"]) == MAX_PAGES and url else "source_exhausted",
                    "request_walltime_seconds": round(sum(p.get("request_walltime_seconds", 0) for p in receipt["pages"]), 6)})
    receipt_path.write_bytes(canonical(receipt))
    return receipt


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def graph_components(nodes):
    graph = UnionFind(len(nodes))
    first = {"id": {}, "sequence": {}, "pfam": {}}
    for i, row in enumerate(nodes):
        for key, values in (("id", [row["id"]]), ("sequence", [row["sequence"]]), ("pfam", row["pfam"])):
            for value in values:
                if value in first[key]:
                    graph.union(i, first[key][value])
                else:
                    first[key][value] = i
    groups = defaultdict(list)
    for i in range(len(nodes)):
        groups[graph.find(i)].append(i)
    out = []
    for indices in groups.values():
        accessions = sorted({nodes[i]["id"] for i in indices})
        key = "acc:" + ",".join(accessions)
        out.append({"key": key, "hash": hash_text(f"pilot5a-component:{key}:{SEED}"),
                    "indices": indices, "accessions": accessions,
                    "pfam": sorted({p for i in indices for p in nodes[i]["pfam"]}),
                    "has_old": any(nodes[i]["old"] for i in indices)})
    return sorted(out, key=lambda x: x["key"])


def normalize(entry):
    accession = entry["primaryAccession"]
    sequence = entry["sequence"]["value"]
    return {"id": accession, "sequence": sequence,
            "pfam": sorted({x["id"] for x in entry.get("uniProtKBCrossReferences", []) if x["database"] == "Pfam"})}


def old_pool():
    path = ROOT / "data" / "proteins.jsonl"
    legacy = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(legacy) != 512:
        raise RuntimeError("Legacy pool no longer contains the frozen 512 records")
    sources = [file_binding(path, len(legacy))]
    nodes = [{"id": x["id"], "sequence": x["sequence"], "pfam": x["pfam"], "old": True, "origin": "pilot1_512"} for x in legacy]
    for pilot in ("pilot2", "pilot3a", "pilot4a"):
        receipt_path = ROOT / "data" / pilot / "http_provenance.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        sources.append(file_binding(receipt_path))
        raw = []
        for page in receipt["pages"]:
            page_path = ROOT / page["path"]
            if page_path.resolve().parent != (ROOT / "data" / pilot / "raw").resolve():
                raise RuntimeError(f"{pilot} raw page escaped expected directory")
            content = page_path.read_bytes()
            entries = json.loads(content)["results"]
            if sha(content) != page["sha256"] or len(entries) != page["records"]:
                raise RuntimeError(f"{pilot} raw page/receipt mismatch")
            raw.extend(normalize(x) for x in entries)
            sources.append(file_binding(page_path, len(entries)))
        if len(raw) != 4096:
            raise RuntimeError(f"{pilot} old pool must include all frozen 4096 raw entries")
        if pilot in ("pilot3a", "pilot4a") and any(a["id"] >= b["id"] for a, b in zip(raw, raw[1:])):
            raise RuntimeError("Old Pilot 4-A raw accession order drifted")
        nodes += [{**x, "old": True, "origin": pilot + "_all_raw_4096"} for x in raw]
    return nodes, sources


def exact_interval(feature, accession, length):
    location = feature.get("location", {})
    for source in (location, feature):
        for key in ("sequence", "sequenceId", "sequenceIdentifier", "isoformId", "isoform"):
            if key in source and source[key] != accession:
                return None, "noncanonical_or_isoform_location"
    start, end = location.get("start", {}), location.get("end", {})
    if any(x.get("modifier", "EXACT") != "EXACT" for x in (start, end)):
        return None, "uncertain_location"
    if any(type(x.get("value")) is not int for x in (start, end)):
        return None, "noninteger_location"
    s, e = start["value"], end["value"]
    if not 1 <= s <= e <= length:
        return None, "out_of_range_location"
    return (s - 1, e - 1), None


def parse_chains(value, length):
    """Parse explicit UniProt-coordinate coverage; no PDB-atom mapping claim."""
    coverage, invalid = [], []
    for segment in value.split(","):
        segment = segment.strip()
        match = re.fullmatch(r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)\s*=\s*(\d+)\s*-\s*(\d+)", segment)
        if not match:
            invalid.append(segment)
            continue
        start, end = int(match[2]), int(match[3])
        if not 1 <= start <= end <= length:
            invalid.append(segment)
            continue
        for chain in match[1].split("/"):
            if chain == "-":
                invalid.append("unknown_chain_placeholder:- in " + segment)
                continue
            coverage.append({"chain": chain, "start_1based": start, "end_1based": end,
                             "start_q_0based": start - 1, "end_q_0based": end - 1})
    return coverage, invalid


def pdb_crossrefs(entry, length):
    refs = defaultdict(list)
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "PDB":
            continue
        properties = defaultdict(list)
        for prop in xref.get("properties", []):
            properties[prop.get("key")].append(prop.get("value", ""))
        methods = properties.get("Method", [])
        coverage, invalid = [], []
        for value in properties.get("Chains", []):
            parsed, bad = parse_chains(value, length)
            coverage.extend(parsed)
            invalid.extend(bad)
        reasons = []
        if len(methods) != 1 or methods[0] not in METHODS:
            reasons.append("unknown_or_nonexperimental_pdb_method")
        if not coverage:
            reasons.append("no_parseable_pdb_chain_coverage")
        refs[xref["id"].upper()].append({"pdb_id": xref["id"].upper(),
            "method": methods[0] if len(methods) == 1 else methods, "coverage": coverage,
            "unparsed_chain_segments": invalid, "exclusion_reasons": reasons, "raw_xref": xref})
    return refs


def site_hash(accession, label, q, evidence):
    return hash_text(f"pilot5a-evidence:{accession}:{q}:{label}:{evidence['pdb_id']}:{evidence['chain']}:{evidence['feature_start_1based']}:{evidence['feature_end_1based']}:{evidence['eco_code']}:{SEED}")


def evidence_tuple(accession, label, q, evidence):
    return (accession, q, label, evidence["pdb_id"], evidence["chain"],
            evidence["feature_start_1based"], evidence["feature_end_1based"], evidence["eco_code"])


def labels_and_pair(entry):
    accession, sequence = entry["primaryAccession"], entry["sequence"]["value"]
    length = len(sequence)
    xrefs = pdb_crossrefs(entry, length)
    annotated = {"HELIX": set(), "STRAND": set()}
    supported = {"HELIX": defaultdict(list), "STRAND": defaultdict(list)}
    feature_audit, evidence_reason_counts = [], Counter()
    for index, feature in enumerate(entry.get("features", [])):
        label = FEATURE_TYPES.get(feature.get("type"))
        if label is None:
            continue
        interval, error = exact_interval(feature, accession, length)
        audit = {"id": accession, "feature_index": index, "label": label,
                 "raw_feature": feature, "exclusion_reasons": [error] if error else [], "evidence_audit": []}
        if interval is not None:
            start, end = interval
            annotated[label].update(range(start, end + 1))
            audit["start_q_0based"], audit["end_q_0based"] = start, end
            for evidence in feature.get("evidences", []):
                reasons = []
                code = evidence.get("evidenceCode", "")
                pdb_id = str(evidence.get("id", "")).upper()
                if evidence.get("source") != "PDB":
                    reasons.append("evidence_source_not_PDB")
                if re.fullmatch(r"ECO:\d{7}", code) is None:
                    reasons.append("missing_or_invalid_ECO_code")
                if pdb_id not in xrefs:
                    reasons.append("evidence_PDB_not_in_entry_crossrefs")
                valid_coverages = []
                if not reasons:
                    for reference in xrefs[pdb_id]:
                        if reference["exclusion_reasons"]:
                            reasons.extend(reference["exclusion_reasons"])
                            continue
                        for coverage in reference["coverage"]:
                            lo = max(start, coverage["start_q_0based"])
                            hi = min(end, coverage["end_q_0based"])
                            if lo <= hi:
                                record = {"pdb_id": pdb_id, "chain": coverage["chain"], "method": reference["method"],
                                          "eco_code": code, "feature_index": index, "source_evidence": evidence,
                                          "feature_start_1based": start + 1, "feature_end_1based": end + 1,
                                          "coverage": coverage, "raw_xref": reference["raw_xref"]}
                                valid_coverages.append(record)
                                for q in range(lo, hi + 1):
                                    supported[label][q].append(record)
                    if not valid_coverages and not reasons:
                        reasons.append("PDB_chain_coverage_does_not_cover_feature")
                evidence_reason_counts.update(set(reasons))
                audit["evidence_audit"].append({"raw_evidence": evidence, "exclusion_reasons": sorted(set(reasons)),
                                                "accepted_coverage_count": len(valid_coverages)})
            if not any(x["accepted_coverage_count"] for x in audit["evidence_audit"]):
                audit["exclusion_reasons"].append("no_accepted_PDB_source_evidence")
        feature_audit.append(audit)
    conflicts = annotated["HELIX"] & annotated["STRAND"]
    sites = {"HELIX": [], "STRAND": []}
    site_loss = Counter()
    for label in sites:
        for q in sorted(annotated[label]):
            reasons = []
            if q in conflicts:
                reasons.append("conflicting_HELIX_STRAND_annotation")
            if sequence[q] != "A":
                reasons.append("center_residue_not_A")
            if not (q >= 7 and q < length - 7):
                reasons.append("local_plusminus7_unavailable")
            if q not in supported[label]:
                reasons.append("no_valid_PDB_evidence_covering_site")
            site_loss.update(reasons)
            if reasons:
                continue
            evidence = min(supported[label][q], key=lambda x: (site_hash(accession, label, q, x), evidence_tuple(accession, label, q, x)))
            visible15 = sequence[max(0, q - 15):min(length, q + 16)]
            sites[label].append({"label": label, "q_0based": q, "uniprot_residue_1based": q + 1,
                "future_token_index_0based": q + 1, "center_aa": "A", "window_plusminus7": sequence[q - 7:q + 8],
                "plusminus15_available": q >= 15 and q < length - 15,
                "window_plusminus15_visible": visible15, "window_plusminus15_visible_length": len(visible15),
                "window_plusminus15_start_q_0based": max(0, q - 15), "window_plusminus15_end_q_0based": min(length - 1, q + 15),
                "legal_evidence_count": len(supported[label][q]),
                "evidence_selection_sha256": site_hash(accession, label, q, evidence), "evidence": evidence})
    pair, pair_count, best_sort_key = None, 0, None
    for h_site in sites["HELIX"]:
        for s_site in sites["STRAND"]:
            hq, sq = h_site["q_0based"], s_site["q_0based"]
            if hq == sq:
                continue
            he, se = h_site["evidence"], s_site["evidence"]
            key = f"{accession}:{hq}:{sq}:{he['pdb_id']}:{he['chain']}:{se['pdb_id']}:{se['chain']}"
            digest = hash_text(f"pilot5a-pair:{key}:{SEED}")
            sort_key = (digest, (accession, hq, sq, he["pdb_id"], he["chain"], se["pdb_id"], se["chain"]))
            candidate = {"id": accession, "h_site": h_site, "s_site": s_site,
                "pair_key": key, "pair_sha256": digest,
                "same_PDB_source": he["pdb_id"] == se["pdb_id"],
                "same_PDB_and_chain": (he["pdb_id"], he["chain"]) == (se["pdb_id"], se["chain"]),
                "site_distance": abs(hq - sq), "plusminus7_windows_overlap": abs(hq - sq) <= 14,
                "both_plusminus15_available": h_site["plusminus15_available"] and s_site["plusminus15_available"]}
            pair_count += 1
            if best_sort_key is None or sort_key < best_sort_key:
                pair, best_sort_key = candidate, sort_key
    counts = {"canonical_exact_annotated_residues": {k: len(v) for k, v in annotated.items()},
              "conflicting_q_0based": sorted(conflicts), "valid_A_sites_plusminus7": {k: len(v) for k, v in sites.items()},
              "valid_A_sites_plusminus15": {k: sum(x["plusminus15_available"] for x in v) for k, v in sites.items()},
              "site_exclusion_reason_counts_nonexclusive": dict(site_loss),
              "evidence_exclusion_reason_counts_nonexclusive": dict(evidence_reason_counts), "legal_pair_count": pair_count}
    return pair, sites, feature_audit, counts


def select_representatives(representatives):
    """Apply the frozen component order and all-or-nothing 128-component gate."""
    ordered = sorted(representatives, key=lambda x: (x["component_sha256"], x["component_key"]))
    if len(ordered) < REQUIRED_COMPONENTS:
        return []
    selected = []
    for row in ordered[:REQUIRED_COMPONENTS]:
        pair = row["pair"]
        a, b = sorted((pair["h_site"], pair["s_site"]), key=lambda site: site["q_0based"])
        selected.append({"id": row["id"], "sequence": row["sequence"], "pfam": row["pfam"], "no_Pfam": not row["pfam"],
            "split": "T_confirm", "component_key": row["component_key"], "component_sha256": row["component_sha256"],
            "protein_sha256": row["protein_sha256"], **{k: v for k, v in pair.items() if k != "id"},
            "pair_order": "canonical_q_ascending", "a_site": a, "b_site": b,
            "y_a": int(a["label"] == "HELIX"), "y_b": int(b["label"] == "HELIX")})
    return selected


def rebuild(lock):
    receipt_path = OUT / "http_provenance.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not receipt.get("complete") or receipt["protocol_sha256"] != lock["protocol_sha256"]:
        raise RuntimeError("Incomplete download or protocol mismatch")
    if receipt.get("terminal_failure") or receipt.get("pending_request"):
        raise RuntimeError("Terminal/unresolved source response cannot be used for build")
    entries, releases, following = receipt_pages(receipt, lock)
    if not releases:
        raise RuntimeError("No official source page/release available")
    ids = [e["primaryAccession"] for e in entries]
    if ids and (ids[0] <= lock["continuation"]["old_last_accession"] or any(a >= b for a, b in zip(ids, ids[1:]))):
        raise RuntimeError("Source strict continuation accession order not respected")
    old_nodes, old_sources = old_pool()
    if old_sources != lock["old_pool_sources"]:
        raise RuntimeError("Old pool sources drifted after DATA rule lock")
    new_nodes = [{**normalize(e), "old": False, "origin": "pilot5a_raw", "raw_index": i} for i, e in enumerate(entries)]
    nodes = old_nodes + new_nodes
    components = graph_components(nodes)
    node_component = {i: c for c in components for i in c["indices"]}
    old_ids, old_sequences = {x["id"] for x in old_nodes}, {x["sequence"] for x in old_nodes}
    audits, feature_rows, site_rows, pair_rows = [], [], [], []
    eligible_by_component = defaultdict(list)
    reason_counts, feature_reasons, site_reasons, evidence_reasons = Counter(), Counter(), Counter(), Counter()
    stages = Counter({"raw_records": len(entries)})
    seen_ids = set()
    for i, (entry, node) in enumerate(zip(entries, new_nodes)):
        sequence, accession = node["sequence"], node["id"]
        component = node_component[len(old_nodes) + i]
        reasons = []
        if accession in seen_ids:
            reasons.append("duplicate_candidate_accession")
        seen_ids.add(accession)
        if accession in old_ids:
            reasons.append("old_pool_accession")
        if sequence in old_sequences:
            reasons.append("old_pool_exact_sequence")
        if component["has_old"]:
            reasons.append("component_connected_to_old_pool")
        sequence_ok = True
        if not 64 <= len(sequence) <= 256:
            reasons.append("length_outside_64_256")
            sequence_ok = False
        if not set(sequence) <= AA:
            reasons.append("nonstandard_amino_acid")
            sequence_ok = False
        if entry["sequence"].get("length") != len(sequence):
            reasons.append("source_sequence_length_mismatch")
            sequence_ok = False
        if entry.get("entryType") != "UniProtKB reviewed (Swiss-Prot)":
            reasons.append("source_not_reviewed")
            sequence_ok = False
        pair, sites, features, label_counts = labels_and_pair(entry)
        if not sites["HELIX"]:
            reasons.append("no_eligible_HELIX_A_site")
        if not sites["STRAND"]:
            reasons.append("no_eligible_STRAND_A_site")
        if pair is None:
            reasons.append("no_legal_pair")
        for feature in features:
            feature["raw_index"] = i
            feature_reasons.update(feature["exclusion_reasons"])
        feature_rows.extend(features)
        for label in sites:
            site_rows.extend({"id": accession, "raw_index": i, **x} for x in sites[label])
        if pair is not None:
            pair_rows.append({"raw_index": i, "component_key": component["key"], "eligible_for_selection": not reasons, **pair})
        site_reasons.update(label_counts["site_exclusion_reason_counts_nonexclusive"])
        evidence_reasons.update(label_counts["evidence_exclusion_reason_counts_nonexclusive"])
        reasons = sorted(set(reasons))
        reason_counts.update(reasons)
        audit = {"raw_index": i, "id": accession, "sequence_sha256": sha(sequence.encode("utf-8")),
            "length": len(sequence), "pfam": node["pfam"], "no_Pfam": not node["pfam"],
            "nonstandard_symbols": sorted(set(sequence) - AA), "component_key": component["key"],
            "component_connected_to_old_pool": component["has_old"], "exclusion_reasons": reasons, **label_counts}
        audits.append(audit)
        stages["standard_reviewed_sequence_records"] += sequence_ok
        stages["standard_sequence_and_old_disconnected_records"] += sequence_ok and not component["has_old"]
        stages["standard_old_disconnected_with_both_sites_records"] += sequence_ok and not component["has_old"] and bool(sites["HELIX"] and sites["STRAND"])
        stages["eligible_pair_records"] += not reasons
        if not reasons:
            eligible_by_component[component["key"]].append({**node, "pair": pair, "component_key": component["key"],
                "component_sha256": component["hash"], "protein_sha256": hash_text(f"pilot5a-protein:{accession}:{SEED}")})
    component_rows, representatives = [], []
    for component in components:
        new_indices = [i for i in component["indices"] if not nodes[i]["old"]]
        if not new_indices:
            continue
        eligible = eligible_by_component[component["key"]]
        representative = min(eligible, key=lambda x: (x["protein_sha256"], x["id"])) if eligible else None
        if representative:
            representatives.append(representative)
        component_rows.append({"component_key": component["key"], "component_sha256": component["hash"],
            "all_member_accessions": component["accessions"], "pfam": component["pfam"], "no_Pfam": not component["pfam"],
            "old_members": sorted({nodes[i]["id"] for i in component["indices"] if nodes[i]["old"]}),
            "old_node_records": sum(nodes[i]["old"] for i in component["indices"]),
            "new_member_accessions": sorted({nodes[i]["id"] for i in new_indices}), "new_raw_indices": [nodes[i]["raw_index"] for i in new_indices],
            "connected_to_old_pool": component["has_old"], "eligible_protein_records": len(eligible),
            "representative": representative["id"] if representative else None,
            "representative_protein_sha256": representative["protein_sha256"] if representative else None})
    enough = len(representatives) >= REQUIRED_COMPONENTS
    selected = select_representatives(representatives)
    funnel_order = ("raw_records", "standard_reviewed_sequence_records", "standard_sequence_and_old_disconnected_records",
                    "standard_old_disconnected_with_both_sites_records", "eligible_pair_records")
    losses = [{"from": a, "to": b, "lost_records": stages[a] - stages[b]} for a, b in zip(funnel_order, funnel_order[1:])]
    counts = {"old_pool_records": len(old_nodes), "old_unique_accessions": len(old_ids),
        "old_unique_sequences": len(old_sequences), "raw_response_records": len(entries),
        "raw_no_Pfam_records": sum(not x["pfam"] for x in new_nodes),
        "stage_survivor_counts": dict(stages), "stage_loss_counts_ordered": losses,
        "eligible_new_known_Pfam_or_sequence_components": len(representatives),
        "all_components_touching_new": len(component_rows), "components_connected_to_old": sum(x["connected_to_old_pool"] for x in component_rows),
        "record_exclusion_reason_counts_nonexclusive": dict(reason_counts),
        "feature_exclusion_reason_counts_nonexclusive": dict(feature_reasons),
        "site_exclusion_reason_counts_nonexclusive": dict(site_reasons),
        "evidence_exclusion_reason_counts_nonexclusive": dict(evidence_reasons),
        "conflicting_residue_positions_all_raw_records": sum(len(x["conflicting_q_0based"]) for x in audits),
        "raw_records_with_conflicting_labels": sum(bool(x["conflicting_q_0based"]) for x in audits),
        "raw_feature_records": len(feature_rows),
        "raw_canonical_exact_features": sum("start_q_0based" in x for x in feature_rows),
        "selected_by_split": dict(Counter(x["split"] for x in selected)),
        "selected_no_Pfam": sum(x["no_Pfam"] for x in selected),
        "selected_same_PDB_source": sum(x["same_PDB_source"] for x in selected),
        "selected_different_PDB_source": sum(not x["same_PDB_source"] for x in selected),
        "selected_both_plusminus15_available": sum(x["both_plusminus15_available"] for x in selected),
        "selected_plusminus7_windows_overlap": sum(x["plusminus7_windows_overlap"] for x in selected)}
    data_bytes = jsonl(selected)
    checks = {
        "bounded_raw_response_records": len(entries) <= MAX_RECORDS,
        "old_pool_includes_512_plus_pilot2_all4096_plus_pilot3a_all4096_plus_pilot4a_all4096": len(old_nodes) == 12800,
        "maximum_nine_successful_source_pages": len(receipt["pages"]) <= MAX_PAGES,
        "source_release_matches_old_pilot4a": releases == {EXPECTED_RELEASE},
        "strict_source_continuation_after_old_last_accession": (not ids or ids[0] > lock["continuation"]["old_last_accession"]) and all(a < b for a, b in zip(ids, ids[1:])),
        "unique_selected_accessions": len({x["id"] for x in selected}) == len(selected),
        "unique_selected_sequences": len({x["sequence"] for x in selected}) == len(selected),
        "one_pair_per_component": len({x["component_key"] for x in selected}) == len(selected),
        "no_selected_old_accession": all(x["id"] not in old_ids for x in selected),
        "no_selected_old_exact_sequence": all(x["sequence"] not in old_sequences for x in selected),
        "no_selected_old_connected_component": all(not c["connected_to_old_pool"] for c in component_rows if c["component_key"] in {x["component_key"] for x in selected}),
        "center_A_and_explicit_coordinates": all(x[k]["center_aa"] == x["sequence"][x[k]["q_0based"]] == "A" and x[k]["uniprot_residue_1based"] == x[k]["future_token_index_0based"] == x[k]["q_0based"] + 1 for x in selected for k in ("h_site", "s_site")),
        "different_sites_and_local7": all(x["h_site"]["q_0based"] != x["s_site"]["q_0based"] and all(len(x[k]["window_plusminus7"]) == 15 for k in ("h_site", "s_site")) for x in selected),
        "selected_source_coverage_and_frozen_method": all(x[k]["evidence"]["coverage"]["start_q_0based"] <= x[k]["q_0based"] <= x[k]["evidence"]["coverage"]["end_q_0based"] and x[k]["evidence"]["method"] in METHODS for x in selected for k in ("h_site", "s_site")),
        "no_selected_conflicting_q": all(x[k]["q_0based"] not in next(a["conflicting_q_0based"] for a in audits if a["id"] == x["id"]) for x in selected for k in ("h_site", "s_site")),
        "canonical_q_order_and_binary_labels": all(x["a_site"]["q_0based"] < x["b_site"]["q_0based"] and x["y_a"] == int(x["a_site"]["label"] == "HELIX") and x["y_b"] == 1 - x["y_a"] for x in selected),
        "monotonic_stage_funnel": all(x["lost_records"] >= 0 for x in losses),
        "fixed_split_sizes_if_sufficient": dict(Counter(x["split"] for x in selected)) == (dict(SPLIT_SIZES) if enough else {}),
        "zero_model_forwards": True,
    }
    status = "DATA_PASS" if enough and all(checks.values()) else "DATA_INSUFFICIENT" if all(checks.values()) else "DATA_INTEGRITY_ERROR"
    manifest = {"schema_version": 1, "status": status, "configuration": config(), "protocol_lock": lock,
        "source": {"endpoint": SOURCE, "release": next(iter(releases))[0], "release_date": next(iter(releases))[1],
                   "http_provenance_path": receipt_path.relative_to(ROOT).as_posix(), "http_provenance_sha256": sha(receipt_path.read_bytes()),
                   "stop_reason": receipt["stop_reason"], "request_walltime_seconds": receipt["request_walltime_seconds"],
                   "successful_pages": len(receipt["pages"]), "continuation": lock["continuation"],
                   "license": "CC BY 4.0", "license_evidence": "docs/ASSETS_AND_PRIOR.md; data/raw/uniprot_license.json"},
        "old_pool_sources": old_sources, "counts": counts, "data_path": "data/pilot5a/proteins.jsonl", "data_sha256": sha(data_bytes),
        "split_ids": {s: [x["id"] for x in selected if x["split"] == s] for s, _ in SPLIT_SIZES},
        "hash_rules": {"encoding": "ASCII without newline; lowercase SHA256 hex ascending",
            "component_edges": "accession OR exact full sequence OR any shared Pfam; all old and raw-new nodes before label filtering",
            "component_key": "acc: + comma-joined lexicographically sorted unique member accessions",
            "site_evidence": "pilot5a-evidence:{accession}:{q}:{class}:{PDB}:{chain}:{start_1based}:{end_1based}:{ECO}:2026101605",
            "pair": "pilot5a-pair:{accession}:{Hq}:{Sq}:{HPDB}:{Hchain}:{SPDB}:{Schain}:2026101605",
            "protein": "pilot5a-protein:{accession}:2026101605",
            "component": "pilot5a-component:{component_key}:2026101605"},
        "coordinate_mapping": "UniProt 1-based inclusive feature/source coverage -> q=position-1 -> future ESM token index (0-based with BOS)=q+1",
        "label_interpretation": {"HELIX": "UniProt Helix aggregate H/G/I", "STRAND": "UniProt Beta strand aggregate E/B", "unannotated_or_TURN": "never used as negative"},
        "position_order": {"rule": "a_site precedes b_site in canonical q; HELIX=1 and STRAND=0", "y_a_randomized_or_balanced": False},
        "limitations": ["Known Pfam/exact-sequence component independence does not establish remote-homology independence; no_Pfam is explicit.",
            "Source PDB/chain coverage is UniProt-coordinate provenance, not recomputed DSSP or atom-level SIFTS verification.",
            "Different-source HELIX/STRAND pairs are not asserted to represent one structural state.",
            "ESM/SAE pretraining overlap is unknown; no model values were read or generated."],
        "model_forward_count": 0}
    save_frozen(OUT / "proteins.jsonl", data_bytes)
    save_frozen(OUT / "candidate_records.jsonl", jsonl(new_nodes))
    save_frozen(OUT / "candidate_filter_audit.jsonl", jsonl(audits))
    save_frozen(OUT / "feature_provenance.jsonl", jsonl(feature_rows))
    save_frozen(OUT / "site_manifest.jsonl", jsonl(site_rows))
    save_frozen(OUT / "pair_manifest.jsonl", jsonl(pair_rows))
    save_frozen(OUT / "component_manifest.jsonl", jsonl(component_rows))
    save_frozen(OUT / "selected_manifest.json", canonical(manifest))
    acceptance = {"status": status, "checks": checks, "counts": counts,
        "data_sha256": sha(data_bytes), "selected_manifest_sha256": sha(canonical(manifest)),
        "protocol_sha256": lock["protocol_sha256"], "model_forward_count": 0,
        "scope": "Source-only build boundary checks; supervisor owns the single consolidated acceptance. This data gate is not statistical/scientific acceptance."}
    save_frozen(OUT / "DATA_BUILD_CHECKS.json", canonical(acceptance))
    paths = [p for p in sorted(OUT.rglob("*")) if p.is_file() and "run_receipts" not in p.parts and p.name != "asset_manifest.json"]
    save_frozen(OUT / "asset_manifest.json", canonical({"schema_version": 1, "assets": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())} for p in paths], "model_forward_count": 0}))
    return acceptance


def self_test():
    """Only Pilot 5-A deltas; reuse accepted Pilot 4-A mapping/transport checks."""
    script_bytes = Path(__file__).read_bytes()
    script_text = script_bytes.decode("utf-8")
    previous_text = (ROOT / "scripts/prepare_pilot4a_data.py").read_text(encoding="utf-8")
    normalized_previous = (previous_text.replace("pilot4a", "pilot5a").replace("Pilot4A", "Pilot5A")
        .replace("Pilot 4-A", "Pilot 5-A").replace("Pilot 3-A", "Pilot 4-A").replace("2026101604", "2026101605"))
    def function_asts(source):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    current_asts, prior_asts = function_asts(script_text), function_asts(normalized_previous)
    reused = ("public_url", "page_size", "next_url", "PublicSearchRedirects", "source_page", "receipt_pages", "download",
              "UnionFind", "graph_components", "normalize", "exact_interval", "parse_chains", "pdb_crossrefs", "site_hash",
              "evidence_tuple", "labels_and_pair", "select_representatives")
    checks = {"accepted_pilot4a_transport_graph_mapping_selection_unchanged_except_namespace":
              all(current_asts[name] == prior_asts[name] for name in reused)}
    anchor = continuation_binding()
    official = url_params(anchor["official_next_url"])
    checks["pilot4a_official_continuation_and_only_size_rewritten"] = (
        anchor["old_last_accession"] == "P92533" and anchor["cursor"] == "nnn1hsuzifwzrv8k2rhir4e7i2"
        and anchor["old_receipt_path"] == "data/pilot4a/http_provenance.json" and anchor["old_final_page_records"] == 96
        and official["size"] == "96" and url_params(anchor["start_request_url"]) == {**official, "size": "500"}
        and anchor["previous_same_query_raw_records"] == 8192)
    sizes, total = [], 0
    while total < MAX_RECORDS and len(sizes) < MAX_PAGES:
        size = min(500, MAX_RECORDS - total)
        sizes.append(size)
        total += size
    checks["unchanged_raw_page_attempt_caps_and_new_hash_seed"] = (
        sizes == [500] * 8 + [96] and total == 4096 and MAX_ATTEMPTS == 3
        and SEED == 2026101605 and REQUIRED_COMPONENTS == 128 and SPLIT_SIZES == (("T_confirm", 128),))
    def feature(kind, q):
        return {"type": kind, "location": {"start": {"value": q + 1}, "end": {"value": q + 1}},
                "evidences": [{"evidenceCode": "ECO:0007829", "source": "PDB", "id": "1ABC"}]}
    sample = {"primaryAccession": "SYNTH1", "sequence": {"value": "A" * 64, "length": 64},
        "features": [feature("Helix", 7), feature("Beta strand", 39)],
        "uniProtKBCrossReferences": [{"database": "PDB", "id": "1ABC", "properties": [
            {"key": "Method", "value": "X-ray"}, {"key": "Chains", "value": "A=1-64"}]}]}
    pair = labels_and_pair(sample)[0]
    groups = graph_components([{"id": "NEW", "sequence": "A", "pfam": [], "old": False}])
    checks["pilot5a_evidence_pair_component_namespaces"] = (
        pair["h_site"]["evidence_selection_sha256"] == hash_text("pilot5a-evidence:SYNTH1:7:HELIX:1ABC:A:8:8:ECO:0007829:2026101605")
        and pair["pair_sha256"] == hash_text(f"pilot5a-pair:{pair['pair_key']}:2026101605")
        and groups[0]["hash"] == hash_text("pilot5a-component:acc:NEW:2026101605"))
    reps = [{"id": f"SYNTH{i:03d}", "sequence": "A" * 64, "pfam": [], "pair": pair,
        "component_key": f"acc:SYNTH{i:03d}", "component_sha256": hash_text(f"pilot5a-component:acc:SYNTH{i:03d}:{SEED}"),
        "protein_sha256": hash_text(f"pilot5a-protein:SYNTH{i:03d}:{SEED}")} for i in range(129)]
    selected = select_representatives(reps)
    checks["new_seed_first128_T_q_order_and_no_smaller_fallback"] = (
        len(selected) == 128 and selected == select_representatives(list(reversed(reps)))
        and select_representatives(reps[:127]) == []
        and all(x["split"] == "T_confirm" and x["a_site"]["q_0based"] < x["b_site"]["q_0based"]
                and x["y_a"] == 1 and x["y_b"] == 0 for x in selected)
        and [x["component_key"] for x in selected] == [x["component_key"] for x in sorted(reps, key=lambda x: (x["component_sha256"], x["component_key"]))[:128]])
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(ast.parse(script_text)) if isinstance(n, ast.Import)} | {
        n.module.split(".")[0] for n in ast.walk(ast.parse(script_text)) if isinstance(n, ast.ImportFrom)}
    checks["stdlib_only_no_model_imports"] = not (imported & {"torch", "transformers", "numpy", "requests"})
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "scope": "Pilot5-A delta checks; original mapping/transport tests are reused, not rerun",
        "reused_check_receipts": [file_binding(ROOT / "data/pilot4a/implementation_checks_9e18d6c4d91a.json"),
                                  file_binding(ROOT / "data/pilot4a/DATA_BUILD_CHECKS.json")],
        "network_requests": 0, "model_forward_count": 0, "script_sha256": sha(script_bytes)}
    report_path = OUT / ("implementation_checks_" + sha(script_bytes)[:12] + ".json")
    save_frozen(report_path, canonical(result))
    result["receipt_path"] = report_path.relative_to(ROOT).as_posix()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--describe-lock", action="store_true", help="Print the full offline lock payload; supervisor adds frozen_utc and saves DATA_RULE_LOCK.json")
    args = parser.parse_args()
    if args.describe_lock:
        if args.download or args.self_test:
            parser.error("--describe-lock cannot request a download or run checks")
        path = args.protocol or ROOT / "docs/PILOT5A_DATA_PROTOCOL.md"
        print(json.dumps(rule_inputs(path), sort_keys=True, indent=2))
        return 0
    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 2
    if args.protocol is None or not args.protocol_sha256:
        parser.error("--protocol and --protocol-sha256 are required, including offline rebuilds")
    started = time.perf_counter()
    try:
        lock = lock_protocol(args.protocol, args.protocol_sha256)
        if args.download:
            download(lock)
        result = rebuild(lock)
    except Exception as exc:
        source_failure = isinstance(exc, (HTTPError, URLError, TimeoutError, OSError)) or str(exc).startswith("UniProt HTTP")
        result = {"status": getattr(exc, "status", "DATA_SOURCE_BLOCKED" if source_failure else "DATA_INTEGRITY_ERROR"),
                  "error_type": type(exc).__name__, "reason": str(exc), "model_forward_count": 0,
                  "walltime_seconds": round(time.perf_counter() - started, 6), "query_changed": False}
        result["run_receipt_path"] = run_receipt("stopped", result)
        print(json.dumps(result, indent=2))
        return 2
    result["run_receipt_path"] = run_receipt("data_run", {"status": result["status"], "protocol_sha256": lock["protocol_sha256"],
        "script_sha256": sha(Path(__file__).read_bytes()), "walltime_seconds": round(time.perf_counter() - started, 6),
        "data_sha256": result["data_sha256"], "selected_manifest_sha256": result["selected_manifest_sha256"]})
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "DATA_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
