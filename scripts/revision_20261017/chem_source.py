"""Acquire the frozen public source; this program never decompresses molecules."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/revision_20261017/chemistry/source"
URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
SIZE = 1486837335
MODIFIED = "Sun, 27 Sep 2026 07:55:20 GMT"
EXPECTED = "36b683a59272b196ed7ece140903a249f7df475a06c312c963a5c761f1037f1a"


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(record):
    (OUT / "SOURCE_RECEIPT.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "CID-SMILES.gz"
    if path.exists() or (OUT / "SOURCE_RECEIPT.json").exists():
        raise SystemExit("Existing acquisition evidence; do not start a second acquisition")
    rule = ROOT / "docs/revision_20261017/CHEMISTRY_SOURCE_RULE.md"
    record = dict(status="RUNNING_SOURCE_ONLY", started_utc=utc(), url=URL,
                  expected_bytes=SIZE, expected_last_modified=MODIFIED,
                  expected_sha256=EXPECTED, rule_sha256=hashlib.sha256(rule.read_bytes()).hexdigest(),
                  source_only_no_decompression=True, attempts=[], received_bytes=0)
    save(record)
    digest = hashlib.sha256()
    size = 0
    last_report = time.monotonic()
    with path.open("xb") as dst:
        for attempt in range(1, 4):
            info = dict(number=attempt, started_utc=utc(), offset=size)
            record["attempts"].append(info)
            headers = {"User-Agent": "NativeUnitAuditResearch/1.0"}
            if size:
                headers["Range"] = f"bytes={size}-"
            try:
                with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=60) as response:
                    info.update(status=response.status, final_url=response.geturl(),
                                content_length=response.headers.get("Content-Length"),
                                last_modified=response.headers.get("Last-Modified"),
                                content_range=response.headers.get("Content-Range"))
                    if response.geturl() != URL or response.headers.get("Last-Modified") != MODIFIED:
                        raise ValueError("SOURCE_DRIFT")
                    if size == 0 and (response.status != 200 or int(response.headers.get("Content-Length", -1)) != SIZE):
                        raise ValueError("SOURCE_LENGTH_OR_STATUS")
                    if size and (response.status != 206 or response.headers.get("Content-Range") != f"bytes {size}-{SIZE - 1}/{SIZE}"):
                        raise ValueError("RANGE_NOT_CONFIRMED")
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                        if size > SIZE:
                            raise ValueError("SOURCE_EXCEEDED_SIZE")
                        if time.monotonic() - last_report > 20:
                            record["received_bytes"] = size
                            save(record)
                            print(json.dumps({"downloaded_bytes": size, "expected_bytes": SIZE}), flush=True)
                            last_report = time.monotonic()
                    if size != SIZE:
                        raise OSError("INCOMPLETE_BODY")
                    info["complete"] = True
                    break
            except Exception as exc:
                info["error_type"] = type(exc).__name__
                info["complete"] = False
                record["received_bytes"] = size
                save(record)
                if isinstance(exc, ValueError) or attempt == 3:
                    record.update(status="SOURCE_FAILED", completed_utc=utc())
                    save(record)
                    raise SystemExit("Source acquisition failed; retained partial source and safe receipt") from None
                time.sleep(2)
        dst.flush()
        os.fsync(dst.fileno())
    record.update(received_bytes=size, sha256=digest.hexdigest(), completed_utc=utc())
    record["status"] = "PASS_SOURCE_IDENTITY" if digest.hexdigest() == EXPECTED and size == SIZE else "SOURCE_IDENTITY_MISMATCH"
    save(record)
    print(json.dumps({k: record[k] for k in ("status", "received_bytes", "sha256", "completed_utc")}), flush=True)
    if record["status"] != "PASS_SOURCE_IDENTITY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
