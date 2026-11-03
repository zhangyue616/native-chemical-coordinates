"""Protocol v6: MoLFormer-XL-both-10pct on H4. For the pretrained model and K untrained twins, best-lift vectors S at
all 12 layers, native and under R rotations; the tests of gates_v6.py then take, for each network, the maximum over
layers of its same-layer mean-rank statistic (J1) and the J2 statistic at that same layer, so no layer is chosen in
advance and none is chosen by looking at the pretrained model.

usage: python -I audit_v6.py <tsv> <concept_dir> <label_tag> <out_root> [--K 199] [--R 9] [--K_hf 39] [--K_orf 9]
                               [--n_mol N] [--blind] [--lift_batch 256] [--freeze_dir DIR]
Sets (all under <out_root>/molformer/):
  pretrained/<layer>/S_native.npy, S_rot_{r}.npy, pre_native_full.npy      the released model, natives and R rotations
  twins/<layer>/S_native_{k}.npy, S_rot_{k}_{r}.npy                         K twins, seeds 4000+k, pretraining init
  twins_hf/<layer>/S_native_{k}.npy                                         K_hf twins, seeds 6000+k, HF-port init
                                                                            (natives only; descriptive)
  pretrained_orf/<layer>/S_native_{s}.npy                                   the released weights with the linear
                                                                            attention's random-feature matrices redrawn
                                                                            under seeds 7000+s (natives only; descriptive)
  fingerprints.json, run_record.json
--blind (dry runs only): the pretrained model is replaced by one more twin (seed 4999) and the ORF set is skipped, so
no readability of the released model is computed before the freeze. --n_mol N slices the first N molecules.
Freeze binding: for tag H4 (confirmatory) the script requires the protocol's K, R, K_hf, K_orf, no --n_mol and no
--blind, verifies <freeze_dir>/PROTOCOL_v6.sha256 and <freeze_dir>/FREEZE_v6_MANIFEST.sha256 against the files on
disk, checks that its inputs are the manifest's H4 files and that the determinism check passed, and records the
freeze. --freeze_dir with another tag rehearses the same check on a scratch freeze (dry run).
Every array is written as soon as it is computed and skipped when present (resumable); the log is appended.
"""

# R13: usage can be read without importing optional model dependencies.
if __name__ == "__main__":
    import sys as _help_sys
    if "--help" in _help_sys.argv[1:] or "-h" in _help_sys.argv[1:]:
        print(__doc__)
        raise SystemExit(0)

import sys, os, csv, json, time, hashlib, argparse, platform

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import models  # noqa: E402
import lift  # noqa: E402
from audit import rotation  # noqa: E402

LAYERS = [f"L{i}" for i in range(12)]
LAST = "L11"
TWIN_SEED0, HF_SEED0, ORF_SEED0, BLIND_SEED, ROT_SEED0 = 4000, 6000, 7000, 4999, 200
WIDTH, ACT = 768, "gelu"
PROTOCOL_DESIGN = dict(K=199, R=9, K_hf=39, K_orf=9, N_atoms=245749, N_mol=10000)
SCRIPTS = ("audit_v6.py", "gates_v6.py", "models.py", "lift.py", "audit.py", "gates_v2.py",
           "check_molformer_alignment.py", "molformer_checkpoint_facts.py", "molformer_stereo_check.py",
           "molformer_determinism_check.py", "token_visible_concepts.py")
FREEZE_LIST = os.path.join(ROOT, "lineB", "results_v6", "FREEZE_v6_FILES.txt")   # the paths the manifest must hold
MODEL_FILES = ("config.json", "configuration_molformer.py", "modeling_molformer.py", "tokenizer.json",
               "tokenizer_config.json", "model.safetensors")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(m):
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def write_json(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def read_manifest(man):
    """{relative path: sha256} of a sha256sum -b manifest (lines 'hash *path'; '#' lines ignored)."""
    out = {}
    for line in open(man, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, p = line.split(maxsplit=1)
        out[p.lstrip("*")] = h
    return out


def verify_freeze(freeze_dir):
    """Recompute every hash of the freeze manifest; abort on any mismatch. Returns the record to store."""
    prot = os.path.join(freeze_dir, "PROTOCOL_v6.sha256")
    man = os.path.join(freeze_dir, "FREEZE_v6_MANIFEST.sha256")
    assert os.path.exists(prot) and os.path.exists(man), "protocol v6 is not frozen (PROTOCOL_v6.sha256 / manifest missing)"
    pline = open(prot, encoding="utf-8").read().strip()
    phash, pfile = pline.split()[0], pline.split()[1]
    assert sha(os.path.join(freeze_dir, pfile)) == phash, "the protocol file differs from its frozen hash"
    entries = read_manifest(man)
    for p, h in entries.items():
        assert sha(os.path.join(ROOT, p)) == h, f"manifest mismatch: {p}"
    return dict(protocol_line=pline, manifest_sha256=sha(man), manifest_entries=len(entries),
                manifest_header=open(man, encoding="utf-8").readline().strip(), entries=entries)


def network_list(K, K_hf, K_orf, blind):
    nets = [("pretrained", "pretrained", BLIND_SEED if blind else None, "pretraining" if blind else None, True)]
    nets += [("twins", f"twin_{k}", TWIN_SEED0 + k, "pretraining", True) for k in range(K)]
    nets += [("twins_hf", f"hf_{k}", HF_SEED0 + k, "hf", False) for k in range(K_hf)]
    if not blind:
        nets += [("pretrained_orf", f"orf_{s}", ORF_SEED0 + s, "orf", False) for s in range(K_orf)]
    return nets   # (sub, name, seed, init, with_rotations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv"); ap.add_argument("cdir"); ap.add_argument("tag"); ap.add_argument("out_root")
    ap.add_argument("--K", type=int, default=199); ap.add_argument("--R", type=int, default=9)
    ap.add_argument("--K_hf", type=int, default=39); ap.add_argument("--K_orf", type=int, default=9)
    ap.add_argument("--n_mol", type=int, default=None); ap.add_argument("--blind", action="store_true")
    ap.add_argument("--lift_batch", type=int, default=256)
    ap.add_argument("--freeze_dir", default=None, help="directory of PROTOCOL_v6.sha256 and the manifest "
                                                       "(default lineB for tag H4)")
    a = ap.parse_args()
    K, R, K_hf, K_orf = a.K, a.R, a.K_hf, a.K_orf
    confirmatory = a.tag == "H4"
    base = os.path.join(a.out_root, "molformer")
    os.makedirs(base, exist_ok=True)
    t0 = time.time()
    mf = models.WEIGHTS["molformer"]
    start = dict(protocol="lineB PROTOCOL_v6", args=vars(a), confirmatory=confirmatory,
                 started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 versions={k: __import__(k).__version__ for k in ("torch", "transformers", "tokenizers", "rdkit", "numpy", "scipy")},
                 cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
                 script_sha256={f: sha(os.path.join(HERE, f)) for f in SCRIPTS},
                 model_file_sha256={f: sha(os.path.join(mf, f)) for f in MODEL_FILES},
                 input_sha256=dict(tsv=sha(a.tsv), labels=sha(os.path.join(a.cdir, f"labels_{a.tag}.npz")),
                                   concepts=sha(os.path.join(a.cdir, "concepts_v1.json"))),
                 seeds=dict(twins=[TWIN_SEED0, TWIN_SEED0 + K - 1], twins_hf=[HF_SEED0, HF_SEED0 + K_hf - 1],
                            pretrained_orf=[ORF_SEED0, ORF_SEED0 + K_orf - 1], blind_twin=BLIND_SEED if a.blind else None,
                            rotations=[ROT_SEED0, ROT_SEED0 + R - 1]))
    freeze_dir = a.freeze_dir or (os.path.join(ROOT, "lineB") if confirmatory else None)
    if confirmatory:
        assert not a.blind and a.n_mol is None, "the confirmatory run on H4 takes neither --blind nor --n_mol"
        assert (K, R, K_hf, K_orf) == tuple(PROTOCOL_DESIGN[k] for k in ("K", "R", "K_hf", "K_orf")), \
            "the confirmatory run must use the protocol's K, R, K_hf and K_orf"
        assert os.path.abspath(freeze_dir) == os.path.abspath(os.path.join(ROOT, "lineB")), "confirmatory freeze dir is lineB/"
        assert os.path.abspath(a.out_root) == os.path.abspath(os.path.join(ROOT, "lineB", "results_v6")), \
            "the confirmatory run writes to lineB/results_v6"
    if freeze_dir:
        fz = verify_freeze(freeze_dir)
        if confirmatory:
            ent = fz["entries"]
            listed = [l.strip() for l in open(FREEZE_LIST, encoding="utf-8") if l.strip()]
            assert sorted(ent) == sorted(listed), "the manifest does not hold exactly the listed files"
            for key, rel in (("tsv", "data/derived/lineB/corpus/H4_used.tsv"), ("labels", "data/derived/lineB/concepts/labels_H4.npz"),
                             ("concepts", "data/derived/lineB/concepts/concepts_v1.json")):
                assert start["input_sha256"][key] == ent[rel], f"input {key} is not the frozen file {rel}"
            for f in SCRIPTS:
                assert start["script_sha256"][f] == ent[f"scripts/lineB/{f}"], f"script {f} differs from the frozen one"
            det = json.load(open(os.path.join(ROOT, "lineB", "results_v6", "determinism_check.json")))
            assert det.get("passed") is True, "determinism check did not pass"
        start["freeze"] = {k: v for k, v in fz.items() if k != "entries"}
    rec_path = os.path.join(base, "run_record.json")
    record = json.load(open(rec_path)) if os.path.exists(rec_path) else {"starts": []}
    record["starts"].append(start)
    write_json(rec_path, record)
    print(json.dumps(start, indent=1), flush=True)
    with open(a.tsv, encoding="utf-8") as fh:
        smiles = [r["smiles"] for r in csv.DictReader(fh, delimiter="\t")]
    concepts = json.load(open(os.path.join(a.cdir, "concepts_v1.json"), encoding="utf-8"))
    Lsp = sp.load_npz(os.path.join(a.cdir, f"labels_{a.tag}.npz")).tocsr()
    if a.n_mol is not None:
        atoms = np.load(os.path.join(a.cdir, f"atoms_{a.tag}.npz"))
        rows = np.nonzero(atoms["mol_idx"] < a.n_mol)[0]
        assert rows.max() == len(rows) - 1, "atoms are not in molecule order"
        smiles, Lsp = smiles[:a.n_mol], Lsp[rows]
    Lsp = Lsp.tocsc()
    N = Lsp.shape[0]
    if confirmatory:
        assert (N, len(smiles)) == (PROTOCOL_DESIGN["N_atoms"], PROTOCOL_DESIGN["N_mol"]), (N, len(smiles))
    pos = np.asarray(Lsp.sum(0)).ravel()
    rd = np.array([c["id"] for c in concepts if c["partition"] == "readable" and pos[c["id"]] > 0])
    print(f"molecules {len(smiles)}, atoms {N}, readable concepts with positives {len(rd)} of 173", flush=True)
    np.save(os.path.join(base, "readable_concept_ids.npy"), rd)
    P = lift.Positives(torch.from_numpy(Lsp[:, rd].toarray().astype(np.uint8)).cuda())
    prev = pos[rd] / N
    rots = {}

    def rot(r):
        if r not in rots:
            rots[r] = torch.from_numpy(rotation(WIDTH, ROT_SEED0 + r).astype(np.float32)).cuda()
        return rots[r]

    def path(sub, layer, name):
        return os.path.join(base, sub, layer, f"{name}.npy")

    def save(sub, layer, name, arr):
        os.makedirs(os.path.join(base, sub, layer), exist_ok=True)
        tmp = path(sub, layer, name)[:-4] + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path(sub, layer, name))

    def lifts(Z):
        return lift.lift_array_batched(Z, P, prev, batch=a.lift_batch, act=ACT)

    fp_path = os.path.join(base, "fingerprints.json")
    fps = json.load(open(fp_path)) if os.path.exists(fp_path) else {}

    def build(name, seed, init):
        if init == "orf":
            m, _ = models.load_model("molformer", molformer_orf_seed=seed)
        elif seed is None:
            m, _ = models.load_model("molformer")
        else:
            m, _ = models.load_model("molformer", twin_seed=seed, molformer_init=init)
        fp = fingerprint(m)
        if name in fps:
            assert fps[name]["state_dict_sha256_16"] == fp, f"{name}: rebuilt network differs from its first build"
        else:
            fps[name] = dict(seed=seed, init=init, state_dict_sha256_16=fp)
            write_json(fp_path, fps)
        return m

    nets = network_list(K, K_hf, K_orf, a.blind)
    done = 0
    for sub, name, seed, init, with_rot in nets:
        k = None if sub == "pretrained" else name.split("_")[1]
        nat = lambda l: path(sub, l, "S_native" if k is None else f"S_native_{k}")
        rt = lambda l, r: path(sub, l, f"S_rot_{r}" if k is None else f"S_rot_{k}_{r}")
        need = [nat(l) for l in LAYERS] + ([rt(l, r) for l in LAYERS for r in range(R)] if with_rot else [])
        if all(os.path.exists(p) for p in need):
            print(f"  skip {name} (present)", flush=True)
            continue
        t1 = time.time()
        m = build(name, seed, init)
        Zs = models.extract_multi(m, "molformer", smiles, LAYERS, batch_size=256)
        del m
        for l in LAYERS:
            Z = Zs.pop(l)
            assert Z.shape == (N, WIDTH), (Z.shape, N)
            if not os.path.exists(nat(l)):
                full = lifts(Z)
                if sub == "pretrained":
                    save(sub, l, "pre_native_full", full.astype(np.float32))
                save(sub, l, os.path.basename(nat(l))[:-4], full.max(axis=(0, 1)).astype(np.float64))
            if with_rot:
                for r in range(R):
                    if not os.path.exists(rt(l, r)):
                        save(sub, l, os.path.basename(rt(l, r))[:-4], lifts(Z @ rot(r)).max(axis=(0, 1)).astype(np.float64))
            del Z
            torch.cuda.empty_cache()
        done += 1
        print(f"  {name} done ({'natives + rotations' if with_rot else 'natives'}) in {time.time() - t1:.0f}s; "
              f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB; {time.time() - t0:.0f}s total", flush=True)
    record = json.load(open(rec_path))
    record["completed"] = dict(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seconds=time.time() - t0,
                               networks=len(nets), computed_this_start=done, K=K, R=R, K_hf=K_hf,
                               K_orf=0 if a.blind else K_orf, blind=a.blind, N_atoms=int(N), N_mol=len(smiles),
                               peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 2**30)
    write_json(rec_path, record)
    print(f"molformer v6: complete, K={K}, R={R}, K_hf={K_hf}, K_orf={0 if a.blind else K_orf}, blind={a.blind}, "
          f"readable {len(rd)}, atoms {N}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
