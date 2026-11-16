"""
Export the reproducible data pipeline (code + raw inputs + reference checksums) into zenodo_release/pipeline/.
The repository `src/` and `scripts/` are canonical; this exporter copies them verbatim and records SHA-256
hashes in CODE_MANIFEST.json so that the released code can be traced to the repository version.

    python scripts/export_release_pipeline.py
"""
import os
import sys
import glob
import json
import shutil
import hashlib
import zipfile
import datetime
import importlib.metadata as md

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src", "data_generation", "base"))
import curate_spectral_library as cur  # noqa: E402

OUT = os.path.join(ROOT, "zenodo_release", "pipeline")
CODE = [
    "run_pipeline.py",
    "src/config.py",
    "src/data_generation/base/curate_spectral_library.py",
    "src/data_generation/base/make_split.py",
    "src/data_generation/base/automate_modtran.py",
    "src/data_generation/base/synthesize_forward_dataset.py",
    "scripts/build_dataset_facts.py",
    "scripts/validate_dataset_integrity.py",
    "modtran/05vs15nadirRAD.ltn",
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for b in iter(lambda: fp.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    manifest = {"exported_at": datetime.datetime.now().isoformat(timespec="seconds"), "files": {}}

    for rel in CODE:
        dst = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), dst)
        manifest["files"][rel] = sha256(dst)

    # raw spectra actually read by the curator (same keyword logic as curate_rebalanced_library)
    raw_dir = os.path.join(ROOT, "data", "raw", "ecospeclib", "ecospeclib-all")
    kws = [k for t in cur.TARGET_DEFS for k in t["keywords"]] + list(cur.BOTANICAL_KEYWORDS)
    files = [f for f in glob.glob(os.path.join(raw_dir, "**", "*.*"), recursive=True)
             if f.lower().endswith((".spectrum.txt", ".txt")) and not f.lower().endswith(".ancillary.txt")
             and any(k in os.path.basename(f).lower() for k in kws)]
    inp = os.path.join(OUT, "inputs")
    os.makedirs(inp)
    with zipfile.ZipFile(os.path.join(inp, "spectral_library_raw.zip"), "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(files):
            zf.write(f, os.path.relpath(f, raw_dir))
    manifest["raw_spectra_files"] = len(files)

    os.makedirs(os.path.join(inp, "srf", "aster"))
    for f in sorted(glob.glob(os.path.join(ROOT, "bibliografic_references", "aster", "*_ch1[1-5].txt"))):
        shutil.copy(f, os.path.join(inp, "srf", "aster"))
    os.makedirs(os.path.join(inp, "srf", "hytes"))
    shutil.copy(os.path.join(ROOT, "data", "processed", "hytes_real_parameters.csv"), os.path.join(inp, "srf", "hytes"))
    os.makedirs(os.path.join(inp, "atmospheric_luts"))
    for f in glob.glob(os.path.join(ROOT, "data", "processed", "atmospheric_luts", "*")):
        if f.endswith((".csv", ".npz")):
            shutil.copy(f, os.path.join(inp, "atmospheric_luts"))
    for f in sorted(glob.glob(os.path.join(inp, "**", "*"), recursive=True)):
        if os.path.isfile(f):
            manifest["files"][os.path.relpath(f, OUT).replace("\\", "/")] = sha256(f)

    # MODTRAN decks actually executed (archived by automate_modtran.py --archive-decks) + reference tape7.scn
    cards_src = os.path.join(ROOT, "data", "processed", "modtran_cards")
    if os.path.isdir(cards_src):
        cards_dst = os.path.join(inp, "modtran_cards")
        os.makedirs(os.path.join(cards_dst, "tape5"))
        for f in glob.glob(os.path.join(cards_src, "tape5", "*.tp5")):
            shutil.copy(f, os.path.join(cards_dst, "tape5"))
        for name in ("deck_index.csv", "README_MODTRAN_CARDS.md"):
            shutil.copy(os.path.join(cards_src, name), cards_dst)
        with zipfile.ZipFile(os.path.join(cards_dst, "tape7_reference.zip"), "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for f in sorted(glob.glob(os.path.join(cards_src, "tape7_reference", "*.7sc"))):
                zf.write(f, os.path.basename(f))
        for f in sorted(glob.glob(os.path.join(cards_dst, "**", "*"), recursive=True)):
            if os.path.isfile(f):
                manifest["files"][os.path.relpath(f, OUT).replace("\\", "/")] = sha256(f)
        # discoverable copy next to the published LUTs (decks + index + documentation, without tape7 outputs)
        pub = os.path.join(ROOT, "zenodo_release", "data", "atmospheric_luts", "modtran_cards")
        if os.path.isdir(pub):
            shutil.rmtree(pub)
        shutil.copytree(os.path.join(cards_dst, "tape5"), os.path.join(pub, "tape5"))
        for name in ("deck_index.csv", "README_MODTRAN_CARDS.md"):
            shutil.copy(os.path.join(cards_dst, name), pub)
        manifest["modtran_decks"] = len(glob.glob(os.path.join(cards_dst, "tape5", "*.tp5")))

    # N4 interpolation check grid (24 MODTRAN profiles at intermediate H2O scales + their decks), read by test B8
    chk_src = os.path.join(ROOT, "data", "validation", "n4_interp_check")
    if os.path.isdir(chk_src):
        chk_dst = os.path.join(inp, "n4_interp_check")
        shutil.copytree(chk_src, chk_dst, ignore=shutil.ignore_patterns("tape7_reference"))
        for f in sorted(glob.glob(os.path.join(chk_dst, "**", "*"), recursive=True)):
            if os.path.isfile(f):
                manifest["files"][os.path.relpath(f, OUT).replace("\\", "/")] = sha256(f)

    # reference outputs (canonical build) for the reproducibility stage
    import pandas as pd
    facts = json.load(open(os.path.join(ROOT, "data", "dataset_facts.json"), encoding="utf-8"))
    cat = pd.read_csv(os.path.join(ROOT, "data", "processed", "spectral_libraries", "library_metadata_catalog.csv"))
    split = json.load(open(os.path.join(ROOT, "splits", "split_v1.1.json"), encoding="utf-8"))
    ref = {"dataset_version": facts["dataset_version"],
           "library_emissivity_hashes": dict(zip(cat.spectrum_id, cat.emissivity_hash)),
           "split_val_spectrum_ids": [sorted(f["val_spectrum_ids"]) for f in split["folds"]],
           "canonical_counts_fold0": {k: v for k, v in facts["canonical_counts_fold0"].items() if k != "plain_language"},
           "partition_md5": {s: {p: d["md5"] for p, d in parts.items()} for s, parts in facts["partitions"].items()}}
    json.dump(ref, open(os.path.join(OUT, "REFERENCE_CHECKSUMS.json"), "w", encoding="utf-8"), indent=1)

    pkgs = ["numpy", "pandas", "scipy", "torch", "scikit-learn", "tqdm"]
    with open(os.path.join(OUT, "requirements.txt"), "w") as fp:
        fp.write("# Versions used to build the reference release\n")
        for p in pkgs:
            fp.write(f"{p}=={md.version(p)}\n")
    json.dump(manifest, open(os.path.join(OUT, "CODE_MANIFEST.json"), "w"), indent=1)
    shutil.copy(os.path.join(ROOT, "docs", "PIPELINE_README.md"), os.path.join(OUT, "README.md"))
    shutil.copy(os.path.join(ROOT, "docs", "dataset_validation_protocol.md"), os.path.join(OUT, "dataset_validation_protocol.md"))
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(f))
    print(f"exported {len(CODE)} code files, {len(files)} raw spectra, {len(manifest['files'])} hashed files, "
          f"{size / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()

