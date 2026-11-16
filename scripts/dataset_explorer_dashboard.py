"""
OpenTIR-PINN Benchmark: Interactive Spectral Dataset Explorer & Curation Auditor
==================================================================================
Author: Dr. Atilio Grondona (UFRGS) & GeoAI Team
Associated Publication: IEEE Transactions on Geoscience and Remote Sensing (TGRS)
Zenodo Benchmark DOI: 10.5281/zenodo.23966244 | Version: 1.1.1 (Double-Quarantine Suite)

Interactive Streamlit application for exploring laboratory thermal infrared (TIR) spectra,
validating physical emissivity/reflectance derivations, inspecting sensor band convolutions (HyTES & ASTER),
auditing the 3rd Validation Gate (quarantined transmission / KBr pellet measurements),
visualizing MODTRAN 5.2 atmospheric look-up tables, and analyzing synthetic forward training/validation samples
with dynamic component mixing.

Launch instructions:
    streamlit run dataset_explorer_dashboard.py
"""

import os
import sys
import glob
import json
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import torch
except ImportError:
    torch = None

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OpenTIR-PINN Spectral & Dataset Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #f1f5f9;
        font-size: 0.92rem;
        line-height: 1.6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-card b {
        color: #93c5fd;
    }
    .metric-card code {
        background-color: #1e293b;
        color: #38bdf8;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #334155;
    }
    .badge-golden {
        background-color: rgba(245, 158, 11, 0.22);
        color: #fcd34d;
        border: 1px solid #f59e0b;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.82rem;
    }
    .badge-dev {
        background-color: rgba(59, 130, 246, 0.22);
        color: #bfdbfe;
        border: 1px solid #3b82f6;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.82rem;
    }
    .badge-quar {
        background-color: rgba(239, 68, 68, 0.22);
        color: #fca5a5;
        border: 1px solid #ef4444;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.82rem;
    }
    .spec-table-container {
        overflow-x: auto;
        border: 1px solid #334155;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .spec-table {
        width: 100%;
        border-collapse: collapse;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 0.86rem;
        background-color: #0f172a;
        color: #f1f5f9;
        text-align: left;
    }
    .spec-table th {
        background-color: #1e293b;
        color: #94a3b8;
        font-weight: 600;
        padding: 10px 14px;
        border-bottom: 2px solid #334155;
        white-space: nowrap;
    }
    .spec-table td {
        padding: 9px 14px;
        border-bottom: 1px solid #1e293b;
        vertical-align: middle;
        white-space: nowrap;
    }
    .spec-table tr:hover {
        background-color: rgba(51, 65, 85, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Robust Multi-Path Asset Resolution
# -----------------------------------------------------------------------------
@st.cache_resource
def resolve_dataset_paths():
    """Locates dataset assets whether running from repo root or zenodo_release/ folder."""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # Inside zenodo_release
        {
            "catalog": os.path.join(curr_dir, "data", "processed", "spectral_libraries", "library_metadata_catalog.csv"),
            "npz": os.path.join(curr_dir, "data", "processed", "spectral_libraries", "rebalanced_spectral_library.npz"),
            "excluded_dir": os.path.join(curr_dir, "data", "raw_laboratory_spectra", "excluded_data"),
            "golden_dir": os.path.join(curr_dir, "data", "raw_laboratory_spectra", "golden_targets"),
            "srf_dir": os.path.join(curr_dir, "data", "srf"),
            "atm_dir": os.path.join(curr_dir, "data", "atmospheric_luts"),
            "atm_catalog": os.path.join(curr_dir, "data", "atmospheric_luts", "lut_catalog.csv"),
            "splits_json": os.path.join(curr_dir, "data", "splits", "split_v1.1.json"),
            "forward_aster_npz": os.path.join(curr_dir, "data", "aster_5band_sobolev_pinn_dataset.npz"),
            "forward_hytes_npz": os.path.join(curr_dir, "data", "hytes_256band_sobolev_pinn_dataset.npz"),
            "final_datasets_dir": os.path.join(curr_dir, "..", "data", "processed", "final_datasets")
        },
        # From repo root
        {
            "catalog": os.path.join(curr_dir, "..", "data", "processed", "spectral_libraries", "library_metadata_catalog.csv"),
            "npz": os.path.join(curr_dir, "..", "data", "processed", "spectral_libraries", "rebalanced_spectral_library.npz"),
            "excluded_dir": os.path.join(curr_dir, "..", "zenodo_release", "data", "raw_laboratory_spectra", "excluded_data"),
            "golden_dir": os.path.join(curr_dir, "..", "zenodo_release", "data", "raw_laboratory_spectra", "golden_targets"),
            "srf_dir": os.path.join(curr_dir, "..", "zenodo_release", "data", "srf"),
            "atm_dir": os.path.join(curr_dir, "..", "zenodo_release", "data", "atmospheric_luts"),
            "atm_catalog": os.path.join(curr_dir, "..", "zenodo_release", "data", "atmospheric_luts", "lut_catalog.csv"),
            "splits_json": os.path.join(curr_dir, "..", "splits", "split_v1.1.json"),
            "forward_aster_npz": os.path.join(curr_dir, "..", "zenodo_release", "data", "aster_5band_sobolev_pinn_dataset.npz"),
            "forward_hytes_npz": os.path.join(curr_dir, "..", "zenodo_release", "data", "hytes_256band_sobolev_pinn_dataset.npz"),
            "final_datasets_dir": os.path.join(curr_dir, "..", "data", "processed", "final_datasets")
        },
        # Direct absolute fallback
        {
            "catalog": os.path.abspath(r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\data\processed\spectral_libraries\library_metadata_catalog.csv"),
            "npz": os.path.abspath(r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\data\processed\spectral_libraries\rebalanced_spectral_library.npz"),
            "excluded_dir": os.path.abspath(r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\zenodo_release\data\raw_laboratory_spectra\excluded_data"),
            "golden_dir": os.path.abspath(r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\zenodo_release\data\raw_laboratory_spectra\golden_targets"),
            "srf_dir": os.path.abspath(r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\zenodo_release\data\srf"),
            "atm_dir": os.path.abspath(r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\zenodo_release\data\atmospheric_luts"),
            "atm_catalog": os.path.abspath(r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\zenodo_release\data\atmospheric_luts\lut_catalog.csv"),
            "splits_json": os.path.abspath(r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\splits\split_v1.1.json"),
            "forward_aster_npz": os.path.abspath(r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\zenodo_release\data\aster_5band_sobolev_pinn_dataset.npz"),
            "forward_hytes_npz": os.path.abspath(r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\zenodo_release\data\hytes_256band_sobolev_pinn_dataset.npz"),
            "final_datasets_dir": os.path.abspath(r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\data\processed\final_datasets")
        }
    ]

    for p in candidates:
        if os.path.exists(p["catalog"]) and os.path.exists(p["npz"]):
            return p

    raise FileNotFoundError("Could not automatically locate library_metadata_catalog.csv and rebalanced_spectral_library.npz.")


def planck_radiance(wavelengths_um, temp_k):
    """Computes spectral blackbody radiance in W/(m^2 sr um) via Planck's Law."""
    c1 = 1.19104282e8  # 2*h*c^2 in W*um^4/(m^2*sr)
    c2 = 14387.77      # h*c/k in um*K
    w = np.asarray(wavelengths_um, dtype=np.float64)
    exp_term = np.exp(c2 / (w * float(temp_k))) - 1.0
    return c1 / (w**5 * exp_term)


@st.cache_data
def load_atmospheric_catalog(catalog_path: str):
    """Loads catalog of 30 MODTRAN 5.2 profiles."""
    if os.path.exists(catalog_path):
        return pd.read_csv(catalog_path)
    return None


@st.cache_data
def load_atmospheric_profile(filepath: str):
    """Loads a high-resolution atmospheric profile and sorts ascending by wavelength."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df.sort_values('wavelength_um').reset_index(drop=True)
    return None


@st.cache_data
def load_spectral_data():
    paths = resolve_dataset_paths()
    catalog_df = pd.read_csv(paths["catalog"])
    npz_data = np.load(paths["npz"], allow_pickle=True)
    wavelengths = npz_data["wavelengths_um"] if "wavelengths_um" in npz_data else npz_data["wavelengths"]
    emissivities = npz_data["emissivity"]

    # Annotate with split assignments if split_v1.1.json exists
    split_path = paths.get("splits_json", "")
    if os.path.exists(split_path):
        try:
            with open(split_path, "r") as f:
                split_info = json.load(f)

            fold_map = {}
            for fold_info in split_info.get("folds", []):
                f_idx = fold_info["fold_idx"]
                for sid in fold_info.get("train_spectrum_ids", []):
                    fold_map.setdefault(sid, {"train": [], "val": []})["train"].append(f"Fold {f_idx}")
                for sid in fold_info.get("val_spectrum_ids", []):
                    fold_map.setdefault(sid, {"train": [], "val": []})["val"].append(f"Fold {f_idx}")

            def get_split_role(row):
                if row["is_golden_target"]:
                    return "🏆 Golden Target (Benchmark OOD)"
                if row.get("is_sibling_quarantined", False):
                    return "🛡️ Quarantined Sibling (Control)"
                return "🧪 Development Pool"

            def get_val_fold(row):
                sid = row["spectrum_id"]
                if sid in fold_map and fold_map[sid]["val"]:
                    return ", ".join(fold_map[sid]["val"])
                return "None"

            def get_train_folds(row):
                sid = row["spectrum_id"]
                if sid in fold_map and fold_map[sid]["train"]:
                    return ", ".join(fold_map[sid]["train"])
                return "None"

            catalog_df["benchmark_partition"] = catalog_df.apply(get_split_role, axis=1)
            catalog_df["validation_fold"] = catalog_df.apply(get_val_fold, axis=1)
            catalog_df["training_folds"] = catalog_df.apply(get_train_folds, axis=1)
        except Exception:
            pass

    return paths, catalog_df, wavelengths, emissivities


@st.cache_resource
def load_forward_dataset(sensor: str, partition_key: str, _paths: dict):
    """Loads simulated forward radiance observations for ASTER or HyTES across benchmark partitions."""
    npz_path = _paths.get("forward_aster_npz" if sensor == "aster" else "forward_hytes_npz", "")
    if not os.path.exists(npz_path):
        return None, None
    npz_data = np.load(npz_path, allow_pickle=True)
    wavelengths = npz_data["wavelengths"]

    if partition_key == "q2":
        prefix = "val_q2_"
        name_tag = "Golden Targets — Quadrant II (Dev Atmospheres)"
    elif partition_key == "q4":
        prefix = "val_q4_"
        name_tag = "Golden Targets — Quadrant IV (Severe Polar Winter OOD)"
    elif partition_key == "train":
        prefix = "train_"
        name_tag = "Master Training Set (Dev Pool × Dev Atmospheres)"
    elif partition_key == "val_oof":
        if "val_oof_Ts" in npz_data:
            prefix = "val_oof_"
            name_tag = "Quadrant I — Out-of-Fold Validation Set (Unseen Dev Validation | 2,024 samples)"
        else:
            pt_path = os.path.join(_paths.get("final_datasets_dir", ""), f"val_dataset_{sensor}.pt")
            if os.path.exists(pt_path) and torch is not None:
                pt_data = torch.load(pt_path, weights_only=False)
                df = pd.DataFrame([{
                    'sample_idx': i,
                    'spectrum_id': s.get('spectrum_id', 'unknown'),
                    'material_class': s.get('material_class', 'unknown'),
                    'target_name': s.get('golden_name', s.get('material_class', 'unknown')),
                    'atmosphere_id': s.get('atmosphere_id', 'unknown'),
                    'pwv': float(s.get('pwv', 0.0)),
                    'ts_k': float(s['target_temp'].item()),
                    'mean_emissivity': float(s['target_emissivity'].mean().item()),
                    'mean_l_sensor': float(s['input_radiance_sensor'].mean().item()),
                    'mean_l_surf': float(s['input_radiance'].mean().item()),
                    'mean_l_down': float(s['downwelling_radiance'].mean().item())
                } for i, s in enumerate(pt_data)])
                return df, {'type': 'pt', 'data': pt_data, 'wavelengths': wavelengths, 'name': 'Quadrant I — Out-of-Fold Validation Set'}
            else:
                return None, None
    else:
        return None, None

    n_samples = len(npz_data[prefix + "Ts"])
    df = pd.DataFrame({
        'sample_idx': np.arange(n_samples),
        'material_class': npz_data[prefix + "materials"],
        'target_name': npz_data[prefix + "golden_name"] if (prefix + "golden_name") in npz_data else npz_data[prefix + "materials"],
        'atmosphere_id': npz_data[prefix + "atmosphere_id"] if (prefix + "atmosphere_id") in npz_data else "atm_modtran",
        'pwv': npz_data[prefix + "pwv"] if (prefix + "pwv") in npz_data else 0.0,
        'ts_k': npz_data[prefix + "Ts"],
        'mean_emissivity': np.mean(npz_data[prefix + "emissivity"], axis=1),
        'mean_l_sensor': np.mean(npz_data[prefix + "L_toa"], axis=1),
        'mean_l_surf': np.mean(npz_data[prefix + "L_obs"], axis=1),
        'mean_l_down': np.mean(npz_data[prefix + "L_down"], axis=1)
    })
    return df, {'type': 'npz', 'data': npz_data, 'prefix': prefix, 'wavelengths': wavelengths, 'name': name_tag}


@st.cache_data
def load_excluded_data(excluded_dir: str):
    """Parses raw spectra from the 3rd Validation Gate excluded directory."""
    if not os.path.exists(excluded_dir):
        return []

    files = glob.glob(os.path.join(excluded_dir, "*.txt"))
    excluded_records = []
    for f in sorted(files):
        fname = os.path.basename(f)
        if fname.lower().startswith("readme"):
            continue

        meta = {
            'filename': fname,
            'name': 'Unknown',
            'sample_id': 'Unknown',
            'measurement': 'Unknown',
            'particle_size': 'Fine' if 'fine' in fname.lower() else ('Coarse' if 'coarse' in fname.lower() else 'Solid'),
            'filepath': f
        }

        wavs, vals = [], []
        with open(f, 'r', encoding='latin-1') as fp:
            for line in fp:
                ll = line.lower().strip()
                if ll.startswith("name:"):
                    meta['name'] = line.split(":", 1)[1].strip()
                elif ll.startswith("sample no.") or ll.startswith("sample:"):
                    meta['sample_id'] = line.split(":", 1)[1].strip()
                elif "measurement:" in ll:
                    meta['measurement'] = line.split(":", 1)[1].strip()

                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        w = float(parts[0])
                        v = float(parts[1])
                        wavs.append(w)
                        vals.append(v)
                    except ValueError:
                        continue

        if len(wavs) > 10:
            wavs = np.array(wavs)
            vals = np.array(vals)
            sidx = np.argsort(wavs)
            wavs = wavs[sidx]
            vals = vals[sidx]
            if vals.max() > 1.5:
                vals = vals / 100.0

            naive_emis = 1.0 - vals
            excluded_records.append({
                'meta': meta,
                'wavs': wavs,
                'raw_vals': vals,
                'naive_emissivity': naive_emis
            })

    return excluded_records


# -----------------------------------------------------------------------------
# 3. Main Data Ingestion
# -----------------------------------------------------------------------------
paths, catalog_df, wavelengths, emissivities = load_spectral_data()
excluded_records = load_excluded_data(paths["excluded_dir"])
atm_catalog_df = load_atmospheric_catalog(paths.get("atm_catalog", ""))

# Global Master Class & Size Lists
all_classes = sorted(catalog_df["material_class"].dropna().unique().tolist())
all_sizes = sorted(catalog_df["particle_size"].dropna().unique().tolist()) if "particle_size" in catalog_df.columns else ["coarse", "fine", "solid", "natural"]

# -----------------------------------------------------------------------------
# 4. Sidebar Controls & Multi-Tier Filters
# -----------------------------------------------------------------------------
st.sidebar.image("https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg", width=40)
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("**OpenTIR-PINN Benchmark (v1.1.1)**")
st.sidebar.markdown("---")

# Display Modality Toggle
st.sidebar.subheader("1. Measurement Modality")
view_mode = st.sidebar.radio(
    "Select Display Curve:",
    options=["Directional Emissivity (ε)", "Hemispherical Reflectance (ρ)", "Dual Synchronized View (ε & ρ)"],
    index=0,
    help="Toggle between Emissivity (ε) and Reflectance (ρ = 1 - ε) via Kirchhoff's Law for opaque solids."
)

# Comprehensive Partition Filter
st.sidebar.subheader("2. Benchmark Partition")
partition_options = [
    "All Partitions (241 Spectra)",
    "🏆 Golden Targets (7 Strict OOD Benchmarks)",
    "🛡️ Quarantined Siblings (1 Sibling Control)",
    "🧪 Development Pool (All 233 Train/Val Spectra)",
    "📂 Fold 0 — Train Split (187 Spectra)",
    "📂 Fold 0 — Val Split (46 Spectra)",
    "📂 Fold 1 — Train Split (187 Spectra)",
    "📂 Fold 1 — Val Split (46 Spectra)",
    "📂 Fold 2 — Train Split (186 Spectra)",
    "📂 Fold 2 — Val Split (47 Spectra)",
    "📂 Fold 3 — Train Split (186 Spectra)",
    "📂 Fold 3 — Val Split (47 Spectra)",
    "📂 Fold 4 — Train Split (186 Spectra)",
    "📂 Fold 4 — Val Split (47 Spectra)",
]
partition_choice = st.sidebar.selectbox("Filter by Partition:", partition_options, index=0)

# 2.1 Dynamic Partition Pre-Filtering (Cascading Filter Engine)
if partition_choice.startswith("🏆 Golden Targets"):
    part_df = catalog_df[catalog_df["is_golden_target"] == True].copy()
elif partition_choice.startswith("🛡️ Quarantined Siblings"):
    part_df = catalog_df[catalog_df["is_sibling_quarantined"] == True].copy()
elif partition_choice.startswith("🧪 Development Pool"):
    part_df = catalog_df[(catalog_df["is_golden_target"] == False) & (catalog_df["is_sibling_quarantined"] == False)].copy()
elif "Fold" in partition_choice and "Train" in partition_choice:
    fold_str = partition_choice.split("—")[0].replace("📂", "").strip()
    if "training_folds" in catalog_df.columns:
        part_df = catalog_df[catalog_df["training_folds"].str.contains(fold_str, na=False)].copy()
    else:
        part_df = catalog_df.copy()
elif "Fold" in partition_choice and "Val" in partition_choice:
    fold_str = partition_choice.split("—")[0].replace("📂", "").strip()
    if "validation_fold" in catalog_df.columns:
        part_df = catalog_df[catalog_df["validation_fold"].str.contains(fold_str, na=False)].copy()
    else:
        part_df = catalog_df.copy()
else:
    part_df = catalog_df.copy()

# Material Class Multi-Select (Dynamically constrained by selected partition)
st.sidebar.subheader("3. Material Classes")
avail_classes = sorted(part_df["material_class"].unique().tolist())
selected_classes = st.sidebar.multiselect(
    "Select Classes:",
    options=avail_classes,
    default=avail_classes,
    help="Filter by diagnostic mineralogical or surface material classes (dynamically cascaded from partition selection)."
)

# Granulometry Filter (Dynamically constrained by selected partition)
st.sidebar.subheader("4. Particle Size / Texture")
avail_sizes = sorted(part_df["particle_size"].dropna().unique().tolist()) if "particle_size" in part_df.columns else ["coarse", "fine", "solid", "natural"]
selected_sizes = st.sidebar.multiselect(
    "Particle Size:",
    options=avail_sizes,
    default=avail_sizes,
    help="Filter by grain size / texture (dynamically cascaded from partition selection)."
)

# Spectral Window Range Slider
st.sidebar.subheader("5. Spectral Range (TIR)")
min_w, max_w = float(wavelengths.min()), float(wavelengths.max())
spectral_range = st.sidebar.slider(
    "Wavelength (μm):",
    min_value=min_w,
    max_value=max_w,
    value=(7.50, 13.50),
    step=0.05
)

# Apply cascaded filters
filtered_df = part_df.copy()
if selected_classes:
    filtered_df = filtered_df[filtered_df["material_class"].isin(selected_classes)]
if selected_sizes and "particle_size" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["particle_size"].isin(selected_sizes)]

# Filter spectral grid mask
wave_mask = (wavelengths >= spectral_range[0]) & (wavelengths <= spectral_range[1])
sub_wavelengths = wavelengths[wave_mask]

# -----------------------------------------------------------------------------
# 5. Top Header & KPI Summary Banner
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🛰️ OpenTIR-PINN Spectral & Dataset Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header"><b>Physics-Informed Radiative Transfer & Double-Quarantine Benchmark</b> | '
    'Interactive Analytical Dashboard for Laboratory Spectra, Atmospheric Profiles & Forward Training Samples</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Curated Spectra", f"{len(catalog_df)}")
    st.caption("✨ Pristine Laboratory FTIR")
with col2:
    st.metric("Golden Targets (OOD)", f"{catalog_df['is_golden_target'].sum()}")
    st.caption("🛡️ Immutable Benchmarks")
with col3:
    st.metric("Development Pool", f"{(~catalog_df['is_golden_target'] & ~catalog_df['is_sibling_quarantined']).sum()}")
    st.caption("🔄 5-Fold Partitioned")
with col4:
    st.metric("Atmospheric LUTs", "30 Profiles")
    st.caption("🌤️ MODTRAN 5.2")
with col5:
    st.metric("Forward Coupled Samples", "17,204")
    st.caption("🛰️ ASTER & HyTES (4 Quadrants)")

# -----------------------------------------------------------------------------
# 6. Main Navigation Tabs (7 Comprehensive Tabs)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Spectral Explorer",
    "🏆 Golden Targets Suite",
    "🔬 Multi-Spectrum Comparison",
    "⚠️ 3rd Validation Gate & Forensic Audit",
    "🌤️ Atmospheric LUTs & Dynamic RTE Studio",
    "📦 Forward Datasets & Sample-Level Inspector",
    "📑 Metadata Catalog & FAIR Table",
    "✅ Dataset Validation Report"
])


# -----------------------------------------------------------------------------
# TAB 1: Spectral Explorer
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Interactive Spectral Curve Visualizer")
    st.caption("Select one or more curated laboratory spectra to analyze continuous FTIR profiles, Reststrahlen bands, and Christiansen frequencies.")

    if len(filtered_df) == 0:
        st.warning("No spectra match the current filter selection in the sidebar.")
    else:
        spectrum_options = filtered_df["spectrum_id"].tolist()
        spec_label_dict = {
            r["spectrum_id"]: f"[{r['material_class'].upper()}] {r['spectrum_id']} - {r['name']} ({r['sample_id']}) {'[GOLDEN]' if r['is_golden_target'] else ''}"
            for _, r in filtered_df.iterrows()
        }

        default_selection = [spectrum_options[0]] if spectrum_options else []
        selected_spectras = st.multiselect(
            "Select Spectra to Plot:",
            options=spectrum_options,
            default=default_selection,
            format_func=lambda x: spec_label_dict.get(x, x)
        )

        col_srf1, col_srf2 = st.columns(2)
        with col_srf1:
            show_aster_srf = st.checkbox("Overlay ASTER 5-Band Filter Windows (SRF)", value=True)
        with col_srf2:
            show_hytes_srf = st.checkbox("Overlay HyTES 256-Channel Spectral Envelope (SRF)", value=False)

        if selected_spectras:
            fig = go.Figure()

            for spec_id in selected_spectras:
                row = catalog_df[catalog_df["spectrum_id"] == spec_id].iloc[0]
                idx = row.name
                emis_full = emissivities[idx]
                refl_full = 1.0 - emis_full

                sub_emis = emis_full[wave_mask]
                sub_refl = refl_full[wave_mask]

                label = f"{row['material_class']} | {row['name']} ({row['sample_id']})"

                if view_mode == "Directional Emissivity (ε)":
                    fig.add_trace(go.Scatter(
                        x=sub_wavelengths, y=sub_emis, mode="lines",
                        name=f"ε: {label}", line=dict(width=2.2)
                    ))
                elif view_mode == "Hemispherical Reflectance (ρ)":
                    fig.add_trace(go.Scatter(
                        x=sub_wavelengths, y=sub_refl, mode="lines",
                        name=f"ρ: {label}", line=dict(width=2.2)
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=sub_wavelengths, y=sub_emis, mode="lines",
                        name=f"ε (Emissivity): {label}", line=dict(width=2.2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=sub_wavelengths, y=sub_refl, mode="lines",
                        name=f"ρ (Reflectance): {label}", line=dict(width=1.5, dash="dash")
                    ))

            if show_aster_srf:
                aster_bands = [
                    (8.125, 8.475, "ASTER Band #10", "8.2910 μm", "354.0 nm"),
                    (8.475, 8.825, "ASTER Band #11", "8.6340 μm", "358.0 nm"),
                    (8.925, 9.275, "ASTER Band #12", "9.0750 μm", "355.0 nm"),
                    (10.250, 10.950, "ASTER Band #13", "10.6570 μm", "702.0 nm"),
                    (10.950, 11.650, "ASTER Band #14", "11.3180 μm", "705.0 nm")
                ]
                aster_shapes = [
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=float(b_start),
                        x1=float(b_end),
                        y0=0,
                        y1=1,
                        fillcolor="rgba(147, 197, 253, 0.16)",
                        line=dict(color="rgba(59, 130, 246, 0.40)", width=1, dash="dot"),
                        layer="below"
                    )
                    for b_start, b_end, _, _, _ in aster_bands
                ]
                current_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
                fig.update_layout(shapes=current_shapes + aster_shapes)

                for b_start, b_end, b_name, b_center, b_bw in aster_bands:
                    x_pts = np.linspace(b_start, b_end, 7)
                    y_pts = np.linspace(0.05, 0.95, 7)
                    X_grid, Y_grid = np.meshgrid(x_pts, y_pts)
                    tooltip = (
                        f"<b>{b_name}</b><br>"
                        f"Center Wavelength: {b_center}<br>"
                        f"Bandwidth (FWHM): {b_bw}<br>"
                        f"Band Start (λ_start): {b_start:.4f} μm<br>"
                        f"Band End (λ_end): {b_end:.4f} μm<br>"
                        f"Interval: [{b_start:.4f}, {b_end:.4f}] μm"
                    )
                    fig.add_trace(go.Scatter(
                        x=X_grid.flatten(),
                        y=Y_grid.flatten(),
                        mode="markers",
                        marker=dict(size=14, opacity=0),
                        hoverinfo="text",
                        hovertext=[tooltip] * len(X_grid.flatten()),
                        hovertemplate="%{hovertext}<extra></extra>",
                        showlegend=False
                    ))

            if show_hytes_srf:
                try:
                    hytes_wavs = None
                    if os.path.exists(paths.get("forward_hytes_npz", "")):
                        hytes_data = np.load(paths["forward_hytes_npz"])
                        hytes_wavs = hytes_data["wavelengths"]
                    if hytes_wavs is None or len(hytes_wavs) != 256:
                        hytes_wavs = np.linspace(7.49237, 11.991634, 256)

                    mid = (hytes_wavs[:-1] + hytes_wavs[1:]) / 2.0
                    l_bounds = np.insert(mid, 0, hytes_wavs[0] - (hytes_wavs[1] - hytes_wavs[0]) / 2.0)
                    r_bounds = np.append(mid, hytes_wavs[-1] + (hytes_wavs[-1] - hytes_wavs[-2]) / 2.0)

                    # 256 individual contiguous channel rectangles with alternating subtle fills
                    hytes_shapes = [
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=float(l_bounds[i]),
                            x1=float(r_bounds[i]),
                            y0=0,
                            y1=1,
                            fillcolor="rgba(251, 146, 60, 0.08)" if i % 2 == 0 else "rgba(251, 146, 60, 0.22)",
                            line=dict(color="rgba(249, 115, 22, 0.35)", width=0.5),
                            layer="below"
                        )
                        for i in range(256)
                    ]
                    current_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
                    fig.update_layout(shapes=current_shapes + hytes_shapes)

                    # Interactive tooltips across full vertical span
                    hytes_y_levels = np.linspace(0.05, 0.95, 7)
                    hytes_x_all = np.repeat(hytes_wavs, len(hytes_y_levels))
                    hytes_y_all = np.tile(hytes_y_levels, len(hytes_wavs))
                    hytes_tooltips = [
                        f"<b>HyTES Channel #{i+1:03d}</b><br>"
                        f"Center Wavelength: {hytes_wavs[i]:.4f} μm<br>"
                        f"Bandwidth: {(r_bounds[i] - l_bounds[i])*1000:.1f} nm<br>"
                        f"Channel Start (λ_start): {l_bounds[i]:.4f} μm<br>"
                        f"Channel End (λ_end): {r_bounds[i]:.4f} μm<br>"
                        f"Interval: [{l_bounds[i]:.4f}, {r_bounds[i]:.4f}] μm"
                        for i in range(256)
                    ]
                    hytes_hover_all = np.repeat(hytes_tooltips, len(hytes_y_levels))

                    fig.add_trace(go.Scatter(
                        x=hytes_x_all,
                        y=hytes_y_all,
                        mode="markers",
                        marker=dict(size=12, opacity=0),
                        hoverinfo="text",
                        hovertext=hytes_hover_all,
                        hovertemplate="%{hovertext}<extra></extra>",
                        showlegend=False
                    ))
                except Exception as e:
                    st.warning(f"Error rendering HyTES channels: {e}")

            y_title = "Emissivity ε" if view_mode == "Directional Emissivity (ε)" else ("Reflectance ρ" if view_mode == "Hemispherical Reflectance (ρ)" else "ε / ρ Value")
            # Always anchor y-axis at 0.0 to prevent clipping of reflectance or deep Reststrahlen dips
            fig.update_layout(
                title=f"Continuous Spectral Curves ({view_mode})",
                xaxis_title="Wavelength λ (μm)",
                yaxis_title=y_title,
                yaxis=dict(range=[0.0, 1.02]),
                xaxis=dict(range=[spectral_range[0], spectral_range[1]]),
                template="plotly_white",
                height=520,
                legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom", bgcolor="rgba(15, 23, 42, 0.88)", bordercolor="#334155", borderwidth=1, font=dict(color="#f8fafc", size=10))
            )
            st.plotly_chart(fig, width="stretch")

            # Details structured telemetry table for selected spectra
            st.markdown("#### 📋 Selected Spectral Metadata & Diagnostics")
            table_rows = []
            for spec_id in selected_spectras:
                r = catalog_df[catalog_df["spectrum_id"] == spec_id].iloc[0]
                badge = "badge-golden" if r["is_golden_target"] else ("badge-quar" if r.get("is_sibling_quarantined", False) else "badge-dev")
                role_label = r.get("benchmark_partition", "Development")
                val_f = r.get("validation_fold", "None")
                train_f = r.get("training_folds", "All")
                md5_short = r["emissivity_hash"][:12]

                row_html = (
                    "<tr>"
                    f'<td style="font-weight: 600; color: #38bdf8;"><code>{r["spectrum_id"]}</code></td>'
                    f'<td style="font-weight: 500; color: #f8fafc;">{r["name"]}</td>'
                    f'<td><span class="{badge}">{role_label}</span></td>'
                    f'<td style="color: #cbd5e1;">{r["material_class"]}</td>'
                    f'<td style="color: #cbd5e1;">{r["particle_size"]}</td>'
                    f'<td style="text-align: center; color: #94a3b8;">{val_f}</td>'
                    f'<td style="text-align: center; color: #94a3b8;">{train_f}</td>'
                    f'<td style="text-align: right; font-family: monospace; color: #34d399;">{r["min_emissivity"]:.4f}</td>'
                    f'<td style="text-align: right; font-family: monospace; color: #60a5fa;">{r["max_emissivity"]:.4f}</td>'
                    f'<td style="text-align: right; font-family: monospace; font-weight: 600; color: #fbbf24;">{r["mean_emissivity"]:.4f}</td>'
                    f'<td><code style="background-color: #1e293b; color: #a78bfa; padding: 2px 6px; border-radius: 4px; border: 1px solid #334155;">{md5_short}...</code></td>'
                    "</tr>"
                )
                table_rows.append(row_html)

            table_html = (
                '<div class="spec-table-container">'
                '<table class="spec-table">'
                '<thead><tr>'
                '<th>Spectrum ID</th>'
                '<th>Specimen Name</th>'
                '<th>Partition Role</th>'
                '<th>Class</th>'
                '<th>Particle Size</th>'
                '<th style="text-align: center;">Val Fold</th>'
                '<th style="text-align: center;">Train Folds</th>'
                '<th style="text-align: right;">Min ε</th>'
                '<th style="text-align: right;">Max ε</th>'
                '<th style="text-align: right;">Mean ε</th>'
                '<th>MD5 Checksum</th>'
                '</tr></thead>'
                '<tbody>' + ''.join(table_rows) + '</tbody>'
                '</table>'
                '</div>'
            )
            st.markdown(table_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: Golden Targets Suite
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🏆 The 7 Golden Targets Benchmark Suite")
    st.markdown(r"""
    The Golden Targets are the **clean-room benchmark endmembers** quarantined from training.
    In v1.1.1, the Feldspar target has been **physically sanitized to pure coarse Anorthite** (`anorthite_1 coarse`, $\epsilon \in [0.957, 0.989]$),
    completely eliminating the KBr transmission pellet artifact.
    """)

    golden_df = catalog_df[catalog_df["is_golden_target"] == True].copy()
    sibling_df = catalog_df[catalog_df.get("is_sibling_quarantined", False) == True].copy()

    fig_gt = go.Figure()
    for _, r in golden_df.iterrows():
        emis = emissivities[r.name]
        fig_gt.add_trace(go.Scatter(
            x=wavelengths, y=emis, mode="lines",
            name=f"{r['golden_target_name'].upper()}: {r['name']} ({r['sample_id']})",
            line=dict(width=2.5)
        ))

    for _, r in sibling_df.iterrows():
        emis = emissivities[r.name]
        fig_gt.add_trace(go.Scatter(
            x=wavelengths, y=emis, mode="lines",
            name=f"SIBLING CONTROL: {r['name']} ({r['sample_id']})",
            line=dict(width=1.8, dash="dot")
        ))

    fig_gt.update_layout(
        title="Pristine Emissivity Curves of the 7 Golden Targets + Sibling Control (v1.1.1 Suite)",
        xaxis_title="Wavelength λ (μm)",
        yaxis_title="Directional Emissivity (ε)",
        yaxis=dict(range=[0.0, 1.02]),
        xaxis=dict(range=[7.5, 13.5]),
        template="plotly_white",
        height=500,
        legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom", bgcolor="rgba(15, 23, 42, 0.88)", bordercolor="#334155", borderwidth=1, font=dict(color="#f8fafc", size=10))
    )
    st.plotly_chart(fig_gt, width="stretch")

    st.markdown("#### 📋 Golden Targets Cryptographic & Physical Verification Matrix")
    st.dataframe(
        golden_df[[
            "golden_target_name", "spectrum_id", "sample_id", "name",
            "material_class", "particle_size", "min_emissivity", "max_emissivity",
            "mean_emissivity", "source_filename", "emissivity_hash"
        ]],
        width="stretch"
    )


# -----------------------------------------------------------------------------
# TAB 3: Multi-Spectrum Comparison
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Multi-Spectrum Statistical Envelope & Class Variability")
    st.caption("Compare intra-class spectral dispersion, standard deviation envelopes, and Christiansen frequency behavior.")

    comp_class = st.selectbox("Select Target Class for Envelope Analysis:", all_classes, index=0)
    class_df = catalog_df[catalog_df["material_class"] == comp_class]

    if len(class_df) == 0:
        st.warning(f"No spectra available for class {comp_class}.")
    else:
        class_indices = class_df.index.tolist()
        class_emissivities = emissivities[class_indices]

        mean_curve = np.mean(class_emissivities, axis=0)
        std_curve = np.std(class_emissivities, axis=0)
        min_curve = np.min(class_emissivities, axis=0)
        max_curve = np.max(class_emissivities, axis=0)

        fig_env = go.Figure()

        # Min-Max Envelope
        fig_env.add_trace(go.Scatter(
            x=wavelengths, y=max_curve, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig_env.add_trace(go.Scatter(
            x=wavelengths, y=min_curve, mode="lines",
            fill="tonexty", fillcolor="rgba(191, 219, 254, 0.4)",
            line=dict(width=0), name="Total Dynamic Range [Min - Max]"
        ))

        # Mean +/- 1 Std
        fig_env.add_trace(go.Scatter(
            x=wavelengths, y=mean_curve + std_curve, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig_env.add_trace(go.Scatter(
            x=wavelengths, y=mean_curve - std_curve, mode="lines",
            fill="tonexty", fillcolor="rgba(59, 130, 246, 0.3)",
            line=dict(width=0), name="Confidence Band [Mean ± 1σ]"
        ))

        # Mean curve
        fig_env.add_trace(go.Scatter(
            x=wavelengths, y=mean_curve, mode="lines",
            name=f"Mean Profile (N = {len(class_df)})",
            line=dict(color="#1E3A8A", width=3)
        ))

        fig_env.update_layout(
            title=f"Spectral Dispersion & Envelope: Class '{comp_class.upper()}' (N = {len(class_df)})",
            xaxis_title="Wavelength λ (μm)",
            yaxis_title="Directional Emissivity (ε)",
            yaxis=dict(range=[0.40, 1.02]),
            xaxis=dict(range=[7.5, 13.5]),
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig_env, width="stretch")


# -----------------------------------------------------------------------------
# TAB 4: 3rd Validation Gate & Forensic Audit
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("⚠️ 3rd Validation Gate: Forensic Audit of Excluded Data")
    st.markdown(r"""
    The **3rd Validation Gate** protects Kirchhoff's Law by actively purging measurements conducted in **transmission** 
    (fine powders suspended in KBr pellets) and unphysical artifacts ($\epsilon_{\min} < 0.40$ or $\epsilon_{\max} < 0.80$).
    """)

    if not excluded_records:
        st.info("No quarantined records found in `zenodo_release/data/raw_laboratory_spectra/excluded_data/`.")
    else:
        st.warning(f"Found {len(excluded_records)} quarantined spectra with severe transmission/unphysical artifacts.")

        ex_options = [r["meta"]["filename"] for r in excluded_records]
        selected_ex = st.selectbox("Select Quarantined Spectrum to Inspect:", ex_options, index=0)

        record = next(r for r in excluded_records if r["meta"]["filename"] == selected_ex)
        m = record["meta"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Sample ID", m["sample_id"])
        with c2:
            st.metric("Mineral Name", m["name"])
        with c3:
            st.metric("Particle Size", m["particle_size"])
        with c4:
            st.metric("Header Measurement", m["measurement"])
            st.caption("⚠️ **Physical Status:** Quarantined Transmission Artifact")

        # Plot artifact vs physical counterpart
        fig_forensic = go.Figure()
        fig_forensic.add_trace(go.Scatter(
            x=record["wavs"], y=record["naive_emissivity"], mode="lines",
            name=f"Quarantined Artifact (1 - Transmittance): {m['name']}",
            line=dict(color="#DC2626", width=2.5, dash="dash")
        ))

        # Dynamically identify physical benchmark counterpart for the selected specimen
        sel_sample_id = m.get("sample_id", "").strip().lower()
        sel_filename = m.get("filename", "").lower()
        
        # Match by exact sample_id (preferring coarse if multiple exist)
        match = catalog_df[catalog_df["sample_id"].str.lower() == sel_sample_id]
        if len(match) > 1 and "coarse" in match["particle_size"].values:
            match = match[match["particle_size"] == "coarse"]
            
        # Match by base mineral keyword (calcite, quartz, kaolinite, etc.)
        if len(match) == 0:
            kw = sel_sample_id.split("_")[0] if "_" in sel_sample_id else sel_sample_id
            match = catalog_df[catalog_df["spectrum_id"].str.contains(kw, case=False, na=False)]
            
        # Match feldspar for albite / anorthite
        if len(match) == 0 and ("albite" in sel_filename or "anorthite" in sel_filename):
            match = catalog_df[catalog_df["material_class"] == "feldspar"]
            
        bench_row = match.iloc[0] if len(match) > 0 else catalog_df.iloc[0]
        bench_emis = emissivities[bench_row.name]

        fig_forensic.add_trace(go.Scatter(
            x=wavelengths, y=bench_emis, mode="lines",
            name=f"True Physical Benchmark: {bench_row['name']} ({bench_row['sample_id']})",
            line=dict(color="#059669", width=2.8)
        ))

        fig_forensic.add_hrect(
            y0=0.0, y1=0.40,
            fillcolor="rgba(239, 68, 68, 0.12)", line_width=1,
            line_color="rgba(239, 68, 68, 0.4)",
            annotation_text="Forbidden Region: ε < 0.40 (Violation of Kirchhoff's Law)",
            annotation_position="bottom right"
        )

        fig_forensic.update_layout(
            title=f"Forensic Comparison: Quarantined {m['name']} Artifact vs. True Physical Benchmark ({bench_row['name']})",
            xaxis_title="Wavelength λ (μm)",
            yaxis_title="Calculated Emissivity (ε)",
            yaxis=dict(range=[0.0, 1.05]),
            xaxis=dict(range=[7.5, 13.5]),
            template="plotly_white",
            height=480,
            legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom", bgcolor="rgba(15, 23, 42, 0.88)", bordercolor="#334155", borderwidth=1, font=dict(color="#f8fafc", size=10))
        )
        st.plotly_chart(fig_forensic, width="stretch")


# -----------------------------------------------------------------------------
# TAB 5: Atmospheric LUTs & Dynamic RTE Studio (The Component Mixer)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("🌤️ Atmospheric Radiative Transfer & Dynamic RTE Studio")
    
    with st.container(border=True):
        st.markdown("##### 🛰️ Coupled Boundary-Value Radiative Transfer Equation (RTE)")
        st.latex(r"L_{\mathrm{sensor}}(\lambda) = \tau(\lambda) \left[ \varepsilon(\lambda) B(\lambda, T_s) + (1 - \varepsilon(\lambda)) L^\downarrow(\lambda) \right] + L^\uparrow(\lambda)")
        info_c1, info_c2, info_c3 = st.columns(3)
        with info_c1:
            st.markdown(
                '<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 10px 14px; height: 100%;">'
                '<div style="font-size: 0.76rem; font-weight: 700; color: #93c5fd; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🔬 Curated Emissivity (241 Spectra)</div>'
                '<div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45;">'
                'Curated from <b>ECOSTRESS v1.0</b>: 141 geological & urban solids + 100 diversity-maximized botanical endmembers (via <i>L</i><sub>2</sub> Euclidean FPS).'
                '</div></div>',
                unsafe_allow_html=True
            )
        with info_c2:
            st.markdown(
                '<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #10b981; border-radius: 6px; padding: 10px 14px; height: 100%;">'
                '<div style="font-size: 0.76rem; font-weight: 700; color: #6ee7b7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🌤️ Atmospheric Profiles (30 LUTs)</div>'
                '<div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45;">'
                'High-fidelity <b>MODTRAN 5.2</b> look-up tables spanning 6 standard geographic models across 5 water vapor scales (PWV: 0.21–8.23 g/cm²).'
                '</div></div>',
                unsafe_allow_html=True
            )
        with info_c3:
            st.markdown(
                '<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #f59e0b; border-radius: 6px; padding: 10px 14px; height: 100%;">'
                '<div style="font-size: 0.76rem; font-weight: 700; color: #fcd34d; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🌡️ Thermodynamic State (Ts)</div>'
                '<div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45;">'
                'Arbitrary kinetic surface temperature <b><i>T</i><sub>s</sub></b> dynamically coupled with Planck blackbody radiation <i>B</i>(λ, <i>T</i><sub>s</sub>).'
                '</div></div>',
                unsafe_allow_html=True
            )

    if atm_catalog_df is None or len(atm_catalog_df) == 0:
        st.warning("Atmospheric look-up table catalog not found in `data/atmospheric_luts/`.")
    else:
        # Step 1: Atmosphere & Sample Selectors
        st.markdown("#### 1. Atmospheric Profile & Surface Sample Pairing")
        row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)

        with row1_c1:
            atm_models = atm_catalog_df["atm_model_name"].unique().tolist()
            selected_model = st.selectbox("1. Climatological Model:", atm_models, index=0)

        with row1_c2:
            model_df = atm_catalog_df[atm_catalog_df["atm_model_name"] == selected_model]
            h2o_scales = model_df["h2o_scale"].tolist()
            scale_labels = {
                r["h2o_scale"]: f"{r['h2o_scale']:.2f}x (PWV = {r['pwv_g_cm2']:.2f} g/cm²)"
                for _, r in model_df.iterrows()
            }
            selected_scale = st.selectbox(
                "2. Water Vapor Scaling (PWV):",
                h2o_scales,
                index=2 if len(h2o_scales) > 2 else 0,
                format_func=lambda x: scale_labels.get(x, f"{x:.2f}x"),
                help="Dynamically cascades to the 5 valid scaled moisture profiles for the selected climatological model."
            )

        with row1_c3:
            sensor_path = st.selectbox(
                "3. Observation Geometry:",
                ["ASTER Spaceborne TOA (100 km)", "HyTES Airborne Path (20 km)"],
                index=0
            )

        with row1_c4:
            catalog_options = catalog_df["spectrum_id"].tolist()
            spec_format = {
                r["spectrum_id"]: f"[{r['material_class'].upper()}] {r['name']} ({r['sample_id']})"
                for _, r in catalog_df.iterrows()
            }
            default_target_idx = [i for i, s in enumerate(catalog_options) if "quartz_1" in s.lower()]
            target_sel = st.selectbox(
                "4. Ground Surface (241 Targets):",
                options=catalog_options,
                index=default_target_idx[0] if default_target_idx else 0,
                format_func=lambda x: spec_format.get(x, x),
                key="rte_target_select_dynamic",
                help="Choose from the complete universe of 241 curated ECOSTRESS laboratory spectra."
            )

        # Climatological Context Guidance Banner
        clima_context = {
            "Tropical": "15°N maritime atmosphere | High moisture column (PWV 2.06 – 8.23 g/cm²) | Typical surface Ts: 295 – 320 K",
            "Mid-Latitude Summer": "45°N summer continental | Moderate moisture column (PWV 1.46 – 5.84 g/cm²) | Typical surface Ts: 290 – 315 K",
            "Mid-Latitude Winter": "45°N winter continental | Low moisture column (PWV 0.42 – 1.70 g/cm²) | Typical surface Ts: 265 – 290 K",
            "Sub-Arctic Summer": "60°N summer sub-polar | Moderate-low moisture (PWV 1.04 – 4.18 g/cm²) | Typical surface Ts: 280 – 300 K",
            "Sub-Arctic Winter": "60°N winter polar vortex | Ultra-dry cold column (PWV 0.21 – 0.84 g/cm²) | Typical surface Ts: 250 – 275 K",
            "1976 US Standard": "Global mean reference atmosphere | Mid moisture (PWV 0.71 – 2.84 g/cm²) | Typical surface Ts: 285 – 305 K"
        }
        st.caption(f"ℹ️ **Climatological Context:** {clima_context.get(selected_model, 'Standard MODTRAN 5.2 Atmosphere')}")

        # Step 2: Surface Temperature Slider
        st.markdown("#### 2. Surface Thermodynamic State")
        temp_col1, temp_col2 = st.columns([3, 1])
        with temp_col1:
            sim_temp = st.slider(
                "Surface Kinetic Temperature Ts (K):",
                min_value=250.0, max_value=340.0, value=300.0, step=0.5,
                help="Kinetic temperature setpoint of the radiating ground."
            )
        with temp_col2:
            st.metric("Surface Temperature", f"{sim_temp:.1f} K", help="Kinetic temperature in Kelvin")
            st.caption(f"Equivalent: **{sim_temp - 273.15:.2f} °C**")

        # Load Selected Atmospheric Profile
        selected_prof_row = model_df[model_df["h2o_scale"] == selected_scale].iloc[0]
        prof_csv_path = os.path.join(paths["atm_dir"], selected_prof_row["csv_file"])
        prof_df = load_atmospheric_profile(prof_csv_path)

        if prof_df is not None:
            tau_col = "tau_aster" if "ASTER" in sensor_path else "tau_hytes"
            lup_col = "L_up_aster" if "ASTER" in sensor_path else "L_up_hytes"

            atm_wavs = prof_df["wavelength_um"].values
            target_row = catalog_df[catalog_df["spectrum_id"] == target_sel].iloc[0]
            target_emis = emissivities[target_row.name]

            # Forward Radiative Transfer Calculations
            sim_emis = np.interp(atm_wavs, wavelengths, target_emis)
            sim_b_t = planck_radiance(atm_wavs, sim_temp)
            sim_l_emit = sim_emis * sim_b_t
            sim_l_refl = (1.0 - sim_emis) * prof_df["L_down_surface"].values
            sim_l_surf = sim_l_emit + sim_l_refl
            sim_tau = prof_df[tau_col].values
            sim_l_up = prof_df[lup_col].values
            sim_l_down = prof_df["L_down_surface"].values
            sim_l_sensor = sim_tau * sim_l_surf + sim_l_up

            # -------------------------------------------------------------
            # Step 3: Radiative Component Mixer (Individual or Grouped)
            # -------------------------------------------------------------
            st.markdown("#### 3. Radiative Component Selector (Individual or Grouped)")

            comp_mode = st.radio(
                "Select Component Grouping Mode:",
                options=[
                    "🌐 All Radiative Components (Complete Signal Decomposition)",
                    "🏜️ Surface Domain (L_surf, L_emit, L_refl, Blackbody B)",
                    "🌤️ Atmospheric Domain (Transmittance τ, Path L↑, Sky Downwelling L↓)",
                    "🛰️ Forward Coupling Balances (L_sensor, L_surf, Path L↑)",
                    "🔬 Ground vs. Sensor Contrast (L_sensor, L_surf, Emissivity ε)",
                    "⚙️ Custom Component Mixer (Choose Individual Terms Below)"
                ],
                index=0,
                horizontal=False
            )

            # Dictionary of available components
            all_comp_defs = {
                "L_sensor": {"label": "At-Sensor Total Radiance: L_sensor = τ·L_surf + L↑", "default": True, "color": "#1E3A8A", "width": 3.0, "dash": "solid", "axis": "primary"},
                "L_surf": {"label": "Surface-Leaving Net Radiance: L_surf = L_emit + L_refl", "default": True, "color": "#059669", "width": 2.5, "dash": "solid", "axis": "primary"},
                "L_emit": {"label": "Direct Surface Thermal Emission: L_emit = ε·B(Ts)", "default": True, "color": "#10B981", "width": 2.0, "dash": "dot", "axis": "primary"},
                "L_refl": {"label": "Reflected Downwelling Sky Radiance: L_refl = (1-ε)·L↓", "default": True, "color": "#F59E0B", "width": 2.0, "dash": "dot", "axis": "primary"},
                "L_up": {"label": "Atmospheric Path Upwelling Radiance: L↑", "default": True, "color": "#7C3AED", "width": 2.0, "dash": "dash", "axis": "primary"},
                "L_down": {"label": "Hemispherical Downwelling Sky Radiance: L↓", "default": False, "color": "#D97706", "width": 1.8, "dash": "dashdot", "axis": "primary"},
                "B_planck": {"label": "Ideal Blackbody Radiance: B(λ, Ts)", "default": False, "color": "#4B5563", "width": 1.5, "dash": "dashdot", "axis": "primary"},
                "emissivity": {"label": "Ground Directional Emissivity: ε(λ) [Secondary Axis]", "default": False, "color": "#DC2626", "width": 2.2, "dash": "solid", "axis": "secondary"},
                "transmittance": {"label": "Atmospheric Transmittance: τ(λ) [Secondary Axis]", "default": False, "color": "#2563EB", "width": 2.2, "dash": "solid", "axis": "secondary"}
            }

            # Pre-select based on mode
            if comp_mode.startswith("🌐 All"):
                active_comps = ["L_sensor", "L_surf", "L_emit", "L_refl", "L_up", "L_down", "emissivity", "transmittance"]
            elif comp_mode.startswith("🏜️ Surface"):
                active_comps = ["L_surf", "L_emit", "L_refl", "B_planck", "emissivity"]
            elif comp_mode.startswith("🌤️ Atmospheric"):
                active_comps = ["transmittance", "L_up", "L_down"]
            elif comp_mode.startswith("🛰️ Forward"):
                active_comps = ["L_sensor", "L_surf", "L_up"]
            elif comp_mode.startswith("🔬 Ground"):
                active_comps = ["L_sensor", "L_surf", "emissivity"]
            else:
                # Custom mode: render checkboxes in 3 columns
                st.caption("Toggle individual terms on/off:")
                m_c1, m_c2, m_c3 = st.columns(3)
                active_comps = []
                with m_c1:
                    if st.checkbox("At-Sensor Radiance: L_sensor", value=True): active_comps.append("L_sensor")
                    if st.checkbox("Surface-Leaving Radiance: L_surf", value=True): active_comps.append("L_surf")
                    if st.checkbox("Direct Emission: ε·B(Ts)", value=True): active_comps.append("L_emit")
                with m_c2:
                    if st.checkbox("Reflected Sky Radiance: (1-ε)·L↓/π", value=True): active_comps.append("L_refl")
                    if st.checkbox("Upwelling Path Radiance: L↑", value=True): active_comps.append("L_up")
                    if st.checkbox("Downwelling Sky Radiance: L↓", value=False): active_comps.append("L_down")
                with m_c3:
                    if st.checkbox("Ideal Blackbody: B(λ, Ts)", value=False): active_comps.append("B_planck")
                    if st.checkbox("Ground Emissivity: ε(λ) [Secondary Axis]", value=False): active_comps.append("emissivity")
                    if st.checkbox("Atmospheric Transmittance: τ(λ) [Secondary Axis]", value=False): active_comps.append("transmittance")

            # -------------------------------------------------------------
            # Step 4: Live Dynamic Plot Rendering (Dual-Axis)
            # -------------------------------------------------------------
            has_secondary = any(all_comp_defs[k]["axis"] == "secondary" for k in active_comps)
            fig_dynamic = make_subplots(specs=[[{"secondary_y": has_secondary}]])

            comp_data_map = {
                "L_sensor": sim_l_sensor,
                "L_surf": sim_l_surf,
                "L_emit": sim_l_emit,
                "L_refl": sim_l_refl,
                "L_up": sim_l_up,
                "L_down": sim_l_down,
                "B_planck": sim_b_t,
                "emissivity": sim_emis,
                "transmittance": sim_tau
            }

            for k in active_comps:
                cdef = all_comp_defs[k]
                y_vals = comp_data_map[k]
                is_sec = (cdef["axis"] == "secondary")

                fig_dynamic.add_trace(
                    go.Scatter(
                        x=atm_wavs,
                        y=y_vals,
                        mode="lines",
                        name=cdef["label"].split(":")[0],
                        line=dict(color=cdef["color"], width=cdef["width"], dash=cdef["dash"])
                    ),
                    secondary_y=is_sec
                )

            # Highlight 9.6 um Ozone absorption
            fig_dynamic.add_vrect(
                x0=9.4, x1=9.8,
                fillcolor="rgba(239, 68, 68, 0.10)",
                line_width=0,
                annotation_text="O3 (9.6 μm)",
                annotation_position="top left"
            )

            fig_dynamic.update_xaxes(title_text="Wavelength λ (μm)", range=[7.5, 13.5])
            fig_dynamic.update_yaxes(title_text="Radiance [W/(m² sr μm)]", secondary_y=False)
            if has_secondary:
                fig_dynamic.update_yaxes(title_text="Emissivity ε / Transmittance τ (0 - 1)", range=[0.0, 1.05], secondary_y=True)

            fig_dynamic.update_layout(
                title=f"Dynamic Coupled Radiative Balance | {target_row['material_class'].upper()} ({target_row['name']}) at Ts = {sim_temp:.1f} K & {selected_model} (PWV = {selected_prof_row['pwv_g_cm2']:.2f} g/cm²)",
                template="plotly_white",
                height=520,
                legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom", bgcolor="rgba(15, 23, 42, 0.88)", bordercolor="#334155", borderwidth=1, font=dict(color="#f8fafc", size=10))
            )
            st.plotly_chart(fig_dynamic, width="stretch")

            # -------------------------------------------------------------
            # Step 5: Energy Budget Breakdown Table & Metrics
            # -------------------------------------------------------------
            tir_m = (atm_wavs >= 8.0) & (atm_wavs <= 12.0)
            tot_sensor_sig = np.mean(sim_l_sensor[tir_m])
            emit_pct = (np.mean(sim_tau[tir_m] * sim_l_emit[tir_m]) / tot_sensor_sig) * 100.0 if tot_sensor_sig > 0 else 0.0
            refl_pct = (np.mean(sim_tau[tir_m] * sim_l_refl[tir_m]) / tot_sensor_sig) * 100.0 if tot_sensor_sig > 0 else 0.0
            path_pct = (np.mean(sim_l_up[tir_m]) / tot_sensor_sig) * 100.0 if tot_sensor_sig > 0 else 0.0
            mean_tau_val = np.mean(sim_tau[tir_m])

            st.markdown("#### 4. Photon Energy Budget Telemetry (8.0 - 12.0 μm Atmospheric Window)")
            eb1, eb2, eb3, eb4 = st.columns(4)
            with eb1:
                st.metric("Direct Ground Emission (τ·ε·B)", f"{emit_pct:.1f}%", help="Percentage of detected photons originating directly from surface kinetic temperature.")
            with eb2:
                st.metric("Reflected Sky Radiance (τ·L_refl)", f"{refl_pct:.1f}%", help="Downwelling sky radiance reflected by the ground into the sensor path.")
            with eb3:
                st.metric("Atmospheric Path Radiance (L↑)", f"{path_pct:.1f}%", help="Upwelling thermal photons emitted by atmospheric gases directly toward the sensor.")
            with eb4:
                st.metric("Mean Window Transmittance τ̄", f"{mean_tau_val:.3f}", help="Average atmospheric transmittance across the 8.0 - 12.0 μm window.")
                st.caption(f"Atmospheric PWV: **{selected_prof_row['pwv_g_cm2']:.2f} g/cm²**")

            # Download Coupled CSV
            sim_export_df = pd.DataFrame({
                "wavelength_um": atm_wavs,
                "emissivity": sim_emis,
                "transmittance": sim_tau,
                "planck_radiance": sim_b_t,
                "l_emit": sim_l_emit,
                "l_refl": sim_l_refl,
                "l_surf": sim_l_surf,
                "l_up": sim_l_up,
                "l_down": sim_l_down,
                "l_sensor": sim_l_sensor
            })
            sim_csv_bytes = sim_export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Coupled Radiative Curves as CSV",
                data=sim_csv_bytes,
                file_name=f"rte_coupling_{target_row['spectrum_id']}_{selected_prof_row['profile_id']}_{int(sim_temp)}K.csv",
                mime="text/csv"
            )


# -----------------------------------------------------------------------------
# TAB 6: Forward Datasets & Sample-Level Inspector (Train / Val / Golden)
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("📦 Forward Coupled Radiative Datasets & Sample-Level Inspector")
    
    with st.container(border=True):
        st.markdown("##### 🛰️ Forward Radiative Transfer Coupling & Observation Synthesis")
        st.caption(
            "Inspect individual synthetic top-of-atmosphere (TOA) and airborne radiance observations generated by coupling "
            "curated pristine FTIR laboratory spectra with high-fidelity MODTRAN 5.2 radiative transfer look-up tables (LUTs):"
        )
        st.latex(r"L_{\mathrm{sensor}}(\lambda) = \tau(\lambda) \left[ \varepsilon(\lambda) B(\lambda, T_s) + (1 - \varepsilon(\lambda)) L^\downarrow(\lambda) \right] + L^\uparrow(\lambda)")
        part_c1, part_c2, part_c3, part_c4 = st.columns(4)
        with part_c1:
            st.markdown(
                '<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 12px 14px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">'
                '<div>'
                '<div style="font-size: 0.72rem; font-weight: 700; color: #93c5fd; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🏋️ Quadrant I: Master Train</div>'
                '<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">5,500 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>'
                '<div style="font-family: monospace; font-size: 0.73rem; color: #60a5fa; background-color: #1e293b; border-radius: 4px; padding: 3px 6px; margin-bottom: 6px; border: 1px solid #3b82f633;">500/class × 11 classes = 5,500</div>'
                '<div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 6px;">Master Training: Continuous Ts ∈ [275, 335] K across 4 nominal atmospheres.</div>'
                '</div>'
                '<div style="font-size: 0.71rem; font-weight: 600; color: #93c5fd; border-top: 1px dashed #334155; padding-top: 6px; margin-top: 4px;">⚡ + Stochastic ε<sub>aug</sub> + Sensor Noise (0.5×–1.5× NEΔT)</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with part_c2:
            st.markdown(
                '<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #10b981; border-radius: 8px; padding: 12px 14px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">'
                '<div>'
                '<div style="font-size: 0.72rem; font-weight: 700; color: #6ee7b7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🧪 Quadrant I: Val OOF</div>'
                '<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">2,024 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>'
                '<div style="font-family: monospace; font-size: 0.73rem; color: #34d399; background-color: #1e293b; border-radius: 4px; padding: 3px 6px; margin-bottom: 6px; border: 1px solid #10b98133;">46 specs × 4 atms × 11 Ts = 2,024</div>'
                '<div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 6px;">Out-of-Fold Val: 0% intra-sample leakage across 11 discrete Ts [280–330 K].</div>'
                '</div>'
                '<div style="font-size: 0.71rem; font-weight: 600; color: #6ee7b7; border-top: 1px dashed #334155; padding-top: 6px; margin-top: 4px;">🛡️ + Pure FTIR + Nominal Sensor Noise (1.0× NEΔT = 0.1 K)</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with part_c3:
            st.markdown(
                '<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #f59e0b; border-radius: 8px; padding: 12px 14px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">'
                '<div>'
                '<div style="font-size: 0.72rem; font-weight: 700; color: #fcd34d; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🏆 Quadrant II: Pure OOD</div>'
                '<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">308 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>'
                '<div style="font-family: monospace; font-size: 0.73rem; color: #fbbf24; background-color: #1e293b; border-radius: 4px; padding: 3px 6px; margin-bottom: 6px; border: 1px solid #f59e0b33;">7 targets × 4 atms × 11 Ts = 308</div>'
                '<div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 6px;">Zero-Shot Material: Quarantined Golden Targets on seen atmospheres.</div>'
                '</div>'
                '<div style="font-size: 0.71rem; font-weight: 600; color: #fcd34d; border-top: 1px dashed #334155; padding-top: 6px; margin-top: 4px;">🛡️ + Pure FTIR + Nominal Sensor Noise (1.0× NEΔT = 0.1 K)</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with part_c4:
            st.markdown(
                '<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #a855f7; border-radius: 8px; padding: 12px 14px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">'
                '<div>'
                '<div style="font-size: 0.72rem; font-weight: 700; color: #d8b4fe; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">❄️ Quadrant IV: Double OOD</div>'
                '<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">770 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>'
                '<div style="font-family: monospace; font-size: 0.73rem; color: #c084fc; background-color: #1e293b; border-radius: 4px; padding: 3px 6px; margin-bottom: 6px; border: 1px solid #a855f733;">7 targets × 10 atms × 11 Ts = 770</div>'
                '<div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 6px;">Joint OOD Stress Test: Novel materials on Arctic winter profiles.</div>'
                '</div>'
                '<div style="font-size: 0.71rem; font-weight: 600; color: #d8b4fe; border-top: 1px dashed #334155; padding-top: 6px; margin-top: 4px;">🛡️ + Pure FTIR + Nominal Sensor Noise (1.0× NEΔT = 0.1 K)</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with st.expander("📖 Comprehensive Scientific Specification: The 2×2 Generalization Matrix, Quadrant Topology & Simulation Parameters", expanded=False):
            st.markdown(r"""
#### 1. The 2×2 Orthogonal Generalization Matrix & Validation Objectives
In terrestrial thermal infrared (TIR) remote sensing and GeoAI modeling, predictive architectures frequently suffer from **spectral autocorrelation** and **atmospheric confounding**—learning spurious statistical correlations between specific laboratory emissivity features and atmospheric water vapor loadings rather than inverting the underlying physical radiative transfer.

To rigorously prevent data leakage and benchmark real-world transferability, the OpenTIR-PINN benchmark organizes all forward coupled radiative observations across an **orthogonal 2×2 matrix** spanning two independent physical axes:
* **Axis 1 (Surface Material Distribution):** In-Distribution (233 development endmembers spanning 11 geological, pedological, and botanical classes) vs. Out-of-Distribution (7 structurally quarantined Golden Targets: quartz, calcite, kaolinite, feldspar, urban concrete, asphalt, and non-photosynthetic vegetation bark).
* **Axis 2 (Atmospheric Column Distribution):** In-Distribution (Nominal MODTRAN 5.2 profiles across Tropical, Mid-Latitude Summer, Sub-Arctic Summer, and 1976 US Standard) vs. Out-of-Distribution (Severe sub-arctic and polar winter regimes with extreme thermal inversions and ultra-dry columns $\mathrm{PWV} < 0.3\,\mathrm{g/cm}^2$).

```
                      [ Atmospheric Distribution Axis ]
                      In-Distribution (Nominal)     OOD (Severe Polar Winter)
                   +-----------------------------+-----------------------------+
   In-Distribution |         QUADRANT I          |        QUADRANT III         |
   (Dev Pool 233)  |  • Master Training: 5,500   |  • LOCO Generalization      |
                   |  • Out-of-Fold Val: 2,024   |  • Tab 5 Dynamic RTE Studio |
                   +-----------------------------+-----------------------------+
   Out-of-Dist.    |         QUADRANT II         |        QUADRANT IV          |
   (Golden Targets)|  • Pure Spectral OOD: 308   |  • Joint Extreme OOD: 770   |
   (7 Immutable)   |  • Zero-Shot Material Eval  |  • Double-Blind Stress Test |
                   +-----------------------------+-----------------------------+
```

---

#### 2. Quadrant-by-Quadrant Architecture & Verification Roles
* **🏋️ Quadrant I — Master Training Pool ($N = 5,500$ observations):**
  * **Role & What It Validates:** In-distribution representation learning. Validates the physics-informed neural network's capacity to minimize multi-task loss (supervised emissivity MSE, kinetic temperature error, Spectral Angle Mapper, and Sobolev gradient/curvature regularizers) without optimization collapse.
  * **Composition:** 500 spectral profiles sampled across the 233 development materials, coupled across 4 nominal MODTRAN atmospheres at 11 discrete surface kinetic temperatures ($T_s \in [280, 330]\,\mathrm{K}$ in $5\,\mathrm{K}$ steps).
* **🧪 Quadrant I — Out-of-Fold Validation ($N = 2,024$ observations):**
  * **Role & What It Validates:** Leak-free in-distribution interpolation. Validates generalization on unseen rock, soil, and vegetation specimens under standard atmospheric regimes.
  * **Composition:** 46 unseen development spectra isolated strictly in Fold 0 via `StratifiedGroupKFold(n_splits=5)` grouped by specimen `sample_id` (guaranteeing 0% intra-sample leakage across different preparation granulometries), coupled across the 4 nominal atmospheres at the 11 kinetic temperatures ($46 \times 4 \times 11 = 2,024$).
* **🏆 Quadrant II — Golden Target Benchmark ($N = 308$ observations):**
  * **Role & What It Validates:** **Zero-Shot Material Transferability.** Validates whether the network can accurately separate surface temperature $T_s$ and recover complex Reststrahlen doublets and Christiansen frequencies from completely novel materials never encountered during optimization, while operating under standard atmospheres.
  * **Composition:** The 7 permanently quarantined Golden Targets coupled across 4 nominal atmospheres at 11 temperature levels ($7 \times 4 \times 11 = 308$).
* **❄️ Quadrant III — Atmospheric Out-of-Distribution Stress Test:**
  * **Role & What It Validates:** Pure atmospheric transferability. Evaluates network resilience against extreme dry atmospheres, polar vortex conditions, and high path transmittance independently of material variations. Evaluated dynamically via Leave-One-Cluster-Out (LOCO) benchmarking and accessible interactively in Tab 5 (Dynamic RTE Studio).
* **❄️ Quadrant IV — Double-Blind Golden Benchmark ($N = 770$ observations):**
  * **Role & What It Validates:** **The Ultimate Double-Quarantine Stress Test.** Evaluates joint domain shift where *both* the ground material AND the atmospheric column are entirely out of distribution. This benchmark provides the $-79.3\%$ error reduction proof demonstrating that Sobolev-PINN fundamentally outperforms NASA's classical TES algorithm under adverse thermodynamics.
  * **Composition:** The 7 Golden Targets coupled exclusively with 10 severe sub-arctic and mid-latitude winter MODTRAN LUTs (quarantined Model 3 & Model 5 across water vapor scales $0.50\times$ to $2.00\times$, $\mathrm{PWV} \in [0.21, 1.70]\,\mathrm{g/cm}^2$) across all 11 temperatures ($7 \times 10 \times 11 = 770$).

---

#### 3. Data Synthesis Methodology & Exact Physical Values
1. **The Boundary-Value Radiative Transfer Formulation:**
   Every synthetic observation is generated via strict integration of the coupled RTE:
   $$L_{\mathrm{sensor}}(\lambda) = \tau(\lambda) \left[ \varepsilon(\lambda) B(\lambda, T_s) + (1 - \varepsilon(\lambda)) L^\downarrow(\lambda) \right] + L^\uparrow(\lambda)$$
   * $\varepsilon(\lambda)$: High-resolution pristine directional emissivity measured by laboratory FTIR spectrophotometry ($7.5\text{--}13.5\,\mu\text{m}$, $\Delta\lambda \le 1.0\,\mathrm{nm}$).
   * $B(\lambda, T_s)$: Planck blackbody function evaluated at kinetic temperature $T_s$:
     $$B(\lambda, T_s) = \frac{2hc^2}{\lambda^5 \left( \exp\left( \frac{hc}{\lambda k_B T_s} \right) - 1 \right)}$$
   * $(1 - \varepsilon(\lambda)) L^\downarrow(\lambda)$: Reflected downwelling sky radiance, adhering strictly to Kirchhoff's law for opaque terrestrial surfaces ($\rho(\lambda) = 1 - \varepsilon(\lambda)$).
   * $\tau(\lambda)$ and $L^\uparrow(\lambda)$: MODTRAN 5.2 total nadir atmospheric transmittance and upwelling thermal path radiance.
2. **Sensor Convolution:**
   * **ASTER (5 Discrete Bands):** Convolved using JPL laboratory Spectral Response Functions (SRFs): Band 10 ($8.291\,\mu\mathrm{m}$, FWHM $354\,\mathrm{nm}$), Band 11 ($8.634\,\mu\mathrm{m}$, FWHM $358\,\mathrm{nm}$), Band 12 ($9.075\,\mu\mathrm{m}$, FWHM $355\,\mathrm{nm}$), Band 13 ($10.657\,\mu\mathrm{m}$, FWHM $702\,\mathrm{nm}$), Band 14 ($11.318\,\mu\mathrm{m}$, FWHM $705\,\mathrm{nm}$).
   * **HyTES (256 Hyperspectral Channels):** Convolved across continuous contiguous channels ($7.49\text{--}12.00\,\mu\mathrm{m}$, $\Delta\lambda \approx 17.5\,\mathrm{nm}$, Gaussian instrument SRF with nominal FWHM $\approx 17.7\,\mathrm{nm}$).
3. **Kinetic Surface Temperature Grid:**
   * Range: $280.0\,\mathrm{K}$ ($6.85\,^\circ\mathrm{C}$) to $330.0\,\mathrm{K}$ ($56.85\,^\circ\mathrm{C}$) in uniform $5.0\,\mathrm{K}$ increments:
     $$\{280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330\}\,\mathrm{K}$$
   * Represents natural diurnal surface temperature variations across temperate, desert, and urban environments.
4. **Sample Accounting Summary:**
   * Forward samples per sensor: $5,500\text{ (Train)} + 2,024\text{ (Val OOF)} + 308\text{ (Q2)} + 770\text{ (Q4)} = 8,602$ observations.
   * Total benchmark corpus across both sensor architectures: $8,602 \times 2 = 17,204$ coupled radiative observations.

---

#### 4. 💡 Intuitive Combinatorial Summary Formula ("The Quick Rule-of-Thumb")
For instantaneous, high-level verification of every observation count across the benchmark, remember this simple governing relation:

$$\mathbf{N_{\text{observations}} = (N_{\text{targets}} \times N_{\text{atmospheres}} \times N_{\text{temperatures}}) + \text{Sensor Radiometric Noise } (\mathrm{NE}\Delta T)}$$

* **🏋️ Quadrant I (Master Train):** $500\text{ samples/class} \times 11\text{ classes} = \mathbf{5,500\text{ observations}}$ *(continuous $T_s \in [275, 335]\,\mathrm{K}$, stochastic shape jitter $\varepsilon_{\text{aug}}$, dynamic sensor noise $0.5\times\text{--}1.5\times \mathrm{NE}\Delta T$)*.
* **🧪 Quadrant I (Val OOF):** $46\text{ unseen spectra} \times 4\text{ nominal atmospheres} \times 11\text{ temperatures} = \mathbf{2,024\text{ observations}} + \text{nominal sensor noise } (\mathrm{NE}\Delta T = 0.1\,\mathrm{K})$.
* **🏆 Quadrant II (Golden OOD):** $7\text{ golden targets} \times 4\text{ nominal atmospheres} \times 11\text{ temperatures} = \mathbf{308\text{ observations}} + \text{nominal sensor noise } (\mathrm{NE}\Delta T = 0.1\,\mathrm{K})$.
* **❄️ Quadrant IV (Double OOD):** $7\text{ golden targets} \times 10\text{ polar winter atmospheres} \times 11\text{ temperatures} = \mathbf{770\text{ observations}} + \text{nominal sensor noise } (\mathrm{NE}\Delta T = 0.1\,\mathrm{K})$.
* **⚡ Benchmark Total per Sensor:** $5,500 + 2,024 + 308 + 770 = \mathbf{8,602\text{ observations}}$ ($\times 2\text{ sensors (ASTER + HyTES)} = \mathbf{17,204\text{ total forward samples}}$).

---

#### 5. 🔬 The Two-Tier Forward Perturbation Architecture (Noise vs. Augmentation)
Remote sensing models trained exclusively on noise-free synthetic radiances fail on physical instruments due to detector sensitivity thresholds. Our simulation strictly enforces two distinct physical perturbation layers:
1. **Tier 1 — Instrumental Radiometric Noise Injection ($\mathrm{NE}\Delta T$-Grounded):**
   * **Active in ALL partitions** (Train, Val OOF, Q-II, Q-III, Q-IV).
   * Scaled analytically using the Planck temperature derivative at $300\,\mathrm{K}$:
     $$\sigma_L(\lambda) = \left. \frac{\partial B(\lambda, T)}{\partial T} \right|_{T = 300\,\mathrm{K}} \times \mathrm{NE}\Delta T \times \text{noise\_multiplier}$$
     with sensor-specification $\mathrm{NE}\Delta T$ = 0.3 K (ASTER) and 0.2 K (HyTES), multiplier $\mathcal{U}(0.5, 1.5)$ in training and 1.0 in every evaluation partition. Injected into $L_{\mathrm{surf}}$, $L_{\mathrm{sensor}}$, and $L^\downarrow$ (N4, v1.3.0: the network receives $L^\downarrow$, $\tau$ and $L^\uparrow$ of an *estimated* atmosphere with an absolute PWV error of 2.97 mm (evaluation) or $\mathcal{U}(1.84, 4.1)$ mm (training), after Wang et al. 2020 and Seemann et al. 2003; sensor noise is applied to $L_{\mathrm{sensor}}$ and propagated into the corrected $L_{\mathrm{surf}}$).
2. **Tier 2 — Emissivity perturbation: REMOVED in v1.2.2:**
   * The white jitter (N5) and the $\pm 0.5\,\%$ scale (N6) were removed: no verified instrumental/FTIR uncertainty supports them. Training emissivity is the pristine laboratory spectrum; training diversity comes from the temperature, atmosphere and noise draws.
   * **Quarantined Test Vaults (Val OOF, Q2, Q4) Use PURE FTIR Spectra:** Emissivities remain 100% pristine laboratory curves without shape jitter (`is_augmented = False`), preserving genuine mineral Reststrahlen lattice doublets for double-blind benchmarking.
""")


    # Sensor and Partition Selectors
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        sensor_choice = st.radio(
            "Sensor Architecture:",
            options=["ASTER (5 Bands | Multispectral Spaceborne)", "HyTES (256 Channels | Hyperspectral Airborne)"],
            index=0,
            horizontal=True
        )
        sensor_name = "aster" if "ASTER" in sensor_choice else "hytes"

    with s_col2:
        part_dict = {
            "🏋️ Quadrant I — Master Training Set (Dev Pool × Dev Atmospheres | 5,500 samples)": "train",
            "🧪 Quadrant I — Out-of-Fold Validation (Unseen Dev Pool × Dev Atmospheres | 2,024 samples)": "val_oof",
            "🏆 Quadrant II — Golden Target Benchmark (Spectral OOD Targets × Dev Atmospheres | 308 samples)": "q2",
            "❄️ Quadrant IV — Double-Blind Golden Benchmark (Spectral OOD × Severe Polar Winter OOD | 770 samples)": "q4"
        }
        part_label = st.selectbox(
            "Select Forward Simulation Dataset Partition:",
            options=list(part_dict.keys()),
            index=2,
            help="Select benchmark partition across the 2×2 Generalization Matrix."
        )
        part_key = part_dict[part_label]

    sample_df, bundle = load_forward_dataset(sensor_name, part_key, paths)

    if sample_df is None or len(sample_df) == 0:
        st.warning(f"Could not load forward dataset for {sensor_name.upper()} ({part_label}). Ensure dataset assets exist in `data/`.")
    else:
        # Multi-tier filter inside the forward tab
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            avail_classes = sorted(sample_df["material_class"].unique().tolist())
            sel_sample_classes = st.multiselect("Filter by Material Class:", avail_classes, default=avail_classes)
        with f_col2:
            t_min, t_max = float(sample_df["ts_k"].min()), float(sample_df["ts_k"].max())
            sel_temp_range = st.slider("Filter Surface Temp Ts (K):", min_value=t_min, max_value=t_max, value=(t_min, t_max), step=1.0)
        with f_col3:
            p_min, p_max = float(sample_df["pwv"].min()), float(sample_df["pwv"].max())
            if p_min == p_max:
                st.info(f"PWV fixed at {p_min:.2f} g/cm²")
                sel_pwv_range = (p_min, p_max)
            else:
                sel_pwv_range = st.slider("Filter PWV (g/cm²):", min_value=p_min, max_value=p_max, value=(p_min, p_max), step=0.1)

        # Apply sample filters
        filtered_samples = sample_df[
            (sample_df["material_class"].isin(sel_sample_classes)) &
            (sample_df["ts_k"] >= sel_temp_range[0]) &
            (sample_df["ts_k"] <= sel_temp_range[1]) &
            (sample_df["pwv"] >= sel_pwv_range[0]) &
            (sample_df["pwv"] <= sel_pwv_range[1])
        ]

        # Summary KPIs for Forward Samples
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Filtered Samples", f"{len(filtered_samples)}")
            st.caption(f"Subset of **{len(sample_df)}** partition samples")
        with kpi2:
            mean_t = filtered_samples["ts_k"].mean() if len(filtered_samples) > 0 else 0.0
            st.metric("Mean Surface Ts", f"{mean_t:.1f} K")
            st.caption(f"Equivalent: **{mean_t - 273.15:.1f} °C**")
        with kpi3:
            mean_e = filtered_samples["mean_emissivity"].mean() if len(filtered_samples) > 0 else 0.0
            st.metric("Mean Emissivity ε̄", f"{mean_e:.3f}")
        with kpi4:
            mean_rad = filtered_samples["mean_l_sensor"].mean() if len(filtered_samples) > 0 else 0.0
            st.metric("Mean Sensor Radiance", f"{mean_rad:.2f} W/(m² sr μm)")

        st.markdown("#### 📋 Sample Index Ledger")
        st.dataframe(
            filtered_samples[[
                "sample_idx", "material_class", "target_name", "ts_k",
                "mean_emissivity", "atmosphere_id", "pwv",
                "mean_l_sensor", "mean_l_surf", "mean_l_down"
            ]],
            width="stretch",
            height=200,
            hide_index=True
        )

        # Deep Dive Sample Inspector
        if len(filtered_samples) > 0:
            st.markdown("---")
            st.markdown("#### 🔬 Individual Sample Photon Decomposition & Spectral Diagnostic")

            sample_ids_list = filtered_samples["sample_idx"].tolist()
            sel_idx = st.selectbox(
                "Select Sample Index to Inspect:",
                options=sample_ids_list,
                format_func=lambda i: f"Sample #{i:04d} — {sample_df.loc[i, 'target_name']} (Ts = {sample_df.loc[i, 'ts_k']:.1f} K | {sample_df.loc[i, 'atmosphere_id']})"
            )

            wavs = bundle["wavelengths"]
            if bundle["type"] == "npz":
                prefix = bundle["prefix"]
                npz_data = bundle["data"]
                l_sensor = npz_data[prefix + "L_toa"][sel_idx]
                l_surf = npz_data[prefix + "L_obs"][sel_idx]
                l_down = npz_data[prefix + "L_down"][sel_idx]
                emis = npz_data[prefix + "emissivity"][sel_idx]
                ts_val = float(npz_data[prefix + "Ts"][sel_idx])
                mat_name = str(npz_data[prefix + "golden_name"][sel_idx]) if (prefix + "golden_name") in npz_data else str(npz_data[prefix + "materials"][sel_idx])
                mat_class = str(npz_data[prefix + "materials"][sel_idx])
                atm_id = str(npz_data[prefix + "atmosphere_id"][sel_idx]) if (prefix + "atmosphere_id") in npz_data else "MODTRAN Profile"
                pwv_val = float(npz_data[prefix + "pwv"][sel_idx]) if (prefix + "pwv") in npz_data else 0.0
            else:
                s_item = bundle["data"][sel_idx]
                l_sensor = s_item["input_radiance_sensor"].numpy()
                l_surf = s_item["input_radiance"].numpy()
                l_down = s_item["downwelling_radiance"].numpy()
                emis = s_item["target_emissivity"].numpy()
                ts_val = float(s_item["target_temp"].item())
                mat_name = s_item.get("golden_name", s_item.get("material_class", "unknown"))
                mat_class = s_item.get("material_class", "unknown")
                atm_id = s_item.get("atmosphere_id", "unknown")
                pwv_val = float(s_item.get("pwv", 0.0))

            b_t = planck_radiance(wavs, ts_val)
            l_emit = emis * b_t
            l_refl = (1.0 - emis) * l_down

            # Sample dynamic component toggle & Band Mask Policy
            st.caption("Customize Radiative Curves for Sample #" + str(sel_idx) + " & Atmospheric Transmission Mask:")
            sc_c1, sc_c2, sc_c3, sc_c4, sc_c5 = st.columns(5)
            with sc_c1:
                show_s_sensor = st.checkbox("Show L_sensor (At-Sensor)", value=True, key=f"s_sens_{sel_idx}")
            with sc_c2:
                show_s_surf = st.checkbox("Show L_surf (Surface Net)", value=True, key=f"s_surf_{sel_idx}")
            with sc_c3:
                show_s_emit = st.checkbox("Show ε·B(Ts) (Emitted)", value=True, key=f"s_emit_{sel_idx}")
            with sc_c4:
                show_s_refl = st.checkbox("Show (1-ε)·L↓ (Reflected Sky)", value=True, key=f"s_refl_{sel_idx}")
            with sc_c5:
                has_sample_mask = (bundle["type"] != "npz") and ("band_mask" in s_item) and (not s_item["band_mask"].all())
                apply_band_mask = st.checkbox(
                    "Apply Band Mask (N3)",
                    value=has_sample_mask,
                    key=f"s_mask_{sel_idx}",
                    help="Exclude opaque telluric water vapor bands (tau < 0.10) where surface radiance explodes."
                )

            # Filter data based on band mask
            if apply_band_mask and has_sample_mask:
                b_mask = s_item["band_mask"].numpy()
                plot_wavs = wavs[b_mask]
                plot_l_sensor = l_sensor[b_mask]
                plot_l_surf = l_surf[b_mask]
                plot_l_emit = l_emit[b_mask]
                plot_l_refl = l_refl[b_mask]
                plot_emis = emis[b_mask]
                st.info(f"🛡️ **N3 Opaque-Band Mask Active:** Displaying **{int(b_mask.sum())}** clean transparent channels (**{int((~b_mask).sum())}** opaque telluric absorption channels excluded).")
            else:
                plot_wavs = wavs
                plot_l_sensor = l_sensor
                plot_l_surf = l_surf
                plot_l_emit = l_emit
                plot_l_refl = l_refl
                plot_emis = emis

            f_col_left, f_col_right = st.columns(2)

            with f_col_left:
                fig_samp_rad = go.Figure()
                if show_s_sensor:
                    fig_samp_rad.add_trace(go.Scatter(
                        x=plot_wavs, y=plot_l_sensor, mode="lines+markers",
                        name="L_sensor (At-Sensor)",
                        line=dict(color="#1E3A8A", width=2.5),
                        marker=dict(size=7)
                    ))
                if show_s_surf:
                    fig_samp_rad.add_trace(go.Scatter(
                        x=plot_wavs, y=plot_l_surf, mode="lines+markers",
                        name="L_surf (Surface Net)",
                        line=dict(color="#059669", width=2),
                        marker=dict(size=6)
                    ))
                if show_s_emit:
                    fig_samp_rad.add_trace(go.Scatter(
                        x=plot_wavs, y=plot_l_emit, mode="lines+markers",
                        name="ε·B(Ts) (Direct Emission)",
                        line=dict(color="#10B981", width=1.8, dash="dot"),
                        marker=dict(size=5)
                    ))
                if show_s_refl:
                    fig_samp_rad.add_trace(go.Scatter(
                        x=plot_wavs, y=plot_l_refl, mode="lines+markers",
                        name="(1-ε)·L↓ (Reflected Sky)",
                        line=dict(color="#F59E0B", width=1.8, dash="dot"),
                        marker=dict(size=5)
                    ))
                fig_samp_rad.update_layout(
                    title=f"Sample #{sel_idx}: Radiance Energy Decomposition",
                    xaxis_title="Wavelength λ (μm)",
                    yaxis_title="Radiance [W/(m² sr μm)]",
                    template="plotly_white",
                    height=420,
                    legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom", bgcolor="rgba(15, 23, 42, 0.88)", bordercolor="#334155", borderwidth=1, font=dict(color="#f8fafc", size=10))
                )
                st.plotly_chart(fig_samp_rad, width="stretch")

            with f_col_right:
                fig_samp_emis = go.Figure()
                fig_samp_emis.add_trace(go.Scatter(
                    x=plot_wavs, y=plot_emis, mode="lines+markers",
                    name=f"True Emissivity ε ({mat_name})",
                    line=dict(color="#DC2626", width=2.5),
                    marker=dict(size=8, symbol="diamond")
                ))
                fig_samp_emis.update_layout(
                    title=f"Sample #{sel_idx}: True Ground Emissivity Spectrum",
                    xaxis_title="Wavelength λ (μm)",
                    yaxis_title="Emissivity (0 - 1)",
                    yaxis=dict(range=[max(0.0, float(emis.min() - 0.05)), min(1.02, float(emis.max() + 0.05))]),
                    template="plotly_white",
                    height=420,
                    legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom", bgcolor="rgba(15, 23, 42, 0.88)", bordercolor="#334155", borderwidth=1, font=dict(color="#f8fafc", size=10))
                )
                st.plotly_chart(fig_samp_emis, width="stretch")

            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Kinetic Temp Ts", f"{ts_val:.2f} K")
                st.caption(f"Equivalent: **{ts_val - 273.15:.2f} °C**")
            with p2:
                mmd_val = float(emis.max() - emis.min())
                st.metric("Spectral Contrast (MMD)", f"{mmd_val:.4f}", help="Min-Max Difference: measure of spectral diagnostic strength.")
            with p3:
                emit_fraction = (np.mean(l_emit) / np.mean(l_surf)) * 100.0 if np.mean(l_surf) > 0 else 0.0
                st.metric("Emitted Photon Fraction", f"{emit_fraction:.1f}%", help="Percentage of surface signal originating from thermal blackbody emission.")
            with p4:
                st.metric("Atmospheric Profile", atm_id)
                st.caption(f"Water Vapor Column: **{pwv_val:.2f} g/cm²**")


# -----------------------------------------------------------------------------
# TAB 7: Metadata Catalog & FAIR Table
# -----------------------------------------------------------------------------
with tab7:
    st.subheader("📑 Master Spectral Catalog (N = 241 Physical Spectra)")
    st.caption("Searchable, filterable inventory adhering to FAIR (Findable, Accessible, Interoperable, Reusable) data standards.")

    search_query = st.text_input("🔍 Search Catalog (by sample ID, mineral name, partition, or source file):", "")

    catalog_display = catalog_df.copy()
    if search_query:
        q = search_query.lower()
        catalog_display = catalog_display[
            catalog_display["sample_id"].str.lower().str.contains(q, na=False) |
            catalog_display["name"].str.lower().str.contains(q, na=False) |
            catalog_display["source_filename"].str.lower().str.contains(q, na=False) |
            catalog_display["material_class"].str.lower().str.contains(q, na=False) |
            catalog_display.get("benchmark_partition", pd.Series([""]*len(catalog_display))).str.lower().str.contains(q, na=False)
        ]

    cols_to_show = [
        "spectrum_id", "sample_id", "material_class", "superclass",
        "particle_size", "benchmark_partition", "validation_fold",
        "min_emissivity", "max_emissivity", "mean_emissivity", "source_filename", "emissivity_hash"
    ]
    present_cols = [c for c in cols_to_show if c in catalog_display.columns]

    st.dataframe(catalog_display[present_cols], width="stretch")

    csv_bytes = catalog_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Catalog as CSV",
        data=csv_bytes,
        file_name="opentir_spectral_catalog_filtered.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# TAB 8: Dataset Validation Report (obtained values vs pass/fail limits)
# -----------------------------------------------------------------------------
with tab8:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from validation_report_tab import render_validation_tab
        render_validation_tab()
    except Exception as exc:  # keep the rest of the dashboard usable
        st.error(f"Validation report tab failed to render: {exc}")

st.markdown("---")
st.caption("OpenTIR-PINN Benchmark Suite v1.1.1 | Developed by Dr. Atilio Grondona | Zenodo DOI: 10.5281/zenodo.23966244")
