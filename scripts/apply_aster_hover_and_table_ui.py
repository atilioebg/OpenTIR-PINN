import os
import re

css_addition = """    .spec-table-container {
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
</style>"""

aster_patch_target = """                for b_start, b_end, b_name, b_center, b_bw in aster_bands:
                    fig.add_trace(go.Scatter(
                        x=[b_start, b_end, b_end, b_start, b_start],
                        y=[0.0, 0.0, 1.02, 1.02, 0.0],
                        fill="toself",
                        fillcolor="rgba(147, 197, 253, 0.18)",
                        line=dict(color="rgba(59, 130, 246, 0.45)", width=1, dash="dot"),
                        mode="lines",
                        hoveron="fills",
                        name=b_name,
                        hovertemplate=(
                            f"<b>{b_name}</b><br>"
                            f"Center Wavelength: {b_center}<br>"
                            f"Bandwidth: {b_bw}<br>"
                            f"Interval: [{b_start:.4f}, {b_end:.4f}] μm"
                            "<extra></extra>"
                        ),
                        showlegend=False
                    ))"""

aster_patch_replacement = """                for b_start, b_end, b_name, b_center, b_bw in aster_bands:
                    fig.add_trace(go.Scatter(
                        x=[b_start, b_end, b_end, b_start, b_start],
                        y=[0.0, 0.0, 1.02, 1.02, 0.0],
                        fill="toself",
                        fillcolor="rgba(147, 197, 253, 0.18)",
                        line=dict(color="rgba(59, 130, 246, 0.45)", width=1, dash="dot"),
                        mode="lines",
                        hoveron="points+fills",
                        hoverinfo="text",
                        name=b_name,
                        hovertext=(
                            f"<b>{b_name}</b><br>"
                            f"Center Wavelength: {b_center}<br>"
                            f"Bandwidth (FWHM): {b_bw}<br>"
                            f"Band Start (λ_start): {b_start:.4f} μm<br>"
                            f"Band End (λ_end): {b_end:.4f} μm<br>"
                            f"Interval: [{b_start:.4f}, {b_end:.4f}] μm"
                        ),
                        hovertemplate=(
                            f"<b>{b_name}</b><br>"
                            f"Center Wavelength: {b_center}<br>"
                            f"Bandwidth (FWHM): {b_bw}<br>"
                            f"Band Start (λ_start): {b_start:.4f} μm<br>"
                            f"Band End (λ_end): {b_end:.4f} μm<br>"
                            f"Interval: [{b_start:.4f}, {b_end:.4f}] μm"
                            "<extra></extra>"
                        ),
                        showlegend=False
                    ))"""

hytes_patch_target = """                    fig.add_trace(go.Scatter(
                        x=hytes_wavs,
                        y=[0.5] * 256,
                        mode="markers",
                        marker=dict(size=12, opacity=0),
                        hoverinfo="text",
                        hovertext=[
                            f"<b>HyTES Channel #{i+1:03d}</b><br>"
                            f"Center Wavelength: {hytes_wavs[i]:.4f} μm<br>"
                            f"Bandwidth: {(r_bounds[i] - l_bounds[i])*1000:.1f} nm<br>"
                            f"Interval: [{l_bounds[i]:.4f}, {r_bounds[i]:.4f}] μm"
                            for i in range(256)
                        ],
                        showlegend=False
                    ))"""

hytes_patch_replacement = """                    fig.add_trace(go.Scatter(
                        x=hytes_wavs,
                        y=[0.5] * 256,
                        mode="markers",
                        marker=dict(size=12, opacity=0),
                        hoverinfo="text",
                        hovertext=[
                            f"<b>HyTES Channel #{i+1:03d}</b><br>"
                            f"Center Wavelength: {hytes_wavs[i]:.4f} μm<br>"
                            f"Bandwidth: {(r_bounds[i] - l_bounds[i])*1000:.1f} nm<br>"
                            f"Channel Start (λ_start): {l_bounds[i]:.4f} μm<br>"
                            f"Channel End (λ_end): {r_bounds[i]:.4f} μm<br>"
                            f"Interval: [{l_bounds[i]:.4f}, {r_bounds[i]:.4f}] μm"
                            for i in range(256)
                        ],
                        hovertemplate="%{hovertext}<extra></extra>",
                        showlegend=False
                    ))"""

table_patch_target = """            # Details card for selected spectrum
            st.markdown("#### Selected Spectral Metadata & Diagnostics")
            for spec_id in selected_spectras:
                r = catalog_df[catalog_df["spectrum_id"] == spec_id].iloc[0]
                badge = "badge-golden" if r["is_golden_target"] else ("badge-quar" if r.get("is_sibling_quarantined", False) else "badge-dev")
                role_label = r.get("benchmark_partition", "Development")

                st.markdown(f\"\"\"
                <div class="metric-card">
                    <b>{r['spectrum_id']}</b> — {r['name']} | 
                    <span class="{badge}">{role_label}</span> |
                    <b>Class:</b> {r['material_class']} | 
                    <b>Particle Size:</b> {r['particle_size']} | 
                    <b>Validation Fold:</b> {r.get('validation_fold', 'None')} |
                    <b>Training Folds:</b> {r.get('training_folds', 'All')} |
                    <b>Min ε:</b> {r['min_emissivity']:.3f} | <b>Max ε:</b> {r['max_emissivity']:.3f} | <b>Mean ε:</b> {r['mean_emissivity']:.3f} |
                    <b>MD5:</b> <code>{r['emissivity_hash'][:12]}...</code>
                </div>
                \"\"\", unsafe_allow_html=True)"""

table_patch_replacement = """            # Details structured telemetry table for selected spectra
            st.markdown("#### 📋 Selected Spectral Metadata & Diagnostics")
            table_rows = []
            for spec_id in selected_spectras:
                r = catalog_df[catalog_df["spectrum_id"] == spec_id].iloc[0]
                badge = "badge-golden" if r["is_golden_target"] else ("badge-quar" if r.get("is_sibling_quarantined", False) else "badge-dev")
                role_label = r.get("benchmark_partition", "Development")
                val_f = r.get("validation_fold", "None")
                train_f = r.get("training_folds", "All")
                md5_short = r["emissivity_hash"][:12]

                row_html = f\"\"\"
                <tr>
                    <td style="font-weight: 600; color: #38bdf8;"><code>{r['spectrum_id']}</code></td>
                    <td style="font-weight: 500; color: #f8fafc;">{r['name']}</td>
                    <td><span class="{badge}">{role_label}</span></td>
                    <td style="color: #cbd5e1;">{r['material_class']}</td>
                    <td style="color: #cbd5e1;">{r['particle_size']}</td>
                    <td style="text-align: center; color: #94a3b8;">{val_f}</td>
                    <td style="text-align: center; color: #94a3b8;">{train_f}</td>
                    <td style="text-align: right; font-family: monospace; color: #34d399;">{r['min_emissivity']:.4f}</td>
                    <td style="text-align: right; font-family: monospace; color: #60a5fa;">{r['max_emissivity']:.4f}</td>
                    <td style="text-align: right; font-family: monospace; font-weight: 600; color: #fbbf24;">{r['mean_emissivity']:.4f}</td>
                    <td><code style="background-color: #1e293b; color: #a78bfa; padding: 2px 6px; border-radius: 4px; border: 1px solid #334155;">{md5_short}...</code></td>
                </tr>
                \"\"\"
                table_rows.append(row_html)

            table_html = f\"\"\"
            <div class="spec-table-container">
                <table class="spec-table">
                    <thead>
                        <tr>
                            <th>Spectrum ID</th>
                            <th>Specimen Name</th>
                            <th>Partition Role</th>
                            <th>Class</th>
                            <th>Particle Size</th>
                            <th style="text-align: center;">Val Fold</th>
                            <th style="text-align: center;">Train Folds</th>
                            <th style="text-align: right;">Min ε</th>
                            <th style="text-align: right;">Max ε</th>
                            <th style="text-align: right;">Mean ε</th>
                            <th>MD5 Checksum</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>
            </div>
            \"\"\"
            st.markdown(table_html, unsafe_allow_html=True)"""

target_files = [
    "zenodo_release/dataset_explorer_dashboard.py",
    "scripts/dataset_explorer_dashboard.py"
]

for file_path in target_files:
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (does not exist)")
        continue
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Normalize CRLF
    is_crlf = "\r\n" in content
    norm_content = content.replace("\r\n", "\n")

    # 1. Update CSS
    if ".spec-table-container" not in norm_content:
        norm_content = norm_content.replace("</style>", css_addition)
        print(f"Added CSS to {file_path}")

    # 2. Update ASTER patch
    if aster_patch_target in norm_content:
        norm_content = norm_content.replace(aster_patch_target, aster_patch_replacement)
        print(f"Applied ASTER hover patch to {file_path}")
    else:
        print(f"ASTER target not found in {file_path}")

    # 3. Update HyTES patch
    if hytes_patch_target in norm_content:
        norm_content = norm_content.replace(hytes_patch_target, hytes_patch_replacement)
        print(f"Applied HyTES hover patch to {file_path}")
    else:
        print(f"HyTES target not found in {file_path}")

    # 4. Update Table UI
    if table_patch_target in norm_content:
        norm_content = norm_content.replace(table_patch_target, table_patch_replacement)
        print(f"Applied Table UI patch to {file_path}")
    else:
        print(f"Table target not found in {file_path}")

    # Restore line endings
    if is_crlf:
        final_content = norm_content.replace("\n", "\r\n")
    else:
        final_content = norm_content

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_content)

print("All dashboard patches applied successfully!")

