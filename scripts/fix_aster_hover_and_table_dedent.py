import os

new_code_block = '''            if show_aster_srf:
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
                    f\'<td style="font-weight: 600; color: #38bdf8;"><code>{r["spectrum_id"]}</code></td>\'
                    f\'<td style="font-weight: 500; color: #f8fafc;">{r["name"]}</td>\'
                    f\'<td><span class="{badge}">{role_label}</span></td>\'
                    f\'<td style="color: #cbd5e1;">{r["material_class"]}</td>\'
                    f\'<td style="color: #cbd5e1;">{r["particle_size"]}</td>\'
                    f\'<td style="text-align: center; color: #94a3b8;">{val_f}</td>\'
                    f\'<td style="text-align: center; color: #94a3b8;">{train_f}</td>\'
                    f\'<td style="text-align: right; font-family: monospace; color: #34d399;">{r["min_emissivity"]:.4f}</td>\'
                    f\'<td style="text-align: right; font-family: monospace; color: #60a5fa;">{r["max_emissivity"]:.4f}</td>\'
                    f\'<td style="text-align: right; font-family: monospace; font-weight: 600; color: #fbbf24;">{r["mean_emissivity"]:.4f}</td>\'
                    f\'<td><code style="background-color: #1e293b; color: #a78bfa; padding: 2px 6px; border-radius: 4px; border: 1px solid #334155;">{md5_short}...</code></td>\'
                    "</tr>"
                )
                table_rows.append(row_html)

            table_html = (
                \'<div class="spec-table-container">\'
                \'<table class="spec-table">\'
                \'<thead><tr>\'
                \'<th>Spectrum ID</th>\'
                \'<th>Specimen Name</th>\'
                \'<th>Partition Role</th>\'
                \'<th>Class</th>\'
                \'<th>Particle Size</th>\'
                \'<th style="text-align: center;">Val Fold</th>\'
                \'<th style="text-align: center;">Train Folds</th>\'
                \'<th style="text-align: right;">Min ε</th>\'
                \'<th style="text-align: right;">Max ε</th>\'
                \'<th style="text-align: right;">Mean ε</th>\'
                \'<th>MD5 Checksum</th>\'
                \'</tr></thead>\'
                \'<tbody>\' + \'\'.join(table_rows) + \'</tbody>\'
                \'</table>\'
                \'</div>\'
            )
            st.markdown(table_html, unsafe_allow_html=True)'''

target_files = [
    "zenodo_release/dataset_explorer_dashboard.py",
    "scripts/dataset_explorer_dashboard.py"
]

start_marker = "            if show_aster_srf:\n"
end_marker = "            st.markdown(table_html, unsafe_allow_html=True)\n"

for fp in target_files:
    if not os.path.exists(fp):
        continue
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()

    is_crlf = "\r\n" in content
    norm_content = content.replace("\r\n", "\n")

    p1 = norm_content.find(start_marker)
    p2 = norm_content.find(end_marker, p1)
    if p1 != -1 and p2 != -1:
        p2 += len(end_marker)
        norm_content = norm_content[:p1] + new_code_block + "\n" + norm_content[p2:]
        if is_crlf:
            final_content = norm_content.replace("\n", "\r\n")
        else:
            final_content = norm_content
        with open(fp, "w", encoding="utf-8") as f:
            f.write(final_content)
        print(f"Updated {fp} successfully!")
    else:
        print(f"Could not find boundaries in {fp}: p1={p1}, p2={p2}")

print("Batch replacement complete.")

