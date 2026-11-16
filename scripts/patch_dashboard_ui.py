import os

def patch_file(fpath):
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    is_crlf = "\r\n" in content
    norm = content.replace("\r\n", "\n")

    # 1. Patch Tab 1 ASTER and HyTES overlay
    old_tab1_overlay = """            if show_aster_srf:
                aster_bands = [
                    (8.125, 8.475, "ASTER B10 (8.3 μm)"),
                    (8.475, 8.825, "ASTER B11 (8.6 μm)"),
                    (8.925, 9.275, "ASTER B12 (9.1 μm)"),
                    (10.25, 10.95, "ASTER B13 (10.6 μm)"),
                    (10.95, 11.65, "ASTER B14 (11.3 μm)")
                ]
                for b_start, b_end, b_name in aster_bands:
                    fig.add_vrect(
                        x0=b_start, x1=b_end,
                        fillcolor="rgba(147, 197, 253, 0.18)",
                        line_width=1,
                        line_color="rgba(59, 130, 246, 0.4)",
                        annotation_text=b_name,
                        annotation_position="top left"
                    )

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

                    fig.add_annotation(
                        x=9.74, y=1.0,
                        xref="x", yref="paper",
                        text="HyTES 256 Spectrometer Channels (7.49 – 11.99 μm, Δλ ≈ 17.5 nm)",
                        showarrow=False,
                        font=dict(size=11, color="#C2410C"),
                        bgcolor="rgba(255, 237, 213, 0.90)",
                        bordercolor="#F97316",
                        borderwidth=1,
                        borderpad=3,
                        yanchor="bottom"
                    )
                except Exception as e:
                    st.warning(f"Error rendering HyTES channels: {e}")"""

    new_tab1_overlay = """            if show_aster_srf:
                aster_bands = [
                    (8.125, 8.475, "ASTER Band 10", "8.3 μm"),
                    (8.475, 8.825, "ASTER Band 11", "8.6 μm"),
                    (8.925, 9.275, "ASTER Band 12", "9.1 μm"),
                    (10.25, 10.95, "ASTER Band 13", "10.6 μm"),
                    (10.95, 11.65, "ASTER Band 14", "11.3 μm")
                ]
                for b_start, b_end, b_name, b_center in aster_bands:
                    fig.add_trace(go.Scatter(
                        x=[b_start, b_end, b_end, b_start],
                        y=[0.0, 0.0, 1.02, 1.02],
                        fill="toself",
                        fillcolor="rgba(147, 197, 253, 0.18)",
                        line=dict(color="rgba(59, 130, 246, 0.40)", width=1),
                        hoveron="fills",
                        hoverinfo="text",
                        hovertext=(
                            f"<b>{b_name} ({b_center})</b><br>"
                            f"Spectral Window: {b_start:.3f} – {b_end:.3f} μm<br>"
                            f"Center Wavelength: {(b_start+b_end)/2:.3f} μm<br>"
                            f"Bandwidth (FWHM): {(b_end - b_start)*1000:.0f} nm"
                        ),
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

                    # Interactive tooltip across channel centers without static text collision
                    fig.add_trace(go.Scatter(
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
                    ))
                except Exception as e:
                    st.warning(f"Error rendering HyTES channels: {e}")"""

    if old_tab1_overlay in norm:
        norm = norm.replace(old_tab1_overlay, new_tab1_overlay)
        print(f"Patched Tab 1 in {fpath}")
    else:
        print(f"Tab 1 target not found in {fpath}")

    # 2. Patch Tab 2 legend position to bottom right corner
    old_tab2_legend = 'legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)'
    new_tab2_legend = 'legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom", bgcolor="rgba(255, 255, 255, 0.85)", bordercolor="#CBD5E1", borderwidth=1)'
    # Note: in Tab 2, line 659
    if old_tab2_legend in norm:
        # replace only the first occurrence (Tab 2)
        norm = norm.replace(old_tab2_legend, new_tab2_legend, 1)
        print(f"Patched Tab 2 legend in {fpath}")

    # 3. Patch Tab 3 y-axis range to [0.40, 1.02]
    old_tab3_yaxis = """        fig_env.update_layout(
            title=f"Spectral Dispersion & Envelope: Class '{comp_class.upper()}' (N = {len(class_df)})",
            xaxis_title="Wavelength λ (μm)",
            yaxis_title="Directional Emissivity (ε)",
            yaxis=dict(range=[0.0, 1.02]),
            xaxis=dict(range=[7.5, 13.5]),
            template="plotly_white",
            height=500
        )"""

    new_tab3_yaxis = """        fig_env.update_layout(
            title=f"Spectral Dispersion & Envelope: Class '{comp_class.upper()}' (N = {len(class_df)})",
            xaxis_title="Wavelength λ (μm)",
            yaxis_title="Directional Emissivity (ε)",
            yaxis=dict(range=[0.40, 1.02]),
            xaxis=dict(range=[7.5, 13.5]),
            template="plotly_white",
            height=500
        )"""

    if old_tab3_yaxis in norm:
        norm = norm.replace(old_tab3_yaxis, new_tab3_yaxis)
        print(f"Patched Tab 3 yaxis in {fpath}")
    else:
        print(f"Tab 3 target not found in {fpath}")

    # 4. Patch Tab 4 dynamic benchmark matching
    old_tab4_block = """        anorthite_row = catalog_df[catalog_df["spectrum_id"].str.contains("feldspar", case=False, na=False)].iloc[0]
        anorthite_emis = emissivities[anorthite_row.name]
        fig_forensic.add_trace(go.Scatter(
            x=wavelengths, y=anorthite_emis, mode="lines",
            name=f"True Physical Benchmark: {anorthite_row['name']} ({anorthite_row['sample_id']})",
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
            title=f"Forensic Comparison: Quarantined Artifact vs. True Physical Anorthite",
            xaxis_title="Wavelength λ (μm)",
            yaxis_title="Calculated Emissivity (ε)",
            yaxis=dict(range=[0.0, 1.05]),
            xaxis=dict(range=[7.5, 13.5]),
            template="plotly_white",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )"""

    new_tab4_block = """        # Dynamically identify physical benchmark counterpart for the selected specimen
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
            legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom", bgcolor="rgba(255, 255, 255, 0.85)", bordercolor="#CBD5E1", borderwidth=1)
        )"""

    if old_tab4_block in norm:
        norm = norm.replace(old_tab4_block, new_tab4_block)
        print(f"Patched Tab 4 in {fpath}")
    else:
        print(f"Tab 4 target not found in {fpath}")

    updated = norm.replace("\n", "\r\n") if is_crlf else norm
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"Saved {fpath}")

if __name__ == "__main__":
    patch_file("zenodo_release/dataset_explorer_dashboard.py")
    patch_file("scripts/dataset_explorer_dashboard.py")

