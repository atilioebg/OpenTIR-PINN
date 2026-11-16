import os

def update_file(fpath):
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    # Normalize newlines to find target reliably
    is_crlf = "\r\n" in content
    content_norm = content.replace("\r\n", "\n")

    target = """            if show_hytes_srf:
                fig.add_vrect(
                    x0=7.50, x1=12.00,
                    fillcolor="rgba(254, 215, 170, 0.15)",
                    line_width=1.5,
                    line_color="rgba(249, 115, 22, 0.5)",
                    line_dash="dash",
                    annotation_text="HyTES 256 Channels (7.50 - 12.00 μm, Δλ ≈ 17.5 nm)",
                    annotation_position="bottom right"
                )"""

    replacement = """            if show_hytes_srf:
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

    if target not in content_norm:
        print(f"Target not found in {fpath}!")
        return False

    new_content_norm = content_norm.replace(target, replacement)
    new_content = new_content_norm.replace("\n", "\r\n") if is_crlf else new_content_norm
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Successfully updated {fpath}")
    return True

if __name__ == "__main__":
    update_file("zenodo_release/dataset_explorer_dashboard.py")
    update_file("scripts/dataset_explorer_dashboard.py")

