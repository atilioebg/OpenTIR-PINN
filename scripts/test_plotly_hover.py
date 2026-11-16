import plotly.graph_objects as go

b_start, b_end = 8.125, 8.475
b_name = "ASTER Band #10"
b_center = "8.2910 μm"
b_bw = "354.0 nm"

tooltip = (
    f"<b>{b_name}</b><br>"
    f"Center Wavelength: {b_center}<br>"
    f"Bandwidth (FWHM): {b_bw}<br>"
    f"Band Start (λ_start): {b_start:.4f} μm<br>"
    f"Band End (λ_end): {b_end:.4f} μm<br>"
    f"Interval: [{b_start:.4f}, {b_end:.4f}] μm"
)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=[b_start, b_end, b_end, b_start, b_start],
    y=[0.0, 0.0, 1.02, 1.02, 0.0],
    fill="toself",
    fillcolor="rgba(147, 197, 253, 0.18)",
    mode="lines",
    hoveron="points+fills",
    hoverinfo="text",
    name=b_name,
    hovertext=tooltip,
    hovertemplate=tooltip + "<extra></extra>"
))

d1 = fig1.to_dict()
print("Trace 1 hoverinfo:", d1["data"][0].get("hoverinfo"))
print("Trace 1 hoveron:", d1["data"][0].get("hoveron"))
print("Trace 1 name:", d1["data"][0].get("name"))

