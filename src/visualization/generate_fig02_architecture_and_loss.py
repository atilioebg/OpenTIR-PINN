"""
Publication-grade (Q1 / Remote Sensing of Environment) generator for Figure 2:
Sobolev-regularized PINN Architecture & Multi-Objective Loss Decomposition.

Strictly grounded against:
  - manuscript_rse/sections/04_methods.tex
  - src/models/losses.py
"""

import os
from pathlib import Path

# Set writable Matplotlib cache dir inside scratch to avoid permission warnings
_MPL_CACHE = Path(__file__).resolve().parent / "mpl_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath


def setup_rcparams():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def add_shadow_card(
    ax, x, y, w, h, fc="#FFFFFF", ec="#1E3A8A", lw=1.4, r=1.0,
    shadow_dx=0.32, shadow_dy=-0.32, shadow_color="#CBD5E1", shadow_alpha=0.55,
    zorder=2
):
    sh = FancyBboxPatch(
        (x + shadow_dx, y + shadow_dy), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=shadow_color, ec="none", alpha=shadow_alpha, zorder=zorder - 0.5
    )
    ax.add_patch(sh)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, zorder=zorder
    )
    ax.add_patch(box)
    return box


def add_header_strip(
    ax, x, y_top, w, header_h, color, title, right_badge=None,
    badge_w=None, r=1.0, zorder=3
):
    bar = FancyBboxPatch(
        (x + 0.14, y_top - header_h), w - 0.28, header_h - 0.12,
        boxstyle=f"round,pad=0,rounding_size={max(0.35, r - 0.25)}",
        fc=color, ec="none", zorder=zorder
    )
    ax.add_patch(bar)
    if right_badge:
        ax.text(
            x + 0.95, y_top - header_h * 0.52, title,
            ha="left", va="center", color="#FFFFFF",
            fontsize=9.0, fontweight="bold", zorder=zorder + 0.5
        )
        bw = badge_w if badge_w is not None else max(4.2, len(right_badge) * 0.36 + 1.4)
        bx = x + w - bw - 0.65
        by = y_top - header_h * 0.80
        bh = header_h * 0.58
        pill = FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0,rounding_size=0.45",
            fc="#FFFFFF", ec="none", alpha=0.20, zorder=zorder + 0.3
        )
        ax.add_patch(pill)
        ax.text(
            bx + bw * 0.5, by + bh * 0.48, right_badge,
            ha="center", va="center", color="#FFFFFF",
            fontsize=7.1, fontweight="bold", zorder=zorder + 0.6
        )
    else:
        ax.text(
            x + w * 0.5, y_top - header_h * 0.52, title,
            ha="center", va="center", color="#FFFFFF",
            fontsize=9.0, fontweight="bold", zorder=zorder + 0.5
        )


def add_arrow(ax, p1, p2, color="#1E293B", lw=1.6, ls="-", rad=0.0, hl=5.5, hw=3.5, zorder=5):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle=f"-|>,head_length={hl},head_width={hw}",
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw, linestyle=ls,
        mutation_scale=1.0, zorder=zorder
    )
    ax.add_patch(arr)
    return arr


def add_rounded_ortho_arrow(ax, pts, radius=1.2, color="#1E293B", lw=1.5, ls="-", zorder=4.5):
    """Draw a sleek orthogonal polyline with rounded corners ending in a crisp arrowhead."""
    verts = []
    codes = []
    n = len(pts)
    assert n >= 2
    verts.append(pts[0])
    codes.append(MplPath.MOVETO)

    for i in range(1, n - 1):
        p_prev = np.array(pts[i - 1], dtype=float)
        p_curr = np.array(pts[i], dtype=float)
        p_next = np.array(pts[i + 1], dtype=float)

        v_in = p_prev - p_curr
        v_out = p_next - p_curr
        len_in = np.linalg.norm(v_in)
        len_out = np.linalg.norm(v_out)
        r_eff = min(radius, len_in * 0.45, len_out * 0.45)

        c_start = p_curr + (v_in / len_in) * r_eff
        c_end = p_curr + (v_out / len_out) * r_eff

        verts.append(tuple(c_start))
        codes.append(MplPath.LINETO)
        verts.append(tuple(p_curr))
        codes.append(MplPath.CURVE3)
        verts.append(tuple(c_end))
        codes.append(MplPath.CURVE3)

    p_last = np.array(pts[-1], dtype=float)
    p_before = np.array(pts[-2], dtype=float)
    v_final = p_last - p_before
    v_len = np.linalg.norm(v_final)
    head_gap = min(0.9, v_len * 0.35)
    p_stem_end = p_last - (v_final / v_len) * head_gap

    verts.append(tuple(p_stem_end))
    codes.append(MplPath.LINETO)

    path = MplPath(verts, codes)
    patch = patches.PathPatch(
        path, facecolor="none", edgecolor=color, linewidth=lw,
        linestyle=ls, capstyle="round", zorder=zorder
    )
    ax.add_patch(patch)
    add_arrow(ax, tuple(p_stem_end), tuple(p_last), color=color, lw=lw, ls="-", zorder=zorder + 0.1)


def build_figure(out_png: Path, out_pdf: Path):
    setup_rcparams()

    # Aspect ratio matches .docx frame cx/cy = 5943600 / 4036176 = 1.47258187
    fig_w = 12.2
    fig_h = fig_w / (5943600.0 / 4036176.0)  # 8.2848 in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#FFFFFF")

    ax_a = fig.add_axes([0.010, 0.518, 0.980, 0.468])
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 48)
    ax_a.axis("off")

    ax_b = fig.add_axes([0.010, 0.014, 0.980, 0.486])
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(0, 50)
    ax_b.axis("off")

    # Subtle horizontal section rule between Panel (a) and Panel (b)
    fig.add_artist(
        plt.Line2D([0.018, 0.982], [0.509, 0.509], transform=fig.transFigure,
                   color="#CBD5E1", lw=1.1, ls="-")
    )

    # Cohesive scientific palette
    C_NAVY = "#0F172A"
    C_COBALT = "#1E3A8A"
    C_BLUE = "#1D4ED8"
    C_SKY = "#0284C7"
    C_TEAL = "#0F766E"
    C_EMERALD = "#047857"
    C_CRIMSON = "#B91C1C"
    C_AMBER = "#B45309"
    C_PURPLE = "#6D28D9"
    C_SLATE = "#475569"
    C_MUTED = "#64748B"

    # =========================================================================
    # PANEL (a): SOBOLEV-PINN ARCHITECTURE & RADIATIVE COUPLING
    # =========================================================================
    badge_a = FancyBboxPatch((0.6, 44.3), 3.3, 3.0, boxstyle="round,pad=0,rounding_size=0.6",
                             fc=C_NAVY, ec="none", zorder=4)
    ax_a.add_patch(badge_a)
    ax_a.text(2.25, 45.8, "(a)", ha="center", va="center", color="#FFFFFF",
              fontsize=10.5, fontweight="bold", zorder=5)
    ax_a.text(4.6, 45.8, "Sobolev-PINN Modular Architecture & Forward Radiative Coupling",
              ha="left", va="center", color=C_NAVY, fontsize=11.6, fontweight="bold")

    pill_norm = FancyBboxPatch((64.2, 44.5), 35.0, 2.7, boxstyle="round,pad=0,rounding_size=0.6",
                               fc="#F1F5F9", ec="#94A3B8", lw=0.9, zorder=3)
    ax_a.add_patch(pill_norm)
    ax_a.text(81.7, 45.85, "No BatchNorm / LayerNorm  |  Radiometric Scale Preserved",
              ha="center", va="center", color="#334155", fontsize=7.9, fontweight="bold", zorder=4)

    # --- STAGE 1: Input Conditioning Tiers ---
    s1_x, s1_y, s1_w, s1_h = 0.6, 13.5, 25.4, 29.6
    add_shadow_card(ax_a, s1_x, s1_y, s1_w, s1_h, fc="#FFFFFF", ec=C_COBALT, lw=1.4, r=1.0)
    add_header_strip(ax_a, s1_x, s1_y + s1_h, s1_w, 3.6, C_COBALT, "1. Input Conditioning")

    tiers = [
        ("Single (n=1)", "HyTES 223 ch.", r"$\mathbf{X} = [\hat{\mathbf{L}}_{\mathrm{surf}}]$", "#0F766E", "#F0FDFA"),
        ("Dual (n=2)", "TES Parity", r"$\mathbf{X} = [\hat{\mathbf{L}}_{\mathrm{surf}},\,\hat{\mathbf{L}}^{\downarrow}]$", "#0284C7", "#F0F9FF"),
        ("Triple (n=3)", "ASTER 5 ch.", r"$\mathbf{X} = [\hat{\mathbf{L}}_{\mathrm{surf}},\,\hat{\boldsymbol{\tau}},\,\hat{\mathbf{L}}^{\downarrow}]$", "#1D4ED8", "#EFF6FF"),
        ("Quad (n=4)", "Full TOA", r"$\mathbf{X} = [\mathbf{L}_{\mathrm{sensor}},\,\hat{\boldsymbol{\tau}},\,\hat{\mathbf{L}}^{\uparrow},\,\hat{\mathbf{L}}^{\downarrow}]$", "#6D28D9", "#F5F3FF"),
    ]
    row_h = 5.15
    row_gap = 0.65
    start_y = s1_y + s1_h - 4.3 - row_h
    for idx, (t_name, t_mode, t_eq, t_col, t_bg) in enumerate(tiers):
        ry = start_y - idx * (row_h + row_gap)
        rx = s1_x + 0.82
        rw = s1_w - 1.64
        rbox = FancyBboxPatch((rx, ry), rw, row_h, boxstyle="round,pad=0,rounding_size=0.55",
                              fc=t_bg, ec=t_col, lw=0.9, zorder=3)
        ax_a.add_patch(rbox)
        astripe = FancyBboxPatch((rx + 0.24, ry + 0.55), 0.52, row_h - 1.1,
                                 boxstyle="round,pad=0,rounding_size=0.2",
                                 fc=t_col, ec="none", zorder=4)
        ax_a.add_patch(astripe)
        ax_a.text(rx + 1.30, ry + row_h * 0.72, t_name, ha="left", va="center",
                  color=t_col, fontsize=7.9, fontweight="bold", zorder=4)
        ax_a.text(rx + rw - 0.55, ry + row_h * 0.72, t_mode, ha="right", va="center",
                  color=C_SLATE, fontsize=7.1, fontweight="bold", zorder=4)
        ax_a.text(rx + 1.30, ry + row_h * 0.27, t_eq, ha="left", va="center",
                  color=C_NAVY, fontsize=7.7, zorder=4)

    f1 = FancyBboxPatch((s1_x + 0.82, s1_y + 0.65), s1_w - 1.64, 2.35,
                        boxstyle="round,pad=0,rounding_size=0.4",
                        fc="#F1F5F9", ec="#CBD5E1", lw=0.7, zorder=3)
    ax_a.add_patch(f1)
    ax_a.text(s1_x + s1_w * 0.5, s1_y + 1.82,
              r"Flatten: $\mathbf{x} = \mathrm{vec}(\mathbf{X}) \in \mathbb{R}^{u},\; u = m \times n$",
              ha="center", va="center", color=C_NAVY, fontsize=7.4, zorder=4)

    # Arrow Stage 1 -> Stage 2
    add_arrow(ax_a, (s1_x + s1_w, 28.3), (28.3, 28.3), color=C_NAVY, lw=1.6)
    ax_a.text(27.15, 29.8, r"$\mathbf{x}$", ha="center", va="center", color=C_NAVY, fontsize=8.5, fontweight="bold")

    # --- STAGE 2: Standardization ---
    s2_x, s2_y, s2_w, s2_h = 28.3, 15.2, 11.4, 26.2
    add_shadow_card(ax_a, s2_x, s2_y, s2_w, s2_h, fc="#F8FAFC", ec=C_SKY, lw=1.4, r=1.0)
    add_header_strip(ax_a, s2_x, s2_y + s2_h, s2_w, 3.6, C_SKY, "2. Standardization")

    ax_a.text(s2_x + s2_w * 0.5, s2_y + 19.3,
              r"$\tilde{\mathbf{x}} = (\mathbf{x} - \boldsymbol{\mu}_x) \oslash \boldsymbol{\sigma}_x$",
              ha="center", va="center", color=C_NAVY, fontsize=8.3, fontweight="bold", zorder=4)

    gx0, gx1 = s2_x + 1.0, s2_x + s2_w - 1.0
    gy0, gy1 = s2_y + 7.6, s2_y + 15.4
    gframe = FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0,
                            boxstyle="round,pad=0,rounding_size=0.4",
                            fc="#FFFFFF", ec="#BAE6FD", lw=0.8, zorder=3)
    ax_a.add_patch(gframe)
    zv = np.linspace(-2.8, 2.8, 80)
    gv = np.exp(-0.5 * zv ** 2)
    px = gx0 + 0.5 + (zv + 2.8) / 5.6 * (gx1 - gx0 - 1.0)
    py = gy0 + 1.2 + gv * (gy1 - gy0 - 2.3)
    ax_a.fill_between(px, gy0 + 1.2, py, color="#BAE6FD", alpha=0.65, zorder=3.5)
    ax_a.plot(px, py, color=C_SKY, lw=1.5, zorder=4)
    ax_a.plot([s2_x + s2_w * 0.5, s2_x + s2_w * 0.5], [gy0 + 1.0, gy1 - 0.6],
              color=C_SLATE, lw=0.8, ls="--", zorder=3.8)
    ax_a.text(s2_x + s2_w * 0.5, gy0 + 0.35, r"$\mu=0,\;\sigma=1$",
              ha="center", va="bottom", color=C_SLATE, fontsize=6.6, zorder=4)

    ax_a.text(s2_x + s2_w * 0.5, s2_y + 3.8,
              "Fold-train statistics\nLeak-free transform",
              ha="center", va="center", color=C_SLATE, fontsize=7.3, linespacing=1.4, zorder=4)

    # Arrow Stage 2 -> Stage 3
    add_arrow(ax_a, (s2_x + s2_w, 28.3), (42.0, 28.3), color=C_NAVY, lw=1.6)
    ax_a.text(40.85, 29.8, r"$\tilde{\mathbf{x}}$", ha="center", va="center", color=C_NAVY, fontsize=8.5, fontweight="bold")

    # --- STAGE 3: Sobolev-MLP Backbone ---
    s3_x, s3_y, s3_w, s3_h = 42.0, 13.5, 26.8, 29.6
    add_shadow_card(ax_a, s3_x, s3_y, s3_w, s3_h, fc="#FFFFFF", ec=C_COBALT, lw=1.4, r=1.0)
    add_header_strip(ax_a, s3_x, s3_y + s3_h, s3_w, 3.6, C_COBALT,
                     "3. Sobolev-MLP Backbone", right_badge=r"$\mathbf{h} \in \mathbb{R}^d$", badge_w=4.8)

    nn_x0, nn_y0, nn_w, nn_h = s3_x + 0.8, s3_y + 0.8, 12.4, 24.4
    nn_bg = FancyBboxPatch((nn_x0, nn_y0), nn_w, nn_h, boxstyle="round,pad=0,rounding_size=0.6",
                           fc="#F8FAFC", ec="#E2E8F0", lw=0.8, zorder=2.5)
    ax_a.add_patch(nn_bg)

    layer_xs = [nn_x0 + 1.45, nn_x0 + 4.4, nn_x0 + 8.1, nn_x0 + 10.95]
    layer_counts = [4, 5, 5, 4]
    layer_cols = [C_SKY, C_BLUE, C_COBALT, C_PURPLE]
    layer_labels = [r"$\tilde{\mathbf{x}}$", r"$\mathbf{z}_1$", r"$\mathbf{z}_n$", r"$\mathbf{h}$"]

    def node_ys(count, y_mid=s3_y + 18.6, span=9.0):
        return np.linspace(y_mid + span * 0.5, y_mid - span * 0.5, count)

    all_pos = [list(zip([lx] * cnt, node_ys(cnt))) for lx, cnt in zip(layer_xs, layer_counts)]

    for x1, y1 in all_pos[0]:
        for x2, y2 in all_pos[1]:
            ax_a.plot([x1, x2], [y1, y2], color="#93C5FD", lw=0.55, alpha=0.65, zorder=3)
    for x1, y1 in all_pos[2]:
        for x2, y2 in all_pos[3]:
            ax_a.plot([x1, x2], [y1, y2], color="#A5B4FC", lw=0.55, alpha=0.65, zorder=3)
    for dy in [-2.6, 0.0, 2.6]:
        ax_a.plot([layer_xs[1] + 0.7, layer_xs[2] - 0.7], [s3_y + 18.6 + dy, s3_y + 18.6 + dy],
                  color="#64748B", lw=0.9, ls=":", zorder=3)
    ax_a.text((layer_xs[1] + layer_xs[2]) * 0.5, s3_y + 24.1, r"$\times\, n$ layers",
              ha="center", va="center", color=C_COBALT, fontsize=7.4, fontweight="bold", zorder=4)

    for col_idx, (pts, ncol, lbl) in enumerate(zip(all_pos, layer_cols, layer_labels)):
        for px_n, py_n in pts:
            circ = patches.Ellipse((px_n, py_n), width=0.90, height=1.12,
                                   fc=ncol, ec="#FFFFFF", lw=0.9, zorder=4)
            ax_a.add_patch(circ)
        ax_a.text(layer_xs[col_idx], s3_y + 12.6, lbl, ha="center", va="center",
                  color=ncol, fontsize=7.8, fontweight="bold", zorder=4)

    sx0, sx1 = nn_x0 + 0.65, nn_x0 + nn_w - 0.65
    sy0, sy1 = nn_y0 + 0.8, nn_y0 + 9.2
    sbox = FancyBboxPatch((sx0, sy0), sx1 - sx0, sy1 - sy0,
                          boxstyle="round,pad=0,rounding_size=0.45",
                          fc="#FFFFFF", ec="#CBD5E1", lw=0.8, zorder=3)
    ax_a.add_patch(sbox)
    ax_a.text(sx0 + 0.45, sy1 - 1.15, r"$\mathrm{SiLU}(z) = z \cdot \sigma(z)$",
              ha="left", va="center", color=C_COBALT, fontsize=7.0, fontweight="bold", zorder=4)
    ax_a.text(sx1 - 0.45, sy1 - 1.15, r"$C^{\infty}$",
              ha="right", va="center", color=C_TEAL, fontsize=7.1, fontweight="bold", zorder=4)
    zs = np.linspace(-4.0, 4.0, 90)
    silu = zs / (1.0 + np.exp(-zs))
    spx = sx0 + 0.6 + (zs + 4.0) / 8.0 * (sx1 - sx0 - 1.2)
    spy = sy0 + 1.0 + (silu - silu.min()) / (silu.max() - silu.min()) * (sy1 - sy0 - 3.0)
    zero_x = sx0 + 0.6 + 0.5 * (sx1 - sx0 - 1.2)
    zero_y = sy0 + 1.0 + (0.0 - silu.min()) / (silu.max() - silu.min()) * (sy1 - sy0 - 3.0)
    ax_a.plot([sx0 + 0.6, sx1 - 0.6], [zero_y, zero_y], color="#CBD5E1", lw=0.7, zorder=3.2)
    ax_a.plot([zero_x, zero_x], [sy0 + 0.8, sy1 - 2.0], color="#CBD5E1", lw=0.7, zorder=3.2)
    ax_a.plot(spx, spy, color=C_BLUE, lw=1.6, zorder=4)

    sp_x0, sp_y0, sp_w = s3_x + 13.8, s3_y + 0.8, 12.2
    p_stack = FancyBboxPatch((sp_x0, s3_y + 21.7), sp_w, 3.3, boxstyle="round,pad=0,rounding_size=0.45",
                             fc="#DBEAFE", ec="#3B82F6", lw=0.8, zorder=3)
    ax_a.add_patch(p_stack)
    ax_a.text(sp_x0 + sp_w * 0.5, s3_y + 23.85, "Layer Block:", ha="center", va="center",
              color=C_COBALT, fontsize=7.0, fontweight="bold", zorder=4)
    ax_a.text(sp_x0 + sp_w * 0.5, s3_y + 22.45, "[Linear \u2192 SiLU \u2192 Dropout]", ha="center", va="center",
              color=C_NAVY, fontsize=6.7, fontweight="bold", zorder=4)

    p_nobn = FancyBboxPatch((sp_x0, s3_y + 18.2), sp_w, 2.7, boxstyle="round,pad=0,rounding_size=0.45",
                            fc="#FEF2F2", ec="#FCA5A5", lw=0.8, zorder=3)
    ax_a.add_patch(p_nobn)
    ax_a.text(sp_x0 + sp_w * 0.5, s3_y + 19.55, "No Batch / LayerNorm", ha="center", va="center",
              color=C_CRIMSON, fontsize=7.1, fontweight="bold", zorder=4)

    p_hpo = FancyBboxPatch((sp_x0, s3_y + 9.8), sp_w, 7.6, boxstyle="round,pad=0,rounding_size=0.45",
                           fc="#F8FAFC", ec="#CBD5E1", lw=0.8, zorder=3)
    ax_a.add_patch(p_hpo)
    ax_a.text(sp_x0 + sp_w * 0.5, s3_y + 16.2, "Optuna Search Space", ha="center", va="center",
              color=C_NAVY, fontsize=7.2, fontweight="bold", zorder=4)
    ax_a.text(sp_x0 + 0.55, s3_y + 14.3, r"Depth: $n \in \{2,\dots,7\}$", ha="left", va="center",
              color=C_SLATE, fontsize=7.1, zorder=4)
    ax_a.text(sp_x0 + 0.55, s3_y + 12.5, r"Width: $d \in \{256,\dots,1024\}$", ha="left", va="center",
              color=C_SLATE, fontsize=7.0, zorder=4)
    ax_a.text(sp_x0 + 0.55, s3_y + 10.7, r"Dropout: $p \in [0, 0.10]$", ha="left", va="center",
              color=C_SLATE, fontsize=7.1, zorder=4)

    p_prod = FancyBboxPatch((sp_x0, sp_y0), sp_w, 8.2, boxstyle="round,pad=0,rounding_size=0.45",
                            fc="#EFF6FF", ec="#93C5FD", lw=0.9, zorder=3)
    ax_a.add_patch(p_prod)
    ax_a.text(sp_x0 + sp_w * 0.5, sp_y0 + 6.95, "Selected Production", ha="center", va="center",
              color=C_COBALT, fontsize=7.2, fontweight="bold", zorder=4)
    ax_a.text(sp_x0 + 0.55, sp_y0 + 5.05, "ASTER (5 ch.):", ha="left", va="center",
              color=C_NAVY, fontsize=6.9, fontweight="bold", zorder=4)
    ax_a.text(sp_x0 + 0.55, sp_y0 + 3.70, r"$5 \times 384,\; p=0.020$ (599,814)", ha="left", va="center",
              color=C_BLUE, fontsize=6.4, zorder=4)
    ax_a.text(sp_x0 + 0.55, sp_y0 + 2.05, "HyTES (223 ch.):", ha="left", va="center",
              color=C_NAVY, fontsize=6.9, fontweight="bold", zorder=4)
    ax_a.text(sp_x0 + 0.55, sp_y0 + 0.72, r"$3 \times 768,\; p=0.047$ (1,525,472)", ha="left", va="center",
              color=C_BLUE, fontsize=6.3, zorder=4)

    # Bifurcated Rounded Arrows Stage 3 -> 4a / 4b
    ax_a.text(69.8, 29.8, r"$\mathbf{h}$", ha="center", va="center", color=C_NAVY, fontsize=8.5, fontweight="bold")
    add_rounded_ortho_arrow(ax_a, [(s3_x + s3_w, 28.3), (70.5, 28.3), (70.5, 36.2), (72.6, 36.2)],
                            radius=0.8, color=C_CRIMSON, lw=1.6, zorder=4.5)
    add_rounded_ortho_arrow(ax_a, [(s3_x + s3_w, 28.3), (70.5, 28.3), (70.5, 20.4), (72.6, 20.4)],
                            radius=0.8, color=C_EMERALD, lw=1.6, zorder=4.5)

    # --- STAGE 4a: Temperature Head ---
    h4_x, h4_w, h4a_y, h4_h = 72.6, 23.1, 29.1, 14.0
    add_shadow_card(ax_a, h4_x, h4a_y, h4_w, h4_h, fc="#FFF5F5", ec=C_CRIMSON, lw=1.4, r=1.0)
    add_header_strip(ax_a, h4_x, h4a_y + h4_h, h4_w, 3.4, C_CRIMSON,
                     "4a. Temperature Head", right_badge=r"$\hat{T}_s\,(\mathrm{K})$", badge_w=4.4)

    ax_a.text(h4_x + 0.85, h4a_y + 8.3,
              r"$\hat{T}_s = 200 + \mathrm{Softplus}(\mathbf{w}_T^{\top}\mathbf{h} + b_T)$",
              ha="left", va="center", color="#7F1D1D", fontsize=7.6, fontweight="bold", zorder=4)
    ax_a.text(h4_x + 0.85, h4a_y + 5.0,
              r"$\bullet$ Hard bound: $\hat{T}_s > 200\,\mathrm{K}$",
              ha="left", va="center", color=C_NAVY, fontsize=7.3, zorder=4)
    ax_a.text(h4_x + 0.85, h4a_y + 2.3,
              r"$\bullet$ Prior bias init: $\mu_T$ center",
              ha="left", va="center", color=C_SLATE, fontsize=7.2, zorder=4)

    tx0, tx1 = h4_x + 16.4, h4_x + h4_w - 0.7
    ty0, ty1 = h4a_y + 0.9, h4a_y + 7.2
    tbox = FancyBboxPatch((tx0, ty0), tx1 - tx0, ty1 - ty0,
                          boxstyle="round,pad=0,rounding_size=0.4",
                          fc="#FFFFFF", ec="#FCA5A5", lw=0.8, zorder=3)
    ax_a.add_patch(tbox)
    xs_t = np.linspace(-3.0, 3.0, 60)
    ys_t = np.log1p(np.exp(xs_t))
    tpx = tx0 + 0.5 + (xs_t + 3.0) / 6.0 * (tx1 - tx0 - 1.0)
    tpy = ty0 + 1.5 + ys_t / ys_t.max() * (ty1 - ty0 - 2.3)
    ax_a.plot([tx0 + 0.4, tx1 - 0.4], [ty0 + 1.5, ty0 + 1.5], color="#EF4444", lw=0.8, ls="--", zorder=3.5)
    ax_a.plot(tpx, tpy, color=C_CRIMSON, lw=1.6, zorder=4)
    ax_a.text((tx0 + tx1) * 0.5, ty0 + 0.55, "200 K floor", ha="center", va="center",
              color=C_CRIMSON, fontsize=6.0, fontweight="bold", zorder=4)

    # --- STAGE 4b: Emissivity Head ---
    h4b_y = 13.5
    add_shadow_card(ax_a, h4_x, h4b_y, h4_w, h4_h, fc="#F0FDF4", ec=C_EMERALD, lw=1.4, r=1.0)
    add_header_strip(ax_a, h4_x, h4b_y + h4_h, h4_w, 3.4, C_EMERALD,
                     "4b. Emissivity Head", right_badge=r"$\hat{\boldsymbol{\varepsilon}} \in (0,1)$", badge_w=4.9)

    ax_a.text(h4_x + 0.85, h4b_y + 8.3,
              r"$\hat{\varepsilon}_i = \mathrm{Sigmoid}(\mathbf{w}_{\varepsilon,i}^{\top}\mathbf{h} + b_{\varepsilon,i})$",
              ha="left", va="center", color="#064E3B", fontsize=7.9, fontweight="bold", zorder=4)
    ax_a.text(h4_x + 0.85, h4b_y + 5.0,
              r"$\bullet$ Hard bound: $\hat{\varepsilon}_i \in (0, 1)$",
              ha="left", va="center", color=C_NAVY, fontsize=7.3, zorder=4)
    ax_a.text(h4_x + 0.85, h4b_y + 2.3,
              r"$\bullet$ Prior bias: $\mathrm{logit}(\boldsymbol{\mu}_{\varepsilon})$",
              ha="left", va="center", color=C_SLATE, fontsize=7.2, zorder=4)

    ex0, ex1 = h4_x + 16.4, h4_x + h4_w - 0.7
    ey0, ey1 = h4b_y + 0.9, h4b_y + 7.2
    ebox = FancyBboxPatch((ex0, ey0), ex1 - ex0, ey1 - ey0,
                          boxstyle="round,pad=0,rounding_size=0.4",
                          fc="#FFFFFF", ec="#6EE7B7", lw=0.8, zorder=3)
    ax_a.add_patch(ebox)
    xs_e = np.linspace(-4.5, 4.5, 60)
    ys_e = 1.0 / (1.0 + np.exp(-xs_e))
    epx = ex0 + 0.5 + (xs_e + 4.5) / 9.0 * (ex1 - ex0 - 1.0)
    epy = ey0 + 1.3 + ys_e * (ey1 - ey0 - 2.6)
    ax_a.plot([ex0 + 0.4, ex1 - 0.4], [ey0 + 1.3, ey0 + 1.3], color="#10B981", lw=0.8, ls="--", zorder=3.5)
    ax_a.plot([ex0 + 0.4, ex1 - 0.4], [ey1 - 1.3, ey1 - 1.3], color="#10B981", lw=0.8, ls="--", zorder=3.5)
    ax_a.plot(epx, epy, color=C_EMERALD, lw=1.6, zorder=4)
    ax_a.text(ex0 + 0.55, ey1 - 0.85, "1", ha="left", va="center", color=C_EMERALD, fontsize=6.1, fontweight="bold", zorder=4)
    ax_a.text(ex0 + 0.55, ey0 + 0.70, "0", ha="left", va="center", color=C_EMERALD, fontsize=6.1, fontweight="bold", zorder=4)

    # --- STAGE 5: Forward Radiative Planck Consistency Loop ---
    s5_x, s5_y, s5_w, s5_h = 28.3, 0.8, 49.6, 9.6
    add_shadow_card(ax_a, s5_x, s5_y, s5_w, s5_h, fc="#F5F3FF", ec=C_PURPLE, lw=1.4, r=1.0)
    s5_bar = FancyBboxPatch((s5_x + 0.14, s5_y + s5_h - 2.85), s5_w - 0.28, 2.72,
                            boxstyle="round,pad=0,rounding_size=0.6",
                            fc=C_PURPLE, ec="none", zorder=3)
    ax_a.add_patch(s5_bar)
    ax_a.text(s5_x + 1.2, s5_y + s5_h - 1.48,
              "5. Forward Radiative Planck Consistency (PINN Physical Closure)",
              ha="left", va="center", color="#FFFFFF", fontsize=8.8, fontweight="bold", zorder=4)

    ax_a.text(s5_x + s5_w * 0.5, s5_y + 4.85,
              r"$\hat{L}_{\mathrm{surf,pred},i} = \hat{\varepsilon}_i\, B(\lambda_i, \hat{T}_s) + (1 - \hat{\varepsilon}_i)\,\hat{L}^{\downarrow}_i$",
              ha="center", va="center", color="#4C1D95", fontsize=8.8, fontweight="bold", zorder=4)
    ax_a.text(s5_x + s5_w * 0.5, s5_y + 1.95,
              r"Brightness-Temperature Residual: $\mathcal{L}_{\mathrm{phys}} = \frac{1}{s_{\mathrm{phys}} N_{\mathrm{eff}}} \sum_i \left[\mathrm{BT}(\lambda_i,\hat{L}_{\mathrm{surf,pred},i}) - \mathrm{BT}(\lambda_i,\hat{L}_{\mathrm{surf,in},i})\right]^2$",
              ha="center", va="center", color=C_NAVY, fontsize=7.5, zorder=4)

    add_rounded_ortho_arrow(ax_a, [(13.3, s1_y), (13.3, 5.6), (s5_x, 5.6)],
                            radius=1.3, color=C_PURPLE, lw=1.5, ls="--", zorder=4.5)
    ax_a.text(20.9, 7.2, r"$\hat{\mathbf{L}}^{\downarrow},\;\hat{\mathbf{L}}_{\mathrm{surf,in}}$",
              ha="center", va="center", color=C_PURPLE, fontsize=8.4, fontweight="bold", zorder=5)

    add_rounded_ortho_arrow(ax_a, [(h4_x + h4_w, 36.2), (98.5, 36.2), (98.5, 6.8), (s5_x + s5_w, 6.8)],
                            radius=1.2, color=C_CRIMSON, lw=1.5, ls="-", zorder=4.5)
    ax_a.text(88.4, 8.25, r"$\hat{T}_s$ (K)", ha="center", va="center",
              color=C_CRIMSON, fontsize=8.2, fontweight="bold", zorder=5)

    add_rounded_ortho_arrow(ax_a, [(h4_x + h4_w, 20.4), (96.9, 20.4), (96.9, 3.4), (s5_x + s5_w, 3.4)],
                            radius=1.1, color=C_EMERALD, lw=1.5, ls="-", zorder=4.5)
    ax_a.text(88.4, 1.85, r"$\hat{\boldsymbol{\varepsilon}}$", ha="center", va="center",
              color=C_EMERALD, fontsize=8.6, fontweight="bold", zorder=5)

    add_arrow(ax_a, (55.4, s5_y + s5_h), (55.4, s3_y), color=C_PURPLE, lw=1.6, ls="--")
    ax_a.text(56.4, 11.9, r"Backprop $\nabla_{\theta}\mathcal{L}_{\mathrm{phys}}$",
              ha="left", va="center", color=C_PURPLE, fontsize=7.6, fontweight="bold", zorder=5)

    # =========================================================================
    # PANEL (b): MULTI-OBJECTIVE TRAINING LOSS DECOMPOSITION
    # =========================================================================
    badge_b = FancyBboxPatch((0.6, 46.2), 3.3, 3.0, boxstyle="round,pad=0,rounding_size=0.6",
                             fc=C_NAVY, ec="none", zorder=4)
    ax_b.add_patch(badge_b)
    ax_b.text(2.25, 47.7, "(b)", ha="center", va="center", color="#FFFFFF",
              fontsize=10.5, fontweight="bold", zorder=5)
    ax_b.text(4.6, 47.7, r"Multi-Objective Training Loss Decomposition ($\mathcal{L}_{\mathrm{total}}$)",
              ha="left", va="center", color=C_NAVY, fontsize=11.6, fontweight="bold")

    pill_b = FancyBboxPatch((64.2, 46.4), 35.0, 2.7, boxstyle="round,pad=0,rounding_size=0.6",
                            fc="#F1F5F9", ec="#94A3B8", lw=0.9, zorder=3)
    ax_b.add_patch(pill_b)
    ax_b.text(81.7, 47.75, "4 Physical & Spectral Pillars  |  Unit-Free Variance Scaling",
              ha="center", va="center", color="#334155", fontsize=7.9, fontweight="bold", zorder=4)

    add_shadow_card(ax_b, 0.6, 37.5, 98.8, 7.6, fc="#F8FAFC", ec="#334155", lw=1.3, r=0.9)
    ax_b.text(
        50.0, 42.3,
        r"$\mathcal{L}_{\mathrm{total}} \;=\; \frac{w_{\varepsilon}}{\overline{\sigma_{\varepsilon}^2}}\,\mathcal{L}_{\mathrm{MSE},\varepsilon} \;+\; \frac{w_T}{\sigma_T^2}\,\mathcal{L}_{\mathrm{MSE},T} \;+\; w_{\mathrm{sam}}\,\mathcal{L}_{\mathrm{sam}} \;+\; w_{\mathrm{grad}}\,\mathcal{L}_{\mathrm{grad}} \;+\; w_{\mathrm{curv}}\,\mathcal{L}_{\mathrm{curv}} \;+\; w_{\mathrm{phys}}\,\mathcal{L}_{\mathrm{phys}}$",
        ha="center", va="center", color=C_NAVY, fontsize=10.3, fontweight="bold", zorder=4
    )
    ax_b.text(
        50.0, 39.0,
        r"Normalization: $w_T = 1.0$; training-fold target variances $(\overline{\sigma_{\varepsilon}^2},\, \sigma_T^2)$ and BT floor $s_{\mathrm{phys}} = 1.6\,\mathrm{K}^2$ unify Kelvin and unitless $\varepsilon$ scales.",
        ha="center", va="center", color=C_SLATE, fontsize=7.7, zorder=4
    )

    card_w = 23.6
    card_gap = 1.466
    card_y = 6.3
    card_h = 28.7
    xs_cards = [0.6 + i * (card_w + card_gap) for i in range(4)]
    card_colors = [C_BLUE, C_TEAL, C_AMBER, C_PURPLE]
    card_bgs = ["#EFF6FF", "#F0FDFA", "#FFFBEB", "#F5F3FF"]

    for cx_card, ccol in zip(xs_cards, card_colors):
        mid_x = cx_card + card_w * 0.5
        ax_b.scatter([mid_x], [37.5], s=26, color=ccol, zorder=6)
        add_arrow(ax_b, (mid_x, 37.5), (mid_x, card_y + card_h), color=ccol, lw=1.5)

    def make_uv_mapper(ix0, iy0, ix1, iy1):
        def m(u, v):
            return ix0 + np.asarray(u) * (ix1 - ix0), iy0 + np.asarray(v) * (iy1 - iy0)
        return m

    # --- PILLAR 1: Supervised Amplitude ---
    cx, ccol, cbg = xs_cards[0], card_colors[0], card_bgs[0]
    add_shadow_card(ax_b, cx, card_y, card_w, card_h, fc="#FFFFFF", ec=ccol, lw=1.4, r=1.0)
    add_header_strip(ax_b, cx, card_y + card_h, card_w, 3.6, ccol, "1. Supervised Amplitude")
    ax_b.text(cx + card_w * 0.5, card_y + 22.1,
              r"$\mathcal{L}_{\mathrm{MSE},\varepsilon} = \frac{1}{N_{\mathrm{eff}}}\sum_{i=1}^{N_{\mathrm{eff}}} (\hat{\varepsilon}_i - \varepsilon_i)^2$",
              ha="center", va="center", color=C_NAVY, fontsize=7.9, zorder=4)
    ax_b.text(cx + card_w * 0.5, card_y + 17.3,
              r"$\mathcal{L}_{\mathrm{MSE},T} = (\hat{T}_s - T_s)^2$",
              ha="center", va="center", color=C_NAVY, fontsize=8.2, zorder=4)

    ix0, iy0, ix1, iy1 = cx + 1.35, card_y + 4.3, cx + card_w - 1.35, card_y + 14.3
    mframe = FancyBboxPatch((ix0, iy0), ix1 - ix0, iy1 - iy0, boxstyle="round,pad=0,rounding_size=0.45",
                            fc=cbg, ec="#BFDBFE", lw=0.8, zorder=3)
    ax_b.add_patch(mframe)
    mp = make_uv_mapper(ix0 + 0.9, iy0 + 0.9, ix1 - 0.9, iy1 - 1.0)
    u_x = np.linspace(0.02, 0.60, 50)
    v_true = 0.52 + 0.12 * np.sin(8 * u_x) - 0.18 * np.exp(-((u_x - 0.30) / 0.08) ** 2)
    v_pred = v_true + 0.11 * np.cos(4 * u_x) + 0.04
    pxt, pyt = mp(u_x, v_true)
    pxp, pyp = mp(u_x, v_pred)
    ax_b.fill_between(pxt, pyt, pyp, color="#93C5FD", alpha=0.55, zorder=3.5)
    ax_b.plot(pxt, pyt, color=C_NAVY, lw=1.4, zorder=4)
    ax_b.plot(pxp, pyp, color=C_BLUE, lw=1.4, ls="--", zorder=4.2)
    dvx0, dvy0 = mp(0.67, 0.10)
    dvx1, dvy1 = mp(0.67, 0.92)
    ax_b.plot([dvx0, dvx1], [dvy0, dvy1], color="#BFDBFE", lw=0.9, zorder=3.5)
    bx1_0, by1_0 = mp(0.73, 0.14)
    bx1_1, by1_1 = mp(0.81, 0.72)
    bx2_0, by2_0 = mp(0.85, 0.14)
    bx2_1, by2_1 = mp(0.93, 0.60)
    ax_b.add_patch(patches.Rectangle((bx1_0, by1_0), bx1_1 - bx1_0, by1_1 - by1_0, fc=C_NAVY, alpha=0.85, zorder=4))
    ax_b.add_patch(patches.Rectangle((bx2_0, by2_0), bx2_1 - bx2_0, by2_1 - by2_0, fc=C_BLUE, alpha=0.85, zorder=4))
    ref_x0, ref_y = mp(0.71, 0.72)
    ref_x1, _ = mp(0.95, 0.72)
    ax_b.plot([ref_x0, ref_x1], [ref_y, ref_y], color=C_NAVY, lw=0.8, ls=":", zorder=4.3)
    tx_lbl, ty_lbl = mp(0.31, 0.89)
    ax_b.text(tx_lbl, ty_lbl, r"$\hat{\varepsilon}$ vs $\varepsilon$", ha="center", va="center", color=C_NAVY, fontsize=6.6, fontweight="bold", zorder=4)
    tx_lbl2, ty_lbl2 = mp(0.83, 0.89)
    ax_b.text(tx_lbl2, ty_lbl2, r"$T_s\;/\;\hat{T}_s$", ha="center", va="center", color=C_BLUE, fontsize=6.6, fontweight="bold", zorder=4)

    ax_b.text(cx + card_w * 0.5, card_y + 2.15,
              "Anchors absolute emissivity level\nand scalar kinetic temperature",
              ha="center", va="center", color=C_SLATE, fontsize=7.1, linespacing=1.28, zorder=4)

    # --- PILLAR 2: Spectral Orientation (SAM) ---
    cx, ccol, cbg = xs_cards[1], card_colors[1], card_bgs[1]
    add_shadow_card(ax_b, cx, card_y, card_w, card_h, fc="#FFFFFF", ec=ccol, lw=1.4, r=1.0)
    add_header_strip(ax_b, cx, card_y + card_h, card_w, 3.6, ccol, "2. Spectral Orientation")
    ax_b.text(cx + card_w * 0.5, card_y + 22.0,
              r"$\mathcal{L}_{\mathrm{sam}} = 1 - \frac{\langle \hat{\boldsymbol{\varepsilon}}, \boldsymbol{\varepsilon} \rangle}{\|\hat{\boldsymbol{\varepsilon}}\|_2\,\|\boldsymbol{\varepsilon}\|_2 + 10^{-8}}$",
              ha="center", va="center", color=C_NAVY, fontsize=7.9, zorder=4)
    ax_b.text(cx + card_w * 0.5, card_y + 17.2,
              r"Cosine distance: $1 - \cos(\theta_{\mathrm{SAM}})$",
              ha="center", va="center", color=C_TEAL, fontsize=7.5, fontweight="bold", zorder=4)

    ix0, iy0, ix1, iy1 = cx + 1.35, card_y + 4.3, cx + card_w - 1.35, card_y + 14.3
    mframe = FancyBboxPatch((ix0, iy0), ix1 - ix0, iy1 - iy0, boxstyle="round,pad=0,rounding_size=0.45",
                            fc=cbg, ec="#99F6E4", lw=0.8, zorder=3)
    ax_b.add_patch(mframe)
    mp = make_uv_mapper(ix0 + 1.0, iy0 + 0.9, ix1 - 1.0, iy1 - 0.9)
    ox, oy = mp(0.14, 0.14)
    v1x, v1y = mp(0.84, 0.36)
    v2x, v2y = mp(0.64, 0.78)
    add_arrow(ax_b, (ox, oy), (v1x, v1y), color=C_NAVY, lw=1.7, hl=5.0, hw=3.2, zorder=4.5)
    add_arrow(ax_b, (ox, oy), (v2x, v2y), color=C_TEAL, lw=1.7, ls="--", hl=5.0, hw=3.2, zorder=4.5)
    ax_b.scatter([ox], [oy], s=18, color=C_NAVY, zorder=5)

    ang1 = np.arctan2(v1y - oy, v1x - ox)
    ang2 = np.arctan2(v2y - oy, v2x - ox)
    thetas = np.linspace(ang1, ang2, 30)
    r_arc = 3.6
    ax_b.plot(ox + r_arc * np.cos(thetas), oy + r_arc * np.sin(thetas), color=C_TEAL, lw=1.4, zorder=4)
    ax_b.fill(np.append([ox], ox + r_arc * np.cos(thetas)), np.append([oy], oy + r_arc * np.sin(thetas)),
              color="#99F6E4", alpha=0.55, zorder=3.5)

    l1x, l1y = mp(0.87, 0.36)
    l2x, l2y = mp(0.56, 0.87)
    tax, tay = mp(0.52, 0.44)
    ax_b.text(l1x, l1y, r"$\boldsymbol{\varepsilon}$", ha="left", va="center", color=C_NAVY, fontsize=7.8, fontweight="bold", zorder=5)
    ax_b.text(l2x, l2y, r"$\hat{\boldsymbol{\varepsilon}}$", ha="right", va="center", color=C_TEAL, fontsize=7.8, fontweight="bold", zorder=5)
    ax_b.text(tax, tay, r"$\theta_{\mathrm{SAM}} \to 0$", ha="center", va="center", color=C_TEAL, fontsize=7.1, fontweight="bold", zorder=5)

    ax_b.text(cx + card_w * 0.5, card_y + 2.15,
              "Scale-invariant global alignment\nacross spectral channels",
              ha="center", va="center", color=C_SLATE, fontsize=7.1, linespacing=1.28, zorder=4)

    # --- PILLAR 3: Sobolev Differential Geometry ---
    cx, ccol, cbg = xs_cards[2], card_colors[2], card_bgs[2]
    add_shadow_card(ax_b, cx, card_y, card_w, card_h, fc="#FFFFFF", ec=ccol, lw=1.4, r=1.0)
    add_header_strip(ax_b, cx, card_y + card_h, card_w, 3.6, ccol, r"3. Sobolev ($W^{1,1}, W^{2,1}$)")
    ax_b.text(cx + card_w * 0.5, card_y + 22.3,
              r"$s_i = \frac{\varepsilon_{i+1}-\varepsilon_i}{\lambda_{i+1}-\lambda_i},\;\mathcal{L}_{\mathrm{grad}} = \frac{1}{N_{\mathrm{eff}}-1}\sum_i |\hat{s}_i - s_i|$",
              ha="center", va="center", color=C_NAVY, fontsize=7.0, zorder=4)
    ax_b.text(cx + card_w * 0.5, card_y + 17.5,
              r"$\mathcal{L}_{\mathrm{curv}} = \frac{1}{N_{\mathrm{eff}}-2}\sum_i \left|\frac{\hat{s}_i-\hat{s}_{i-1}}{\bar{h}_i} - \frac{s_i-s_{i-1}}{\bar{h}_i}\right|$",
              ha="center", va="center", color=C_NAVY, fontsize=7.1, zorder=4)

    ix0, iy0, ix1, iy1 = cx + 1.35, card_y + 4.3, cx + card_w - 1.35, card_y + 14.3
    mframe = FancyBboxPatch((ix0, iy0), ix1 - ix0, iy1 - iy0, boxstyle="round,pad=0,rounding_size=0.45",
                            fc=cbg, ec="#FDE68A", lw=0.8, zorder=3)
    ax_b.add_patch(mframe)
    mp = make_uv_mapper(ix0 + 0.85, iy0 + 0.85, ix1 - 0.85, iy1 - 0.9)
    u_w = np.linspace(0.0, 1.0, 64)
    v_rest = 0.58 - 0.44 * np.exp(-((u_w - 0.44) / 0.13) ** 2) - 0.22 * np.exp(-((u_w - 0.73) / 0.08) ** 2)
    saw = 0.075 * np.sin(40 * u_w)
    pxr, pyr = mp(u_w, v_rest)
    pxs, pys = mp(u_w, np.clip(v_rest + saw, 0.05, 0.67))
    ax_b.plot(pxs, pys, color="#94A3B8", lw=1.0, ls=":", zorder=3.8)
    ax_b.plot(pxr, pyr, color=C_AMBER, lw=1.7, zorder=4.2)
    tlx, tly = mp(0.03, 0.87)
    tlx2, tly2 = mp(0.97, 0.87)
    ax_b.text(tlx, tly, r"Smooth $W^{1,1}/W^{2,1}$", ha="left", va="center", color=C_AMBER, fontsize=6.4, fontweight="bold", zorder=5)
    ax_b.text(tlx2, tly2, "Sawtooth", ha="right", va="center", color=C_MUTED, fontsize=6.4, fontweight="bold", zorder=5)

    ax_b.text(cx + card_w * 0.5, card_y + 2.15,
              r"$\Delta\lambda_i,\;\bar{h}_i$ scaled $L_1$ geometry;"
              + "\nprevents Reststrahlen sawtooth",
              ha="center", va="center", color=C_SLATE, fontsize=7.0, linespacing=1.28, zorder=4)

    # --- PILLAR 4: Forward Planck Consistency ---
    cx, ccol, cbg = xs_cards[3], card_colors[3], card_bgs[3]
    add_shadow_card(ax_b, cx, card_y, card_w, card_h, fc="#FFFFFF", ec=ccol, lw=1.4, r=1.0)
    add_header_strip(ax_b, cx, card_y + card_h, card_w, 3.6, ccol, "4. Planck Consistency")
    ax_b.text(cx + card_w * 0.5, card_y + 22.3,
              r"$\hat{L}_{\mathrm{pred},i} = \hat{\varepsilon}_i B(\lambda_i, \hat{T}_s) + (1-\hat{\varepsilon}_i)\hat{L}^{\downarrow}_i$",
              ha="center", va="center", color=C_NAVY, fontsize=7.4, zorder=4)
    ax_b.text(cx + card_w * 0.5, card_y + 17.5,
              r"$\mathcal{L}_{\mathrm{phys}} = \frac{\|\mathrm{BT}(\hat{L}_{\mathrm{pred}}) - \mathrm{BT}(\hat{L}_{\mathrm{in}})\|_2^2}{s_{\mathrm{phys}}\, N_{\mathrm{eff}}}$",
              ha="center", va="center", color=C_NAVY, fontsize=7.7, zorder=4)

    ix0, iy0, ix1, iy1 = cx + 1.35, card_y + 4.3, cx + card_w - 1.35, card_y + 14.3
    mframe = FancyBboxPatch((ix0, iy0), ix1 - ix0, iy1 - iy0, boxstyle="round,pad=0,rounding_size=0.45",
                            fc=cbg, ec="#DDD6FE", lw=0.8, zorder=3)
    ax_b.add_patch(mframe)
    mp = make_uv_mapper(ix0 + 0.85, iy0 + 0.85, ix1 - 0.85, iy1 - 0.9)
    u_bt = np.linspace(0.02, 0.98, 55)
    v_bt1 = 0.54 - 0.25 * np.exp(-((u_bt - 0.45) / 0.14) ** 2)
    v_bt2 = v_bt1 + 0.06 * np.sin(5 * u_bt) + 0.06
    p1x, p1y = mp(u_bt, v_bt1)
    p2x, p2y = mp(u_bt, v_bt2)
    ax_b.fill_between(p1x, p1y, p2y, color="#C4B5FD", alpha=0.65, zorder=3.5)
    ax_b.plot(p1x, p1y, color=C_NAVY, lw=1.4, zorder=4)
    ax_b.plot(p2x, p2y, color=C_PURPLE, lw=1.4, ls="--", zorder=4.2)
    bt_lx, bt_ly = mp(0.50, 0.87)
    bt_rx, bt_ry = mp(0.50, 0.12)
    ax_b.text(bt_lx, bt_ly, r"$\mathrm{BT}(\hat{L}_{\mathrm{pred}}) \to \mathrm{BT}(\hat{L}_{\mathrm{in}})$",
              ha="center", va="center", color=C_PURPLE, fontsize=6.5, fontweight="bold", zorder=5)
    ax_b.text(bt_rx, bt_ry, r"$s_{\mathrm{phys}} = 1.6\,\mathrm{K}^2$",
              ha="center", va="center", color=C_NAVY, fontsize=6.5, fontweight="bold", zorder=5)

    ax_b.text(cx + card_w * 0.5, card_y + 2.15,
              "Couples retrieved outputs via\nKirchhoff surface radiance law",
              ha="center", va="center", color=C_SLATE, fontsize=7.1, linespacing=1.28, zorder=4)

    # Bottom Synthesis Strip
    add_shadow_card(ax_b, 0.6, 0.6, 98.8, 4.2, fc="#F1F5F9", ec="#64748B", lw=1.0, r=0.7,
                    shadow_dx=0.25, shadow_dy=-0.25)
    for div_x in [33.53, 66.47]:
        ax_b.plot([div_x, div_x], [1.15, 4.25], color="#94A3B8", lw=0.9, zorder=4)

    ax_b.text(17.0, 3.55, r"Sobolev $L_1$ Regularization", ha="center", va="center",
              color=C_NAVY, fontsize=7.3, fontweight="bold", zorder=4)
    ax_b.text(17.0, 1.85, r"Preserves sharp mineral bands; avoids $L_2$ rounding.",
              ha="center", va="center", color=C_SLATE, fontsize=6.9, zorder=4)

    ax_b.text(50.0, 3.55, "Training vs. Model Selection", ha="center", va="center",
              color=C_NAVY, fontsize=7.3, fontweight="bold", zorder=4)
    ax_b.text(50.0, 1.85, r"$\mathcal{L}_{\mathrm{total}}$ updates $\theta$; validation score $\mathcal{J}$ guides Optuna.",
              ha="center", va="center", color=C_SLATE, fontsize=6.9, zorder=4)

    ax_b.text(83.0, 3.55, "Cross-Sensor Grid Invariance", ha="center", va="center",
              color=C_NAVY, fontsize=7.3, fontweight="bold", zorder=4)
    ax_b.text(83.0, 1.85, r"$\Delta\lambda_i,\,\bar{h}_i$ unify ASTER (5 ch.) and HyTES (223 ch.).",
              ha="center", va="center", color=C_SLATE, fontsize=6.9, zorder=4)

    fig.savefig(out_png, dpi=300, facecolor="#FFFFFF")
    fig.savefig(out_pdf, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Saved: {out_png} and {out_pdf}")


if __name__ == "__main__":
    root = Path(r"c:\Users\Atilio\Desktop\PROJETOS\AITIR")
    build_figure(
        root / "fig02_pinn_architecture_and_loss.png",
        root / "fig02_pinn_architecture_and_loss.pdf",
    )
