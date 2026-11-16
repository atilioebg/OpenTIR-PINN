"""
Generate an updated, minimalist neural-network node topology diagram for Figure 2 (PINN architecture).
Updates based on PI review:
1. Streamlined bottom-left box: kept only Input Space Decomposition, Features, and Channels.
2. Expanded box width and refined font size so text has generous internal padding on all sides.
3. Completely clean, minimalist layout for RSE publication.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

def create_minimalist_pinn_figure(output_path):
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    fig, ax = plt.subplots(figsize=(14.2, 7.6), dpi=300)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 7.6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Color Palette - Professional Academic
    c_in_fill = '#E0F2FE'       # Soft Blue
    c_in_edge = '#0284C7'
    c_hid_fill = '#F3E8FF'      # Soft Purple
    c_hid_edge = '#7C3AED'
    c_t_fill = '#FFEDD5'        # Soft Orange
    c_t_edge = '#EA580C'
    c_eps_fill = '#DCFCE7'      # Soft Emerald
    c_eps_edge = '#16A34A'
    c_edge_line = '#CBD5E1'
    c_text = '#0F172A'
    c_sub = '#475569'

    # X coordinates of layers
    x_in = 1.7
    x_h1 = 4.3
    x_h2 = 6.4
    x_hn = 8.6
    x_out = 11.1
    x_pred = 12.7

    # Y coordinates of nodes
    # Input Layer (4 nodes + ellipsis)
    y_in = [5.7, 4.7, 3.7, 2.1]
    y_in_dots = 2.9

    # Hidden Layers (5 nodes + ellipsis)
    y_hid = [6.0, 5.0, 4.0, 3.0, 1.6]
    y_hid_dots = 2.3

    # Output Heads
    # Temperature head (top, 1 node)
    y_out_T = [5.85]
    # Emissivity head (bottom, 3 nodes + ellipsis)
    y_out_eps = [4.3, 3.3, 1.8]
    y_out_eps_dots = 2.55

    r_node = 0.36

    # Helper to connect layers
    def connect_layers(x1, ys1, x2, ys2, alpha=0.35, lw=0.9, style='-'):
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1 + r_node, x2 - r_node], [y1, y2],
                        color=c_edge_line, alpha=alpha, lw=lw, linestyle=style, zorder=1)

    # 1. Synaptic Connections
    connect_layers(x_in, y_in, x_h1, y_hid, alpha=0.40, lw=0.9)
    connect_layers(x_h1, y_hid, x_h2, y_hid, alpha=0.40, lw=0.9)
    connect_layers(x_h2, y_hid, x_hn, y_hid, alpha=0.25, lw=0.8, style='--')

    # Connections to Output Heads
    for yh in y_hid:
        ax.plot([x_hn + r_node, x_out - r_node], [yh, y_out_T[0]],
                color='#EA580C', alpha=0.45, lw=1.0, zorder=1)
        for ye in y_out_eps:
            ax.plot([x_hn + r_node, x_out - r_node], [yh, ye],
                    color='#16A34A', alpha=0.35, lw=0.9, zorder=1)

    # Helper to draw a node
    def draw_node(x, y, label, facecolor, edgecolor, fontsize=8.5, fontweight='bold', textcolor=c_text, radius=r_node):
        circ = Circle((x, y), radius, facecolor=facecolor, edgecolor=edgecolor, lw=1.6, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=textcolor, zorder=4)

    # 2. Draw Input Nodes
    in_labels = [r"$\tilde{x}_1$", r"$\tilde{x}_2$", r"$\tilde{x}_3$", r"$\tilde{x}_u$"]
    for y, lbl in zip(y_in, in_labels):
        draw_node(x_in, y, lbl, c_in_fill, c_in_edge, fontsize=10, fontweight='normal')
    ax.text(x_in, y_in_dots, r"$\vdots$", ha='center', va='center', fontsize=15, color=c_sub, fontweight='bold')

    # Input arrows
    for y in y_in:
        ax.annotate('', xy=(x_in - r_node - 0.05, y), xytext=(x_in - r_node - 0.55, y),
                    arrowprops=dict(arrowstyle="-|>", color=c_in_edge, lw=1.4))

    # 3. Draw Hidden Backbone Nodes (with 'SiLU' inside)
    for x_h in [x_h1, x_h2, x_hn]:
        for y in y_hid:
            draw_node(x_h, y, "SiLU", c_hid_fill, c_hid_edge, fontsize=8.2, fontweight='bold', textcolor='#4C1D95')
        ax.text(x_h, y_hid_dots, r"$\vdots$", ha='center', va='center', fontsize=15, color='#6D28D9', fontweight='bold')

    # Horizontal ellipsis between Layer 2 and Layer n
    ax.text((x_h2 + x_hn) / 2.0, 4.0, r"$\cdots$", ha='center', va='center',
            fontsize=24, color='#4C1D95', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9))

    # 4. Draw Output Heads
    # Temperature Head Node (Softplus)
    draw_node(x_out, y_out_T[0], "Softplus", c_t_fill, c_t_edge, fontsize=7.5, fontweight='bold', textcolor='#9A3412', radius=0.40)
    box_T = FancyBboxPatch((x_out - 0.62, y_out_T[0] - 0.58), 1.24, 1.16,
                           boxstyle="round,pad=0.05,rounding_size=0.1",
                           facecolor='none', edgecolor=c_t_edge, lw=1.2, linestyle='--', zorder=2)
    ax.add_patch(box_T)
    ax.text(x_out, y_out_T[0] + 0.75, "Temperature Head", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#C2410C')

    # Emissivity Head Nodes (Sigmoid)
    for y in y_out_eps:
        draw_node(x_out, y, "Sigmoid", c_eps_fill, c_eps_edge, fontsize=7.3, fontweight='bold', textcolor='#166534', radius=0.40)
    ax.text(x_out, y_out_eps_dots, r"$\vdots$", ha='center', va='center', fontsize=15, color='#15803D', fontweight='bold')
    box_eps = FancyBboxPatch((x_out - 0.62, y_out_eps[-1] - 0.55), 1.24, (y_out_eps[0] - y_out_eps[-1]) + 1.10,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor='none', edgecolor=c_eps_edge, lw=1.2, linestyle='--', zorder=2)
    ax.add_patch(box_eps)
    ax.text(x_out, y_out_eps[0] + 0.75, "Emissivity Head", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#15803D')

    # 5. Output Predictions
    # Arrow and text for Temperature
    ax.annotate('', xy=(x_pred - 0.35, y_out_T[0]), xytext=(x_out + 0.42, y_out_T[0]),
                arrowprops=dict(arrowstyle="-|>", color=c_t_edge, lw=1.6))
    ax.text(x_pred + 0.25, y_out_T[0] + 0.12, r"$\hat{T}_s \in \mathbb{R}^{1}$",
            ha='center', va='center', fontsize=10, fontweight='bold', color='#9A3412')
    ax.text(x_pred + 0.25, y_out_T[0] - 0.22, r"$(\hat{T}_s > 200\,\mathrm{K})$",
            ha='center', va='center', fontsize=8, color=c_sub)

    # Arrows and text for Emissivity
    eps_out_labels = [r"$\hat{\varepsilon}_1$", r"$\hat{\varepsilon}_2$", r"$\hat{\varepsilon}_m$"]
    for y, lbl in zip(y_out_eps, eps_out_labels):
        ax.annotate('', xy=(x_pred - 0.45, y), xytext=(x_out + 0.42, y),
                    arrowprops=dict(arrowstyle="-|>", color=c_eps_edge, lw=1.4))
        ax.text(x_pred - 0.15, y, lbl, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#166534')
    ax.text(x_pred - 0.15, y_out_eps_dots, r"$\vdots$", ha='center', va='center', fontsize=13, color='#15803D')

    # Emissivity space badge
    ax.text(x_pred + 0.65, 3.05,
            r"$\hat{\varepsilon} \in (0,1)^{m}$" + "\n" +
            r"$m \in \{5, 223\}$",
            ha='center', va='center', fontsize=8.5, color='#166534',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F0FDF4', edgecolor='#86EFAC', lw=1.0))

    # 6. Top Titles & Mathematical Headers
    y_top = 7.05
    ax.text(x_in, y_top,
            r"$\mathbf{Input\;Layer}$" + "\n" +
            r"$\tilde{\mathbf{x}} \in \mathbb{R}^{u} \cong \mathbb{R}^{n \times m}$",
            ha='center', va='center', fontsize=10.2, fontweight='bold', color='#0369A1')

    ax.text((x_h1 + x_hn) / 2.0, y_top,
            r"$\mathbf{Hidden\;Backbone}\;(n\;\mathrm{layers}\times d\;\mathrm{units})$" + "\n" +
            r"$\mathbf{h}^{(l)} = \mathrm{Dropout}(\mathrm{SiLU}(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}))$",
            ha='center', va='center', fontsize=10.0, fontweight='bold', color='#4C1D95')

    ax.text(x_h1, 6.5, r"Layer 1 ($\mathbb{R}^d$)", ha='center', va='center', fontsize=8.2, color=c_sub)
    ax.text(x_h2, 6.5, r"Layer 2 ($\mathbb{R}^d$)", ha='center', va='center', fontsize=8.2, color=c_sub)
    ax.text(x_hn, 6.5, r"Layer $n$ ($\mathbf{h} \in \mathbb{R}^d$)", ha='center', va='center', fontsize=8.2, color=c_sub)

    ax.text(x_out, y_top,
            r"$\mathbf{Bounded\;Output\;Heads}$" + "\n" +
            r"$(\hat{T}_s, \hat{\varepsilon}) \in \mathbb{R}^{1 + m}$",
            ha='center', va='center', fontsize=10.2, fontweight='bold', color=c_text)

    # 7. Bottom-Left: Streamlined Cardinality Box (With Generous Margins)
    b_card = FancyBboxPatch((0.15, 0.20), 4.15, 0.85, boxstyle="round,pad=0.08,rounding_size=0.1",
                            facecolor='#F8FAFC', edgecolor='#94A3B8', lw=1.1, zorder=2)
    ax.add_patch(b_card)
    ax.text(0.30, 0.625,
            r"$\mathbf{Input\;Space\;Decomposition:}\;\mathbf{x} \in \mathbb{R}^{n \times m}\;(u = n \times m)$" + "\n" +
            r"$\mathbf{Features}\;(n \in \{1,2,3\}):$ Single ($n{=}1$), Dual ($n{=}2$), Triple ($n{=}3$)" + "\n" +
            r"$\mathbf{Channels}\;(m \in \{5,223\}):$ ASTER ($m{=}5$), HyTES ($m{=}223$)",
            ha='left', va='center', fontsize=7.4, color='#1E293B', linespacing=1.4)

    # 8. Bottom-Right: Redesigned Multi-Objective Loss Function Box
    b_loss = FancyBboxPatch((8.8, 0.15), 5.15, 0.85, boxstyle="round,pad=0.08,rounding_size=0.1",
                            facecolor='#FEF2F2', edgecolor='#DC2626', lw=1.2, zorder=2)
    ax.add_patch(b_loss)

    ax.text(11.37, 0.76,
            r"$\mathbf{Multi-Objective\;Loss\;Formulation}\;(\mathcal{L}_{\mathrm{total}})$",
            ha='center', va='center', fontsize=8.6, fontweight='bold', color='#991B1B')

    ax.text(11.37, 0.48,
            r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{sup}} + \mathcal{L}_{\mathrm{Sobolev}} + \mathcal{L}_{\mathrm{phys}}$",
            ha='center', va='center', fontsize=8.4, fontweight='bold', color='#0F172A')

    ax.text(11.37, 0.26,
            r"$\mathcal{L}_{\mathrm{sup}}:\;\mathrm{MSE}_{(T_s,\varepsilon)} + \mathrm{SAM} \quad\vert\quad "
            r"\mathcal{L}_{\mathrm{Sobolev}}:\;W^{1,1}(D^{(1)}) + W^{2,1}(D^{(2)}) \quad\vert\quad "
            r"\mathcal{L}_{\mathrm{phys}}:\;T_b(\hat{L}_{\mathrm{surf}})$",
            ha='center', va='center', fontsize=7.5, color='#450A0A')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Successfully generated updated {output_path}")

if __name__ == '__main__':
    out_dir = r"manuscript\figures"
    out_file = os.path.join(out_dir, "fig02_pinn_architecture.png")
    create_minimalist_pinn_figure(out_file)
