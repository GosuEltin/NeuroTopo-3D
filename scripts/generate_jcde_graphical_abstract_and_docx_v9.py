"""Generate Official JCDE Submission Companion Files (Graphical Abstract, Highlights.docx, Cover_Letter.docx, Figure_Alt_Text.docx).

Strictly adheres to Oxford University Press / Journal of Computational Design and Engineering (JCDE) author guidelines:
  1. Graphical Abstract:
     - A4 portrait format (`8.27 x 11.69 in`) exported as PDF (for peer review) and 300-DPI JPG + TIFF (for production).
     - Strictly pictorial & visual (NO algorithm flowcharts permitted by JCDE!).
  2. Highlights (`3_Highlights_JCDE.docx`):
     - A4 portrait MS Word (.docx) with heading "Highlights" on the first line.
     - 3 to 5 bullet points, each strictly <= 15 words AND <= 100 characters (including spaces).
  3. Cover Letter (`4_Cover_Letter_JCDE.docx`):
     - Addressed to the Editor-in-Chief of JCDE, stating novelty, originality, and explicit Conflicts of Interest declaration.
  4. Figure Alt Text (`5_Figure_Alt_Text_JCDE.docx`):
     - 1-2 descriptive accessibility sentences per figure (`Figure 1` to `Figure 10` + `Graphical Abstract`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import docx
from docx.shared import Inches, Pt, Mm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from src.cad.step_exporter_v9 import fit_face_plane, orient_faces_outward

RUN_DIR = ROOT / "runs_q1_journal_v9"
CAD_DIR = RUN_DIR / "cad_exports"
PKG_DIR = ROOT / "jcde_submission_package"
PKG_DIR.mkdir(parents=True, exist_ok=True)

C_NAVY = "#0f172a"
C_BLUE = "#1d4ed8"
C_TEAL = "#0d9488"
C_EMERALD = "#059669"
C_AMBER = "#d97706"
C_RED = "#dc2626"
C_INDIGO = "#4338ca"
C_SLATE = "#334155"


def _align_procrustes_3d(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    p_mean = P.mean(axis=0, keepdims=True)
    q_mean = Q.mean(axis=0, keepdims=True)
    P0 = P - p_mean
    Q0 = Q - q_mean
    p_norm = max(float(np.linalg.norm(P0)), 1e-12)
    q_norm = max(float(np.linalg.norm(Q0)), 1e-12)
    P_unit = P0 / p_norm
    Q_unit = Q0 / q_norm
    H = P_unit.T @ Q_unit
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    return (P_unit @ R) * q_norm + q_mean


def _set_equal_3d_box(ax, pts: np.ndarray, zoom: float = 0.56) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    mid = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)) * zoom, 1e-3)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#e2e8f0")
    ax.yaxis.pane.set_edgecolor("#e2e8f0")
    ax.zaxis.pane.set_edgecolor("#e2e8f0")
    ax.grid(False)


def _plot_2d_drawing(ax, px: np.ndarray, adj: np.ndarray, esty: np.ndarray, title: str) -> None:
    N = px.shape[0]
    y_flip = -px[:, 1]
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] > 0.5:
                st = int(esty[i, j]) if esty is not None else 1
                if st == 2:
                    ax.plot([px[i, 0], px[j, 0]], [y_flip[i], y_flip[j]], color="#64748b", lw=1.25, ls=(0, (3.2, 2.0)), zorder=2)
                else:
                    ax.plot([px[i, 0], px[j, 0]], [y_flip[i], y_flip[j]], color=C_NAVY, lw=1.75, zorder=3)
    ax.scatter(px[:, 0], y_flip, s=18, color=C_TEAL, edgecolors="white", linewidths=0.6, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=7.5, fontweight="bold", color=C_NAVY, pad=2)


def _plot_3d_solid(ax, P: np.ndarray, faces: list[list[int]], base_rgb: tuple[float, float, float], edge_color: str, title: str) -> None:
    oriented = orient_faces_outward(P, faces)
    light_dir = np.array([0.42, -0.52, 0.74], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    polys, fcolors = [], []
    for f in oriented:
        polys.append(P[f])
        _, n_f, _, _ = fit_face_plane(P, f)
        shade = 0.48 + 0.50 * max(0.0, float(np.dot(n_f, light_dir)))
        rgb = np.clip(np.array(base_rgb) * shade + 0.05, 0.0, 1.0)
        fcolors.append((*rgb.tolist(), 0.90))
    coll = Poly3DCollection(polys, facecolors=fcolors, edgecolors=edge_color, linewidths=0.95)
    ax.add_collection3d(coll)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=10, color=edge_color, zorder=5)
    ax.view_init(elev=23, azim=-48)
    _set_equal_3d_box(ax, P, zoom=0.54)
    ax.set_title(title, fontsize=7.5, fontweight="bold", color=edge_color, pad=-2)


def generate_graphical_abstract() -> dict:
    """Create A4 Portrait Pictorial Graphical Abstract (8.27 x 11.69 inches, NO flowcharts)."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
    })
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
    fig.patch.set_facecolor("white")

    # Outer layout: 4 pictorial visual zones on A4 portrait
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[0.46, 1.38, 1.58, 1.58],
        hspace=0.28,
        left=0.075, right=0.965, top=0.968, bottom=0.055,
    )

    # Zone 1: Visual Title Header Banner
    ax_hdr = fig.add_subplot(gs[0, 0])
    ax_hdr.set_xlim(0, 100)
    ax_hdr.set_ylim(0, 20)
    ax_hdr.axis("off")
    hdr_box = FancyBboxPatch((0.5, 1.0), 99.0, 18.0, boxstyle="round,pad=0.3,rounding_size=1.5", facecolor="#f0fdfa", edgecolor=C_TEAL, linewidth=1.6)
    ax_hdr.add_patch(hdr_box)
    ax_hdr.text(
        50.0, 13.2,
        "NeuroTopo-3D: Spectral Normal-Flow Manifold Projection for Single-View Polyhedral and CAD B-Rep Reconstruction",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_NAVY,
    )
    ax_hdr.text(
        50.0, 5.8,
        "Bridging 2D Engineering Line Drawings & Symbolic GD&T Predicates to Watertight ISO 10303-21 (.STEP) Solids",
        ha="center", va="center", fontsize=7.6, style="italic", color=C_SLATE,
    )

    # Zone 2: Pictorial Comparison of 2D Drawings vs. Warped Neural vs. Certified NeuroTopo-3D .STEP B-Rep Solids
    gs_z2 = gs[1, 0].subgridspec(1, 4, wspace=0.14)
    l_npz = np.load(CAD_DIR / "l_bracket_extrusion_sample00_viz.npz", allow_pickle=True)
    v_npz = np.load(CAD_DIR / "v_groove_guide_sample00_viz.npz", allow_pickle=True)

    ax_z2_0 = fig.add_subplot(gs_z2[0, 0])
    _plot_2d_drawing(
        ax_z2_0, l_npz["px_2d"], l_npz["adj"], l_npz["estyle"],
        title="(A) 2D CAD Blueprint\n(Solid vs. Dashed Lines)",
    )

    ax_z2_1 = fig.add_subplot(gs_z2[0, 1], projection="3d")
    l_gt = l_npz["gt_3d"]
    l_raw = _align_procrustes_3d(l_npz["raw_3d"], l_gt)
    l_v9 = _align_procrustes_3d(l_npz["v9_3d"], l_gt)
    l_faces = [list(f) for f in l_npz["faces"].tolist()]
    _plot_3d_solid(
        ax_z2_1, l_raw, l_faces,
        base_rgb=(0.86, 0.30, 0.24), edge_color=C_RED,
        title="(B) Raw Neural 3D Mesh\n28.7 mm Warp (0% Sewable)",
    )

    ax_z2_2 = fig.add_subplot(gs_z2[0, 2], projection="3d")
    _plot_3d_solid(
        ax_z2_2, l_v9, l_faces,
        base_rgb=(0.05, 0.58, 0.52), edge_color=C_NAVY,
        title="(C) NeuroTopo-3D (Ours)\n<0.01 mm Planar (.STEP)",
    )

    ax_z2_3 = fig.add_subplot(gs_z2[0, 3], projection="3d")
    v_gt = v_npz["gt_3d"]
    v_v9 = _align_procrustes_3d(v_npz["v9_3d"], v_gt)
    v_faces = [list(f) for f in v_npz["faces"].tolist()]
    _plot_3d_solid(
        ax_z2_3, v_v9, v_faces,
        base_rgb=(0.12, 0.44, 0.78), edge_color=C_NAVY,
        title="(D) Non-Convex V-Groove\nPA = 0.087 (Newell Valid)",
    )

    # Zone 3: Spectral Manifold Geometry (Left) & Parametric Nullspace .STEP Family Explorer (Right)
    gs_z3 = gs[2, 0].subgridspec(1, 2, width_ratios=[1.08, 1.22], wspace=0.24)
    ax_geo = fig.add_subplot(gs_z3[0, 0])
    t = np.linspace(-1.2, 1.25, 200)
    mx = t
    my = 0.22 * (t**2) - 0.18
    ax_geo.plot(mx, my, color=C_EMERALD, lw=2.5, label="Exact Constraint Manifold $\\mathcal{M}(\\mathcal{C})$")
    ax_geo.fill_between(mx, my - 0.045, my + 0.045, color=C_EMERALD, alpha=0.15)
    p0 = np.array([-0.24, 0.86])
    p_star = np.array([-0.18, 0.22 * (-0.18) ** 2 - 0.18])
    p_tik = np.array([-0.21, 0.34])
    p_msda = np.array([0.84, 0.22 * (0.84) ** 2 - 0.18])
    s_c = np.linspace(0, 1, 40)
    ax_geo.plot(
        (1 - s_c) * p0[0] + s_c * p_star[0] + 0.04 * s_c * (1 - s_c),
        (1 - s_c) * p0[1] + s_c * p_star[1],
        color=C_TEAL, lw=2.4, label="Ours: Normal-Flow $\\operatorname{row}(\\tilde{J}_z)$ ($\\mathcal{O}(\\epsilon^2)$ Drift)",
    )
    ax_geo.plot([p0[0], p_tik[0]], [p0[1], p_tik[1]], color=C_RED, lw=1.8, ls="--", label="Soft Tikhonov ($\\Omega(\\lambda)$ Gap, 0% Sewable)")
    ax_geo.plot([p0[0], p_msda[0]], [p0[1], p_msda[1]], color=C_INDIGO, lw=1.8, ls="-.", label="Classical MSDA / GN ($\\mathcal{O}(\\epsilon)$ Shear)")
    ax_geo.scatter([p0[0]], [p0[1]], s=55, color=C_BLUE, zorder=6)
    ax_geo.text(p0[0] + 0.08, p0[1] - 0.02, "Occlusion-Asymmetric\nNeural Anchor $\\hat{P}_0$", fontsize=7.4, fontweight="bold", color=C_BLUE)
    ax_geo.scatter([p_tik[0]], [p_tik[1]], s=48, color=C_RED, marker="X", zorder=6)
    ax_geo.scatter([p_msda[0]], [p_msda[1]], s=48, color=C_INDIGO, marker="^", zorder=6)
    ax_geo.scatter([p_star[0]], [p_star[1]], s=75, color=C_TEAL, marker="*", zorder=7)
    ax_geo.text(p_star[0] + 0.10, p_star[1] + 0.12, "Certified Root $P^\\star$\n($P_{\\ker}\\delta\\zeta^{(k)} = 0$)", fontsize=7.2, fontweight="bold", color=C_TEAL)
    ax_geo.set_xlim(-1.22, 1.25)
    ax_geo.set_ylim(-0.86, 1.18)
    ax_geo.set_xticks([])
    ax_geo.set_yticks([])
    ax_geo.set_xlabel("Generalized Bas-Relief Nullspace $\\ker(\\tilde{J}_z)$", fontsize=8.1, fontweight="bold")
    ax_geo.set_ylabel("Constraint Row-Space $\\operatorname{row}(\\tilde{J}_z)$", fontsize=8.1, fontweight="bold")
    ax_geo.set_title("(E) Spectral Manifold Geometry (Theorems 2 & 3)", fontsize=8.6, fontweight="bold", color=C_NAVY)
    ax_geo.legend(loc="lower left", fontsize=6.5, frameon=True)

    # Right subpanel of Zone 3: 3-Class Dual-Gauge Rigidity Performance (792 Multi-Constraint Views)
    ax_rig = fig.add_subplot(gs_z3[0, 1])
    s_a2 = json.loads((RUN_DIR / "suite_a_12family_792views_v9.json").read_text(encoding="utf-8"))["pooled_792_views"]["by_rigidity_class"]
    cls_a2 = json.loads((RUN_DIR / "classical_msda_lipson_benchmarks_v9.json").read_text(encoding="utf-8"))["part2_12family_792views"]["by_rigidity_class"]
    c_names = [
        "Class I: Ray-Rigid\n($d_{\\mathrm{null}}^{\\mathrm{ray}}=0$, 193 Views)",
        "Class II: Coupled Tilt\n($d_{\\mathrm{null}}^{\\mathrm{aff}}=0 < d_{\\mathrm{null}}^{\\mathrm{ray}}$, 178 Views)",
        "Class III: Bas-Relief\n($d_{\\mathrm{null}}^{\\mathrm{aff}}\\geq 1$, 421 Views)",
    ]
    ckeys = ["Class_I_RayRigid", "Class_II_CoupledTilt", "Class_III_BasRelief"]
    rx = np.arange(3)
    rw = 0.24
    v_msda = [cls_a2[k]["lipson_shpitalni_pa"] for k in ckeys]
    v_raw = [s_a2[k]["v6_raw_pa"] for k in ckeys]
    v_v9 = [s_a2[k]["v6_v9cert_pa"] for k in ckeys]
    ax_rig.bar(rx - rw, v_msda, rw, color=C_INDIGO, label="Lipson-Shpitalni '96 (Non-Neural)")
    ax_rig.bar(rx, v_raw, rw, color=C_AMBER, label="Raw Neural Lifter (PairGT-Asym)")
    b_r = ax_rig.bar(rx + rw, v_v9, rw, color=C_TEAL, label="NeuroTopo-3D (Certified)")
    for b, val in zip(b_r, v_v9):
        ax_rig.text(b.get_x() + b.get_width() / 2, val + 0.008, f"{val:.4f}", ha="center", fontsize=7.2, fontweight="bold", color=C_TEAL)
    ax_rig.set_xticks(rx)
    ax_rig.set_xticklabels(c_names, fontsize=7.2)
    ax_rig.set_ylim(0, 0.34)
    ax_rig.set_ylabel("3D Procrustes Error (Shape PA) $\\downarrow$", fontsize=8.1, fontweight="bold")
    ax_rig.set_title("(F) Dual-Gauge Rigidity Classes Across 792 Views", fontsize=8.6, fontweight="bold", color=C_NAVY)
    ax_rig.legend(loc="upper left", fontsize=6.9, frameon=True)
    ax_rig.grid(True, axis="y", alpha=0.35)

    # Zone 4: Quantitative Breakthrough Summary Across All Benchmarks (Left: CAD & Real-World, Right: Key Metrics)
    gs_z4 = gs[3, 0].subgridspec(1, 2, width_ratios=[1.12, 1.18], wspace=0.24)
    ax_cad = fig.add_subplot(gs_z4[0, 0])
    methods_c = [
        "Raw Neural\n(PairGT-Asym)",
        "Marill-MSDA\n(1991)",
        "Soft Tikhonov\n(L-BFGS)",
        "Two-Stage\nSLSQP",
        "Lipson-Shpitalni\n(1996 CAD)",
        "NeuroTopo-3D\n(Ours)",
    ]
    pa_c = [0.3868, 0.3553, 0.3384, 0.2863, 0.1534, 0.0696]
    cols_c = [C_AMBER, "#64748b", C_RED, C_BLUE, C_INDIGO, C_TEAL]
    xc = np.arange(len(methods_c))
    bars_c = ax_cad.bar(xc, pa_c, width=0.56, color=cols_c, edgecolor=C_NAVY, linewidth=0.7)
    for b, val in zip(bars_c, pa_c):
        ax_cad.text(b.get_x() + b.get_width() / 2, val + 0.010, f"{val:.4f}", ha="center", fontsize=6.9, fontweight="bold", color=C_NAVY)
    ax_cad.set_xticks(xc)
    ax_cad.set_xticklabels(methods_c, rotation=20, ha="right", fontsize=6.9)
    ax_cad.set_ylim(0, 0.46)
    ax_cad.set_ylabel("3D Shape PA on 120 CAD Parts $\\downarrow$", fontsize=8.0, fontweight="bold")
    ax_cad.set_title("(G) 120-Part Mechanical CAD .STEP Benchmark", fontsize=8.6, fontweight="bold", color=C_NAVY)
    ax_cad.grid(True, axis="y", alpha=0.35)

    ax_cov = fig.add_subplot(gs_z4[0, 1])
    suites_lbl = [
        "3-Seed Held-Out\nMirror Pairs (1,536)",
        "Tetrahedra Rescue\n(3 Seeds)",
        "Unseen OOD Tier-1\nComplex-8",
        "External GDP-29K\nTextbook (9 Seeds)",
        "120 CAD Parts\nSewable (<=0.10mm)",
    ]
    v843_cov = [92.97, 82.29, 56.25, 69.75, 20.00]
    v90_cov = [96.35, 95.31, 95.83, 93.83, 89.17]
    sx = np.arange(len(suites_lbl))
    sw = 0.34
    ax_cov.bar(sx - sw / 2, v843_cov, sw, color=C_BLUE, label="Post-Hoc Two-Stage SLSQP")
    b_cv = ax_cov.bar(sx + sw / 2, v90_cov, sw, color=C_TEAL, label="NeuroTopo-3D (0 Invalid Views)")
    for b, val in zip(b_cv, v90_cov):
        ax_cov.text(b.get_x() + b.get_width() / 2, val + 1.8, f"{val:.1f}%", ha="center", fontsize=7.0, fontweight="bold", color=C_TEAL)
    ax_cov.set_xticks(sx)
    ax_cov.set_xticklabels(suites_lbl, rotation=18, ha="right", fontsize=7.1)
    ax_cov.set_ylim(0, 118)
    ax_cov.set_ylabel("Certified Valid / Sewable Rate (%) $\\uparrow$", fontsize=8.1, fontweight="bold")
    ax_cov.set_title("(H) Certified Coverage & CAD B-Rep Sewing Gains", fontsize=8.8, fontweight="bold", color=C_NAVY)
    ax_cov.legend(loc="upper right", fontsize=6.9, frameon=True)
    ax_cov.grid(True, axis="y", alpha=0.35)

    pdf_out = PKG_DIR / "2_Graphical_Abstract_A4_Portrait.pdf"
    jpg_out = PKG_DIR / "2_Graphical_Abstract_HighRes_300dpi.jpg"
    png_out = PKG_DIR / "2_Graphical_Abstract_Preview.png"
    fig.savefig(pdf_out, format="pdf")
    fig.savefig(jpg_out, format="jpg", dpi=300)
    fig.savefig(png_out, format="png", dpi=220)
    plt.close(fig)
    return {
        "pdf": str(pdf_out),
        "jpg": str(jpg_out),
        "png": str(png_out),
        "pdf_bytes": pdf_out.stat().st_size,
    }


def generate_highlights_docx() -> dict:
    """Create A4 portrait MS Word Highlights file strictly obeying <=15 words and <=100 chars per bullet."""
    highlights = [
        "Spectral theory classifies single-view polyhedral rigidity into three exact regimes.",
        "Normal-flow row-space continuation eliminates first-order Bas-Relief nullspace shear.",
        "Multi-branch continuation reaches 96.35% certified coverage with zero invalid views.",
        "Outperforms equivariant GNNs, Lipson-Shpitalni solvers, and frontier multimodal VLMs.",
        "Converts 89.17% of 120 CAD blueprints into sewable ISO 10303-21 STEP B-Rep solids.",
    ]
    audit = []
    for idx, h in enumerate(highlights, 1):
        words = len(h.split())
        chars = len(h)
        assert words <= 15, f"Highlight {idx} exceeds 15 words: {words}"
        assert chars <= 100, f"Highlight {idx} exceeds 100 chars: {chars}"
        audit.append({"index": idx, "text": h, "words": words, "chars_with_spaces": chars})

    doc = docx.Document()
    sec = doc.sections[0]
    sec.page_width = Mm(210)
    sec.page_height = Mm(297)
    sec.top_margin = Inches(1.0)
    sec.bottom_margin = Inches(1.0)
    sec.left_margin = Inches(1.0)
    sec.right_margin = Inches(1.0)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    p_head = doc.add_paragraph()
    run_h = p_head.add_run("Highlights")
    run_h.bold = True
    run_h.font.name = "Times New Roman"
    run_h.font.size = Pt(14)

    for h in highlights:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(h)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)

    out_path = PKG_DIR / "3_Highlights_JCDE.docx"
    doc.save(out_path)
    return {"path": str(out_path), "bullets": audit}


def generate_cover_letter_docx() -> str:
    """Create official JCDE Cover Letter (.docx) with mandatory Conflicts of Interest disclosure."""
    doc = docx.Document()
    sec = doc.sections[0]
    sec.page_width = Mm(210)
    sec.page_height = Mm(297)
    sec.top_margin = Inches(1.0)
    sec.bottom_margin = Inches(1.0)
    sec.left_margin = Inches(1.0)
    sec.right_margin = Inches(1.0)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    def add_p(text: str, bold: bool = False, space_after: float = 8.0) -> None:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.35
        p.paragraph_format.space_after = Pt(space_after)
        r = p.add_run(text)
        r.bold = bold
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)

    add_p("November 2, 2026", space_after=12)
    add_p("To the Editor-in-Chief\nJournal of Computational Design and Engineering (JCDE)\nOxford University Press", bold=True, space_after=12)
    add_p(
        "Subject: Submission of Original Research Article — \"NeuroTopo-3D: Spectral Normal-Flow Manifold Projection for Single-View Polyhedral and CAD B-Rep Reconstruction\"",
        bold=True,
        space_after=12,
    )
    add_p("Dear Editor-in-Chief,")
    add_p(
        "We are pleased to submit our original research article entitled \"NeuroTopo-3D: Spectral Normal-Flow Manifold Projection for Single-View Polyhedral and CAD B-Rep Reconstruction\" by Tin Nguyen Anh and Thuyen Phan Thi Le (FPT University, Quy Nhon AI Campus, Vietnam) for consideration for publication as a Research Article in the Journal of Computational Design and Engineering (JCDE)."
    )
    add_p(
        "Reconstructing exact, watertight 3D boundary-representation (B-Rep) solids from a single 2D engineering blueprint, textbook diagram, or freehand sketch with symbolic geometric dimensioning and tolerancing (GD&T) predicates is a foundational problem in computational design and reverse engineering. Existing paradigms face a sharp divide: classical non-neural regularity optimizers (such as Marill MSDA and Lipson-Shpitalni) distort asymmetric or non-convex parts along unobservable Generalized Bas-Relief (GBR) viewing-ray nullspaces, whereas modern 2D-to-3D neural lifters exhibit 7 to 47 mm of out-of-plane face warp on a 100 mm part, causing immediate B-Rep sewing failure in industrial CAD kernels."
    )
    add_p(
        "In this work, we resolve this tension both theoretically and algorithmically through five core contributions directly aligned with the scope of JCDE:"
    )
    bullets = [
        "1. Spectral Theory of Single-View Lifting & Dual-Gauge Rigidity: We prove planar z = 0 saddle collapse for reflection-symmetric lifters (Theorem 1), a visibility-diameter message-passing depth lower bound (Lemma 1), and an exact Schur-complement rank identity (Theorem 2) classifying single-view constrained polyhedra into strictly ray-rigid (Class I), coupled tilt-shear (Class II), and bas-relief families (Class III). All symbolic and spectral identities are machine-verified in SymPy and IEEE-754 float64.",
        "2. Iterative Normal-Flow Row-Space Levenberg-Marquardt (NF-RSLM) & Crossing Disambiguation: We prove that whereas soft Tikhonov penalties suffer an O(lambda) infeasibility floor, restricting every step to the coupled viewing-ray row space enforces exact zero stepwise nullspace velocity and pure O(epsilon^2) curvature drift (Theorem 3), paired with multi-branch Necker continuation and a target-free solid-over-dashed 2D ray-crossing verifier (Proposition 1).",
        "3. Comprehensive Multi-Seed & Classical CAD Benchmarks: Across 6,144 three-seed mirror-pair evaluations and 792 twelve-family multi-constraint views, we benchmark seven equivariant/Transformer lifters (EGNN, GATv2-Edge, GraphGPS, PairGT), four frontier multimodal VLMs, two classical non-neural CAD solvers (Marill-MSDA 1991 and Lipson-Shpitalni 1996), and eight manifold optimization baselines, achieving 96.35% +/- 1.13% certified mirror-pair coverage, 0.1107 +/- 0.0014 Shape PA (p = 2.22 x 10^-5), and zero invalid updates.",
        "4. Mechanical CAD Blueprint Reverse Engineering & ISO 10303-21 (.STEP) Synthesis: Across 120 convex and non-convex machined components (chamfered blocks, slanted wedges, L-brackets, V-groove guideways, stepped fixtures, and hopper dies), NeuroTopo-3D achieves 89.17% (107/120) direct CAD sewing validity (<= 0.10 mm planarity tolerance) and 2.2x lower Shape PA than Lipson-Shpitalni (1996), while supporting an interactive Parametric Nullspace CAD Family Explorer at 0.000 px 2D reprojection drift.",
        "5. Zero-Shot Real-World Generalization: Without domain fine-tuning, our framework transfers zero-shot to external scanned GDP-29K solid-geometry textbook diagrams (93.83% +/- 1.75% certified coverage across 9 seed pairs; 100/100 valid on disjoint test scans) and smartphone-photographed hand-drawn sketches (20/20 valid, 100% crossing occlusion consistency)."
    ]
    for b in bullets:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.line_spacing = 1.28
        p.paragraph_format.space_after = Pt(5)
        r = p.add_run(b)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11.5)

    add_p(
        "Reproducibility and Open Source Availability: In full compliance with JCDE's Replication of Results standards, all source code, automated SymPy/float64 theorem proof verifiers, benchmark evaluation logs, and exported ISO 10303-21 (.STEP) and Wavefront (.OBJ) CAD solid models are provided in the Supplementary Reproducibility Archive and publicly hosted at https://github.com/GosuEltin/NeuroTopo-3D."
    )
    add_p(
        "Originality and Submission Declarations: We confirm that this manuscript is original, has not been published previously, and is not currently under consideration for publication elsewhere. All authors have approved the manuscript and agree with its submission to the Journal of Computational Design and Engineering."
    )
    add_p(
        "Conflicts of Interest Disclosure: In accordance with JCDE policy, the authors declare no conflict of interest. Neither author has any financial or personal relationships with other people or organizations that could inappropriately influence or bias this work.",
        bold=True,
    )
    add_p("Thank you very much for your time and consideration. We look forward to hearing from you.")
    add_p(
        "Sincerely,\n\nTin Nguyen Anh (Corresponding Author)\nORCID: https://orcid.org/0009-0009-4454-9541\nEmail: ti93ck2@gmail.com\n\nThuyen Phan Thi Le\nEmail: thuyenptl@fe.edu.vn\n\nFPT University, Quy Nhon AI Campus, Vietnam",
        space_after=4,
    )

    out_path = PKG_DIR / "4_Cover_Letter_JCDE.docx"
    doc.save(out_path)
    return str(out_path)


def generate_alt_text_docx() -> str:
    """Create official JCDE Figure Alt Text MS Word file (`5_Figure_Alt_Text_JCDE.docx`)."""
    alt_items = [
        ("Graphical Abstract", "Visual summary contrasting single-view 2D engineering blueprints and warped raw neural meshes against watertight ISO 10303-21 STEP B-Rep solids reconstructed via spectral normal-flow manifold projection across three rigidity classes."),
        ("Figure 1", "Four-panel schematic showing the five-stage NeuroTopo-3D perception-to-STEP pipeline, normal-flow row-space trajectory versus Tikhonov and Gauss-Newton trajectories on the constraint manifold, multi-branch Necker reflection continuation, and target-free 2D solid-over-dashed crossing verification."),
        ("Figure 2", "Three quantitative verification plots demonstrating zero visibility-cue propagation below the four-hop visibility diameter (Theorem 1 and Lemma 1), exact Schur-complement dual-gauge nullspace decomposition across 12 polyhedral families (Theorem 2), and log-log slope-2 quadratic nullspace curvature drift scaling (Theorem 3)."),
        ("Figure 3", "Three-panel benchmark chart comparing three-seed held-out 3D Shape Procrustes Alignment across seven Module 2 neural lifters, end-to-end graph recovery and 3D error against four frontier multimodal vision-language models, and the Module 3 Pareto frontier of Shape PA versus certified mirror-pair coverage."),
        ("Figure 4", "Qualitative 3D wireframe gallery across four polyhedral families comparing perceived 2D graphs, raw neural anchors, post-hoc two-stage SLSQP, and certified NeuroTopo-3D reconstructions, paired with bar charts showing tetrahedral mirror-pair coverage rising from 15.6% to 95.3%."),
        ("Figure 5", "Two-panel bar chart of the 792-view 12-family multi-constraint benchmark stratified by Dual-Gauge Rigidity Classes I, II, and III, comparing non-neural Lipson-Shpitalni optimization, raw neural anchors, two-stage SLSQP, and NeuroTopo-3D in 3D Shape PA and certified valid coverage."),
        ("Figure 6", "Three-panel bar chart of three-tier out-of-distribution generalization across unseen composite scenes, three-seed leave-one-scene-out cross-validation, and 100 procedural random 3D convex hulls, showing up to 96 percentage-point coverage rescue with bounded nullspace drift."),
        ("Figure 7", "Six-column reverse-engineering gallery of convex and non-convex mechanical CAD blueprints showing 2D drawings, warped zero-shot neural meshes, and watertight NeuroTopo-3D STEP B-Rep solids, alongside log-scale face non-planarity against the 0.10 mm sewing tolerance and 3D dihedral angle error."),
        ("Figure 8", "Visual and quantitative demonstration of the Parametric Nullspace CAD Family Explorer on an affine-constrained frustum hopper die, showing five watertight STEP solid variants generated along the primary Bas-Relief kernel mode at 0.000 px 2D reprojection drift."),
        ("Figure 9", "Real-world transfer gallery on external scanned GDP-29K solid-geometry textbook diagrams and smartphone-photographed hand-drawn paper sketches, accompanied by a bar chart showing certified valid coverage increasing from 69.8% to 93.8% across nine seed pairs."),
        ("Figure 10", "Six-panel stress-test and sensitivity plot showing graceful degradation under 2D pixel jitter, topological edge/vertex drops, solid-dashed visibility flips, and 15% spurious false constraints, alongside four-order-of-magnitude stability plateaus across spectral SVD thresholds."),
    ]
    doc = docx.Document()
    sec = doc.sections[0]
    sec.page_width = Mm(210)
    sec.page_height = Mm(297)
    sec.top_margin = Inches(1.0)
    sec.bottom_margin = Inches(1.0)
    sec.left_margin = Inches(1.0)
    sec.right_margin = Inches(1.0)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    p_title = doc.add_paragraph()
    r_t = p_title.add_run("Alternative Text (Alt Text) for Figures and Graphical Abstract — JCDE Submission")
    r_t.bold = True
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(13.5)

    for label, desc in alt_items:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.4
        p.paragraph_format.space_after = Pt(6)
        r1 = p.add_run(f"{label}: ")
        r1.bold = True
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(12)
        r2 = p.add_run(desc)
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(12)

    out_path = PKG_DIR / "5_Figure_Alt_Text_JCDE.docx"
    doc.save(out_path)
    return str(out_path)


if __name__ == "__main__":
    ga_info = generate_graphical_abstract()
    hl_info = generate_highlights_docx()
    cl_path = generate_cover_letter_docx()
    at_path = generate_alt_text_docx()
    print(json.dumps({"graphical_abstract": ga_info, "highlights": hl_info, "cover_letter": cl_path, "alt_text": at_path}, indent=2))
