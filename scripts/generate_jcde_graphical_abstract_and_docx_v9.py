"""Generate Official JCDE Submission Companion Files (100% Aligned with Official Templates V2.22 / V3.1):
  - Graphical_abstract.pdf, Graphical_abstract.jpg (3000x1200 px, 5:2 ratio, 300 DPI), Graphical_abstract.docx
    (plus 2_Graphical_Abstract_A4_Portrait.pdf & 2_Graphical_Abstract_HighRes_300dpi.jpg)
  - 3_Highlights_JCDE.docx & Highlights.docx (from official Highlights-example 1.docx)
  - 4_Submission_Checklist_JCDE.docx & JCDE submission checklist (JCDE-202X-XXX) V2.22.docx
    (populated directly from official Submission Checklist_V2.22)
  - 5_Figure_Alt_Text_JCDE.docx & JCDE_Alt_Text_Template V1.docx
    (populated directly from official Alt Text/JCDE_Alt_Text_Template V1.docx with embedded thumbnails)
  - 6_Cover_Letter_JCDE.docx & 4_Cover_Letter_JCDE.docx
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import docx
from docx.shared import Inches, Pt, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH

from src.cad.step_exporter_v9 import fit_face_plane, orient_faces_outward

RUN_DIR = ROOT / "runs_q1_journal_v9"
CAD_DIR = RUN_DIR / "cad_exports"
FIG_DIR = ROOT / "manuscript_q1_v9" / "figures"
PKG_DIR = ROOT / "jcde_submission_package"
PKG_DIR.mkdir(parents=True, exist_ok=True)
TPL_DIR = PKG_DIR / "official_templates"
TPL_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS_TPL_ROOT = Path(
    r"C:\Users\GosuEltin Lap\Downloads\jcde-all-templates-and-checklists\_JCDE_All_Template_and_Checklist_September_29_2026"
)

C_NAVY = "#0f172a"
C_BLUE = "#1d4ed8"
C_TEAL = "#0d9488"
C_EMERALD = "#059669"
C_AMBER = "#d97706"
C_RED = "#dc2626"
C_INDIGO = "#4338ca"
C_SLATE = "#334155"


def _sync_official_templates() -> dict[str, Path]:
    """Copy official blank templates into jcde_submission_package/official_templates for self-contained reproducibility."""
    mapping = {
        "checklist": (
            DOWNLOADS_TPL_ROOT / "Submission Checklist_V2.22" / "JCDE submission checklist (JCDE-202X-XXX) V2.22.docx",
            TPL_DIR / "blank_JCDE_submission_checklist_V2.22.docx",
        ),
        "highlights": (
            DOWNLOADS_TPL_ROOT / "Highlights" / "Highlights-example 1.docx",
            TPL_DIR / "blank_Highlights_example1.docx",
        ),
        "graphical_abstract": (
            DOWNLOADS_TPL_ROOT / "Graphical Abstract" / "Graphical Abstract-example 1.docx",
            TPL_DIR / "blank_Graphical_Abstract_example1.docx",
        ),
        "alt_text": (
            DOWNLOADS_TPL_ROOT / "Alt Text" / "JCDE_Alt_Text_Template V1.docx",
            TPL_DIR / "blank_JCDE_Alt_Text_Template_V1.docx",
        ),
    }
    resolved: dict[str, Path] = {}
    for key, (src, dst) in mapping.items():
        if src.exists():
            shutil.copy2(src, dst)
        assert dst.exists(), f"Missing official template for {key}: {dst}"
        resolved[key] = dst
    return resolved


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
    ax.set_title(title, fontsize=7.8, fontweight="bold", color=C_NAVY, pad=2)


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
    ax.set_title(title, fontsize=7.8, fontweight="bold", color=edge_color, pad=-2)


def generate_graphical_abstract(tpl_ga_path: Path) -> dict:
    """Create 5:2 Aspect Ratio Graphical Abstract (10.0 x 4.0 in @ 300 DPI = 3000 x 1200 px) per JCDE Table 5 R6-R13.
    - Font: Arial / Helvetica (Table 5 R7)
    - Size: 3000 x 1200 px >= 1250 x 500 px at 300 DPI, 5:2 aspect ratio (Table 5 R8)
    - No 'Graphical Abstract' heading or title banner inside the image, and no outer border (Table 5 R9 & examples)
    - Saved as Graphical_abstract.pdf, Graphical_abstract.jpg, Graphical_abstract.docx (Table 5 R10-R11)
    - Also copied to 2_Graphical_Abstract_A4_Portrait.pdf & 2_Graphical_Abstract_HighRes_300dpi.jpg.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
    })
    fig = plt.figure(figsize=(10.0, 4.0), dpi=300)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.28, 1.05, 1.15],
        wspace=0.23,
        left=0.025, right=0.985, top=0.915, bottom=0.135,
    )

    # Column 1 (2x2 Grid): 2D CAD Blueprint & Warped Raw Mesh vs. Certified NeuroTopo-3D STEP Solids
    gs_c1 = gs[0, 0].subgridspec(2, 2, hspace=0.26, wspace=0.08)
    l_npz = np.load(CAD_DIR / "l_bracket_extrusion_sample00_viz.npz", allow_pickle=True)
    v_npz = np.load(CAD_DIR / "v_groove_guide_sample00_viz.npz", allow_pickle=True)

    ax_c1_0 = fig.add_subplot(gs_c1[0, 0])
    _plot_2d_drawing(
        ax_c1_0, l_npz["px_2d"], l_npz["adj"], l_npz["estyle"],
        title="(A) 2D CAD Blueprint\n(Solid & Dashed Lines)",
    )

    ax_c1_1 = fig.add_subplot(gs_c1[0, 1], projection="3d")
    l_gt = l_npz["gt_3d"]
    l_raw = _align_procrustes_3d(l_npz["raw_3d"], l_gt)
    l_v9 = _align_procrustes_3d(l_npz["v9_3d"], l_gt)
    l_faces = [list(f) for f in l_npz["faces"].tolist()]
    _plot_3d_solid(
        ax_c1_1, l_raw, l_faces,
        base_rgb=(0.86, 0.30, 0.24), edge_color=C_RED,
        title="(B) Raw Neural Mesh\n28.7 mm Warp (0% Sew)",
    )

    ax_c1_2 = fig.add_subplot(gs_c1[1, 0], projection="3d")
    _plot_3d_solid(
        ax_c1_2, l_v9, l_faces,
        base_rgb=(0.05, 0.58, 0.52), edge_color=C_NAVY,
        title="(C) NeuroTopo-3D (.STEP)\n<0.01 mm Planar (Sewable)",
    )

    ax_c1_3 = fig.add_subplot(gs_c1[1, 1], projection="3d")
    v_gt = v_npz["gt_3d"]
    v_v9 = _align_procrustes_3d(v_npz["v9_3d"], v_gt)
    v_faces = [list(f) for f in v_npz["faces"].tolist()]
    _plot_3d_solid(
        ax_c1_3, v_v9, v_faces,
        base_rgb=(0.12, 0.44, 0.78), edge_color=C_NAVY,
        title="(D) Non-Convex V-Groove\nPA = 0.087 (Newell Valid)",
    )

    # Column 2: Spectral Normal-Flow Manifold Geometry (Theorems 2 & 3)
    ax_geo = fig.add_subplot(gs[0, 1])
    t = np.linspace(-1.2, 1.25, 200)
    mx = t
    my = 0.22 * (t**2) - 0.18
    ax_geo.plot(mx, my, color=C_EMERALD, lw=2.4, label="Feasible Manifold $\\mathcal{M}_0(\\mathcal{C})$")
    ax_geo.fill_between(mx, my - 0.048, my + 0.048, color=C_EMERALD, alpha=0.15)
    p0 = np.array([-0.24, 0.86])
    p_star = np.array([-0.18, 0.22 * (-0.18) ** 2 - 0.18])
    p_tik = np.array([-0.21, 0.34])
    p_msda = np.array([0.84, 0.22 * (0.84) ** 2 - 0.18])
    s_c = np.linspace(0, 1, 40)
    ax_geo.plot(
        (1 - s_c) * p0[0] + s_c * p_star[0] + 0.04 * s_c * (1 - s_c),
        (1 - s_c) * p0[1] + s_c * p_star[1],
        color=C_TEAL, lw=2.3, label="Ours: Normal-Flow ($\\mathcal{O}(\\epsilon^2)$ Drift)",
    )
    ax_geo.plot([p0[0], p_tik[0]], [p0[1], p_tik[1]], color=C_RED, lw=1.7, ls="--", label="Soft Tikhonov ($\\Omega(\\lambda)$ Gap)")
    ax_geo.plot([p0[0], p_msda[0]], [p0[1], p_msda[1]], color=C_INDIGO, lw=1.7, ls="-.", label="Unconstrained GN ($\\mathcal{O}(\\epsilon)$ Shear)")
    ax_geo.scatter([p0[0]], [p0[1]], s=52, color=C_BLUE, zorder=6)
    ax_geo.text(p0[0] + 0.08, p0[1] - 0.03, "Asymmetric Lifter\nAnchor $\\hat{P}_0$", fontsize=7.3, fontweight="bold", color=C_BLUE)
    ax_geo.scatter([p_tik[0]], [p_tik[1]], s=46, color=C_RED, marker="X", zorder=6)
    ax_geo.scatter([p_msda[0]], [p_msda[1]], s=46, color=C_INDIGO, marker="^", zorder=6)
    ax_geo.scatter([p_star[0]], [p_star[1]], s=72, color=C_TEAL, marker="*", zorder=7)
    ax_geo.text(p_star[0] + 0.06, p_star[1] + 0.08, "Certified Root $P^\\star$\n($P_{\\ker}\\delta\\zeta^{(k)} = 0$)", fontsize=6.8, fontweight="bold", color=C_TEAL)
    ax_geo.set_xlim(-1.22, 1.25)
    ax_geo.set_ylim(-0.88, 1.18)
    ax_geo.set_xticks([])
    ax_geo.set_yticks([])
    ax_geo.set_xlabel("Bas-Relief Nullspace $\\ker(\\tilde{J}_z)$", fontsize=8.0, fontweight="bold")
    ax_geo.set_ylabel("Constraint Row-Space $\\operatorname{row}(\\tilde{J}_z)$", fontsize=8.0, fontweight="bold")
    ax_geo.set_title("(E) Spectral Manifold Projection\n(Theorems 2 & 3: Zero Stepwise Shear)", fontsize=8.4, fontweight="bold", color=C_NAVY)
    ax_geo.legend(loc="lower left", fontsize=6.5, frameon=True)

    # Column 3: 120-Part Mechanical CAD .STEP Reverse-Engineering & Multi-Suite Coverage
    ax_cad = fig.add_subplot(gs[0, 2])
    methods_c = [
        "Raw Neural\n(PairGT)",
        "Marill-MSDA\n(1991)",
        "Tikhonov\n(L-BFGS)",
        "Two-Stage\nSLSQP",
        "Lipson-\nShpitalni '96",
        "NeuroTopo-3D\n(Ours)",
    ]
    pa_c = [0.3868, 0.3553, 0.3384, 0.2863, 0.1534, 0.0696]
    cols_c = [C_AMBER, "#64748b", C_RED, C_BLUE, C_INDIGO, C_TEAL]
    xc = np.arange(len(methods_c))
    bars_c = ax_cad.bar(xc, pa_c, width=0.56, color=cols_c, edgecolor=C_NAVY, linewidth=0.7)
    for b, val in zip(bars_c, pa_c):
        ax_cad.text(b.get_x() + b.get_width() / 2, val + 0.010, f"{val:.4f}", ha="center", fontsize=6.9, fontweight="bold", color=C_NAVY)
    ax_cad.text(
        2.65, 0.448,
        "96.35% Held-Out Pair Cov · 93.83% GDP-29K Cov\n89.17% (107/120) Sewable STEP · 0 Invalid Updates",
        ha="center", va="center", fontsize=6.6, fontweight="bold", color=C_EMERALD,
        bbox=dict(boxstyle="round,pad=0.25", fc="#ecfdf5", ec=C_EMERALD, lw=0.85),
    )
    ax_cad.set_xticks(xc)
    ax_cad.set_xticklabels(methods_c, rotation=22, ha="right", fontsize=7.0)
    ax_cad.set_ylim(0, 0.50)
    ax_cad.set_ylabel("3D Shape PA on 120 CAD Parts $\\downarrow$", fontsize=8.0, fontweight="bold")
    ax_cad.set_title("(F) 120-Part CAD .STEP Benchmark\n(2.2x Lower Error vs. Lipson-Shpitalni)", fontsize=8.4, fontweight="bold", color=C_NAVY)
    ax_cad.grid(True, axis="y", alpha=0.35)

    # Save exact JCDE filenames + numbered package aliases
    ga_pdf = PKG_DIR / "Graphical_abstract.pdf"
    ga_jpg = PKG_DIR / "Graphical_abstract.jpg"
    ga_docx = PKG_DIR / "Graphical_abstract.docx"

    pdf_out = PKG_DIR / "2_Graphical_Abstract_A4_Portrait.pdf"
    jpg_out = PKG_DIR / "2_Graphical_Abstract_HighRes_300dpi.jpg"
    png_out = PKG_DIR / "2_Graphical_Abstract_Preview.png"
    docx_out2 = PKG_DIR / "2_Graphical_Abstract_JCDE.docx"

    fig.savefig(ga_pdf, format="pdf")
    fig.savefig(ga_jpg, format="jpg", dpi=300)
    fig.savefig(png_out, format="png", dpi=300)
    plt.close(fig)

    shutil.copy2(ga_pdf, pdf_out)
    shutil.copy2(ga_jpg, jpg_out)

    # Also build Graphical_abstract.docx directly from official Graphical Abstract-example 1.docx
    doc_ga = docx.Document(tpl_ga_path)
    for p in list(doc_ga.paragraphs):
        p._element.getparent().remove(p._element)
    p_img = doc_ga.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.add_run().add_picture(str(ga_jpg), width=Inches(6.3))
    doc_ga.save(ga_docx)
    shutil.copy2(ga_docx, docx_out2)

    return {
        "pdf": str(ga_pdf),
        "jpg": str(ga_jpg),
        "docx": str(ga_docx),
        "numbered_pdf": str(pdf_out),
        "numbered_jpg": str(jpg_out),
        "pixel_dimensions": "3000x1200 (5:2 aspect ratio at 300 DPI)",
        "pdf_bytes": ga_pdf.stat().st_size,
    }


def generate_highlights_docx(tpl_hl_path: Path) -> dict:
    """Populate Highlights directly from official Highlights-example 1.docx (<=15 words and <=100 chars per bullet)."""
    highlights = [
        "Spectral theory classifies single-view polyhedral rigidity into three local regimes.",
        "Normal-flow row-space continuation eliminates first-order Bas-Relief nullspace shear.",
        "Multi-branch continuation reaches 96.35% tolerance-certified mirror-pair coverage.",
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

    doc = docx.Document(tpl_hl_path)
    # Keep paragraph 0 ('Highlights' title) and replace bullet paragraphs 1..5
    while len(doc.paragraphs) > 1:
        p = doc.paragraphs[-1]
        p._element.getparent().remove(p._element)

    for h in highlights:
        p = doc.add_paragraph(style="List Paragraph")
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run("• " + h)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)

    out_path = PKG_DIR / "3_Highlights_JCDE.docx"
    out_exact = PKG_DIR / "Highlights.docx"
    doc.save(out_path)
    shutil.copy2(out_path, out_exact)
    return {"path": str(out_path), "exact_path": str(out_exact), "bullets": audit}


def _set_cell_text_preserve_shading(cell, text: str, font_size: float = 10.0, bold: bool = False) -> None:
    if not cell.paragraphs:
        cell.add_paragraph()
    p0 = cell.paragraphs[0]
    while len(cell.paragraphs) > 1:
        p_extra = cell.paragraphs[-1]
        p_extra._element.getparent().remove(p_extra._element)
    p0.text = ""
    r = p0.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(font_size)
    r.bold = bold


def _check_box_in_cell(cell) -> None:
    replaced = False
    for p in cell.paragraphs:
        for r in p.runs:
            if "☐" in r.text:
                r.text = r.text.replace("☐", "☑")
                replaced = True
    if not replaced:
        _set_cell_text_preserve_shading(cell, "☑", font_size=10.0)


def generate_submission_checklist_docx(tpl_chk_path: Path) -> str:
    """Populate the exact official JCDE Submission Checklist V2.22 (.docx) across all 6 tables (Tables 0-5)."""
    doc = docx.Document(tpl_chk_path)
    assert len(doc.tables) == 6, f"Expected 6 tables in official JCDE checklist V2.22, got {len(doc.tables)}"

    # Table 0: Manuscript Information (22 rows x 6 cols; col 3 = Content, col 4 = MS under Review)
    t0 = doc.tables[0]
    t0_values = {
        1: "JCDE-202X-XXX (Initial Submission)",
        2: "NeuroTopo-3D: Spectral Normal-Flow Manifold Projection for Single-View Polyhedral and CAD B-Rep Reconstruction",
        3: "Tin Nguyen Anh, Thuyen Phan Thi Le",
        4: "ti93ck2@gmail.com, thuyenptl@fe.edu.vn",
        6: "Tin Nguyen Anh",
        7: "FPT University, Quy Nhon AI Campus, Gia Lai, Vietnam",
        8: "ti93ck2@gmail.com",
        9: "https://orcid.org/0009-0009-4454-9541",
        10: "N/A",
        11: "N/A",
        12: "N/A",
        13: "N/A",
        14: "Tin Nguyen Anh",
        15: "ti93ck2@gmail.com",
        16: "FPT University, Quy Nhon AI Campus",
        17: "An Phu Thinh New Urban Area, Nhon Binh Ward",
        18: "Quy Nhon",
        19: "Gia Lai",
        20: "55113",
        21: "Vietnam",
    }
    for r_idx, val in t0_values.items():
        _set_cell_text_preserve_shading(t0.rows[r_idx].cells[3], val, font_size=9.5)
        if r_idx not in (10, 11, 12, 13):
            _check_box_in_cell(t0.rows[r_idx].cells[4])

    # Table 1: Required Files (5 rows x 5 cols; col 3 = MS under Review)
    t1 = doc.tables[1]
    for r_idx in (1, 4):
        _check_box_in_cell(t1.rows[r_idx].cells[3])

    # Table 2: Common Checklist (19 rows x 6 cols; col 4 = MS under Review)
    # R1..R3 checked; R4=Yes (leave blank), R5=No (check); R7=Yes (blank), R8=No (check); R10, R12, R14..R18 checked
    t2 = doc.tables[2]
    for r_idx in (1, 2, 3, 5, 8, 10, 12, 14, 15, 16, 17, 18):
        _check_box_in_cell(t2.rows[r_idx].cells[4])

    # Table 3: Format and Style for Text (33 rows x 4 cols; col 2 = MS under Review)
    t3 = doc.tables[3]
    for r_idx in range(1, len(t3.rows)):
        _check_box_in_cell(t3.rows[r_idx].cells[2])

    # Table 4: Equations, Figures, and Tables (19 rows x 4 cols; col 2 = MS under Review)
    t4 = doc.tables[4]
    for r_idx in range(1, len(t4.rows)):
        _check_box_in_cell(t4.rows[r_idx].cells[2])

    # Table 5: Checklists for Individual Files (42 rows x 4 cols; col 2 = MS under Review)
    # Leave revised-only rows (3, 20, 21) and video row (35) unchecked for Initial Submission
    t5 = doc.tables[5]
    skip_t5 = {3, 20, 21, 35}
    for r_idx in range(1, len(t5.rows)):
        if r_idx not in skip_t5:
            _check_box_in_cell(t5.rows[r_idx].cells[2])

    out_path = PKG_DIR / "4_Submission_Checklist_JCDE.docx"
    out_exact = PKG_DIR / "JCDE submission checklist (JCDE-202X-XXX) V2.22.docx"
    doc.save(out_path)
    shutil.copy2(out_path, out_exact)
    return str(out_path)


def generate_cover_letter_docx() -> str:
    """Create official JCDE Cover Letter (.docx) with Oxford spelling (-ize), Conflicts of Interest, and ATiGB 2026 disclosure."""
    doc = docx.Document()
    sec = doc.sections[0]
    sec.page_width = Mm(210)
    sec.page_height = Mm(297)
    sec.top_margin = Inches(0.95)
    sec.bottom_margin = Inches(0.95)
    sec.left_margin = Inches(1.0)
    sec.right_margin = Inches(1.0)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11.5)

    def add_p(text: str, bold: bool = False, space_after: float = 7.0) -> None:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.30
        p.paragraph_format.space_after = Pt(space_after)
        r = p.add_run(text)
        r.bold = bold
        r.font.name = "Times New Roman"
        r.font.size = Pt(11.5)

    add_p("November 3, 2026", space_after=10)
    add_p("To the Editor-in-Chief\nJournal of Computational Design and Engineering (JCDE)\nOxford University Press", bold=True, space_after=10)
    add_p(
        "Subject: Submission of Original Research Article — \"NeuroTopo-3D: Spectral Normal-Flow Manifold Projection for Single-View Polyhedral and CAD B-Rep Reconstruction\"",
        bold=True,
        space_after=10,
    )
    add_p("Dear Editor-in-Chief,")
    add_p(
        "We are pleased to submit our original research article entitled \"NeuroTopo-3D: Spectral Normal-Flow Manifold Projection for Single-View Polyhedral and CAD B-Rep Reconstruction\" by Tin Nguyen Anh and Thuyen Phan Thi Le (FPT University, Quy Nhon AI Campus, Gia Lai, Vietnam) for consideration for publication as a Research Article in the Journal of Computational Design and Engineering (JCDE)."
    )
    add_p(
        "Reconstructing watertight 3D boundary-representation (B-Rep) solids from a single 2D engineering blueprint, textbook diagram, or freehand sketch with symbolic geometric dimensioning and tolerancing (GD&T) predicates is a foundational problem in computational design and reverse engineering. Existing paradigms face a sharp divide: classical non-neural regularity optimizers (Marill MSDA and Lipson-Shpitalni) distort asymmetric or non-convex parts along unobservable Generalized Bas-Relief (GBR) viewing-ray nullspaces, whereas pure 2D-to-3D neural lifters exhibit 7 to 47 mm of out-of-plane face warp on a 100 mm part, causing immediate B-Rep sewing failure in CAD kernels."
    )
    add_p(
        "In this work, we resolve this dilemma through five core contributions directly aligned with the scope of JCDE:"
    )
    bullets = [
        "1. Spectral Theory of Single-View Lifting & Dual-Gauge Rigidity: We prove planar z = 0 saddle collapse for reflection-symmetric lifters (Theorem 1), a visibility-diameter message-passing depth lower bound (Lemma 1), and an exact Schur-complement rank identity (Theorem 2) classifying regular local neighborhoods of constrained polyhedra into viewing-ray rigid (Class I), coupled tilt-shear (Class II), and bas-relief submanifolds (Class III). All symbolic and spectral identities are verified in SymPy and IEEE-754 float64.",
        "2. Iterative Normal-Flow Row-Space Levenberg-Marquardt (NF-RSLM) & Crossing Disambiguation: We prove that whereas soft Tikhonov penalties suffer an O(lambda) infeasibility floor, restricting every step to the coupled viewing-ray row space enforces zero stepwise nullspace velocity and O(epsilon^2) curvature drift (Theorem 3), paired with multi-branch Necker continuation and a target-free solid-over-dashed 2D ray-crossing verifier (Proposition 1).",
        "3. Multi-Seed Equivariant Lifting & Classical CAD Solver Benchmarks: Across 6,144 three-seed mirror-pair evaluations and 792 twelve-family multi-constraint views, we benchmark seven equivariant/Transformer lifters (EGNN, GATv2-Edge, GraphGPS, PairGT), four multimodal VLMs, two classical non-neural CAD solvers (Marill-MSDA 1991 and Lipson-Shpitalni 1996), and eight manifold optimization baselines, achieving 96.35% +/- 1.13% tolerance-certified mirror-pair coverage, 0.1107 +/- 0.0014 Shape PA (p = 2.22 x 10^-5), and zero unabstained tolerance-violating updates.",
        "4. Mechanical CAD Blueprint Reverse Engineering & ISO 10303-21 (.STEP) Synthesis: Across 120 convex and non-convex machined components, NeuroTopo-3D jointly achieves 89.17% (107/120) nonlinear GD&T semantic feasibility and ISO 10303-21 (.STEP) B-Rep sewing validity (<0.01 mm face planarity on all certified solids) with 2.2x lower 3D Shape PA (0.0696 vs. 0.1534) and 2.1x lower dihedral error (3.03 deg vs. 6.35 deg) than Lipson-Shpitalni (1996), plus a Parametric Nullspace CAD Family Explorer at 0.000 px 2D reprojection drift.",
        "5. Zero-Shot Real-World Generalization: Without domain fine-tuning, our framework transfers zero-shot to external scanned GDP-29K solid-geometry textbook diagrams (93.83% +/- 1.75% tolerance-certified coverage across 9 seed pairs; 100/100 non-degenerate on disjoint test scans) and smartphone-photographed hand-drawn sketches (20/20 valid, 100% crossing occlusion consistency)."
    ]
    for b in bullets:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.22)
        p.paragraph_format.line_spacing = 1.24
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(b)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)

    add_p(
        "Disclosure of Related Prior Conference Work: A preliminary 2D synthetic benchmark study by the authors appeared in the 2026 IEEE International Conference on Advanced Technologies in Information and Green Business (ATiGB 2026, pp. 421-431, DOI: 10.1109/ATiGB70203.2026.11628217; cited in the manuscript as Nguyen Anh & Phan Thi Le, 2026). That conference paper was restricted strictly to 2D image-plane keypoint localization and 2D projected planar graph refinement on four basic synthetic templates without recovering 3D vertex depths. The present JCDE submission is a substantially new theoretical, algorithmic, and CAD engineering investigation introducing 2D-to-3D Equivariant Graph Transformer lifting, Dual-Gauge Spectral Rigidity Theory (Theorems 1-3), Normal-Flow Row-Space Levenberg-Marquardt 3D manifold projection, 12-family and real-world GDP-29K/sketch evaluations, classical MSDA/Lipson-Shpitalni comparisons, and ISO 10303-21 (.STEP) B-Rep solid synthesis.",
        bold=False,
    )
    add_p(
        "Reproducibility and Open Source Availability: All source code, automated SymPy/float64 theorem verifiers, benchmark evaluation logs, and exported ISO 10303-21 (.STEP) and Wavefront (.OBJ) CAD solid models are publicly hosted at https://github.com/GosuEltin/NeuroTopo-3D and included in the accompanying Supplementary Reproducibility Archive."
    )
    add_p(
        "Originality and Conflicts of Interest Disclosure: We confirm that this manuscript is original, has not been published previously, and is not under consideration elsewhere. In accordance with JCDE policy, the authors declare no conflict of interest.",
        bold=True,
    )
    add_p(
        "Sincerely,\n\nTin Nguyen Anh (Corresponding Author, ORCID: https://orcid.org/0009-0009-4454-9541, Email: ti93ck2@gmail.com)\nThuyen Phan Thi Le (Co-Author, Email: thuyenptl@fe.edu.vn)\nFPT University, Quy Nhon AI Campus, Gia Lai, Vietnam",
        space_after=4,
    )

    out_path_4 = PKG_DIR / "4_Cover_Letter_JCDE.docx"
    out_path_6 = PKG_DIR / "6_Cover_Letter_JCDE.docx"
    doc.save(out_path_4)
    shutil.copy2(out_path_4, out_path_6)
    return str(out_path_6)


def generate_alt_text_docx(tpl_alt_path: Path) -> str:
    """Populate the exact official 4-column JCDE_Alt_Text_Template V1.docx with thumbnails, short captions, and Alt Text."""
    alt_items = [
        (
            "Graphical Abstract",
            PKG_DIR / "Graphical_abstract.jpg",
            "Single-view 2D CAD blueprints to watertight ISO 10303-21 STEP solids via spectral normal-flow manifold projection.",
            "Visual summary contrasting 2D engineering blueprints and warped raw neural meshes against watertight ISO 10303-21 STEP B-Rep solids reconstructed via spectral normal-flow manifold projection across three rigidity classes.",
        ),
        (
            "Figure 1",
            FIG_DIR / "fig1_q1_architecture_and_v9_solver.png",
            "System architecture of NeuroTopo-3D and geometric interpretation of normal-flow manifold projection.",
            "Four-panel schematic showing the five-stage NeuroTopo-3D perception-to-STEP pipeline (A), normal-flow row-space trajectory versus Tikhonov and Gauss-Newton trajectories (B), multi-branch Necker reflection continuation (C), and target-free 2D solid-over-dashed crossing verification (D).",
        ),
        (
            "Figure 2",
            FIG_DIR / "fig2_theorems_spectral_and_drift_scaling.png",
            "Machine-verified empirical validation of Theorem 1, Lemma 1, Theorem 2, and Theorem 3.",
            "Three quantitative verification plots demonstrating zero visibility-cue propagation below the four-hop visibility diameter (A), Schur-complement dual-gauge nullspace decomposition across 12 polyhedral families (B), and log-log slope-2 quadratic nullspace curvature drift scaling (C).",
        ),
        (
            "Figure 3",
            FIG_DIR / "fig3_module2_equivariant_baselines_and_solvers.png",
            "Module 2 equivariant 3D lifter baselines, frontier multimodal VLMs, and Module 3 Pareto frontier.",
            "Three-panel benchmark chart comparing three-seed held-out 3D Shape Procrustes Alignment across seven Module 2 neural lifters (A), end-to-end graph recovery and 3D error against four frontier multimodal vision-language models (B), and the Module 3 Pareto frontier of Shape PA versus tolerance-certified mirror-pair coverage (C).",
        ),
        (
            "Figure 4",
            FIG_DIR / "fig4_basic_solids_and_rescued_tetrahedra.png",
            "Qualitative 3D wireframe reconstructions and per-family coverage on four polyhedral families.",
            "Qualitative 3D wireframe gallery across four polyhedral families comparing perceived 2D graphs, raw neural anchors, two-stage SLSQP, and tolerance-certified NeuroTopo-3D reconstructions, paired with bar charts showing tetrahedral mirror-pair coverage rising from 15.6% to 95.3% (E, F).",
        ),
        (
            "Figure 5",
            FIG_DIR / "fig5_complex_12family_rigidity_classes.png",
            "792-view 12-family multi-constraint benchmark stratified by Dual-Gauge Rigidity Classes I, II, and III.",
            "Two-panel bar chart of the 792-view 12-family multi-constraint benchmark stratified by Dual-Gauge Rigidity Classes I, II, and III, comparing non-neural Lipson-Shpitalni optimization, raw neural anchors, two-stage SLSQP, and NeuroTopo-3D in 3D Shape PA (A) and tolerance-certified valid coverage (B).",
        ),
        (
            "Figure 6",
            FIG_DIR / "fig6_3tier_ood_and_nullspace_drift.png",
            "Three-tier out-of-distribution generalization and viewing-ray nullspace drift protection.",
            "Three-panel bar chart of three-tier out-of-distribution generalization across unseen composite scenes (A), three-seed leave-one-scene-out cross-validation and 100 procedural random 3D convex hulls (B), and certified OOD coverage rescue with bounded nullspace drift (C).",
        ),
        (
            "Figure 7",
            FIG_DIR / "fig7_cad_step_reverse_engineering_6archetypes.png",
            "Reverse engineering of 120 convex and non-convex mechanical CAD blueprints into STEP B-Rep solids.",
            "Six-column reverse-engineering gallery of convex and non-convex mechanical CAD blueprints showing 2D drawings, warped zero-shot neural meshes, and watertight NeuroTopo-3D STEP B-Rep solids, alongside log-scale face non-planarity against the 0.10 mm sewing tolerance (D) and 3D dihedral angle error (E).",
        ),
        (
            "Figure 8",
            FIG_DIR / "fig8_parametric_nullspace_cad_explorer.png",
            "Parametric Nullspace CAD Family Explorer on an affine-constrained frustum hopper die at 0.000 px drift.",
            "Visual and quantitative demonstration of the Parametric Nullspace CAD Family Explorer on an affine-constrained frustum hopper die, showing five watertight STEP solid variants generated along the primary Bas-Relief kernel mode at 0.000 px 2D reprojection drift.",
        ),
        (
            "Figure 9",
            FIG_DIR / "fig9_realworld_gdp29k_and_handdrawn_sketches.png",
            "Zero-shot transfer to external scanned GDP-29K textbook diagrams and hand-drawn paper sketches.",
            "Real-world transfer gallery on external scanned GDP-29K solid-geometry textbook diagrams (A-C) and smartphone-photographed hand-drawn paper sketches (E-H), accompanied by a bar chart showing tolerance-certified valid coverage increasing from 69.8% to 93.8% across nine seed pairs (D).",
        ),
        (
            "Figure 10",
            FIG_DIR / "fig10_corruption_robustness_and_spectral_sensitivity.png",
            "14-condition corruption robustness stress tests and spectral hyperparameter sensitivity sweeps.",
            "Six-panel stress-test and sensitivity plot showing graceful degradation under 2D pixel jitter (A), topological edge/vertex drops (B), and visibility flips or spurious false constraints (C), alongside four-order-of-magnitude stability plateaus across spectral SVD thresholds and continuation radii (D-F).",
        ),
    ]

    doc = docx.Document(tpl_alt_path)
    tbl = doc.tables[0]
    while len(tbl.rows) < len(alt_items) + 1:
        tbl.add_row()

    for idx, (item_lbl, img_path, short_cap, alt_txt) in enumerate(alt_items, start=1):
        row = tbl.rows[idx]
        _set_cell_text_preserve_shading(row.cells[0], item_lbl, font_size=10.5, bold=True)
        # Col 1: embed image thumbnail if available
        c_img = row.cells[1]
        for p in list(c_img.paragraphs):
            p.text = ""
        if img_path.exists():
            p_pic = c_img.paragraphs[0]
            p_pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_pic.add_run().add_picture(str(img_path), width=Inches(1.45))
        else:
            _set_cell_text_preserve_shading(c_img, f"[{item_lbl}]", font_size=9.5)
        _set_cell_text_preserve_shading(row.cells[2], short_cap, font_size=10.0, bold=False)
        _set_cell_text_preserve_shading(row.cells[3], alt_txt, font_size=10.0, bold=False)

    out_path_5 = PKG_DIR / "5_Figure_Alt_Text_JCDE.docx"
    out_path_7 = PKG_DIR / "7_Figure_Alt_Text_JCDE.docx"
    out_exact = PKG_DIR / "JCDE_Alt_Text_Template V1.docx"
    doc.save(out_path_5)
    shutil.copy2(out_path_5, out_path_7)
    shutil.copy2(out_path_5, out_exact)
    return str(out_path_5)


if __name__ == "__main__":
    tpls = _sync_official_templates()
    ga_info = generate_graphical_abstract(tpls["graphical_abstract"])
    hl_info = generate_highlights_docx(tpls["highlights"])
    chk_path = generate_submission_checklist_docx(tpls["checklist"])
    cl_path = generate_cover_letter_docx()
    at_path = generate_alt_text_docx(tpls["alt_text"])
    print(json.dumps({
        "graphical_abstract": ga_info,
        "highlights": hl_info,
        "submission_checklist": chk_path,
        "cover_letter": cl_path,
        "alt_text": at_path,
    }, indent=2))
