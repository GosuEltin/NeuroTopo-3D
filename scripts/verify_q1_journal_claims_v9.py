"""Automated End-to-End Q1 Journal Reproducibility, Integrity, and Claim-to-JSON Verifier (V9.0).

Executes five strict certification gates before Q1 manuscript release:
  Gate 1: Bit-for-bit SHA-256 integrity of frozen prior submission (`overleaf_neurotopo_v845/`).
  Gate 2: Pytest execution across V9 normal-flow projector, Q1 lifting baselines, classical MSDA/Lipson-Shpitalni solvers, and ISO 10303-21 STEP exporter.
  Gate 3: Presence and PASS status of all 10 executed JSON benchmark & theorem proof reports.
  Gate 4: Verification of exported ISO 10303-21 `.STEP` / `.OBJ` solids, all 10 vector PDF figures (`fig1`..`fig10`),
          compiled `overleaf_neurotopo_q1_journal/main.pdf`, and zero undefined refs/citations in `main.log`.
  Gate 5: Dynamic claim-to-JSON numerical match audit across `main.tex` + `sections/*.tex` plus zero-ACIIDS mention check.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_q1_journal_v9"
Q1_DIR = ROOT / "overleaf_neurotopo_q1_journal"
FROZEN_SHA_JSON = ROOT / "aciids2027_submitted_frozen" / "SHA256_FROZEN_ACIIDS2027.json"
OUT_REPORT = RUN_DIR / "final_q1_audit_report.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_gate1_frozen_aciids() -> dict:
    manifest = json.loads(FROZEN_SHA_JSON.read_text(encoding="utf-8"))
    mismatches = []
    checked = 0
    for name, meta in manifest.items():
        orig = Path(meta["original_path"])
        checked += 1
        if not orig.exists():
            mismatches.append({"file": name, "error": "missing_original_file"})
            continue
        live_sha = _sha256(orig)
        live_bytes = orig.stat().st_size
        if live_sha != meta["sha256"] or live_bytes != meta["bytes"]:
            mismatches.append(
                {
                    "file": name,
                    "expected_sha256": meta["sha256"],
                    "live_sha256": live_sha,
                    "expected_bytes": meta["bytes"],
                    "live_bytes": live_bytes,
                }
            )
    return {
        "passed": len(mismatches) == 0 and checked >= 7,
        "files_checked": checked,
        "mismatches": mismatches,
    }


def check_gate2_pytest() -> dict:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_normal_flow_projector_v9.py",
        "tests/test_q1_lifting_baselines_v9.py",
        "tests/test_classical_msda_v9.py",
        "tests/test_step_exporter_v9.py",
        "-q",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def check_gate3_json_reports() -> tuple[dict, dict]:
    expected = {
        "theorems": (RUN_DIR / "theorem_verification_report.json", "all_q1_theorems_machine_verified"),
        "module2_3seeds": (RUN_DIR / "module2_modern_baselines_3seeds.json", "PASS"),
        "module3_3seeds": (RUN_DIR / "module3_solvers_ablations_and_latency_3seeds.json", "PASS"),
        "classical_msda_ls": (RUN_DIR / "classical_msda_lipson_benchmarks_v9.json", "PASS"),
        "suite_a2_792": (RUN_DIR / "suite_a_12family_792views_v9.json", "PASS"),
        "suite_b_ood": (RUN_DIR / "suite_b_3tier_ood_v9.json", "PASS"),
        "suite_c1_realworld": (RUN_DIR / "suite_c1_realworld_gdp_and_handdrawn_v9.json", "PASS"),
        "suite_c2_cad": (RUN_DIR / "suite_c2_cad_reverse_engineering_120parts_v9.json", "PASS"),
        "suite_d_robustness": (RUN_DIR / "suite_d_robustness_and_sensitivity_v9.json", "PASS"),
        "v845_frontier_m1": (ROOT / "runs_v845_frontier_and_baselines" / "analysis.json", "v845_frontier_and_matched_baselines_complete"),
    }
    loaded = {}
    errors = []
    for key, (path, exp_status) in expected.items():
        if not path.exists():
            errors.append(f"Missing JSON report: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[key] = data
        if data.get("status") != exp_status:
            errors.append(f"Unexpected status in {path.name}: {data.get('status')} != {exp_status}")
    return {"passed": len(errors) == 0, "reports_verified": len(loaded), "errors": errors}, loaded


def check_gate4_artifacts_figures_and_pdf() -> dict:
    errors = []
    fig_stems = [
        "fig1_q1_architecture_and_v9_solver",
        "fig2_theorems_spectral_and_drift_scaling",
        "fig3_module2_equivariant_baselines_and_solvers",
        "fig4_basic_solids_and_rescued_tetrahedra",
        "fig5_complex_12family_rigidity_classes",
        "fig6_3tier_ood_and_nullspace_drift",
        "fig7_cad_step_reverse_engineering_6archetypes",
        "fig8_parametric_nullspace_cad_explorer",
        "fig9_realworld_gdp29k_and_handdrawn_sketches",
        "fig10_corruption_robustness_and_spectral_sensitivity",
    ]
    fig_dir = Q1_DIR / "figures"
    for stem in fig_stems:
        pdf_p = fig_dir / f"{stem}.pdf"
        if not pdf_p.exists() or pdf_p.stat().st_size < 8000:
            errors.append(f"Missing or undersized figure PDF: {pdf_p}")

    cad_dir = RUN_DIR / "cad_exports"
    step_files = sorted(cad_dir.glob("*.step"))
    obj_files = sorted(cad_dir.glob("*.obj"))
    if len(step_files) < 11:
        errors.append(f"Expected >= 11 exported .step solids (6 archetypes + 5 nullspace variants), found {len(step_files)}")
    if len(obj_files) < 6:
        errors.append(f"Expected >= 6 exported .obj solids, found {len(obj_files)}")
    for sp in step_files:
        txt = sp.read_text(encoding="utf-8", errors="replace")
        if "ISO-10303-21;" not in txt or "MANIFOLD_SOLID_BREP" not in txt:
            errors.append(f"Invalid ISO 10303-21 STEP structure in {sp.name}")

    main_pdf = Q1_DIR / "main.pdf"
    main_log = Q1_DIR / "main.log"
    page_count = 0
    if not main_pdf.exists() or main_pdf.stat().st_size < 100_000:
        errors.append("Missing or undersized overleaf_neurotopo_q1_journal/main.pdf")
    if not main_log.exists():
        errors.append("Missing overleaf_neurotopo_q1_journal/main.log")
    else:
        log_txt = main_log.read_text(encoding="utf-8", errors="replace")
        if "LaTeX Warning: Reference" in log_txt or "There were undefined references" in log_txt:
            errors.append("Undefined LaTeX references detected in main.log")
        if "LaTeX Warning: Citation" in log_txt or "There were undefined citations" in log_txt:
            errors.append("Undefined LaTeX citations detected in main.log")
        if re.search(r"^! ", log_txt, flags=re.MULTILINE):
            errors.append("LaTeX compilation error (`! `) detected in main.log")
        m_pg = re.search(r"Output written on main\.pdf \((\d+) pages", log_txt)
        if m_pg:
            page_count = int(m_pg.group(1))
        if page_count < 22:
            errors.append(f"Expected unabridged Q1 master edition >= 22 pages, got {page_count}")

    return {
        "passed": len(errors) == 0,
        "pdf_figures_count": len(fig_stems),
        "step_files_count": len(step_files),
        "obj_files_count": len(obj_files),
        "compiled_pdf_pages": page_count,
        "errors": errors,
    }


def check_gate5_claims_against_json(loaded: dict) -> dict:
    tex_files = [Q1_DIR / "main.tex"] + sorted((Q1_DIR / "sections").glob("*.tex"))
    full_tex = "\n".join(p.read_text(encoding="utf-8") for p in tex_files)

    m2 = loaded["module2_3seeds"]["summary_table"]
    m3_h = loaded["module3_3seeds"]["aggregate_3seed"]["heldout"]
    a2 = loaded["suite_a2_792"]["pooled_792_views"]
    b_ood = loaded["suite_b_ood"]
    c1 = loaded["suite_c1_realworld"]
    c2 = loaded["suite_c2_cad"]["overall_120_parts"]
    d_rob = loaded["suite_d_robustness"]["part1_corruption_robustness_14_conditions"]
    v845 = loaded["v845_frontier_m1"]
    cls_rep = loaded["classical_msda_ls"]
    cls_h = cls_rep["part1_heldout_3seeds_1536views"]["aggregate_3seed_heldout"]
    cls_a2 = cls_rep["part2_12family_792views"]
    cls_cad = cls_rep["part3_cad_reverse_engineering_120parts"]["overall_120_parts"]

    required_tokens = {
        # Module 1 & Frontier VLMs
        "m1_pck5": f"{v845['module1_v842_heldout']['overall']['pck5_pct']['mean']:.2f}",
        "m1_edge_f1": f"{v845['module1_v842_heldout']['overall']['edge_f1_pct']['mean']:.2f}",
        "vlm_gpt56_pa": f"{v845['frontier_vlm_comparison']['gpt-5.6-sol']['vlm_plus_v840_m2_shape_pa']:.4f}",
        # Module 2 3-seed baselines
        "m2_pairgt_sym_pa": f"{m2['pair_gt_sym_v6']['heldout']['macro_shape_pa_mean']:.4f}",
        "m2_pairgt_asym_pa": f"{m2['pair_gt_nosym_v6']['heldout']['macro_shape_pa_mean']:.4f}",
        "m2_egnn_pa": f"{m2['egnn_v9']['heldout']['macro_shape_pa_mean']:.4f}",
        "m2_gatv2_pa": f"{m2['gatv2_edge_v9']['heldout']['macro_shape_pa_mean']:.4f}",
        "m2_graphgps_pa": f"{m2['graphgps_v9']['heldout']['macro_shape_pa_mean']:.4f}",
        # Classical non-neural baselines (Marill-MSDA 1991 & Lipson-Shpitalni 1996)
        "cls_h_msda_pa": f"{cls_h['marill_msda_1991']['macro_shape_pa_mean']:.4f}",
        "cls_h_ls_pa": f"{cls_h['lipson_shpitalni_1996']['macro_shape_pa_mean']:.4f}",
        "cls_h_ls_proper_pa": f"{cls_h['lipson_shpitalni_1996']['macro_proper_pa_mean']:.4f}",
        "cls_h_ls_paircov": f"{cls_h['lipson_shpitalni_1996']['pair_cov_pct_mean']:.2f}",
        "cls_a2_overall_ls_pa": f"{cls_a2['overall_792_views']['lipson_shpitalni_pa']:.4f}",
        "cls_a2_class1_ls_pa": f"{cls_a2['by_rigidity_class']['Class_I_RayRigid']['lipson_shpitalni_pa']:.4f}",
        "cls_a2_class3_ls_pa": f"{cls_a2['by_rigidity_class']['Class_III_BasRelief']['lipson_shpitalni_pa']:.4f}",
        "cls_cad_msda_pa": f"{cls_cad['marill_msda']['shape_pa_mean']:.4f}",
        "cls_cad_ls_pa": f"{cls_cad['lipson_shpitalni']['shape_pa_mean']:.4f}",
        "cls_cad_ls_plan": f"{cls_cad['lipson_shpitalni']['max_face_planarity_mm_mean']:.4f}",
        # Module 3 3-seed solvers & ablations
        "m3_v9_paircov": f"{m3_h['v9_certified_full']['pair_cov_mean'] * 100.0:.2f}",
        "m3_v9_pa": f"{m3_h['v9_certified_full']['macro_shape_pa_mean']:.4f}",
        "m3_v9_tetracov": f"{m3_h['v9_certified_full']['family']['tetra_v2']['pair_cov_mean'] * 100.0:.2f}",
        "m3_v843_paircov": f"{m3_h['v843_aciids']['pair_cov_mean'] * 100.0:.2f}",
        "m3_v843_pa": f"{m3_h['v843_aciids']['macro_shape_pa_mean']:.4f}",
        "m3_tikhonov_pa": f"{m3_h['soft_tikhonov_lbfgs']['macro_shape_pa_mean']:.4f}",
        "m3_alm_pa": f"{m3_h['alm_classical']['macro_shape_pa_mean']:.4f}",
        # Suite A2 792 views & Rigidity Classes
        "a2_overall_v9_pa": f"{a2['overall']['v6_v9cert_pa']:.4f}",
        "a2_overall_v9_cov": f"{a2['overall']['v6_v9cert_valid_pct']:.2f}",
        "a2_class1_v9_pa": f"{a2['by_rigidity_class']['Class_I_RayRigid']['v6_v9cert_pa']:.4f}",
        "a2_class2_v9_pa": f"{a2['by_rigidity_class']['Class_II_CoupledTilt']['v6_v9cert_pa']:.4f}",
        "a2_class3_v9_pa": f"{a2['by_rigidity_class']['Class_III_BasRelief']['v6_v9cert_pa']:.4f}",
        # Suite B 3-Tier OOD
        "b_tier1_v9_pa": f"{b_ood['tier1_zeroshot_basic4_to_complex8']['neurotopo_v5_final_confirmation_pilot144']['overall_complex_8']['v9_ood_pa']:.4f}",
        "b_loso_s03_v9_pa": f"{b_ood['tier2_loso_unseen_topologies_3seeds']['scene_03']['mean_3seeds']['v9_ood_pa']:.4f}",
        "b_loso_s05_v9_pa": f"{b_ood['tier2_loso_unseen_topologies_3seeds']['scene_05']['mean_3seeds']['v9_ood_pa']:.4f}",
        "b_tier3_v6_v9_pa": f"{b_ood['tier3_nontemplate_convex_hulls_100']['m2_v6_generalist']['v9_ood_pa']:.4f}",
        # Suite C1 Real-World GDP-29K & Hand-Drawn
        "c1_gdp9_v9_cov": f"{c1['part1_gdp29k_human_reviewed_9cells']['aggregate_9cells']['v9_valid_cov']['mean'] * 100.0:.2f}",
        "c1_gdp9_v843_cov": f"{c1['part1_gdp29k_human_reviewed_9cells']['aggregate_9cells']['v843_valid_cov']['mean'] * 100.0:.2f}",
        "c1_handdrawn_rmse": f"{c1['part3_photographed_handdrawn_sketches_20']['summary']['mean_reprojection_rmse_px']:.2f}",
        # Suite C2 120-Part CAD STEP Reverse Engineering
        "c2_v9_pa": f"{c2['v9_ours']['shape_pa_mean']:.4f}",
        "c2_v9_sew_pct": f"{c2['v9_ours']['cad_sewing_valid_pct']:.2f}" if c2['v9_ours']['cad_sewing_valid_pct'] % 1 else f"{c2['v9_ours']['cad_sewing_valid_pct']:.1f}",
        "c2_raw_pa": f"{c2['raw_neural']['shape_pa_mean']:.4f}",
        # Suite D Corruption Robustness
        "d_clean_v9_pa": f"{d_rob['clean']['v9_macro_pa']:.4f}",
        "d_ed20_v9_pa": f"{d_rob['edge_drop_20']['v9_macro_pa']:.4f}",
        "d_flip20_v9_pa": f"{d_rob['edge_style_flip_20']['v9_macro_pa']:.4f}",
    }

    missing_claims = []
    for k, tok in required_tokens.items():
        if tok not in full_tex:
            # Allow 1-decimal or 2-decimal formatting match for percentages if exact value ends in .00/.10 etc.
            alt_tok = tok.rstrip("0").rstrip(".") if "." in tok else tok
            if alt_tok not in full_tex:
                missing_claims.append({"metric": k, "expected_token": tok})

    no_aciids_mentions = "aciids" not in full_tex.lower()
    stale_param_163m = "1.63\\text{M}" in full_tex

    return {
        "passed": (len(missing_claims) == 0) and no_aciids_mentions and (not stale_param_163m),
        "claims_checked": len(required_tokens),
        "missing_claims": missing_claims,
        "zero_aciids_mentions_in_manuscript": no_aciids_mentions,
        "no_stale_parameter_strings": not stale_param_163m,
    }


def main() -> None:
    g1 = check_gate1_frozen_aciids()
    g2 = check_gate2_pytest()
    g3, loaded = check_gate3_json_reports()
    g4 = check_gate4_artifacts_figures_and_pdf()
    g5 = check_gate5_claims_against_json(loaded) if g3["passed"] else {"passed": False, "error": "gate3_failed"}

    all_passed = all(g["passed"] for g in (g1, g2, g3, g4, g5))
    report = {
        "overall_status": "ALL_5_Q1_AUDIT_GATES_PASSED" if all_passed else "FAILED",
        "all_passed": all_passed,
        "gate1_frozen_aciids2027_sha256": g1,
        "gate2_pytest_unit_tests": g2,
        "gate3_json_reports": g3,
        "gate4_figures_step_and_pdf": g4,
        "gate5_claim_to_json_audit": g5,
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
