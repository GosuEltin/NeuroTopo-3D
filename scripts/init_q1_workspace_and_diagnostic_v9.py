"""Initialize Q1 Journal Workspace, freeze ACIIDS 2027 submission, and run deep spectral/failure diagnostics."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.module3_v843_contract import COHORT_SPECS
from scripts.run_v845_12family_pilot_smoke import load_benchmark_items
from src.lifting.cached_dataset import CachedModule2Dataset
from src.reasoning.constraint_binding_v6 import bind_constraints_from_point_anchors_v6
from src.reasoning.nullspace_guarded_projector_v843 import (
    NullSpaceGuardedProjectorV843,
    depth_jacobian_nullity_v843,
)
from src.reasoning.residuals_v4 import compile_semantic_residuals


def compute_gauge_reduced_nullity_n_minus_3(
    coordinates: torch.Tensor,
    constraints: list,
    tau_svd: float = 1e-3,
    rel_tol: float = 1e-2,
) -> dict:
    """Compute both (N-1) mean-centered nullity and (N-3) P-projected depth nullity."""
    xyz = coordinates.detach().cpu().double()
    n = xyz.shape[0]
    xy = xyz[:, :2]
    z0 = xyz[:, 2].clone()

    # Build P = I - 11^T/N - U(U^T U)^{-1} U^T
    ones = torch.ones((n, 1), dtype=torch.float64)
    xy_centered = xy - xy.mean(dim=0, keepdim=True)
    basis_mat = torch.cat([ones, xy_centered], dim=1)  # (N, 3)
    q_gauge, _ = torch.linalg.qr(basis_mat, mode="reduced")  # (N, 3)
    proj_p = torch.eye(n, dtype=torch.float64) - q_gauge @ q_gauge.T
    evals, evecs = torch.linalg.eigh(proj_p)
    keep = evals > 0.5
    q_z = evecs[:, keep]  # (N, N-3) orthonormal basis of col(P)

    def res_only(z_vec: torch.Tensor) -> torch.Tensor:
        pts = torch.cat((xy, z_vec[:, None]), dim=1)
        bundle = compile_semantic_residuals(pts, constraints, reference_coordinates=xyz)
        return bundle.vector

    z_req = z0.clone().requires_grad_(True)
    b0 = compile_semantic_residuals(xyz, constraints, reference_coordinates=xyz)
    if b0.vector.numel() == 0:
        return {
            "n": n,
            "m_res": 0,
            "d_null_n1": n - 1,
            "d_null_n3": max(0, n - 3),
            "sigma_min_n1": 0.0,
            "sigma_min_n3": 0.0,
        }

    jac_z = torch.autograd.functional.jacobian(res_only, z_req, vectorize=True).detach()  # (M_vec, N)
    # 1) N-1 mean-centered (matching V843)
    jac_n1 = torch.cat([jac_z, torch.ones((1, n), dtype=torch.float64) / n], dim=0)
    sv_n1 = torch.linalg.svdvals(jac_n1)
    thr_n1 = max(tau_svd, rel_tol * float(sv_n1[0].item()))
    rank_n1 = int((sv_n1 >= thr_n1).sum().item())
    d_null_n1 = int(n - rank_n1)
    s_min_n1 = float(sv_n1[n - 1].item()) if sv_n1.numel() >= n else 0.0

    # 2) N-3 affine-gauge-free (J_tilde = J_z @ Q_z)
    if q_z.shape[1] > 0:
        j_tilde = jac_z @ q_z  # (M_vec, N-3)
        sv_n3 = torch.linalg.svdvals(j_tilde)
        thr_n3 = max(tau_svd, rel_tol * float(sv_n3[0].item())) if sv_n3.numel() else tau_svd
        rank_n3 = int((sv_n3 >= thr_n3).sum().item())
        d_null_n3 = int(q_z.shape[1] - rank_n3)
        s_min_n3 = float(sv_n3[q_z.shape[1] - 1].item()) if sv_n3.numel() >= q_z.shape[1] else 0.0
    else:
        d_null_n3 = 0
        s_min_n3 = 0.0

    return {
        "n": n,
        "n_minus_3": int(q_z.shape[1]),
        "m_res": int(jac_z.shape[0]),
        "d_null_n1": d_null_n1,
        "d_null_n3": d_null_n3,
        "sigma_min_n1": s_min_n1,
        "sigma_min_n3": s_min_n3,
    }


def main() -> None:
    src_dir = ROOT / "overleaf_neurotopo_v845"
    freeze_dir = ROOT / "aciids2027_submitted_frozen"
    q1_runs = ROOT / "runs_q1_journal_v9"
    q1_ms = ROOT / "overleaf_neurotopo_q1_journal"
    (q1_ms / "figures").mkdir(parents=True, exist_ok=True)
    q1_runs.mkdir(parents=True, exist_ok=True)
    freeze_dir.mkdir(parents=True, exist_ok=True)

    files_to_freeze = [
        "main.tex",
        "main.pdf",
        "main_lncs.tex",
        "main_lncs.pdf",
        "llncs.cls",
        "splncs04.bst",
        "DescriptionTexts.txt",
    ]
    hashes = {}
    for fname in files_to_freeze:
        sp = src_dir / fname
        if sp.exists():
            dp = freeze_dir / fname
            if not dp.exists():
                shutil.copy2(sp, dp)
            h_src = hashlib.sha256(sp.read_bytes()).hexdigest()
            h_dst = hashlib.sha256(dp.read_bytes()).hexdigest()
            assert h_src == h_dst, f"Hash mismatch for {fname}"
            hashes[fname] = {
                "sha256": h_src,
                "bytes": sp.stat().st_size,
                "original_path": str(sp),
            }

    fig_freeze = freeze_dir / "figures"
    if not fig_freeze.exists() and (src_dir / "figures").exists():
        shutil.copytree(src_dir / "figures", fig_freeze)

    manifest_path = freeze_dir / "SHA256_FROZEN_ACIIDS2027.json"
    manifest_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(f"[OK] Frozen ACIIDS 2027 manifest saved to {manifest_path}")

    # Diagnostic 1: 12-family GT spectral analysis (N-1 vs N-3) & active operators
    cohorts = load_benchmark_items()
    std_items = cohorts["neurotopo_v4_quality_test_pilot144"]
    fam_stats = defaultdict(list)
    fam_ops = defaultdict(Counter)

    for it in std_items:
        f = it["contract"]
        fam = it["family"]
        gt = it["gt_3d_camera"].double() if "gt_3d_camera" in it else None
        if gt is None:
            from src.lifting.v4_dataset import collate_module2_samples
            n = f.graph.point_count
            batch = collate_module2_samples(
                [
                    {
                        "contract": f,
                        "sample_id": it["sample_id"],
                        "family": fam,
                        "geometry_group": it["sample_id"],
                        "topology_group": fam,
                        "target_mask": torch.ones(n, dtype=torch.bool),
                    }
                ]
            )
            gt = batch["targets"]["coordinates_3d_camera"][0, :n].double()
        cs = list(f.constraints)
        for c in cs:
            if c.solver_active:
                fam_ops[fam][c.operator] += 1
        info = compute_gauge_reduced_nullity_n_minus_3(gt, cs)
        fam_stats[fam].append(info)

    summary_12fam = {}
    print("\n=== 12-FAMILY SPECTRAL AUDIT AT GROUND TRUTH ===")
    for fam in sorted(fam_stats.keys()):
        rows = fam_stats[fam]
        d_n1_cnt = dict(Counter(r["d_null_n1"] for r in rows))
        d_n3_cnt = dict(Counter(r["d_null_n3"] for r in rows))
        ops_top = dict(fam_ops[fam].most_common(6))
        summary_12fam[fam] = {
            "count": len(rows),
            "n_range": [min(r["n"] for r in rows), max(r["n"] for r in rows)],
            "d_null_n1_dist": d_n1_cnt,
            "d_null_n3_dist": d_n3_cnt,
            "operators": ops_top,
        }
        print(f"  {fam:28s} | N={summary_12fam[fam]['n_range']} | d_null(N-1)={d_n1_cnt} | d_null(N-3)={d_n3_cnt} | ops={list(ops_top.keys())}")

    # Diagnostic 2: Inspect V843 abstained Tetrahedron views on Held-Out!
    v843_audit_path = ROOT / "runs_module3_v843" / "analysis.json"
    tetra_diag = {}
    if v843_audit_path.exists():
        v843_data = json.loads(v843_audit_path.read_text(encoding="utf-8"))
        ho_branch0 = v843_data["cohorts"]["heldout"]["branches"][0]["arms"]["v843_nullspace_guarded"]
        tetra_diag["heldout_branch0_tetra"] = ho_branch0["family"]["tetra_v2"]
        print("\n=== V843 HELDOUT BRANCH 0 TETRA SUMMARY ===")
        print(json.dumps(tetra_diag["heldout_branch0_tetra"], indent=2))

    # Check exact status distribution on heldout branch_0 tetrahedra from saved tensors/rows if present
    v843_rows_path = ROOT / "runs_module3_v843" / "heldout_branch_0_rows.json"
    if not v843_rows_path.exists():
        # list files in runs_module3_v843
        files_v843 = [p.name for p in (ROOT / "runs_module3_v843").glob("*")]
        tetra_diag["runs_module3_v843_files"] = files_v843
        print("runs_module3_v843 files:", files_v843)

    diag_out = q1_runs / "phase1_spectral_and_failure_diagnostics.json"
    diag_out.write_text(
        json.dumps(
            {
                "frozen_aciids2027_hashes": hashes,
                "summary_12fam_gt_spectra": summary_12fam,
                "v843_tetra_summary": tetra_diag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[OK] Saved Phase 1 diagnostics to {diag_out}")


if __name__ == "__main__":
    main()
