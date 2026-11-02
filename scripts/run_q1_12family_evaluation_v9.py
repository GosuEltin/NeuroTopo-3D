"""Suite A2: Q1 12-Family Multi-Constraint Evaluation & Theorem 2 Rigidity-Class Breakdown (V9.0).

Evaluates Generalist Equivariant Lifter (M2_V6, 1.63M) and Specialist Mirror-Pair Lifter (M2_V840)
paired with:
- Raw Neural Anchor
- Soft Tikhonov L-BFGS (lambda = 0.05)
- Uniform Safe Projector (V8.27, r <= 0.15)
- Two-Stage Nullspace-Guarded Projector (V8.43, ACIIDS 2027)
- V9.0-Fast Pure NF-RSLM + Multi-Branch (zero SLSQP, real-time mode)
- V9.0-Certified Full (Monotone Anchor + NF-RSLM Row-Space + Multi-Branch + Crossing Selector)

Across all 7 benchmark cohorts (792 views total), stratified by:
- All 12 Families, Basic-4 Solids, Complex-8 Scenes, AND
- Theorem 2 Dual-Gauge Rigidity Classes:
  * Class I: Strictly Ray-Rigid (d_null^ray == 0, d_null^aff == 0)
  * Class II: Intrinsically Rigid with Coupled Tilt-Shear (d_null^aff == 0, d_null^ray >= 1)
  * Class III: Under-Constrained Bas-Relief Nullspace (d_null^aff >= 1, d_null^ray >= 2)
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.optimize import minimize

from scripts.analyze_module3_v835_audit import numpy_pa
from scripts.evaluate_feasibility_projector_v6 import _independent_summary
from scripts.export_module2_v6_projector_payload import _model
from scripts.module2_v834_contract import model_inputs
from scripts.run_v845_12family_pilot_smoke import load_benchmark_items
from src.lifting.camera_gauge_v51 import fit_isotropic_orthographic_camera
from src.lifting.v4_dataset import collate_module2_samples
from src.reasoning.normal_flow_projector_v9 import (
    NormalFlowMultiBranchProjectorV9,
    NormalFlowProjectorConfigV9,
    build_ray_orthonormal_basis,
    decompose_depth_displacement_v9,
)
from src.reasoning.nullspace_guarded_projector_v843 import NullSpaceGuardedProjectorV843
from src.reasoning.projector_v6 import FeasibilityProjectorConfigV6
from src.reasoning.residuals_v4 import compile_semantic_residuals
from src.reasoning.shared_anchor_projector_v827 import SharedAnchorProjectorV827

OUT_JSON = ROOT / "runs_q1_journal_v9" / "suite_a_12family_792views_v9.json"


def _classify_dual_gauge(gt_pts: torch.Tensor, constraints) -> tuple[str, int, int]:
    """Compute exact (rigidity_class, d_null_ray, d_null_aff) at ground-truth manifold point."""
    pts = gt_pts.detach().cpu().double()
    n = pts.shape[0]
    xy = pts[:, :2].clone()
    z0 = pts[:, 2].clone()
    q_ray = build_ray_orthonormal_basis(n)

    b0 = compile_semantic_residuals(pts, constraints, reference_coordinates=pts)
    if b0.vector.numel() == 0 or n <= 3:
        return "Class_III_BasRelief", max(0, n - 1), max(0, n - 3)

    def res_ray(zeta: torch.Tensor) -> torch.Tensor:
        zv = z0.mean() + q_ray @ zeta
        return compile_semantic_residuals(torch.cat((xy, zv[:, None]), dim=1), constraints, reference_coordinates=pts).vector

    zeta0 = q_ray.T @ (z0 - z0.mean())
    j_ray = torch.autograd.functional.jacobian(res_ray, zeta0.clone().requires_grad_(True), vectorize=True).detach()
    _, sr, _ = torch.linalg.svd(j_ray, full_matrices=False)
    smax = float(sr[0].item()) if sr.numel() else 0.0
    thr = max(1e-3, 1e-2 * smax)
    d_ray = int((n - 1) - int((sr >= thr).sum().item()))

    # Affine gauge basis orthogonal to span{1, x, y}
    design = torch.cat((torch.ones((n, 1), dtype=torch.float64), xy), dim=1)
    q_full, _ = torch.linalg.qr(design, mode="complete")
    q_aff = q_full[:, 3:]
    if q_aff.shape[1] == 0:
        d_aff = 0
    else:
        def res_aff(xi: torch.Tensor) -> torch.Tensor:
            zv = z0 + q_aff @ xi
            return compile_semantic_residuals(torch.cat((xy, zv[:, None]), dim=1), constraints, reference_coordinates=pts).vector

        xi0 = torch.zeros(q_aff.shape[1], dtype=torch.float64, requires_grad=True)
        j_aff = torch.autograd.functional.jacobian(res_aff, xi0, vectorize=True).detach()
        _, sa, _ = torch.linalg.svd(j_aff, full_matrices=False)
        samax = float(sa[0].item()) if sa.numel() else 0.0
        thra = max(1e-3, 1e-2 * samax)
        d_aff = int(q_aff.shape[1] - int((sa >= thra).sum().item()))

    if d_ray == 0:
        return "Class_I_RayRigid", d_ray, d_aff
    if d_aff == 0 and d_ray >= 1:
        return "Class_II_CoupledTilt", d_ray, d_aff
    return "Class_III_BasRelief", d_ray, d_aff


def _soft_tikhonov(raw_d: torch.Tensor, cs, lam: float = 0.05) -> torch.Tensor:
    b0 = compile_semantic_residuals(raw_d, cs, reference_coordinates=raw_d)
    if not b0.records:
        return raw_d
    x0 = raw_d.numpy().reshape(-1).copy()

    def fg(x_np: np.ndarray):
        xt = torch.from_numpy(x_np).requires_grad_(True)
        pts = xt.reshape_as(raw_d)
        b = compile_semantic_residuals(pts, cs, reference_coordinates=raw_d)
        loss = b.objective + lam * (pts - raw_d).square().sum()
        g = torch.autograd.grad(loss, xt)[0]
        return float(loss.detach().item()), g.detach().numpy()

    sol = minimize(fg, x0, jac=True, method="L-BFGS-B", options={"maxiter": 35, "ftol": 1e-10})
    return torch.from_numpy(sol.x).reshape_as(raw_d)


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m_v6 = _model(
        torch.load(ROOT / "runs_module2_v6/full_seed_20260847/best.pt", map_location="cpu", weights_only=False),
        device,
    ).eval()
    m_v840 = _model(
        torch.load(ROOT / "runs_module2_v840/branch_0/b512_weighted/final.pt", map_location="cpu", weights_only=False),
        device,
    ).eval()

    proj_v827 = SharedAnchorProjectorV827(FeasibilityProjectorConfigV6())
    proj_v843 = NullSpaceGuardedProjectorV843()
    proj_v9_fast = NormalFlowMultiBranchProjectorV9(
        NormalFlowProjectorConfigV9(monotone_stage1_anchor=False, enable_slsqp_fallback_on_rigid=False)
    )
    proj_v9_cert = NormalFlowMultiBranchProjectorV9(NormalFlowProjectorConfigV9(monotone_stage1_anchor=True))

    cohorts = load_benchmark_items()
    out_summary = {}
    all_792_rows = []

    for cname, items in cohorts.items():
        t0 = time.perf_counter()
        rows = []
        fam_rows = defaultdict(list)
        class_rows = defaultdict(list)

        for it in items:
            f = it["contract"]
            fam = it["family"]
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
            tgt = batch["targets"]["coordinates_3d_camera"][0, :n].double().cpu()
            msk = torch.ones(n, dtype=torch.bool)
            px = f.graph.points_2d.double().cpu()
            adj = f.graph.adjacency
            sty = f.graph.edge_style
            cs = list(f.constraints)

            rig_cls, d_ray, d_aff = _classify_dual_gauge(tgt, cs)

            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                p_v6 = m_v6(**model_inputs(batch, device))["coordinates_3d"].float().cpu()[0, :n].double()
                p_v840 = m_v840(**model_inputs(batch, device))["coordinates_3d"].float().cpu()[0, :n].double()

            cam6 = fit_isotropic_orthographic_camera(p_v6, px)
            p_tik = _soft_tikhonov(p_v6, cs, lam=0.05)
            r6_827 = proj_v827.project(p_v6, cs, camera=cam6, points_2d=px)
            r6_843 = proj_v843.project(p_v6, cs, camera=cam6, points_2d=px)

            tf0 = time.perf_counter()
            r6_v9f = proj_v9_fast.project(p_v6, cs, camera=cam6, points_2d=px, adjacency=adj, edge_style=sty)
            lat_v9f_ms = (time.perf_counter() - tf0) * 1000.0

            tc0 = time.perf_counter()
            r6_v9c = proj_v9_cert.project(p_v6, cs, camera=cam6, points_2d=px, adjacency=adj, edge_style=sty)
            lat_v9c_ms = (time.perf_counter() - tc0) * 1000.0

            cam8 = fit_isotropic_orthographic_camera(p_v840, px)
            r8_v9c = proj_v9_cert.project(p_v840, cs, camera=cam8, points_2d=px, adjacency=adj, edge_style=sty)

            ind_tik, _, info_tik, _, _ = _independent_summary(p_tik, cs, p_v6, 0.0011)
            ind6_827, _, info6_827, _, _ = _independent_summary(r6_827.coordinates.double(), cs, p_v6, 0.0011)
            ind6_843, _, info6_843, _, _ = _independent_summary(r6_843.coordinates.double(), cs, p_v6, 0.0011)
            ind6_v9f, _, info6_v9f, _, _ = _independent_summary(r6_v9f.coordinates.double(), cs, p_v6, 0.0011)
            ind6_v9c, _, info6_v9c, _, _ = _independent_summary(r6_v9c.coordinates.double(), cs, p_v6, 0.0011)
            ind8_v9c, _, info8_v9c, _, _ = _independent_summary(r8_v9c.coordinates.double(), cs, p_v840, 0.0011)

            spec_tik = decompose_depth_displacement_v9(p_tik, p_v6, cs)
            raw_pa = float(numpy_pa(p_v6, tgt, msk, True))
            v9c_pa = float(numpy_pa(r6_v9c.coordinates.double(), tgt, msk, True))

            row = {
                "sample_id": it["sample_id"],
                "family": fam,
                "rigidity_class": rig_cls,
                "d_null_ray": d_ray,
                "d_null_aff": d_aff,
                "is_basic_4": fam in ("tetra_v2", "cube_v2", "prism_v2", "polyhedra_v2"),
                "v6_raw_pa": raw_pa,
                "v6_tikhonov_pa": float(numpy_pa(p_tik, tgt, msk, True)),
                "v6_tikhonov_valid": bool(ind_tik and info_tik > 0),
                "v6_tikhonov_null_rms": float(spec_tik["null_rms"]),
                "v6_v827_pa": float(numpy_pa(r6_827.coordinates.double(), tgt, msk, True)),
                "v6_v827_valid": bool((not r6_827.abstained) and ind6_827 and info6_827 > 0),
                "v6_v843_pa": float(numpy_pa(r6_843.coordinates.double(), tgt, msk, True)),
                "v6_v843_valid": bool((not r6_843.abstained) and ind6_843 and info6_843 > 0),
                "v6_v9fast_pa": float(numpy_pa(r6_v9f.coordinates.double(), tgt, msk, True)),
                "v6_v9fast_valid": bool((not r6_v9f.abstained) and ind6_v9f and info6_v9f > 0),
                "v6_v9fast_lat_ms": lat_v9f_ms,
                "v6_v9cert_pa": v9c_pa,
                "v6_v9cert_valid": bool((not r6_v9c.abstained) and ind6_v9c and info6_v9c > 0),
                "v6_v9cert_invalid": bool((not r6_v9c.abstained) and (not ind6_v9c)),
                "v6_v9cert_harm": bool((not r6_v9c.abstained) and (v9c_pa > raw_pa + 0.005)),
                "v6_v9cert_null_rms": float(r6_v9c.diagnostics.get("nullspace_drift_rms", 0.0)),
                "v6_v9cert_lat_ms": lat_v9c_ms,
                "v840_raw_pa": float(numpy_pa(p_v840, tgt, msk, True)),
                "v840_v9cert_pa": float(numpy_pa(r8_v9c.coordinates.double(), tgt, msk, True)),
                "v840_v9cert_valid": bool((not r8_v9c.abstained) and ind8_v9c and info8_v9c > 0),
            }
            rows.append(row)
            fam_rows[fam].append(row)
            class_rows[rig_cls].append(row)
            all_792_rows.append(row)

        def summarize_subset(sub: list[dict]) -> dict:
            if not sub:
                return {}
            return {
                "views": len(sub),
                "v6_raw_pa": round(float(fmean(r["v6_raw_pa"] for r in sub)), 4),
                "v6_tikhonov_pa": round(float(fmean(r["v6_tikhonov_pa"] for r in sub)), 4),
                "v6_tikhonov_valid_pct": round(float(fmean(r["v6_tikhonov_valid"] for r in sub)) * 100.0, 2),
                "v6_tikhonov_null_rms": round(float(fmean(r["v6_tikhonov_null_rms"] for r in sub)), 4),
                "v6_v827_pa": round(float(fmean(r["v6_v827_pa"] for r in sub)), 4),
                "v6_v827_valid_pct": round(float(fmean(r["v6_v827_valid"] for r in sub)) * 100.0, 2),
                "v6_v843_pa": round(float(fmean(r["v6_v843_pa"] for r in sub)), 4),
                "v6_v843_valid_pct": round(float(fmean(r["v6_v843_valid"] for r in sub)) * 100.0, 2),
                "v6_v9fast_pa": round(float(fmean(r["v6_v9fast_pa"] for r in sub)), 4),
                "v6_v9fast_valid_pct": round(float(fmean(r["v6_v9fast_valid"] for r in sub)) * 100.0, 2),
                "v6_v9fast_median_ms": round(float(np.median([r["v6_v9fast_lat_ms"] for r in sub])), 2),
                "v6_v9cert_pa": round(float(fmean(r["v6_v9cert_pa"] for r in sub)), 4),
                "v6_v9cert_valid_pct": round(float(fmean(r["v6_v9cert_valid"] for r in sub)) * 100.0, 2),
                "v6_v9cert_harm_pct": round(float(fmean(r["v6_v9cert_harm"] for r in sub)) * 100.0, 2),
                "v6_v9cert_invalid_count": sum(1 for r in sub if r["v6_v9cert_invalid"]),
                "v6_v9cert_null_rms": round(float(fmean(r["v6_v9cert_null_rms"] for r in sub)), 5),
                "v840_raw_pa": round(float(fmean(r["v840_raw_pa"] for r in sub)), 4),
                "v840_v9cert_pa": round(float(fmean(r["v840_v9cert_pa"] for r in sub)), 4),
                "v840_v9cert_valid_pct": round(float(fmean(r["v840_v9cert_valid"] for r in sub)) * 100.0, 2),
            }

        out_summary[cname] = {
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "all_12_families": summarize_subset(rows),
            "basic_4_families": summarize_subset([r for r in rows if r["is_basic_4"]]),
            "complex_8_scenes": summarize_subset([r for r in rows if not r["is_basic_4"]]),
            "by_rigidity_class": {cls_k: summarize_subset(cls_v) for cls_k, cls_v in sorted(class_rows.items())},
            "per_family": {fam_k: summarize_subset(fam_v) for fam_k, fam_v in sorted(fam_rows.items())},
        }
        a12 = out_summary[cname]["all_12_families"]
        print(
            f"[{cname:38s}] V6+V843: PA={a12['v6_v843_pa']:.4f} ({a12['v6_v843_valid_pct']:5.1f}%) -> "
            f"V6+V9-Cert: PA={a12['v6_v9cert_pa']:.4f} ({a12['v6_v9cert_valid_pct']:5.1f}%, Inv={a12['v6_v9cert_invalid_count']}) | "
            f"V9-Fast: {a12['v6_v9fast_median_ms']:.1f}ms",
            flush=True,
        )

    pooled_by_class = defaultdict(list)
    pooled_by_fam = defaultdict(list)
    for r in all_792_rows:
        pooled_by_class[r["rigidity_class"]].append(r)
        pooled_by_fam[r["family"]].append(r)

    def summarize_global(sub: list[dict]) -> dict:
        return {
            "views": len(sub),
            "v6_raw_pa": round(float(fmean(r["v6_raw_pa"] for r in sub)), 4),
            "v6_tikhonov_pa": round(float(fmean(r["v6_tikhonov_pa"] for r in sub)), 4),
            "v6_tikhonov_valid_pct": round(float(fmean(r["v6_tikhonov_valid"] for r in sub)) * 100.0, 2),
            "v6_v827_pa": round(float(fmean(r["v6_v827_pa"] for r in sub)), 4),
            "v6_v827_valid_pct": round(float(fmean(r["v6_v827_valid"] for r in sub)) * 100.0, 2),
            "v6_v843_pa": round(float(fmean(r["v6_v843_pa"] for r in sub)), 4),
            "v6_v843_valid_pct": round(float(fmean(r["v6_v843_valid"] for r in sub)) * 100.0, 2),
            "v6_v9fast_pa": round(float(fmean(r["v6_v9fast_pa"] for r in sub)), 4),
            "v6_v9fast_valid_pct": round(float(fmean(r["v6_v9fast_valid"] for r in sub)) * 100.0, 2),
            "v6_v9fast_median_ms": round(float(np.median([r["v6_v9fast_lat_ms"] for r in sub])), 2),
            "v6_v9cert_pa": round(float(fmean(r["v6_v9cert_pa"] for r in sub)), 4),
            "v6_v9cert_valid_pct": round(float(fmean(r["v6_v9cert_valid"] for r in sub)) * 100.0, 2),
            "v6_v9cert_harm_pct": round(float(fmean(r["v6_v9cert_harm"] for r in sub)) * 100.0, 2),
            "v6_v9cert_invalid_count": sum(1 for r in sub if r["v6_v9cert_invalid"]),
        }

    report = {
        "status": "PASS",
        "experiment": "Suite_A2_12Family_792Views_and_Theorem2_Classes_V9",
        "pooled_792_views": {
            "overall": summarize_global(all_792_rows),
            "by_rigidity_class": {k: summarize_global(v) for k, v in sorted(pooled_by_class.items())},
            "per_family": {k: summarize_global(v) for k, v in sorted(pooled_by_fam.items())},
        },
        "cohorts": out_summary,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved 792-view 12-family V9 report to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
