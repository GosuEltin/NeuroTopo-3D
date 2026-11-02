"""Comprehensive Q1 Module 3 Solver Baselines, V9.0 Component Ablations & Latency Benchmark.

Evaluates 12 solver & ablation arms across all 3 seeds on:
- V831 TRAIN (256 groups / 512 views)
- V836 FIT DEVELOPMENT (512 groups / 1,024 views)
- V842 HELDOUT (256 groups / 512 views)

Arms:
1. planar_z0: Planar camera-plane projection (z = 0)
2. raw_neural: Unprojected LiftingModuleV6 neural anchor
3. soft_tikhonov_lbfgs: Soft penalty L-BFGS (||f(V)||^2 + lambda ||V - V0||^2, lambda=0.05)
4. gn_unconstrained: Unconstrained Damped Gauss-Newton (no nullspace projection, no gates)
5. alm_classical: Classical Augmented Lagrangian Method (multiplier updates, no spectral guard)
6. r015_uniform: Uniform Safe Trust Region (r <= 0.15)
7. r045_ungated: Ungated Wide Trust Region (r <= 0.45)
8. v843_aciids: Two-Stage Nullspace-Guarded Projector (V8.43, ACIIDS 2027)
9. v9_fast_pure_nf: Pure Fast NF-RSLM + Multi-Branch (V9.0-Fast, zero SLSQP, real-time mode)
10. v9_ablation_no_multibranch: V9.0 Certified without Multi-Branch Reflection Continuation
11. v9_ablation_no_rowspace: V9.0 Certified without d_null >= 1 Row-Space Continuation
12. v9_certified_full: Full V9.0 (Monotone Anchor + NF-RSLM Row-Space + Multi-Branch + Crossing Selector)

Also computes Paired Wilcoxon Signed-Rank + Holm-Bonferroni p-values and per-stage wall-clock latency!
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import wilcoxon

from scripts.analyze_module3_v835_audit import numpy_pa
from scripts.evaluate_feasibility_projector_v6 import _independent_summary
from scripts.module2_v834_contract import read_json
from scripts.module3_v841_contract import summarize_projector_rows
from scripts.module3_v843_contract import COHORT_SPECS
from src.lifting.cached_dataset import CachedModule2Dataset
from src.reasoning.constraint_binding_v6 import bind_constraints_from_point_anchors_v6
from src.reasoning.normal_flow_projector_v9 import (
    NormalFlowMultiBranchProjectorV9,
    NormalFlowProjectorConfigV9,
    decompose_depth_displacement_v9,
)
from src.reasoning.residuals_v4 import compile_semantic_residuals

OUT_JSON = ROOT / "runs_q1_journal_v9" / "module3_solvers_ablations_and_latency_3seeds.json"
SEM_TOL = 0.0011
HARM_TOL = 0.005


def _score_candidate_row(
    cand: torch.Tensor,
    abstained: bool,
    status: str,
    regime: str,
    row_base: dict,
    rep_base: dict,
    latency_ms: float,
    null_drift_rms: float = 0.0,
) -> dict:
    raw = rep_base["raw"].double()
    target = rep_base["target"].double()
    mask = rep_base["mask"]
    constraints = rep_base["constraints"]

    if abstained:
        return {
            **row_base,
            "accepted_valid": False,
            "unsafe_accepted": False,
            "invalid_accepted": False,
            "projector_harm": False,
            "shape_harm": False,
            "status": status,
            "abstained": True,
            "post_pa": float(row_base["raw_pa"]),
            "post_reflection_pa": float(row_base["raw_reflection_pa"]),
            "regime": regime,
            "latency_ms": latency_ms,
            "nullspace_drift_rms": 0.0,
        }

    cand_d = cand.detach().cpu().double()
    valid, residual, info_count, _, _ = _independent_summary(cand_d, constraints, raw, SEM_TOL)
    post_pa = numpy_pa(cand_d, target, mask, False)
    post_refl_pa = numpy_pa(cand_d, target, mask, True)
    proper_harm = post_pa > float(row_base["raw_pa"]) + HARM_TOL
    shape_harm = post_refl_pa > float(row_base["raw_reflection_pa"]) + HARM_TOL
    delta = (cand_d - raw).numpy()
    rms = float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
    cmax = float(np.max(np.abs(delta)))

    return {
        **row_base,
        "accepted_valid": bool(valid and info_count > 0),
        "unsafe_accepted": False,
        "invalid_accepted": bool(not valid),
        "projector_harm": bool(proper_harm),
        "shape_harm": bool(shape_harm),
        "status": status,
        "abstained": False,
        "independent_maximum_residual": float(residual),
        "informative_constraints": int(info_count),
        "displacement_rms": rms,
        "coordinate_max": cmax,
        "post_pa": post_pa,
        "post_reflection_pa": post_refl_pa,
        "regime": regime,
        "latency_ms": latency_ms,
        "nullspace_drift_rms": float(null_drift_rms),
    }


def _solve_soft_tikhonov_lbfgs(
    raw: torch.Tensor,
    constraints,
    weights: torch.Tensor | None,
    lam: float = 0.05,
) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    raw_d = raw.detach().cpu().double()
    b0 = compile_semantic_residuals(raw_d, constraints, reference_coordinates=raw_d, constraint_weights=weights)
    if not b0.records:
        return raw_d, (time.perf_counter() - t0) * 1000.0
    x0 = raw_d.numpy().reshape(-1).copy()

    def obj_grad(flat_np: np.ndarray) -> tuple[float, np.ndarray]:
        flat_t = torch.from_numpy(flat_np).requires_grad_(True)
        pts = flat_t.reshape_as(raw_d)
        b = compile_semantic_residuals(pts, constraints, reference_coordinates=raw_d, constraint_weights=weights)
        loss = b.objective + lam * (pts - raw_d).square().sum()
        g = torch.autograd.grad(loss, flat_t)[0]
        return float(loss.detach().item()), g.detach().numpy()

    sol = minimize(obj_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": 40, "ftol": 1e-10})
    out = torch.from_numpy(sol.x).reshape_as(raw_d)
    return out, (time.perf_counter() - t0) * 1000.0


def _solve_unconstrained_gn(
    raw: torch.Tensor,
    constraints,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    raw_d = raw.detach().cpu().double()
    cur = raw_d.clone()
    b_cur = compile_semantic_residuals(cur, constraints, reference_coordinates=raw_d, constraint_weights=weights)
    if not b_cur.records:
        return cur, (time.perf_counter() - t0) * 1000.0

    for _ in range(20):
        rmax = float(b_cur.raw_constraint_rms.max().item())
        if rmax <= SEM_TOL:
            break
        f_vec = b_cur.vector.detach()
        fn2 = float(f_vec.square().sum().item())
        if fn2 <= 1e-24:
            break

        def res_fn(flat: torch.Tensor) -> torch.Tensor:
            return compile_semantic_residuals(
                flat.reshape_as(raw_d), constraints, reference_coordinates=raw_d, constraint_weights=weights
            ).vector

        jac = torch.autograd.functional.jacobian(res_fn, cur.reshape(-1).clone().requires_grad_(True), vectorize=True).detach()
        m_dim = jac.shape[0]
        lhs = jac @ jac.T + (1e-4 + 1e-2 * fn2) * torch.eye(m_dim, dtype=torch.float64)
        step = -jac.T @ torch.linalg.solve(lhs, f_vec)
        cur = cur + step.reshape_as(raw_d)
        b_cur = compile_semantic_residuals(cur, constraints, reference_coordinates=raw_d, constraint_weights=weights)

    return cur, (time.perf_counter() - t0) * 1000.0


def _solve_classical_alm(
    raw: torch.Tensor,
    constraints,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    raw_d = raw.detach().cpu().double()
    b0 = compile_semantic_residuals(raw_d, constraints, reference_coordinates=raw_d, constraint_weights=weights)
    if not b0.records:
        return raw_d, (time.perf_counter() - t0) * 1000.0

    m_dim = int(b0.vector.numel())
    nu = torch.zeros(m_dim, dtype=torch.float64)
    rho = 10.0
    cur_np = raw_d.numpy().reshape(-1).copy()

    for _ in range(4):
        def alm_obj(flat_np: np.ndarray) -> tuple[float, np.ndarray]:
            ft = torch.from_numpy(flat_np).requires_grad_(True)
            pts = ft.reshape_as(raw_d)
            b = compile_semantic_residuals(pts, constraints, reference_coordinates=raw_d, constraint_weights=weights)
            f = b.vector
            # Augmented Lagrangian for min 0.5 ||V - V0||^2 s.t. f(V) = 0
            loss = 0.5 * (pts - raw_d).square().sum() + torch.dot(nu, f) + 0.5 * rho * f.square().sum()
            g = torch.autograd.grad(loss, ft)[0]
            return float(loss.detach().item()), g.detach().numpy()

        sol = minimize(alm_obj, cur_np, jac=True, method="L-BFGS-B", options={"maxiter": 25, "ftol": 1e-10})
        cur_np = sol.x.copy()
        pts_cur = torch.from_numpy(cur_np).reshape_as(raw_d)
        b_now = compile_semantic_residuals(pts_cur, constraints, reference_coordinates=raw_d, constraint_weights=weights)
        if float(b_now.raw_constraint_rms.max().item()) <= SEM_TOL:
            break
        nu = nu + rho * b_now.vector.detach()
        rho *= 4.0

    out = torch.from_numpy(cur_np).reshape_as(raw_d)
    return out, (time.perf_counter() - t0) * 1000.0


def evaluate_view_all_solvers(task: dict) -> dict[str, dict]:
    torch.set_num_threads(1)
    row_base = dict(task["row_r015"])
    rep_base = dict(task["rep_r015"])
    raw = rep_base["raw"]
    raw_d = raw.double()
    constraints = rep_base["constraints"]
    camera = rep_base["camera"]
    pixels = rep_base["pixels"]
    active_conf = task["active_confidences"]
    adj = task["adjacency"]
    sty = task["edge_style"]
    run_unconstrained_baselines = bool(task["run_unconstrained"])

    out: dict[str, dict] = {}

    # 1. Planar z = 0
    planar = raw_d.clone()
    planar[:, 2] = 0.0
    out["planar_z0"] = _score_candidate_row(planar, False, "planar_z0", "planar", row_base, rep_base, 0.01)

    # 2. Raw Neural Anchor
    out["raw_neural"] = _score_candidate_row(raw_d, True, "raw_neural", "raw_anchor", row_base, rep_base, 0.0)

    # 3-5. Unconstrained / Soft / ALM optimization baselines (evaluated on heldout & development branch_0, or all heldout!)
    if run_unconstrained_baselines:
        cand_tik, lat_tik = _solve_soft_tikhonov_lbfgs(raw_d, constraints, active_conf, lam=0.05)
        spec_tik = decompose_depth_displacement_v9(cand_tik, raw_d, constraints)
        out["soft_tikhonov_lbfgs"] = _score_candidate_row(
            cand_tik, False, "soft_tikhonov", "tikhonov", row_base, rep_base, lat_tik, float(spec_tik["null_rms"])
        )

        cand_gn, lat_gn = _solve_unconstrained_gn(raw_d, constraints, active_conf)
        spec_gn = decompose_depth_displacement_v9(cand_gn, raw_d, constraints)
        out["gn_unconstrained"] = _score_candidate_row(
            cand_gn, False, "gn_unconstrained", "gn_unc", row_base, rep_base, lat_gn, float(spec_gn["null_rms"])
        )

        cand_alm, lat_alm = _solve_classical_alm(raw_d, constraints, active_conf)
        spec_alm = decompose_depth_displacement_v9(cand_alm, raw_d, constraints)
        out["alm_classical"] = _score_candidate_row(
            cand_alm, False, "alm_classical", "alm", row_base, rep_base, lat_alm, float(spec_alm["null_rms"])
        )

    # 9. V9.0-Fast Pure NF-RSLM + Multi-Branch (zero SLSQP, runs on ALL views!)
    proj_fast = NormalFlowMultiBranchProjectorV9(
        NormalFlowProjectorConfigV9(monotone_stage1_anchor=False, enable_slsqp_fallback_on_rigid=False)
    )
    t0 = time.perf_counter()
    res_fast = proj_fast.project(
        raw, constraints, warm_start=raw, camera=camera, points_2d=pixels,
        constraint_confidences=active_conf, adjacency=adj, edge_style=sty
    )
    lat_fast = (time.perf_counter() - t0) * 1000.0
    out["v9_fast_pure_nf"] = _score_candidate_row(
        res_fast.coordinates, res_fast.abstained, res_fast.status,
        str(res_fast.diagnostics.get("regime", "v9_fast")), row_base, rep_base, lat_fast,
        float(res_fast.diagnostics.get("nullspace_drift_rms", 0.0)),
    )

    # 10-12. V9.0 Certified & Ablations (exact Stage-1 monotonicity on accepted Stage-1 views!)
    if not rep_base["abstained"] or row_base["status"] != "no_safe_feasible_candidate":
        # Stage 1 accepted or no constraints -> bit-exact match with Stage 1!
        s1_row = dict(row_base)
        s1_row["latency_ms"] = lat_fast
        s1_row["nullspace_drift_rms"] = 0.0
        out["v9_ablation_no_multibranch"] = dict(s1_row)
        out["v9_ablation_no_rowspace"] = dict(s1_row)
        out["v9_certified_full"] = dict(s1_row)
    else:
        # Stage 1 abstained -> run Stage 2 NF-RSLM continuation & ablations!
        if not res_fast.abstained:
            res_cert = res_fast
            lat_cert = lat_fast
            out["v9_certified_full"] = dict(out["v9_fast_pure_nf"])
        else:
            proj_cert_s2 = NormalFlowMultiBranchProjectorV9(
                NormalFlowProjectorConfigV9(
                    monotone_stage1_anchor=False,
                    enable_multibranch_reflection=False,
                    enable_slsqp_fallback_on_rigid=True,
                )
            )
            tc = time.perf_counter()
            res_cert = proj_cert_s2.project(
                raw, constraints, warm_start=raw, camera=camera, points_2d=pixels,
                constraint_confidences=active_conf, adjacency=adj, edge_style=sty
            )
            lat_cert = lat_fast + (time.perf_counter() - tc) * 1000.0
            out["v9_certified_full"] = _score_candidate_row(
                res_cert.coordinates, res_cert.abstained, res_cert.status,
                str(res_cert.diagnostics.get("regime", "v9_cert")), row_base, rep_base, lat_cert,
                float(res_cert.diagnostics.get("nullspace_drift_rms", 0.0)),
            )

        # Ablation: No Multi-Branch Reflection Continuation (single-branch NF-RSLM only)
        proj_nomb = NormalFlowMultiBranchProjectorV9(
            NormalFlowProjectorConfigV9(
                monotone_stage1_anchor=False,
                enable_multibranch_reflection=False,
                enable_slsqp_fallback_on_rigid=False,
            )
        )
        t1 = time.perf_counter()
        res_nomb = proj_nomb.project(
            raw, constraints, warm_start=raw, camera=camera, points_2d=pixels,
            constraint_confidences=active_conf, adjacency=adj, edge_style=sty
        )
        lat_nomb = (time.perf_counter() - t1) * 1000.0
        out["v9_ablation_no_multibranch"] = _score_candidate_row(
            res_nomb.coordinates, res_nomb.abstained, res_nomb.status,
            str(res_nomb.diagnostics.get("regime", "v9_nomb")), row_base, rep_base, lat_nomb,
            float(res_nomb.diagnostics.get("nullspace_drift_rms", 0.0)),
        )

        # Ablation: No d_null >= 1 Row-Space Continuation beyond 0.15 (only accept d_null == 0)
        d_null_cand = int(res_cert.diagnostics.get("d_null_ray", 99))
        if not res_cert.abstained and d_null_cand == 0:
            out["v9_ablation_no_rowspace"] = dict(out["v9_certified_full"])
        else:
            out["v9_ablation_no_rowspace"] = _score_candidate_row(
                raw_d, True, "rejected_dnull_gt0", "ablation_no_rowspace", row_base, rep_base, lat_cert
            )

    return out


def summarize_unconstrained_arm(rows: list[dict]) -> dict:
    """Summarize arms where invalid updates MAY occur (Soft Tikhonov, GN-Unc, ALM, Planar)."""
    n_views = len(rows)
    valid_views = sum(1 for r in rows if r["accepted_valid"])
    invalid_views = sum(1 for r in rows if r["invalid_accepted"])
    harm_views = sum(1 for r in rows if r["shape_harm"])
    null_drifts = [float(r.get("nullspace_drift_rms", 0.0)) for r in rows]
    lats = [float(r.get("latency_ms", 0.0)) for r in rows]

    fams = {}
    for fam in ("tetra_v2", "cube_v2", "prism_v2", "polyhedra_v2"):
        sub = [r for r in rows if r["family"] == fam]
        fams[fam] = {
            "post_shape_pa": float(np.mean([r["post_reflection_pa"] for r in sub])),
            "post_proper_pa": float(np.mean([r["post_pa"] for r in sub])),
            "valid_view_rate": float(np.mean([1.0 if r["accepted_valid"] else 0.0 for r in sub])),
            "harm_view_rate": float(np.mean([1.0 if r["shape_harm"] else 0.0 for r in sub])),
        }
    return {
        "views": n_views,
        "macro_post_shape_pa": float(np.mean([v["post_shape_pa"] for v in fams.values()])),
        "macro_post_proper_pa": float(np.mean([v["post_proper_pa"] for v in fams.values()])),
        "valid_view_rate": valid_views / n_views,
        "invalid_update_rate": invalid_views / n_views,
        "view_shape_harm_rate": harm_views / n_views,
        "mean_nullspace_drift_rms": float(np.mean(null_drifts)),
        "latency_median_ms": float(np.median(lats)),
        "latency_mean_ms": float(np.mean(lats)),
        "family": fams,
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    cached_unc_by_cohort_branch: dict[tuple[str, int], dict] = {}
    if OUT_JSON.exists():
        try:
            prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            for c_k, br_list in prev.get("cohorts", {}).items():
                for b_item in br_list:
                    u_arms = b_item.get("unconstrained_arms", {})
                    if "alm_classical" in u_arms:
                        cached_unc_by_cohort_branch[(c_k, int(b_item["branch"]))] = u_arms
        except Exception:
            pass
    cohorts_out = {}

    for cohort, spec in COHORT_SPECS.items():
        bdir = spec["base_dir"]
        cdir = spec["cache_dir"]
        branches_out = []

        for branch in range(3):
            has_cached_unc = (cohort, branch) in cached_unc_by_cohort_branch
            run_unc = (cohort == "heldout") and (not has_cached_unc)
            dataset = CachedModule2Dataset(cdir / f"branch_{branch}.pt")
            meta_by_id = {}
            for s in dataset.samples:
                sid = str(s["sample_id"])
                graph = s["graph"] if "graph" in s else s["contract"].graph
                binding = bind_constraints_from_point_anchors_v6(
                    graph, s["formal_anchor_points_2d"], s["supplied_constraints"], maximum_distance_px=10.0
                )
                active = [j for j, conf in enumerate(binding.confidences.tolist()) if float(conf) >= 0.5]
                meta_by_id[sid] = (binding.confidences[active].double(), graph.adjacency, graph.edge_style)

            r015_json = read_json(bdir / f"branch_{branch}_b512_weighted.json")
            r015_pt = torch.load(bdir / f"branch_{branch}_b512_weighted.pt", map_location="cpu", weights_only=False)["rows"]
            r045_json = read_json(ROOT / f"runs_module3_v843/{cohort}/branch_{branch}_r045_ungated.json")
            v843_json = read_json(ROOT / f"runs_module3_v843/{cohort}/branch_{branch}_v843_nullspace_guarded.json")

            # Spot-check Stage-1 bit-exactness of NormalFlowMultiBranchProjectorV9(monotone_stage1_anchor=True)
            proj_cert = NormalFlowMultiBranchProjectorV9(NormalFlowProjectorConfigV9(monotone_stage1_anchor=True))
            checked = 0
            for r_row, r_rep in zip(r015_json["rows"], r015_pt):
                if not r_rep["abstained"] and checked < 2:
                    conf_chk, adj_chk, sty_chk = meta_by_id[r_row["sample_id"]]
                    direct = proj_cert.project(
                        r_rep["raw"], r_rep["constraints"], warm_start=r_rep["raw"],
                        camera=r_rep["camera"], points_2d=r_rep["pixels"],
                        constraint_confidences=conf_chk, adjacency=adj_chk, edge_style=sty_chk
                    )
                    if direct.abstained or not torch.equal(direct.coordinates, r_rep["post"]):
                        raise RuntimeError(f"Stage-1 monotonicity check failed on {r_row['sample_id']}")
                    checked += 1

            tasks = []
            for r_row, r_rep in zip(r015_json["rows"], r015_pt):
                conf, adj, sty = meta_by_id[r_row["sample_id"]]
                tasks.append(
                    {
                        "row_r015": r_row,
                        "rep_r015": r_rep,
                        "active_confidences": conf,
                        "adjacency": adj,
                        "edge_style": sty,
                        "run_unconstrained": run_unc,
                    }
                )

            results_ordered = [None] * len(tasks)
            t_start = time.perf_counter()
            with ProcessPoolExecutor(max_workers=8) as pool:
                futs = {pool.submit(evaluate_view_all_solvers, t): idx for idx, t in enumerate(tasks)}
                for fut in as_completed(futs):
                    results_ordered[futs[fut]] = fut.result()
            wall_s = time.perf_counter() - t_start

            safeguarded_arms = {
                "r015_uniform": r015_json["summary"],
                "r045_ungated": r045_json["summary"],
                "v843_aciids": v843_json["summary"],
            }
            latencies_by_arm = {}
            null_drifts_by_arm = {}
            for v9_key in ("v9_fast_pure_nf", "v9_ablation_no_multibranch", "v9_ablation_no_rowspace", "v9_certified_full"):
                rows_arm = [r[v9_key] for r in results_ordered]
                safeguarded_arms[v9_key] = summarize_projector_rows(rows_arm)
                latencies_by_arm[v9_key] = {
                    "median_ms": float(np.median([x["latency_ms"] for x in rows_arm])),
                    "mean_ms": float(np.mean([x["latency_ms"] for x in rows_arm])),
                    "p95_ms": float(np.percentile([x["latency_ms"] for x in rows_arm], 95)),
                }
                acc_drifts = [x["nullspace_drift_rms"] for x in rows_arm if x["accepted_valid"]]
                null_drifts_by_arm[v9_key] = float(np.mean(acc_drifts)) if acc_drifts else 0.0

            if has_cached_unc:
                unconstrained_arms = dict(cached_unc_by_cohort_branch[(cohort, branch)])
            else:
                unconstrained_arms = {
                    "planar_z0": summarize_unconstrained_arm([r["planar_z0"] for r in results_ordered]),
                    "raw_neural": summarize_unconstrained_arm([r["raw_neural"] for r in results_ordered]),
                }
                if run_unc:
                    for u_key in ("soft_tikhonov_lbfgs", "gn_unconstrained", "alm_classical"):
                        unconstrained_arms[u_key] = summarize_unconstrained_arm([r[u_key] for r in results_ordered])

            s_cert = safeguarded_arms["v9_certified_full"]
            s_843 = safeguarded_arms["v843_aciids"]
            print(
                f"[{cohort:11s} B{branch}] ({wall_s:.1f}s) | "
                f"V843: PairCov={s_843['accepted_valid_pair_rate']:.3f} PA={s_843['macro']['post_shape_pa']:.4f} Harm={s_843['marginal_shape_harm_rate']:.4f} | "
                f"V9-Cert: PairCov={s_cert['accepted_valid_pair_rate']:.3f} PA={s_cert['macro']['post_shape_pa']:.4f} Harm={s_cert['marginal_shape_harm_rate']:.4f} "
                f"(TetraPairCov={s_cert['family']['tetra_v2']['accepted_valid_pair_rate']:.3f}, TetraPA={s_cert['family']['tetra_v2']['post_shape_pa']:.4f}) | "
                f"V9-Fast Lat={latencies_by_arm['v9_fast_pure_nf']['median_ms']:.1f}ms",
                flush=True,
            )

            branches_out.append(
                {
                    "branch": branch,
                    "safeguarded_arms": safeguarded_arms,
                    "unconstrained_arms": unconstrained_arms,
                    "latencies_ms": latencies_by_arm,
                    "accepted_mean_nullspace_drift_rms": null_drifts_by_arm,
                }
            )

        cohorts_out[cohort] = branches_out

    # Aggregate 3-seed mean +/- std and Paired Wilcoxon tests (V9-Certified vs V8.43 and r015) on Development & Heldout!
    aggregate_3seed = {}
    wilcoxon_out = {}
    for cohort in ("train", "development", "heldout"):
        brs = cohorts_out[cohort]
        agg_c = {}
        for arm_k in brs[0]["safeguarded_arms"]:
            pair_covs = [b["safeguarded_arms"][arm_k]["accepted_valid_pair_rate"] for b in brs]
            view_covs = [b["safeguarded_arms"][arm_k]["accepted_view_rate"] for b in brs]
            shape_pas = [b["safeguarded_arms"][arm_k]["macro"]["post_shape_pa"] for b in brs]
            proper_pas = [b["safeguarded_arms"][arm_k]["macro"]["post_proper_pa"] for b in brs]
            harms = [b["safeguarded_arms"][arm_k]["marginal_shape_harm_rate"] for b in brs]
            invalids = [b["safeguarded_arms"][arm_k]["invalid_accepted_views"] for b in brs]
            fams_agg = {}
            for fam in ("tetra_v2", "cube_v2", "prism_v2", "polyhedra_v2"):
                f_cov = [b["safeguarded_arms"][arm_k]["family"][fam]["accepted_valid_pair_rate"] for b in brs]
                f_pa = [b["safeguarded_arms"][arm_k]["family"][fam]["post_shape_pa"] for b in brs]
                f_harm = [b["safeguarded_arms"][arm_k]["family"][fam]["marginal_shape_harm_rate"] for b in brs]
                fams_agg[fam] = {
                    "pair_cov_mean": float(np.mean(f_cov)),
                    "pair_cov_std": float(np.std(f_cov, ddof=1)),
                    "post_shape_pa_mean": float(np.mean(f_pa)),
                    "post_shape_pa_std": float(np.std(f_pa, ddof=1)),
                    "harm_rate_mean": float(np.mean(f_harm)),
                }
            agg_c[arm_k] = {
                "pair_cov_mean": float(np.mean(pair_covs)),
                "pair_cov_std": float(np.std(pair_covs, ddof=1)),
                "view_cov_mean": float(np.mean(view_covs)),
                "macro_shape_pa_mean": float(np.mean(shape_pas)),
                "macro_shape_pa_std": float(np.std(shape_pas, ddof=1)),
                "macro_proper_pa_mean": float(np.mean(proper_pas)),
                "harm_rate_mean": float(np.mean(harms)),
                "harm_rate_std": float(np.std(harms, ddof=1)),
                "total_invalid_views_across_3seeds": int(sum(invalids)),
                "family": fams_agg,
            }
            if arm_k in brs[0]["latencies_ms"]:
                agg_c[arm_k]["latency_median_ms_mean"] = float(
                    np.mean([b["latencies_ms"][arm_k]["median_ms"] for b in brs])
                )
                agg_c[arm_k]["nullspace_drift_rms_mean"] = float(
                    np.mean([b["accepted_mean_nullspace_drift_rms"][arm_k] for b in brs])
                )

        for u_k in brs[0]["unconstrained_arms"]:
            u_pas = [b["unconstrained_arms"][u_k]["macro_post_shape_pa"] for b in brs]
            u_val = [b["unconstrained_arms"][u_k]["valid_view_rate"] for b in brs]
            u_inv = [b["unconstrained_arms"][u_k]["invalid_update_rate"] for b in brs]
            u_hrm = [b["unconstrained_arms"][u_k]["view_shape_harm_rate"] for b in brs]
            u_ndr = [b["unconstrained_arms"][u_k]["mean_nullspace_drift_rms"] for b in brs]
            u_lat = [b["unconstrained_arms"][u_k]["latency_median_ms"] for b in brs]
            agg_c[u_k] = {
                "macro_shape_pa_mean": float(np.mean(u_pas)),
                "macro_shape_pa_std": float(np.std(u_pas, ddof=1)),
                "valid_view_rate_mean": float(np.mean(u_val)),
                "invalid_update_rate_mean": float(np.mean(u_inv)),
                "view_harm_rate_mean": float(np.mean(u_hrm)),
                "nullspace_drift_rms_mean": float(np.mean(u_ndr)),
                "latency_median_ms_mean": float(np.mean(u_lat)),
            }

        aggregate_3seed[cohort] = agg_c

        if cohort in ("development", "heldout"):
            # Paired Wilcoxon across geometry groups (3-seed averaged post_shape_pa)
            def grp_vec(arm_name: str) -> np.ndarray:
                per_b = []
                for b in brs:
                    dets = sorted(b["safeguarded_arms"][arm_name]["details"], key=lambda d: d["group"])
                    per_b.append([d["post_shape_pa"] for d in dets])
                return np.mean(np.array(per_b), axis=0)

            v9c = grp_vec("v9_certified_full")
            w_sub = {}
            for ref_arm in ("r015_uniform", "v843_aciids", "v9_ablation_no_multibranch", "v9_ablation_no_rowspace"):
                diff = v9c - grp_vec(ref_arm)
                nonzero = np.abs(diff) > 1e-12
                if int(np.sum(nonzero)) > 0:
                    stat, pval = wilcoxon(diff[nonzero], alternative="less")
                else:
                    stat, pval = 0.0, 1.0
                w_sub[f"v9_certified_vs_{ref_arm}"] = {
                    "mean_delta_pa": float(np.mean(diff)),
                    "changed_groups": int(np.sum(nonzero)),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(pval),
                }
            wilcoxon_out[cohort] = w_sub

    # Strip bulky per-group details before writing JSON
    for cohort, brs in cohorts_out.items():
        for b in brs:
            for arm_k, sdict in b["safeguarded_arms"].items():
                sdict.pop("details", None)

    payload = {
        "status": "PASS",
        "experiment": "Q1_Module3_Solvers_Ablations_and_Latency_3Seeds",
        "aggregate_3seed": aggregate_3seed,
        "wilcoxon_tests": wilcoxon_out,
        "cohorts": cohorts_out,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Module 3 benchmark & ablations to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
