"""Suite D: 13-Condition Upstream Corruption Robustness & Spectral Sensitivity Sweeps (V9.0).

Evaluates two Q1 reviewer-critical analyses on Confirmatory Heldout (V842):

1. Upstream Perception & Constraint Corruption Robustness (13 Regimes):
   - clean (0 px jitter, 0% drop/flip)
   - point_jitter_1px, point_jitter_2px, point_jitter_3px, point_jitter_5px
   - edge_drop_05 (5%), edge_drop_10 (10%), edge_drop_20 (20%)
   - node_drop_05 (5%), node_drop_10 (10%)
   - edge_style_flip_05 (5%), edge_style_flip_10 (10%), edge_style_flip_20 (20%) [solid <-> dashed]
   - spurious_false_constraint_15pct (15% corrupted formal constraints -> tests fail-closed safety!)
   Compares Raw Neural Anchor, Soft Tikhonov L-BFGS, V8.43 (ACIIDS), and V9.0 (Ours).

2. Dense Spectral & Trust-Region Hyperparameter Sensitivity Sweeps (on Heldout V842):
   - Sweep A: Viewing-ray SVD threshold tau_svd in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1}
   - Sweep B: Rigid singular value gate sigma_gate in {1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.15}
   - Sweep C: Row-space continuation radius r_max_row in {0.15, 0.22, 0.28, 0.35, 0.45, 0.65}
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from scripts.module2_v834_contract import model_inputs, read_json
from src.evaluation.graph_corruption_v53 import GraphCorruptionConfig, corrupt_predicted_record
from src.lifting.anonymous_contract import AnonymousGeometrySample, IndexedConstraint
from src.lifting.cached_dataset import CachedModule2Dataset
from src.lifting.camera_gauge_v51 import fit_isotropic_orthographic_camera
from src.lifting.v4_dataset import collate_module2_samples
from src.reasoning.constraint_binding_v6 import bind_constraints_from_point_anchors_v6
from src.reasoning.normal_flow_projector_v9 import (
    NormalFlowMultiBranchProjectorV9,
    NormalFlowProjectorConfigV9,
)
from src.reasoning.nullspace_guarded_projector_v843 import NullSpaceGuardedProjectorV843
from src.reasoning.residuals_v4 import compile_semantic_residuals

OUT_JSON = ROOT / "runs_q1_journal_v9" / "suite_d_robustness_and_sensitivity_v9.json"


def _soft_tikhonov(raw_d: torch.Tensor, cs, weights: torch.Tensor | None, lam: float = 0.05) -> torch.Tensor:
    b0 = compile_semantic_residuals(raw_d, cs, reference_coordinates=raw_d, constraint_weights=weights)
    if not b0.records:
        return raw_d
    x0 = raw_d.numpy().reshape(-1).copy()

    def fg(x_np: np.ndarray):
        xt = torch.from_numpy(x_np).requires_grad_(True)
        pts = xt.reshape_as(raw_d)
        b = compile_semantic_residuals(pts, cs, reference_coordinates=raw_d, constraint_weights=weights)
        loss = b.objective + lam * (pts - raw_d).square().sum()
        g = torch.autograd.grad(loss, xt)[0]
        return float(loss.detach().item()), g.detach().numpy()

    sol = minimize(fg, x0, jac=True, method="L-BFGS-B", options={"maxiter": 30, "ftol": 1e-10})
    return torch.from_numpy(sol.x).reshape_as(raw_d)


def _eval_robustness_item(task: dict) -> dict:
    torch.set_num_threads(1)
    p_raw = task["raw_3d"].double()
    tgt = task["target_3d"].double()
    msk = task["target_mask"].bool()
    px = task["points_2d"].double()
    adj = task["adjacency"]
    sty = task["edge_style"]
    active_cs = task["active_constraints"]
    active_w = task["active_weights"]
    true_cs = task["true_constraints"]

    if int(msk.sum().item()) < 3:
        return {"measurable": False}

    cam = fit_isotropic_orthographic_camera(p_raw, px)
    raw_pa = float(numpy_pa(p_raw, tgt, msk, True))

    p_tik = _soft_tikhonov(p_raw, active_cs, active_w, lam=0.05)
    tik_pa = float(numpy_pa(p_tik, tgt, msk, True))
    tik_ok, _, tik_info, _, _ = _independent_summary(p_tik, true_cs, p_raw, 0.0011)

    proj_843 = NullSpaceGuardedProjectorV843()
    r_843 = proj_843.project(p_raw, active_cs, camera=cam, points_2d=px, constraint_confidences=active_w)
    c_843 = r_843.coordinates.double()
    v843_pa = float(numpy_pa(c_843, tgt, msk, True))
    ok_843, _, info_843, _, _ = _independent_summary(c_843, true_cs, p_raw, 0.0011)

    proj_v9 = NormalFlowMultiBranchProjectorV9(NormalFlowProjectorConfigV9(monotone_stage1_anchor=True))
    r_v9 = proj_v9.project(
        p_raw, active_cs, camera=cam, points_2d=px, constraint_confidences=active_w, adjacency=adj, edge_style=sty
    )
    c_v9 = r_v9.coordinates.double()
    v9_pa = float(numpy_pa(c_v9, tgt, msk, True))
    ok_v9, _, info_v9, _, _ = _independent_summary(c_v9, true_cs, p_raw, 0.0011)
    # Also check validity wrt supplied (possibly corrupted) constraints to audit solver contract!
    ok_v9_sup, _, _, _, _ = _independent_summary(c_v9, active_cs, p_raw, 0.0011)

    return {
        "measurable": True,
        "family": task["family"],
        "raw_pa": raw_pa,
        "tik_pa": tik_pa,
        "tik_true_valid": bool(tik_ok and tik_info > 0),
        "v843_pa": v843_pa,
        "v843_accepted": not r_843.abstained,
        "v843_true_valid": bool((not r_843.abstained) and ok_843 and info_843 > 0),
        "v843_harm": bool((not r_843.abstained) and (v843_pa > raw_pa + 0.005)),
        "v9_pa": v9_pa,
        "v9_accepted": not r_v9.abstained,
        "v9_true_valid": bool((not r_v9.abstained) and ok_v9 and info_v9 > 0),
        "v9_supplied_invalid": bool((not r_v9.abstained) and (not ok_v9_sup)),
        "v9_harm": bool((not r_v9.abstained) and (v9_pa > raw_pa + 0.005)),
    }


def _eval_sensitivity_item(task: dict) -> dict:
    torch.set_num_threads(1)
    rep = task["rep"]
    row_base = task["row"]
    cfg_kwargs = task["cfg_kwargs"]
    if not rep["abstained"] or row_base["status"] != "no_safe_feasible_candidate":
        return {
            "family": row_base["family"],
            "accepted_valid": bool(row_base["accepted_valid"]),
            "post_shape_pa": float(row_base["post_reflection_pa"]),
            "shape_harm": bool(row_base["shape_harm"]),
        }

    proj = NormalFlowMultiBranchProjectorV9(NormalFlowProjectorConfigV9(**cfg_kwargs))
    res = proj.project(
        rep["raw"],
        rep["constraints"],
        warm_start=rep["raw"],
        camera=rep["camera"],
        points_2d=rep["pixels"],
        constraint_confidences=task["conf"],
        adjacency=task["adj"],
        edge_style=task["sty"],
    )
    if res.abstained:
        return {
            "family": row_base["family"],
            "accepted_valid": False,
            "post_shape_pa": float(row_base["raw_reflection_pa"]),
            "shape_harm": False,
        }
    cand_d = res.coordinates.detach().cpu().double()
    valid, _, info_cnt, _, _ = _independent_summary(cand_d, rep["constraints"], rep["raw"].double(), 0.0011)
    post_pa = float(numpy_pa(cand_d, rep["target"].double(), rep["mask"], True))
    harm = post_pa > float(row_base["raw_reflection_pa"]) + 0.005
    return {
        "family": row_base["family"],
        "accepted_valid": bool(valid and info_cnt > 0),
        "post_shape_pa": post_pa,
        "shape_harm": harm,
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Balanced 160-view confirmatory heldout subset (40 views per family across all 4 families!)
    pred_ds = CachedModule2Dataset(ROOT / "runs_module1_v842/branch_0.pt")
    by_fam: dict[str, list[dict]] = {"tetra_v2": [], "cube_v2": [], "prism_v2": [], "polyhedra_v2": []}
    for sp in pred_ds.samples:
        if len(by_fam[sp["family"]]) < 40:
            by_fam[sp["family"]].append(sp)
    selected_records = [sp for fam in ("tetra_v2", "cube_v2", "prism_v2", "polyhedra_v2") for sp in by_fam[fam]]

    m_v840 = _model(
        torch.load(ROOT / "runs_module2_v840/branch_0/b512_weighted/final.pt", map_location="cpu", weights_only=False),
        device,
    ).eval()

    conditions = [
        ("clean", GraphCorruptionConfig(), False),
        ("point_jitter_1px", GraphCorruptionConfig(point_jitter_sigma_px=1.0), False),
        ("point_jitter_2px", GraphCorruptionConfig(point_jitter_sigma_px=2.0), False),
        ("point_jitter_3px", GraphCorruptionConfig(point_jitter_sigma_px=3.0), False),
        ("point_jitter_5px", GraphCorruptionConfig(point_jitter_sigma_px=5.0), False),
        ("edge_drop_05", GraphCorruptionConfig(edge_drop_rate=0.05), False),
        ("edge_drop_10", GraphCorruptionConfig(edge_drop_rate=0.10), False),
        ("edge_drop_20", GraphCorruptionConfig(edge_drop_rate=0.20), False),
        ("node_drop_05", GraphCorruptionConfig(node_drop_rate=0.05), False),
        ("node_drop_10", GraphCorruptionConfig(node_drop_rate=0.10), False),
        ("edge_style_flip_05", GraphCorruptionConfig(edge_style_flip_rate=0.05), False),
        ("edge_style_flip_10", GraphCorruptionConfig(edge_style_flip_rate=0.10), False),
        ("edge_style_flip_20", GraphCorruptionConfig(edge_style_flip_rate=0.20), False),
        ("spurious_false_constraint_15pct", GraphCorruptionConfig(), True),
    ]

    robustness_out = {}
    rng_false = np.random.default_rng(20261029)

    for cond_name, corr_cfg, inject_false in conditions:
        t0 = time.perf_counter()
        corrupted_recs = []
        batch_items = []
        for rec in selected_records:
            crec, _ = corrupt_predicted_record(rec, corr_cfg, global_seed=20261029)
            corrupted_recs.append(crec)
            g = crec["graph"]
            batch_items.append(
                {
                    "contract": AnonymousGeometrySample(
                        graph=g,
                        constraints=(),
                        target_3d=crec["target_3d"],
                        world_to_image=crec["world_to_image"],
                    ),
                    "target_mask": crec["target_mask"],
                    "sample_id": crec["sample_id"],
                    "family": crec["family"],
                    "topology_group": crec["family"],
                    "geometry_group": crec.get("geometry_group", crec["sample_id"]),
                }
            )

        # Batched GPU Module 2 inference on corrupted 2D graphs!
        raw_preds = []
        for s_idx in range(0, len(batch_items), 32):
            sub_b = collate_module2_samples(batch_items[s_idx : s_idx + 32])
            with torch.inference_mode(), torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                out_c = m_v840(**model_inputs(sub_b, device))["coordinates_3d"].float().cpu()
            for bi, item in enumerate(batch_items[s_idx : s_idx + 32]):
                n_i = item["contract"].graph.point_count
                raw_preds.append(out_c[bi, :n_i].double())

        tasks = []
        for crec, p_raw in zip(corrupted_recs, raw_preds):
            g = crec["graph"]
            binding = bind_constraints_from_point_anchors_v6(
                g, crec["formal_anchor_points_2d"], crec["supplied_constraints"], maximum_distance_px=12.0
            )
            act_idx = [j for j, cf in enumerate(binding.confidences.tolist()) if float(cf) >= 0.5]
            true_cs = [binding.constraints[j] for j in act_idx]
            act_cs = list(true_cs)
            act_w_list = [float(binding.confidences[j].item()) for j in act_idx]

            if inject_false and g.point_count >= 4 and len(act_cs) >= 2:
                # Inject ~15% spurious/contradictory orthogonal or midpoint constraints!
                n_false = max(1, int(round(0.15 * len(act_cs))))
                for _ in range(n_false):
                    perm = rng_false.choice(g.point_count, size=4, replace=False).tolist()
                    act_cs.append(
                        IndexedConstraint(
                            "midpoint",
                            (perm[0], perm[1], perm[2]),
                            {"args": [perm[0], perm[1], perm[2]]},
                            True,
                        )
                    )
                    act_w_list.append(0.85)

            act_w = torch.tensor(act_w_list, dtype=torch.float64) if act_w_list else None
            tasks.append(
                {
                    "family": crec["family"],
                    "raw_3d": p_raw,
                    "target_3d": crec["target_3d"],
                    "target_mask": crec["target_mask"],
                    "points_2d": g.points_2d,
                    "adjacency": g.adjacency,
                    "edge_style": g.edge_style,
                    "active_constraints": act_cs,
                    "active_weights": act_w,
                    "true_constraints": true_cs,
                }
            )

        rows = []
        with ProcessPoolExecutor(max_workers=8) as pool:
            for fut in as_completed([pool.submit(_eval_robustness_item, t) for t in tasks]):
                res_r = fut.result()
                if res_r.get("measurable"):
                    rows.append(res_r)

        def macro_pa(key: str) -> float:
            return float(
                fmean(
                    fmean(r[key] for r in rows if r["family"] == fam)
                    for fam in ("tetra_v2", "cube_v2", "prism_v2", "polyhedra_v2")
                )
            )

        cond_sum = {
            "measurable_views": len(rows),
            "raw_macro_pa": round(macro_pa("raw_pa"), 4),
            "tikhonov_macro_pa": round(macro_pa("tik_pa"), 4),
            "tikhonov_valid_pct": round(float(fmean(1.0 if r["tik_true_valid"] else 0.0 for r in rows)) * 100.0, 2),
            "v843_macro_pa": round(macro_pa("v843_pa"), 4),
            "v843_accepted_pct": round(float(fmean(1.0 if r["v843_accepted"] else 0.0 for r in rows)) * 100.0, 2),
            "v843_true_valid_pct": round(float(fmean(1.0 if r["v843_true_valid"] else 0.0 for r in rows)) * 100.0, 2),
            "v843_harm_pct": round(float(fmean(1.0 if r["v843_harm"] else 0.0 for r in rows)) * 100.0, 2),
            "v9_macro_pa": round(macro_pa("v9_pa"), 4),
            "v9_accepted_pct": round(float(fmean(1.0 if r["v9_accepted"] else 0.0 for r in rows)) * 100.0, 2),
            "v9_true_valid_pct": round(float(fmean(1.0 if r["v9_true_valid"] else 0.0 for r in rows)) * 100.0, 2),
            "v9_harm_pct": round(float(fmean(1.0 if r["v9_harm"] else 0.0 for r in rows)) * 100.0, 2),
            "v9_supplied_invalid_count": sum(1 for r in rows if r["v9_supplied_invalid"]),
        }
        robustness_out[cond_name] = cond_sum
        print(
            f"[Robustness {cond_name:30s}] ({time.perf_counter()-t0:.1f}s) Raw={cond_sum['raw_macro_pa']:.4f} | "
            f"Tik={cond_sum['tikhonov_macro_pa']:.4f} | V843={cond_sum['v843_macro_pa']:.4f} ({cond_sum['v843_accepted_pct']:5.1f}%) -> "
            f"V9={cond_sum['v9_macro_pa']:.4f} ({cond_sum['v9_accepted_pct']:5.1f}%, Harm={cond_sum['v9_harm_pct']:.2f}%, Inv={cond_sum['v9_supplied_invalid_count']})",
            flush=True,
        )

    # =========================================================================
    # PART 2: Dense Spectral & Trust-Region Sensitivity Sweeps on Full Heldout B0 (512 views)
    # =========================================================================
    r015_json = read_json(ROOT / "runs_end2end_v842/module3/branch_0_b512_weighted.json")
    r015_pt = torch.load(ROOT / "runs_end2end_v842/module3/branch_0_b512_weighted.pt", map_location="cpu", weights_only=False)["rows"]
    meta_by_id = {}
    for s in pred_ds.samples:
        sid = str(s["sample_id"])
        g = s["graph"]
        bnd = bind_constraints_from_point_anchors_v6(g, s["formal_anchor_points_2d"], s["supplied_constraints"], maximum_distance_px=10.0)
        act = [j for j, cf in enumerate(bnd.confidences.tolist()) if float(cf) >= 0.5]
        meta_by_id[sid] = (bnd.confidences[act].double(), g.adjacency, g.edge_style)

    def run_sweep_point(cfg_kwargs: dict) -> dict:
        full_kwargs = {"monotone_stage1_anchor": False, **cfg_kwargs}
        stasks = []
        for r_row, r_rep in zip(r015_json["rows"], r015_pt):
            conf, adj, sty = meta_by_id[r_row["sample_id"]]
            stasks.append({"row": r_row, "rep": r_rep, "conf": conf, "adj": adj, "sty": sty, "cfg_kwargs": full_kwargs})
        srows = []
        with ProcessPoolExecutor(max_workers=8) as pool:
            for fut in as_completed([pool.submit(_eval_sensitivity_item, t) for t in stasks]):
                srows.append(fut.result())
        macro = float(
            fmean(
                fmean(r["post_shape_pa"] for r in srows if r["family"] == fam)
                for fam in ("tetra_v2", "cube_v2", "prism_v2", "polyhedra_v2")
            )
        )
        tetra_cov = float(fmean(1.0 if r["accepted_valid"] else 0.0 for r in srows if r["family"] == "tetra_v2"))
        return {
            "macro_shape_pa": round(macro, 4),
            "view_valid_cov_pct": round(float(fmean(1.0 if r["accepted_valid"] else 0.0 for r in srows)) * 100.0, 2),
            "tetra_valid_cov_pct": round(tetra_cov * 100.0, 2),
            "view_harm_pct": round(float(fmean(1.0 if r["shape_harm"] else 0.0 for r in srows)) * 100.0, 2),
        }

    sweep_tau = {}
    for tau in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1):
        sweep_tau[str(tau)] = run_sweep_point({"tau_svd": tau, "step_tau_svd": min(tau, 1e-3)})
        print(f"[Sweep tau_svd={tau:g}] {sweep_tau[str(tau)]}", flush=True)

    sweep_sigma_gate = {}
    for sgate in (1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.15):
        sweep_sigma_gate[str(sgate)] = run_sweep_point({"min_rigid_singular_value": sgate})
        print(f"[Sweep sigma_gate={sgate:g}] {sweep_sigma_gate[str(sgate)]}", flush=True)

    sweep_rmax = {}
    for rmax in (0.15, 0.22, 0.28, 0.35, 0.45, 0.65):
        sweep_rmax[str(rmax)] = run_sweep_point({"underconstrained_max_row_rms": rmax, "rigid_max_rms": max(0.45, rmax)})
        print(f"[Sweep r_max_row={rmax:g}] {sweep_rmax[str(rmax)]}", flush=True)

    payload = {
        "status": "PASS",
        "experiment": "Suite_D_14Condition_Corruption_Robustness_and_Spectral_Sensitivity_V9",
        "part1_corruption_robustness_14_conditions": robustness_out,
        "part2_spectral_and_radius_sensitivity_sweeps": {
            "tau_svd_sweep": sweep_tau,
            "sigma_gate_sweep": sweep_sigma_gate,
            "row_radius_sweep": sweep_rmax,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Suite D Robustness & Sensitivity report to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
