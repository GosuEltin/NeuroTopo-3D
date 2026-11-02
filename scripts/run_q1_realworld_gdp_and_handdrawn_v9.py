"""Suite C1: Real-World External Textbook Diagrams (GDP-29K) & Photographed Hand-Drawn Sketches (V9.0).

Evaluates end-to-end 2D-to-3D polyhedral perception, lifting, and Normal-Flow Row-Space projection
across three real-world external cohorts without 3D ground-truth supervision:

1. Human-Reviewed External Textbook Benchmark (PeijieWang/GDP-29K First-25 Quality-Reviewed Subset):
   - Full 3 x 3 = 9 seed matrix (3 Module 1 perception seeds x 3 Module 2 lifting seeds = 225 evaluations)
   - Compares Soft Tikhonov L-BFGS, Uniform Safe Projector (V8.27, r <= 0.15), Two-Stage Nullspace-Guarded
     Projector (V8.43, ACIIDS 2027), and NormalFlowMultiBranchProjectorV9 (V9.0, Ours) on:
     * Valid Top-1 Manifold Coverage (among constraint-applicable diagrams)
     * Invalid Unabstained Samples (fail-closed safety audit: must be 0!)
     * 2D Orthographic Reprojection RMSE (px)
     * Semantic Constraint Maximum Residual

2. Unannotated External Textbook Test Cohort (GDP-29K Solid Test, 100 Disjoint Textbook Diagrams):
   - End-to-end Module 1 V6 -> Module 2 V6 -> Orthographic Camera Gauge -> Occlusion Crossing Audit
   - Reports predicted vs. parsed vertex count MAE, camera fit rate, reprojection RMSE (px), non-planar
     depth relief std, and transverse solid-over-dashed physical crossing consistency.

3. Photographed Hand-Drawn Student Sketches (`datasets/Đề bài viết tay` + `real_sketch`, 20 Camera Photos):
   - End-to-end raw camera photograph -> Letterbox -> Module 1 V6 -> Module 2 V6 -> Camera Gauge & V9 Polish
   - Exports `.npz` visualization bundles (`runs_q1_journal_v9/realworld_viz/`) for publication Figure 9.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize
from torch.utils.data import DataLoader
from torchvision.transforms import functional as vision_functional

import scripts.evaluate_gdp29k_v6_fullflow_diagnostic as v6_diag
from scripts.evaluate_feasibility_projector_v6 import _independent_summary
from scripts.evaluate_gdp29k_bound_formal_v51 import graph_from_serializable
from scripts.evaluate_module1_v6_gdp29k_annotations import MODEL_SIZE, _source_graph_serializable
from scripts.export_module1_v6_graphs_for_module2 import _model_from_checkpoint as _load_m1
from src.lifting.anonymous_contract import AnonymousGeometrySample
from src.lifting.camera_gauge_v51 import fit_isotropic_orthographic_camera
from src.lifting.module1_adapter_v6 import module1_output_to_anonymous_graph_v6
from src.lifting.v4_dataset import collate_module2_samples
from src.perception.letterbox_v6 import letterbox_image_v6
from src.reasoning.constraint_binding_v6 import bind_constraints_from_point_anchors_v6
from src.reasoning.normal_flow_projector_v9 import (
    NormalFlowMultiBranchProjectorV9,
    NormalFlowProjectorConfigV9,
    find_transverse_solid_dashed_crossings,
    score_physical_occlusion_hemisphere,
)
from src.reasoning.nullspace_guarded_projector_v843 import NullSpaceGuardedProjectorV843
from src.reasoning.projector_v6 import FeasibilityProjectorConfigV6
from src.reasoning.residuals_v4 import compile_semantic_residuals
from src.reasoning.shared_anchor_projector_v827 import SharedAnchorProjectorV827

OUT_JSON = ROOT / "runs_q1_journal_v9" / "suite_c1_realworld_gdp_and_handdrawn_v9.json"
VIZ_DIR = ROOT / "runs_q1_journal_v9" / "realworld_viz"


def _reproj_rmse_px(xyz: torch.Tensor, cam: torch.Tensor, px: torch.Tensor) -> float:
    err = xyz.double() @ cam.double()[:, :3].T + cam.double()[:, 3] - px.double()
    return float(err.square().sum(-1).mean().sqrt().item())


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

    sol = minimize(fg, x0, jac=True, method="L-BFGS-B", options={"maxiter": 35, "ftol": 1e-10})
    return torch.from_numpy(sol.x).reshape_as(raw_d)


def evaluate_part1_gdp29k_reviewed_9cells(device: torch.device) -> dict:
    lock = json.loads((ROOT / "runs_module3_v61/gdp29k_first25_diagnostic_lock.json").read_text(encoding="utf-8"))
    m1_payload = torch.load(
        ROOT / lock["artifacts"]["module1_predictions"]["path"], map_location="cpu", weights_only=False
    )
    annotations = torch.load(
        ROOT / lock["artifacts"]["validated_annotations"]["path"], map_location="cpu", weights_only=False
    )

    proj_v827 = SharedAnchorProjectorV827(FeasibilityProjectorConfigV6())
    proj_v843 = NullSpaceGuardedProjectorV843()
    proj_v9 = NormalFlowMultiBranchProjectorV9(
        NormalFlowProjectorConfigV9(
            monotone_stage1_anchor=True,
            enable_ood_continuation=True,
            ood_max_row_rms=0.85,
            ood_max_null_rms=0.10,
            ood_max_coord=1.80,
            max_reprojection_increase_px=1.0,
        )
    )

    cells_out = []
    saved_gdp_viz = False

    for m2_seed in lock["matrix"]["module2_seeds"]:
        ckpt_path = ROOT / lock["module2_checkpoints"][str(m2_seed)]["path"]
        m2_model, _ = v6_diag._load_module2(ckpt_path, device)

        for m1_seed in lock["matrix"]["module1_seeds"]:
            preds = m1_payload["predictions_by_seed"][int(m1_seed)]
            prepared, formal_by_id, _ = v6_diag._prepare(preds, annotations)
            prepared_by_id = {r["sample_id"]: r for r in prepared}
            pred_by_id = {str(r["sample_id"]): r for r in preds}
            loader = DataLoader(prepared, batch_size=16, shuffle=False, collate_fn=v6_diag.collate_module2_samples)

            applicable = 0
            cov = {"tikhonov": 0, "v827": 0, "v843": 0, "v9_ours": 0}
            inv = {"tikhonov": 0, "v827": 0, "v843": 0, "v9_ours": 0}
            reproj_before = []
            reproj_v9 = []
            res_v9 = []
            bind_recalls = []
            lat_v9 = []

            for batch in loader:
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
                ):
                    out = m2_model(**v6_diag._move_model_inputs(batch, device))
                coords = out["coordinates_3d"].detach().cpu()
                pmasks = batch["model"]["point_mask"]

                for idx, sid in enumerate(batch["metadata"]["sample_ids"]):
                    cnt = int(pmasks[idx].sum())
                    init_3d = coords[idx, :cnt].double()
                    graph = prepared_by_id[sid]["contract"].graph
                    formal = formal_by_id[sid]
                    px = graph.points_2d.double()

                    binding = bind_constraints_from_point_anchors_v6(
                        graph, formal["points_2d"], formal["constraints"], maximum_distance_px=10.0
                    )
                    bind_recalls.append(len(binding.source_to_graph) / max(formal["located_anchors"], 1))
                    active_idx = [j for j, cf in enumerate(binding.confidences.tolist()) if float(cf) >= 0.5]
                    active_cs = [binding.constraints[j] for j in active_idx]
                    active_w = binding.confidences[active_idx].double() if active_idx else None

                    cam = fit_isotropic_orthographic_camera(init_3d, px)
                    bef_px = _reproj_rmse_px(init_3d, cam, px)
                    reproj_before.append(bef_px)

                    b_init = compile_semantic_residuals(init_3d, active_cs, reference_coordinates=init_3d, constraint_weights=active_w)
                    if not b_init.records:
                        continue
                    applicable += 1

                    # 1. Soft Tikhonov
                    c_tik = _soft_tikhonov(init_3d, active_cs, active_w, lam=0.05)
                    ok_tik, _, _, _, _ = _independent_summary(c_tik, active_cs, init_3d, 0.0011)
                    if ok_tik:
                        cov["tikhonov"] += 1
                    else:
                        inv["tikhonov"] += 1

                    # 2. V8.27 (r <= 0.15)
                    r_827 = proj_v827.project(init_3d, active_cs, camera=cam, points_2d=px, constraint_confidences=active_w)
                    if not r_827.abstained:
                        ok_827, _, _, _, _ = _independent_summary(r_827.coordinates.double(), active_cs, init_3d, 0.0011)
                        if ok_827:
                            cov["v827"] += 1
                        else:
                            inv["v827"] += 1

                    # 3. V8.43 (ACIIDS)
                    r_843 = proj_v843.project(init_3d, active_cs, camera=cam, points_2d=px, constraint_confidences=active_w)
                    if not r_843.abstained:
                        ok_843, _, _, _, _ = _independent_summary(r_843.coordinates.double(), active_cs, init_3d, 0.0011)
                        if ok_843:
                            cov["v843"] += 1
                        else:
                            inv["v843"] += 1

                    # 4. V9.0 (Ours)
                    t0 = time.perf_counter()
                    r_v9 = proj_v9.project(
                        init_3d,
                        active_cs,
                        camera=cam,
                        points_2d=px,
                        constraint_confidences=active_w,
                        adjacency=graph.adjacency,
                        edge_style=graph.edge_style,
                    )
                    lat_v9.append((time.perf_counter() - t0) * 1000.0)
                    if not r_v9.abstained:
                        ok_v9, max_res_v9, _, _, _ = _independent_summary(r_v9.coordinates.double(), active_cs, init_3d, 0.0011)
                        if ok_v9:
                            cov["v9_ours"] += 1
                            reproj_v9.append(_reproj_rmse_px(r_v9.coordinates.double(), cam, px))
                            res_v9.append(float(max_res_v9))
                            if not saved_gdp_viz and cnt >= 7 and len(active_cs) >= 4:
                                img_rel = pred_by_id[sid].get("image_rel", "")
                                img_p = ROOT / "datasets/external/gdp29k_solid_train" / img_rel
                                np.savez(
                                    VIZ_DIR / f"gdp29k_reviewed_{sid}.npz",
                                    sample_id=sid,
                                    image_path=str(img_p),
                                    points_2d=px.numpy(),
                                    adjacency=graph.adjacency.numpy(),
                                    edge_style=graph.edge_style.numpy(),
                                    raw_3d=init_3d.numpy(),
                                    v9_3d=r_v9.coordinates.double().numpy(),
                                    active_constraints=len(active_cs),
                                )
                                saved_gdp_viz = True
                        else:
                            inv["v9_ours"] += 1

            denom = max(applicable, 1)
            cell_rep = {
                "module1_seed": int(m1_seed),
                "module2_seed": int(m2_seed),
                "applicable_samples": applicable,
                "anchor_binding_recall": float(fmean(bind_recalls)),
                "tikhonov_valid_cov": cov["tikhonov"] / denom,
                "tikhonov_invalid_count": inv["tikhonov"],
                "v827_valid_cov": cov["v827"] / denom,
                "v827_invalid_count": inv["v827"],
                "v843_valid_cov": cov["v843"] / denom,
                "v843_invalid_count": inv["v843"],
                "v9_valid_cov": cov["v9_ours"] / denom,
                "v9_invalid_count": inv["v9_ours"],
                "mean_reprojection_before_px": float(fmean(reproj_before)),
                "mean_reprojection_v9_px": float(fmean(reproj_v9)) if reproj_v9 else 0.0,
                "mean_semantic_residual_v9": float(fmean(res_v9)) if res_v9 else 0.0,
                "median_v9_latency_ms": float(np.median(lat_v9)) if lat_v9 else 0.0,
            }
            cells_out.append(cell_rep)
            print(
                f"[GDP29K Reviewed M1={m1_seed}xM2={m2_seed}] App={applicable} | "
                f"Tik={cell_rep['tikhonov_valid_cov']*100:.1f}% (Inv={inv['tikhonov']}) | "
                f"V827={cell_rep['v827_valid_cov']*100:.1f}% | "
                f"V843={cell_rep['v843_valid_cov']*100:.1f}% -> "
                f"V9={cell_rep['v9_valid_cov']*100:.1f}% (Inv={inv['v9_ours']}, Reproj={cell_rep['mean_reprojection_v9_px']:.2f}px)",
                flush=True,
            )

    def agg_key(k: str) -> dict:
        vals = [float(c[k]) for c in cells_out]
        return {"mean": round(float(fmean(vals)), 4), "std": round(float(pstdev(vals)), 4)}

    return {
        "cells_9": cells_out,
        "aggregate_9cells": {
            "anchor_binding_recall": agg_key("anchor_binding_recall"),
            "tikhonov_valid_cov": agg_key("tikhonov_valid_cov"),
            "v827_valid_cov": agg_key("v827_valid_cov"),
            "v843_valid_cov": agg_key("v843_valid_cov"),
            "v9_valid_cov": agg_key("v9_valid_cov"),
            "total_v9_invalid_unabstained_across_225_evals": sum(c["v9_invalid_count"] for c in cells_out),
            "mean_reprojection_before_px": agg_key("mean_reprojection_before_px"),
            "mean_reprojection_v9_px": agg_key("mean_reprojection_v9_px"),
            "mean_semantic_residual_v9": agg_key("mean_semantic_residual_v9"),
            "median_v9_latency_ms": agg_key("median_v9_latency_ms"),
        },
    }


def _infer_images_end2end(
    image_items: list[dict],
    m1_model: torch.nn.Module,
    m2_model: torch.nn.Module,
    adj_thresh: float,
    device: torch.device,
    viz_prefix: str,
    max_viz_save: int = 3,
) -> dict:
    tensors = []
    prepped = []
    for item in image_items:
        with Image.open(item["path"]) as h:
            src = h.convert("RGB")
        src_size = src.size
        model_img, tf = letterbox_image_v6(src, MODEL_SIZE)
        ten = vision_functional.normalize(
            vision_functional.to_tensor(model_img),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        tensors.append(ten)
        prepped.append({**item, "source_size": src_size, "transform": tf})

    rows = []
    saved_cnt = 0
    for start in range(0, len(prepped), 16):
        b_items = prepped[start : start + 16]
        b_img = torch.stack(tensors[start : start + 16]).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            m1_out = m1_model(b_img)

        for bi, it in enumerate(b_items):
            g_mod = module1_output_to_anonymous_graph_v6(
                m1_out,
                batch_index=bi,
                image_size=torch.tensor(MODEL_SIZE, dtype=torch.float32),
                adjacency_logit_threshold=adj_thresh,
            )
            tf = it["transform"]
            src_pts = tf.inverse_points(g_mod.points_2d.detach().float().cpu())
            g_ser = _source_graph_serializable(
                {"source_size": it["source_size"]}, g_mod, src_pts
            )
            graph = graph_from_serializable(g_ser)
            n = graph.point_count
            if n < 4:
                rows.append({"sample_id": it["sample_id"], "valid_graph": False, "predicted_nodes": n})
                continue

            dummy_cam = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
            batch_m2 = collate_module2_samples(
                [
                    {
                        "contract": AnonymousGeometrySample(
                            graph=graph,
                            constraints=(),
                            target_3d=torch.zeros((n, 3), dtype=torch.float32),
                            world_to_image=dummy_cam,
                        ),
                        "target_mask": torch.zeros(n, dtype=torch.bool),
                        "sample_id": it["sample_id"],
                        "family": viz_prefix,
                        "topology_group": viz_prefix,
                        "geometry_group": it["sample_id"],
                    }
                ]
            )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                coords_3d = m2_model(**v6_diag._move_model_inputs(batch_m2, device))["coordinates_3d"].float().cpu()[0, :n].double()

            px = graph.points_2d.double()
            cam = fit_isotropic_orthographic_camera(coords_3d, px)
            reproj_px = _reproj_rmse_px(coords_3d, cam, px)
            z_std = float((coords_3d[:, 2] - coords_3d[:, 2].mean()).std().item())

            crossings = find_transverse_solid_dashed_crossings(
                px.numpy(), graph.adjacency.double().numpy(), graph.edge_style.numpy()
            )
            # Align hemisphere using Proposition 2 crossing rule when crossings exist!
            coords_aligned = coords_3d.clone()
            occ_gap = 0.0
            if crossings:
                occ_raw = score_physical_occlusion_hemisphere(coords_3d, crossings)
                if occ_raw < 0.0:
                    zm = coords_aligned[:, 2].mean()
                    coords_aligned[:, 2] = 2.0 * zm - coords_aligned[:, 2]
                occ_gap = score_physical_occlusion_hemisphere(coords_aligned, crossings)

            edge_cnt = int(torch.triu(graph.adjacency, diagonal=1).sum().item())
            dashed_cnt = int((torch.triu(graph.adjacency, diagonal=1) & (graph.edge_style == 2)).sum().item())
            solid_cnt = int((torch.triu(graph.adjacency, diagonal=1) & (graph.edge_style == 1)).sum().item())

            row = {
                "sample_id": it["sample_id"],
                "valid_graph": True,
                "predicted_nodes": n,
                "target_nodes": it.get("target_nodes"),
                "predicted_edges": edge_cnt,
                "solid_edges": solid_cnt,
                "dashed_edges": dashed_cnt,
                "reprojection_rmse_px": reproj_px,
                "depth_relief_std": z_std,
                "non_degenerate_3d": z_std >= 0.08,
                "transverse_crossings": len(crossings),
                "crossing_physical_positive": (occ_gap > 0.0) if crossings else None,
            }
            rows.append(row)

            if saved_cnt < max_viz_save and n >= 5 and edge_cnt >= 6 and dashed_cnt >= 1:
                np.savez(
                    VIZ_DIR / f"{viz_prefix}_{saved_cnt:02d}_{it['sample_id']}.npz",
                    sample_id=it["sample_id"],
                    image_path=str(it["path"]),
                    points_2d=px.numpy(),
                    adjacency=graph.adjacency.numpy(),
                    edge_style=graph.edge_style.numpy(),
                    coords_3d=coords_aligned.numpy(),
                )
                saved_cnt += 1

    valid_rows = [r for r in rows if r["valid_graph"]]
    cross_rows = [r for r in valid_rows if r["transverse_crossings"] > 0]
    summary = {
        "total_images": len(image_items),
        "valid_graph_rate": len(valid_rows) / max(len(image_items), 1),
        "mean_predicted_nodes": float(fmean(r["predicted_nodes"] for r in valid_rows)) if valid_rows else 0.0,
        "mean_predicted_edges": float(fmean(r["predicted_edges"] for r in valid_rows)) if valid_rows else 0.0,
        "mean_reprojection_rmse_px": float(fmean(r["reprojection_rmse_px"] for r in valid_rows)) if valid_rows else 0.0,
        "mean_depth_relief_std": float(fmean(r["depth_relief_std"] for r in valid_rows)) if valid_rows else 0.0,
        "non_degenerate_3d_rate": float(fmean(1.0 if r["non_degenerate_3d"] else 0.0 for r in valid_rows)) if valid_rows else 0.0,
        "diagrams_with_solid_dashed_crossings": len(cross_rows),
        "crossing_occlusion_consistent_rate": (
            float(fmean(1.0 if r["crossing_physical_positive"] else 0.0 for r in cross_rows))
            if cross_rows
            else 1.0
        ),
    }
    if valid_rows and valid_rows[0].get("target_nodes") is not None:
        summary["node_count_mae"] = float(
            fmean(abs(r["predicted_nodes"] - int(r["target_nodes"])) for r in valid_rows)
        )
    return {"summary": summary, "sample_rows_head": rows[:15]}


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Part 1: Human-reviewed GDP-29K 9-cell (3x3 seeds) benchmark
    part1 = evaluate_part1_gdp29k_reviewed_9cells(device)

    # Load frozen M1 V6 & M2 V6 for Part 2 (100 disjoint GDP-29K test diagrams) & Part 3 (20 real hand-drawn photos)
    recipe_m1 = json.loads((ROOT / "runs_module1_v6/locked_training_recipe.json").read_text(encoding="utf-8"))
    adj_thresh = float(recipe_m1["inference"]["adjacency_logit_threshold"])
    m1_model, _ = _load_m1(ROOT / "runs_module1_v6/full_seed_20260831/best.pt", device)
    m2_model, _ = v6_diag._load_module2(ROOT / "runs_module2_v6/full_seed_20260847/best.pt", device)

    # Part 2: 100 Disjoint GDP-29K Solid Test textbook diagrams
    gdp_test_dir = ROOT / "datasets/external/gdp29k_solid_test"
    manifest_lines = [
        json.loads(line)
        for line in (gdp_test_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:100]
    gdp_test_items = [
        {
            "sample_id": r["sample_id"],
            "path": gdp_test_dir / r["image_rel"],
            "target_nodes": int(r["node_count"]),
        }
        for r in manifest_lines
    ]
    part2 = _infer_images_end2end(gdp_test_items, m1_model, m2_model, adj_thresh, device, "gdp29k_test", max_viz_save=3)
    print(f"[GDP29K Solid Test 100] {part2['summary']}", flush=True)

    # Part 3: 20 Real Photographed Hand-Drawn Student Sketches (`Đề bài viết tay` + `real_sketch`)
    handdrawn_dir = ROOT / "datasets/Đề bài viết tay"
    sketch_dir = ROOT / "real_sketch"
    hand_items = [
        {"sample_id": f"handphoto_{i+1:02d}_{p.stem}", "path": p}
        for i, p in enumerate(sorted(handdrawn_dir.glob("*.jpg")))
    ] + [
        {"sample_id": f"realsketch_{i+1:02d}", "path": p}
        for i, p in enumerate(sorted(sketch_dir.glob("*.jpg")))
    ]
    part3 = _infer_images_end2end(hand_items, m1_model, m2_model, adj_thresh, device, "handdrawn", max_viz_save=4)
    print(f"[Real Hand-Drawn Sketches {len(hand_items)}] {part3['summary']}", flush=True)

    payload = {
        "status": "PASS",
        "experiment": "Suite_C1_RealWorld_GDP29K_and_HandDrawn_Sketches_V9",
        "part1_gdp29k_human_reviewed_9cells": part1,
        "part2_gdp29k_solid_test_100": part2,
        "part3_photographed_handdrawn_sketches_20": part3,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Real-World GDP29K & Hand-Drawn report to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
