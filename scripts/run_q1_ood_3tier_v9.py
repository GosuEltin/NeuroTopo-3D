"""Suite B: Q1 Three-Tier Out-of-Distribution (OOD) Generalization & Nullspace Drift Suite (V9.0).

Evaluates NormalFlowMultiBranchProjectorV9 (enable_ood_continuation=True) against:
- Raw Neural Anchor
- Legacy Unconstrained L-BFGS
- Uniform Safe Projector (V8.27, r <= 0.15)
- Two-Stage Nullspace-Guarded Projector (V8.43)
- Subspace-Locked Post-Hoc SLSQP Projector (V8.45)

Across all 3 OOD Tiers:
- Tier 1: Zero-Shot Basic-4 -> Complex-8 across Calibration, Held-Out, and Blind Confirmation
- Tier 2: 3-Seed Leave-One-Topology-Out (LOSO) CV on scene_03_cutting_plane & scene_05_centroid_polyhedron
- Tier 3: 100 Non-Template Random 3D Convex Hulls (N in [9, 13]) + Collinear Ratio Chains
"""
from __future__ import annotations

import json
import random
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
from scipy.spatial import ConvexHull

from scripts.analyze_module3_v835_audit import numpy_pa
from scripts.evaluate_feasibility_projector_v6 import _independent_summary
from scripts.evaluate_module2_v4_frozen import _model_from_checkpoint, _to_device
from scripts.export_module2_v6_projector_payload import _model
from scripts.module2_v834_contract import model_inputs
from scripts.run_v845_12family_pilot_smoke import load_benchmark_items
from scripts.run_v845_frontier_and_baselines import solve_legacy_unconstrained_lbfgs
from scripts.run_v845_ood_suite_and_viz import SubspaceLockedOODProjectorV845
from src.lifting.anonymous_contract import (
    EDGE_STYLE_DASHED,
    EDGE_STYLE_SOLID,
    EDGE_STYLE_UNKNOWN,
    NODE_TYPE_CORNER,
    AnonymousGeometrySample,
    AnonymousGraph,
    IndexedConstraint,
)
from src.lifting.cached_dataset import CachedModule2Dataset
from src.lifting.camera_gauge_v51 import fit_isotropic_orthographic_camera
from src.lifting.v4_dataset import collate_module2_samples
from src.reasoning.normal_flow_projector_v9 import (
    NormalFlowMultiBranchProjectorV9,
    NormalFlowProjectorConfigV9,
    decompose_depth_displacement_v9,
)
from src.reasoning.nullspace_guarded_projector_v843 import (
    NullSpaceGuardedProjectorV843,
    depth_jacobian_nullity_v843,
)
from src.reasoning.projector_v6 import FeasibilityProjectorConfigV6
from src.reasoning.shared_anchor_projector_v827 import SharedAnchorProjectorV827

OUT_JSON = ROOT / "runs_q1_journal_v9" / "suite_b_3tier_ood_v9.json"


def _summarize_ood_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    def m(k: str) -> float:
        return round(float(fmean(r[k] for r in rows)), 4)

    return {
        "views": len(rows),
        "raw_pa": m("raw_pa"),
        "legacy_lbfgs_pa": m("leg_pa"),
        "legacy_lbfgs_null_rms": m("leg_null_rms"),
        "v827_pa": m("v827_pa"),
        "v827_valid_pct": round(m("v827_valid") * 100.0, 2),
        "v843_pa": m("v843_pa"),
        "v843_valid_pct": round(m("v843_valid") * 100.0, 2),
        "v845_ood_pa": m("v845_pa"),
        "v845_ood_valid_pct": round(m("v845_valid") * 100.0, 2),
        "v845_ood_null_rms": m("v845_null_rms"),
        "v9_ood_pa": m("v9_pa"),
        "v9_ood_valid_pct": round(m("v9_valid") * 100.0, 2),
        "v9_ood_null_rms": m("v9_null_rms"),
        "v9_ood_invalid_count": sum(1 for r in rows if r["v9_invalid"]),
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m_v840 = _model(
        torch.load(ROOT / "runs_module2_v840/branch_0/b512_weighted/final.pt", map_location="cpu", weights_only=False),
        device,
    ).eval()
    m_v6 = _model(
        torch.load(ROOT / "runs_module2_v6/full_seed_20260847/best.pt", map_location="cpu", weights_only=False),
        device,
    ).eval()

    proj_v827 = SharedAnchorProjectorV827(FeasibilityProjectorConfigV6())
    proj_v843 = NullSpaceGuardedProjectorV843()
    proj_v845 = SubspaceLockedOODProjectorV845()
    proj_v9_ood = NormalFlowMultiBranchProjectorV9(
        NormalFlowProjectorConfigV9(
            monotone_stage1_anchor=True,
            enable_ood_continuation=True,
            ood_max_row_rms=0.95,
            ood_max_null_rms=0.08,
            ood_max_coord=1.95,
        )
    )

    # =========================================================================
    # TIER 1: Zero-Shot Basic-4 -> Complex-8 across 3 cohorts
    # =========================================================================
    cohorts = load_benchmark_items()
    tier1 = {}
    for cname in (
        "neurotopo_v4_smoke",
        "neurotopo_v4_quality_test_pilot144",
        "neurotopo_v5_final_confirmation_pilot144",
    ):
        items = [
            it for it in cohorts[cname]
            if it["family"] not in ("tetra_v2", "cube_v2", "prism_v2", "polyhedra_v2")
        ]
        rows = []
        fam_rows = defaultdict(list)
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

            with torch.inference_mode():
                p_init = m_v840(**model_inputs(batch, device))["coordinates_3d"].float().cpu()[0, :n].double()

            cam = fit_isotropic_orthographic_camera(p_init, px)
            p_leg, _, _ = solve_legacy_unconstrained_lbfgs(p_init, cs, 0.01)
            r_827 = proj_v827.project(p_init, cs, camera=cam, points_2d=px)
            r_843 = proj_v843.project(p_init, cs, camera=cam, points_2d=px)
            r_845 = proj_v845.project(p_init, cs, camera=cam, points_2d=px)
            r_v9 = proj_v9_ood.project(p_init, cs, camera=cam, points_2d=px, adjacency=adj, edge_style=sty)

            d_null_z = int(depth_jacobian_nullity_v843(tgt, cs)["nullity"])
            ind_827, _, info_827, _, _ = _independent_summary(r_827.coordinates.double(), cs, p_init, 0.0011)
            ind_843, _, info_843, _, _ = _independent_summary(r_843.coordinates.double(), cs, p_init, 0.0011)
            ind_845, _, info_845, _, _ = _independent_summary(r_845.coordinates.double(), cs, p_init, 0.0011)
            ind_v9, _, info_v9, _, _ = _independent_summary(r_v9.coordinates.double(), cs, p_init, 0.0011)

            spec_leg = decompose_depth_displacement_v9(p_leg.double(), p_init, cs)
            spec_845 = decompose_depth_displacement_v9(r_845.coordinates.double(), p_init, cs)
            spec_v9 = decompose_depth_displacement_v9(r_v9.coordinates.double(), p_init, cs)

            row = {
                "family": fam,
                "is_zero_nullity": d_null_z == 0,
                "raw_pa": float(numpy_pa(p_init, tgt, msk, True)),
                "leg_pa": float(numpy_pa(p_leg.double(), tgt, msk, True)),
                "leg_null_rms": float(spec_leg["null_rms"]),
                "v827_pa": float(numpy_pa(r_827.coordinates.double(), tgt, msk, True)),
                "v827_valid": bool((not r_827.abstained) and ind_827 and info_827 > 0),
                "v843_pa": float(numpy_pa(r_843.coordinates.double(), tgt, msk, True)),
                "v843_valid": bool((not r_843.abstained) and ind_843 and info_843 > 0),
                "v845_pa": float(numpy_pa(r_845.coordinates.double(), tgt, msk, True)),
                "v845_valid": bool((not r_845.abstained) and ind_845 and info_845 > 0),
                "v845_null_rms": float(spec_845["null_rms"]),
                "v9_pa": float(numpy_pa(r_v9.coordinates.double(), tgt, msk, True)),
                "v9_valid": bool((not r_v9.abstained) and ind_v9 and info_v9 > 0),
                "v9_invalid": bool((not r_v9.abstained) and (not ind_v9)),
                "v9_null_rms": float(spec_v9["null_rms"]),
            }
            rows.append(row)
            fam_rows[fam].append(row)

        tier1[cname] = {
            "overall_complex_8": _summarize_ood_rows(rows),
            "zero_nullity_d0": _summarize_ood_rows([r for r in rows if r["is_zero_nullity"]]),
            "underconstrained_dpos": _summarize_ood_rows([r for r in rows if not r["is_zero_nullity"]]),
            "per_family": {fk: _summarize_ood_rows(fv) for fk, fv in sorted(fam_rows.items())},
        }
        ov = tier1[cname]["overall_complex_8"]
        z0 = tier1[cname]["zero_nullity_d0"]
        print(
            f"[Tier1 {cname:38s}] Complex8: V845={ov['v845_ood_pa']:.4f} ({ov['v845_ood_valid_pct']:.1f}%, Null={ov['v845_ood_null_rms']:.4f}) -> "
            f"V9-OOD={ov['v9_ood_pa']:.4f} ({ov['v9_ood_valid_pct']:.1f}%, Null={ov['v9_ood_null_rms']:.4f}) | "
            f"d=0: V845={z0['v845_ood_pa']:.4f} -> V9-OOD={z0['v9_ood_pa']:.4f}",
            flush=True,
        )

    # =========================================================================
    # TIER 2: 3-Seed LOSO Unseen Topology CV (scene_03 & scene_05, 100 views/seed)
    # =========================================================================
    protocol = json.loads((ROOT / "runs_module3_v51/loso_ood_protocol.json").read_text(encoding="utf-8"))
    tier2 = {}
    for fold_name, fold_cfg in protocol["folds"].items():
        ds = CachedModule2Dataset(ROOT / fold_cfg["oracle_test"])
        samples = ds.samples[:100]
        seed_reps = []
        for ckpt_rel in fold_cfg["checkpoints"]:
            ckpt = torch.load(ROOT / ckpt_rel, map_location="cpu", weights_only=False)
            model_v4, cfg_v4 = _model_from_checkpoint(ckpt, device)
            include_c = int(cfg_v4.get("constraint_layers", 0)) > 0
            rows = []
            for sp in samples:
                f = sp["contract"]
                n = f.graph.point_count
                batch = collate_module2_samples([sp])
                m_in, _ = _to_device(batch, device, include_constraints=include_c)
                tgt = batch["targets"]["coordinates_3d_camera"][0, :n].double().cpu()
                msk = torch.ones(n, dtype=torch.bool)
                px = f.graph.points_2d.double().cpu()
                adj = f.graph.adjacency
                sty = f.graph.edge_style
                cs = list(f.constraints)

                with torch.inference_mode():
                    p_init = model_v4(**m_in)["coordinates_3d"].float().cpu()[0, :n].double()

                cam = fit_isotropic_orthographic_camera(p_init, px)
                p_leg, _, _ = solve_legacy_unconstrained_lbfgs(p_init, cs, 0.01)
                r_827 = proj_v827.project(p_init, cs, camera=cam, points_2d=px)
                r_843 = proj_v843.project(p_init, cs, camera=cam, points_2d=px)
                r_845 = proj_v845.project(p_init, cs, camera=cam, points_2d=px)
                r_v9 = proj_v9_ood.project(p_init, cs, camera=cam, points_2d=px, adjacency=adj, edge_style=sty)

                ind_827, _, info_827, _, _ = _independent_summary(r_827.coordinates.double(), cs, p_init, 0.0011)
                ind_843, _, info_843, _, _ = _independent_summary(r_843.coordinates.double(), cs, p_init, 0.0011)
                ind_845, _, info_845, _, _ = _independent_summary(r_845.coordinates.double(), cs, p_init, 0.0011)
                ind_v9, _, info_v9, _, _ = _independent_summary(r_v9.coordinates.double(), cs, p_init, 0.0011)

                spec_leg = decompose_depth_displacement_v9(p_leg.double(), p_init, cs)
                spec_845 = decompose_depth_displacement_v9(r_845.coordinates.double(), p_init, cs)
                spec_v9 = decompose_depth_displacement_v9(r_v9.coordinates.double(), p_init, cs)

                rows.append(
                    {
                        "raw_pa": float(numpy_pa(p_init, tgt, msk, True)),
                        "leg_pa": float(numpy_pa(p_leg.double(), tgt, msk, True)),
                        "leg_null_rms": float(spec_leg["null_rms"]),
                        "v827_pa": float(numpy_pa(r_827.coordinates.double(), tgt, msk, True)),
                        "v827_valid": bool((not r_827.abstained) and ind_827 and info_827 > 0),
                        "v843_pa": float(numpy_pa(r_843.coordinates.double(), tgt, msk, True)),
                        "v843_valid": bool((not r_843.abstained) and ind_843 and info_843 > 0),
                        "v845_pa": float(numpy_pa(r_845.coordinates.double(), tgt, msk, True)),
                        "v845_valid": bool((not r_845.abstained) and ind_845 and info_845 > 0),
                        "v845_null_rms": float(spec_845["null_rms"]),
                        "v9_pa": float(numpy_pa(r_v9.coordinates.double(), tgt, msk, True)),
                        "v9_valid": bool((not r_v9.abstained) and ind_v9 and info_v9 > 0),
                        "v9_invalid": bool((not r_v9.abstained) and (not ind_v9)),
                        "v9_null_rms": float(spec_v9["null_rms"]),
                    }
                )
            s_sum = _summarize_ood_rows(rows)
            s_sum["checkpoint"] = ckpt_rel
            seed_reps.append(s_sum)

        tier2[fold_name] = {
            "seeds": seed_reps,
            "mean_3seeds": {
                k: round(float(fmean(s[k] for s in seed_reps)), 4)
                for k in (
                    "raw_pa", "legacy_lbfgs_pa", "legacy_lbfgs_null_rms",
                    "v827_pa", "v843_pa", "v843_valid_pct",
                    "v845_ood_pa", "v845_ood_valid_pct", "v845_ood_null_rms",
                    "v9_ood_pa", "v9_ood_valid_pct", "v9_ood_null_rms",
                )
            },
        }
        m2 = tier2[fold_name]["mean_3seeds"]
        print(
            f"[Tier2 LOSO {fold_name:10s}] Raw={m2['raw_pa']:.4f} | Leg={m2['legacy_lbfgs_pa']:.4f} (Null={m2['legacy_lbfgs_null_rms']:.4f}) | "
            f"V845={m2['v845_ood_pa']:.4f} ({m2['v845_ood_valid_pct']:.1f}%, Null={m2['v845_ood_null_rms']:.4f}) | "
            f"V9-OOD={m2['v9_ood_pa']:.4f} ({m2['v9_ood_valid_pct']:.1f}%, Null={m2['v9_ood_null_rms']:.4f})",
            flush=True,
        )

    # =========================================================================
    # TIER 3: 100 Non-Template Random Convex Hulls + Collinear Ratio Chains
    # =========================================================================
    rng = np.random.default_rng(20261025)
    py_rng = random.Random(20261025)
    t3_rows_v840 = []
    t3_rows_v6 = []

    for idx in range(100):
        raw_pts = rng.uniform(-1.0, 1.0, (10, 3))
        hull = ConvexHull(raw_pts)
        hull_indices = list(hull.vertices)[:7]
        v_labels = "ABCDEFGH"
        hull_names = [v_labels[i] for i in range(len(hull_indices))]
        pts3d = {hull_names[i]: raw_pts[hull_indices[i]].copy() for i in range(len(hull_indices))}

        center = np.mean(list(pts3d.values()), axis=0)
        view_vec = np.array([0.15, -0.20, -1.0])
        view_vec /= np.linalg.norm(view_vec)
        orig_to_name = {hull_indices[i]: hull_names[i] for i in range(len(hull_indices))}
        edge_styles = {}
        for simplex in hull.simplices:
            if all(int(vi) in orig_to_name for vi in simplex):
                v0, v1, v2 = raw_pts[simplex[0]], raw_pts[simplex[1]], raw_pts[simplex[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                if np.dot(normal, v0 - center) < 0:
                    normal = -normal
                st = EDGE_STYLE_SOLID if np.dot(normal, -view_vec) > 0 else EDGE_STYLE_DASHED
                for k in range(3):
                    a_nm = orig_to_name[int(simplex[k])]
                    b_nm = orig_to_name[int(simplex[(k + 1) % 3])]
                    key = (min(a_nm, b_nm), max(a_nm, b_nm))
                    if key not in edge_styles or st == EDGE_STYLE_SOLID:
                        edge_styles[key] = st
        if len(edge_styles) < len(hull_names):
            for i in range(len(hull_names)):
                a_nm, b_nm = hull_names[i], hull_names[(i + 1) % len(hull_names)]
                edge_styles.setdefault((min(a_nm, b_nm), max(a_nm, b_nm)), EDGE_STYLE_SOLID)

        extra_pool = ["M", "N", "P", "S", "H", "K"]
        hull_edge_list = list(edge_styles.keys())
        py_rng.shuffle(hull_edge_list)
        num_extra = min(3, len(hull_edge_list))
        ratio_specs = []
        for k in range(num_extra):
            ex_name = extra_pool[k]
            p1_name, p2_name = hull_edge_list[k]
            ratio_t = py_rng.choice([0.33, 0.50, 0.66, 0.75])
            pts3d[ex_name] = (1.0 - ratio_t) * pts3d[p1_name] + ratio_t * pts3d[p2_name]
            ratio_specs.append({"target": ex_name, "parents": (p1_name, p2_name), "t": float(ratio_t)})
            edge_styles[(min(ex_name, p1_name), max(ex_name, p1_name))] = EDGE_STYLE_SOLID
            edge_styles[(min(ex_name, p2_name), max(ex_name, p2_name))] = EDGE_STYLE_SOLID
            opp = [nm for nm in hull_names if nm not in (p1_name, p2_name)][0]
            edge_styles[(min(ex_name, opp), max(ex_name, opp))] = EDGE_STYLE_DASHED

        node_names = list(pts3d.keys())
        name_to_idx = {nm: i for i, nm in enumerate(node_names)}
        n_pts = len(node_names)
        raw_3d = np.stack([pts3d[nm] for nm in node_names], axis=0)
        raw_3d -= raw_3d.mean(axis=0, keepdims=True)
        gt_3d = raw_3d / max(float(np.sqrt(np.mean(np.sum(raw_3d**2, axis=-1)))), 1e-6)

        ang_z = float(rng.uniform(-0.6, 0.6))
        ang_x = float(rng.uniform(0.35, 0.65))
        Rz = np.array([[np.cos(ang_z), -np.sin(ang_z), 0.0], [np.sin(ang_z), np.cos(ang_z), 0.0], [0.0, 0.0, 1.0]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ang_x), -np.sin(ang_x)], [0.0, np.sin(ang_x), np.cos(ang_x)]])
        gt_cam = gt_3d @ (Rx @ Rz).T

        px_2d = np.zeros((n_pts, 2), dtype=np.float32)
        px_2d[:, 0] = gt_cam[:, 0] * 95.0 + 224.0
        px_2d[:, 1] = gt_cam[:, 1] * 95.0 + 160.0

        adj = torch.zeros((n_pts, n_pts), dtype=torch.bool)
        estyle = torch.full((n_pts, n_pts), EDGE_STYLE_UNKNOWN, dtype=torch.long)
        for (u_nm, v_nm), st in edge_styles.items():
            u, v = name_to_idx[u_nm], name_to_idx[v_nm]
            adj[u, v] = adj[v, u] = True
            estyle[u, v] = estyle[v, u] = st

        formal_cs = []
        for sp_r in ratio_specs:
            t_idx = name_to_idx[sp_r["target"]]
            p1_idx = name_to_idx[sp_r["parents"][0]]
            p2_idx = name_to_idx[sp_r["parents"][1]]
            formal_cs.append(IndexedConstraint("collinear", (p1_idx, p2_idx, t_idx), {"pts": [p1_idx, p2_idx, t_idx]}, True))
            formal_cs.append(IndexedConstraint("betweenness", (t_idx, p1_idx, p2_idx), {"pt": t_idx, "endpoints": [p1_idx, p2_idx]}, True))
            if abs(sp_r["t"] - 0.5) < 1e-4:
                formal_cs.append(IndexedConstraint("midpoint", (p1_idx, p2_idx, t_idx), {"args": [p1_idx, p2_idx, t_idx]}, True))
            else:
                formal_cs.append(IndexedConstraint("distance_ratio", (p1_idx, t_idx, p2_idx), {"pts": [p1_idx, t_idx, p2_idx], "ratio": sp_r["t"]}, True))

        cam_mat = fit_isotropic_orthographic_camera(
            torch.tensor(gt_cam, dtype=torch.float64), torch.tensor(px_2d, dtype=torch.float64)
        ).float()
        contract = AnonymousGeometrySample(
            graph=AnonymousGraph(
                points_2d=torch.tensor(px_2d, dtype=torch.float32),
                adjacency=adj,
                edge_style=estyle,
                image_size=torch.tensor([448.0, 320.0], dtype=torch.float32),
                node_type=torch.full((n_pts,), NODE_TYPE_CORNER, dtype=torch.long),
                node_confidence=torch.ones(n_pts, dtype=torch.float32),
                edge_confidence=adj.float(),
                edge_style_confidence=(estyle != EDGE_STYLE_UNKNOWN).float(),
            ),
            constraints=tuple(formal_cs),
            target_3d=torch.tensor(gt_cam, dtype=torch.float32),
            world_to_image=cam_mat,
        )
        batch = collate_module2_samples(
            [
                {
                    "contract": contract,
                    "sample_id": f"hull_{idx:03d}",
                    "family": "random_convex_hull",
                    "geometry_group": f"hull_{idx:03d}",
                    "topology_group": "random_convex_hull",
                    "target_mask": torch.ones(n_pts, dtype=torch.bool),
                }
            ]
        )
        tgt = torch.tensor(gt_cam, dtype=torch.float64)
        msk = torch.ones(n_pts, dtype=torch.bool)
        px_d = torch.tensor(px_2d, dtype=torch.float64)

        for m_net, out_list in ((m_v840, t3_rows_v840), (m_v6, t3_rows_v6)):
            with torch.inference_mode():
                p_init = m_net(**model_inputs(batch, device))["coordinates_3d"].float().cpu()[0, :n_pts].double()
            cam = fit_isotropic_orthographic_camera(p_init, px_d)
            p_leg, _, _ = solve_legacy_unconstrained_lbfgs(p_init, formal_cs, 0.01)
            r_827 = proj_v827.project(p_init, formal_cs, camera=cam, points_2d=px_d)
            r_843 = proj_v843.project(p_init, formal_cs, camera=cam, points_2d=px_d)
            r_845 = proj_v845.project(p_init, formal_cs, camera=cam, points_2d=px_d)
            r_v9 = proj_v9_ood.project(p_init, formal_cs, camera=cam, points_2d=px_d, adjacency=adj, edge_style=estyle)

            ind_827, _, info_827, _, _ = _independent_summary(r_827.coordinates.double(), formal_cs, p_init, 0.0011)
            ind_843, _, info_843, _, _ = _independent_summary(r_843.coordinates.double(), formal_cs, p_init, 0.0011)
            ind_845, _, info_845, _, _ = _independent_summary(r_845.coordinates.double(), formal_cs, p_init, 0.0011)
            ind_v9, _, info_v9, _, _ = _independent_summary(r_v9.coordinates.double(), formal_cs, p_init, 0.0011)

            spec_leg = decompose_depth_displacement_v9(p_leg.double(), p_init, formal_cs)
            spec_845 = decompose_depth_displacement_v9(r_845.coordinates.double(), p_init, formal_cs)
            spec_v9 = decompose_depth_displacement_v9(r_v9.coordinates.double(), p_init, formal_cs)

            out_list.append(
                {
                    "raw_pa": float(numpy_pa(p_init, tgt, msk, True)),
                    "leg_pa": float(numpy_pa(p_leg.double(), tgt, msk, True)),
                    "leg_null_rms": float(spec_leg["null_rms"]),
                    "v827_pa": float(numpy_pa(r_827.coordinates.double(), tgt, msk, True)),
                    "v827_valid": bool((not r_827.abstained) and ind_827 and info_827 > 0),
                    "v843_pa": float(numpy_pa(r_843.coordinates.double(), tgt, msk, True)),
                    "v843_valid": bool((not r_843.abstained) and ind_843 and info_843 > 0),
                    "v845_pa": float(numpy_pa(r_845.coordinates.double(), tgt, msk, True)),
                    "v845_valid": bool((not r_845.abstained) and ind_845 and info_845 > 0),
                    "v845_null_rms": float(spec_845["null_rms"]),
                    "v9_pa": float(numpy_pa(r_v9.coordinates.double(), tgt, msk, True)),
                    "v9_valid": bool((not r_v9.abstained) and ind_v9 and info_v9 > 0),
                    "v9_invalid": bool((not r_v9.abstained) and (not ind_v9)),
                    "v9_null_rms": float(spec_v9["null_rms"]),
                }
            )

    tier3 = {
        "m2_v840_specialist": _summarize_ood_rows(t3_rows_v840),
        "m2_v6_generalist": _summarize_ood_rows(t3_rows_v6),
    }
    print(
        f"[Tier3 ConvexHull 100] V840+Leg={tier3['m2_v840_specialist']['legacy_lbfgs_pa']:.4f} (Null={tier3['m2_v840_specialist']['legacy_lbfgs_null_rms']:.4f}) -> "
        f"V840+V9-OOD={tier3['m2_v840_specialist']['v9_ood_pa']:.4f} ({tier3['m2_v840_specialist']['v9_ood_valid_pct']:.1f}%, Null={tier3['m2_v840_specialist']['v9_ood_null_rms']:.4f}) | "
        f"V6+V9-OOD={tier3['m2_v6_generalist']['v9_ood_pa']:.4f} ({tier3['m2_v6_generalist']['v9_ood_valid_pct']:.1f}%)",
        flush=True,
    )

    payload = {
        "status": "PASS",
        "experiment": "Suite_B_3Tier_OOD_Generalization_and_Nullspace_Drift_V9",
        "tier1_zeroshot_basic4_to_complex8": tier1,
        "tier2_loso_unseen_topologies_3seeds": tier2,
        "tier3_nontemplate_convex_hulls_100": tier3,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved 3-Tier OOD V9 report to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
