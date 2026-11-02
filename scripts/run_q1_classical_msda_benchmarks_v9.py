"""Comprehensive Q1 Benchmark of Classical Non-Neural Line-Drawing Reconstruction Baselines (V9.0).

Evaluates two seminal non-neural CAD / Computer Vision optimization baselines:
  1. Marill-MSDA (Marill 1991, IJCV): Minimum Standard Deviation of 3D Corner Angles + isometric
     regularity across multi-start spectral/occlusion inflation seeds (no neural prior, no hard GD&T).
  2. Lipson-Shpitalni (Lipson & Shpitalni 1996, Computer-Aided Design): Multi-start MSDA angular &
     isometric regularity + exact symbolic geometric constraint optimization + coupled sub-pixel
     Levenberg-Marquardt manifold polishing + solid-over-dashed crossing orientation (zero neural prior).

Across all three flagship Q1 benchmark suites:
  - Part 1: 3-Seed Confirmatory Held-Out Test (512 views x 3 seeds = 1,536 mirror-pair views)
  - Part 2: Pooled 792-View 12-Family Multi-Constraint Benchmark stratified by Dual-Gauge Rigidity Classes I/II/III
  - Part 3: 120-Part Mechanical CAD Blueprint Reverse-Engineering & ISO 10303-21 .STEP Sewing Benchmark
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.stats import wilcoxon

from scripts.analyze_module3_v835_audit import numpy_pa
from scripts.evaluate_feasibility_projector_v6 import _independent_summary
from scripts.module2_v834_contract import read_json
from scripts.module3_v843_contract import COHORT_SPECS
from scripts.run_q1_12family_evaluation_v9 import _classify_dual_gauge
from scripts.run_q1_cad_reverse_engineering_v9 import (
    _build_cad_archetype,
    _compute_hidden_line_styles,
)
from scripts.run_v845_12family_pilot_smoke import load_benchmark_items
from src.cad.step_exporter_v9 import export_cad_to_step, validate_step_and_brep_quality
from src.lifting.anonymous_contract import EDGE_STYLE_UNKNOWN
from src.lifting.cached_dataset import CachedModule2Dataset
from src.lifting.camera_gauge_v51 import fit_isotropic_orthographic_camera
from src.lifting.v4_dataset import collate_module2_samples
from src.reasoning.classical_line_drawing_msda_v9 import (
    LipsonShpitalniCADSolverV9,
    MarillMSDASolverV9,
)
from src.reasoning.constraint_binding_v6 import bind_constraints_from_point_anchors_v6
from src.reasoning.residuals_v4 import compile_semantic_residuals

OUT_JSON = ROOT / "runs_q1_journal_v9" / "classical_msda_lipson_benchmarks_v9.json"
SEM_TOL = 0.0011


def _worker_heldout_view(task: dict) -> dict:
    torch.set_num_threads(1)
    msda = MarillMSDASolverV9(num_random_seeds=2, max_lbfgs_iter=40)
    ls = LipsonShpitalniCADSolverV9(num_random_seeds=2, stage1_lbfgs_iter=45, stage2_lm_iter=20)

    px = task["pixels"]
    adj = task["adjacency"]
    sty = task["edge_style"]
    cs = task["constraints"]
    cam = task["camera"]
    conf = task["confidences"]
    raw_ref = task["raw_for_ref_scale"].double()
    tgt = task["target"].double()
    mask = task["mask"]

    r_msda = msda.reconstruct(px, adj, sty, constraints=cs, camera=None, reference_for_eval=raw_ref)
    val_m, res_m, info_m, _, _ = _independent_summary(r_msda.coordinates, cs, raw_ref, SEM_TOL)

    r_ls = ls.reconstruct(
        px, adj, sty, constraints=cs, camera=None, constraint_weights=conf, reference_for_eval=raw_ref
    )
    val_l, res_l, info_l, _, _ = _independent_summary(r_ls.coordinates, cs, raw_ref, SEM_TOL)

    return {
        "sample_id": task["sample_id"],
        "family": task["family"],
        "group": task["group"],
        "msda": {
            "shape_pa": float(numpy_pa(r_msda.coordinates, tgt, mask, True)),
            "proper_pa": float(numpy_pa(r_msda.coordinates, tgt, mask, False)),
            "valid": bool(val_m and info_m > 0),
            "invalid": bool(not val_m),
            "max_res": float(res_m),
            "latency_ms": float(r_msda.latency_ms),
        },
        "ls": {
            "shape_pa": float(numpy_pa(r_ls.coordinates, tgt, mask, True)),
            "proper_pa": float(numpy_pa(r_ls.coordinates, tgt, mask, False)),
            "valid": bool(val_l and info_l > 0),
            "invalid": bool(not val_l),
            "max_res": float(res_l),
            "latency_ms": float(r_ls.latency_ms),
        },
    }


def _worker_12fam_view(task: dict) -> dict:
    torch.set_num_threads(1)
    msda = MarillMSDASolverV9(num_random_seeds=2, max_lbfgs_iter=40)
    ls = LipsonShpitalniCADSolverV9(num_random_seeds=2, stage1_lbfgs_iter=45, stage2_lm_iter=20)

    px = task["px"]
    adj = task["adj"]
    sty = task["sty"]
    cs = task["cs"]
    tgt = task["tgt"]
    msk = torch.ones(tgt.shape[0], dtype=torch.bool)

    r_msda = msda.reconstruct(px, adj, sty, constraints=cs, reference_for_eval=tgt)
    val_m, _, info_m, _, _ = _independent_summary(r_msda.coordinates, cs, tgt, SEM_TOL)

    r_ls = ls.reconstruct(px, adj, sty, constraints=cs, reference_for_eval=tgt)
    val_l, _, info_l, _, _ = _independent_summary(r_ls.coordinates, cs, tgt, SEM_TOL)

    return {
        "cohort": task["cohort"],
        "family": task["family"],
        "rig_cls": task["rig_cls"],
        "msda_pa": float(numpy_pa(r_msda.coordinates, tgt, msk, True)),
        "msda_proper_pa": float(numpy_pa(r_msda.coordinates, tgt, msk, False)),
        "msda_valid": bool(val_m and (info_m > 0 or len(cs) == 0)),
        "msda_lat_ms": float(r_msda.latency_ms),
        "ls_pa": float(numpy_pa(r_ls.coordinates, tgt, msk, True)),
        "ls_proper_pa": float(numpy_pa(r_ls.coordinates, tgt, msk, False)),
        "ls_valid": bool(val_l and (info_l > 0 or len(cs) == 0)),
        "ls_lat_ms": float(r_ls.latency_ms),
    }


def _worker_cad_part(task: dict) -> dict:
    torch.set_num_threads(1)
    msda = MarillMSDASolverV9(num_random_seeds=2, max_lbfgs_iter=45)
    ls = LipsonShpitalniCADSolverV9(num_random_seeds=3, stage1_lbfgs_iter=55, stage2_lm_iter=24)

    arch = task["arch"]
    inst = task["inst"]
    gt_cam = task["gt_cam"]
    faces = task["faces"]
    full_cs = task["full_cs"]
    affine_cs = task["affine_cs"]
    px_d = torch.tensor(task["px_2d"], dtype=torch.float64)
    adj = task["adj"]
    estyle = task["estyle"]
    gt_t = torch.tensor(gt_cam, dtype=torch.float64)

    r_msda = msda.reconstruct(px_d, adj, estyle, constraints=full_cs, camera=None, reference_for_eval=gt_t)
    # Canonical Lipson & Shpitalni (1996, CAD): coplanar faces + parallel lines + MSDA/isometric regularity
    r_ls_aff = ls.reconstruct(px_d, adj, estyle, constraints=affine_cs, camera=None, reference_for_eval=gt_t)
    # Augmented Lipson-Shpitalni + explicit 3D metric GD&T (full_cs)
    r_ls_full = ls.reconstruct(px_d, adj, estyle, constraints=full_cs, camera=None, reference_for_eval=gt_t)

    out = {"arch": arch, "inst": inst}
    for key, res_obj in (
        ("marill_msda", r_msda),
        ("lipson_shpitalni", r_ls_aff),
        ("lipson_shpitalni_full_gdt", r_ls_full),
    ):
        coords_np = res_obj.coordinates.detach().cpu().double().numpy()
        step_str = export_cad_to_step(coords_np, faces, part_name=f"{arch.upper()}_{inst:02d}_{key.upper()}")
        qual = validate_step_and_brep_quality(step_str, coords_np, gt_cam, faces)
        b_sem = compile_semantic_residuals(torch.tensor(coords_np, dtype=torch.float64), full_cs, reference_coordinates=gt_t)
        sem_ok = bool(b_sem.records and float(b_sem.raw_constraint_rms.max().item()) <= SEM_TOL)
        b_aff = compile_semantic_residuals(torch.tensor(coords_np, dtype=torch.float64), affine_cs, reference_coordinates=gt_t)
        aff_ok = bool(b_aff.records and float(b_aff.raw_constraint_rms.max().item()) <= SEM_TOL)
        qual["semantically_feasible"] = sem_ok
        qual["affine_feasible"] = aff_ok
        qual["latency_ms"] = float(res_obj.latency_ms)
        out[key] = qual
    return out


def run_part1_heldout_3seeds() -> dict:
    spec = COHORT_SPECS["heldout"]
    bdir = spec["base_dir"]
    cdir = spec["cache_dir"]

    per_seed = []
    for branch in range(3):
        t0 = time.perf_counter()
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

        tasks = []
        for r_row, r_rep in zip(r015_json["rows"], r015_pt):
            conf, adj, sty = meta_by_id[r_row["sample_id"]]
            tasks.append(
                {
                    "sample_id": r_row["sample_id"],
                    "family": r_row["family"],
                    "group": r_row.get("geometry_group", r_row["sample_id"].rsplit("_", 1)[0]),
                    "pixels": r_rep["pixels"],
                    "adjacency": adj,
                    "edge_style": sty,
                    "constraints": r_rep["constraints"],
                    "camera": r_rep["camera"],
                    "confidences": conf,
                    "raw_for_ref_scale": r_rep["raw"],
                    "target": r_rep["target"],
                    "mask": r_rep["mask"],
                }
            )

        rows = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(_worker_heldout_view, t): i for i, t in enumerate(tasks)}
            for fut in as_completed(futs):
                rows[futs[fut]] = fut.result()

        seed_summary = {"branch": branch}
        for m_key in ("msda", "ls"):
            fams = {}
            for fam in ("tetra_v2", "cube_v2", "prism_v2", "polyhedra_v2"):
                sub = [r for r in rows if r["family"] == fam]
                by_grp = defaultdict(list)
                for r in sub:
                    by_grp[r["group"]].append(r[m_key]["valid"])
                pair_cov = fmean([1.0 if all(vs) else 0.0 for vs in by_grp.values()])
                fams[fam] = {
                    "shape_pa": fmean([r[m_key]["shape_pa"] for r in sub]),
                    "proper_pa": fmean([r[m_key]["proper_pa"] for r in sub]),
                    "view_cov": fmean([1.0 if r[m_key]["valid"] else 0.0 for r in sub]),
                    "pair_cov": pair_cov,
                }
            by_grp_all = defaultdict(list)
            for r in rows:
                by_grp_all[r["group"]].append(r[m_key]["valid"])
            seed_summary[m_key] = {
                "macro_shape_pa": fmean([v["shape_pa"] for v in fams.values()]),
                "macro_proper_pa": fmean([v["proper_pa"] for v in fams.values()]),
                "view_cov": fmean([1.0 if r[m_key]["valid"] else 0.0 for r in rows]),
                "pair_cov": fmean([1.0 if all(vs) else 0.0 for vs in by_grp_all.values()]),
                "invalid_views": sum(1 for r in rows if r[m_key]["invalid"]),
                "latency_median_ms": float(np.median([r[m_key]["latency_ms"] for r in rows])),
                "family": fams,
            }
        print(
            f"[Heldout B{branch}] ({time.perf_counter()-t0:.1f}s) | "
            f"MSDA PA={seed_summary['msda']['macro_shape_pa']:.4f} | "
            f"LS PA={seed_summary['ls']['macro_shape_pa']:.4f} PairCov={seed_summary['ls']['pair_cov']*100:.2f}%",
            flush=True,
        )
        per_seed.append(seed_summary)

    agg = {}
    for m_key, name in (("msda", "marill_msda_1991"), ("ls", "lipson_shpitalni_1996")):
        fams_agg = {}
        for fam in ("tetra_v2", "cube_v2", "prism_v2", "polyhedra_v2"):
            fams_agg[fam] = {
                "shape_pa_mean": round(float(np.mean([s[m_key]["family"][fam]["shape_pa"] for s in per_seed])), 4),
                "shape_pa_std": round(float(np.std([s[m_key]["family"][fam]["shape_pa"] for s in per_seed], ddof=1)), 4),
                "pair_cov_pct_mean": round(float(np.mean([s[m_key]["family"][fam]["pair_cov"] * 100.0 for s in per_seed])), 2),
                "view_cov_pct_mean": round(float(np.mean([s[m_key]["family"][fam]["view_cov"] * 100.0 for s in per_seed])), 2),
            }
        agg[name] = {
            "macro_shape_pa_mean": round(float(np.mean([s[m_key]["macro_shape_pa"] for s in per_seed])), 4),
            "macro_shape_pa_std": round(float(np.std([s[m_key]["macro_shape_pa"] for s in per_seed], ddof=1)), 4),
            "macro_proper_pa_mean": round(float(np.mean([s[m_key]["macro_proper_pa"] for s in per_seed])), 4),
            "macro_proper_pa_std": round(float(np.std([s[m_key]["macro_proper_pa"] for s in per_seed], ddof=1)), 4),
            "pair_cov_pct_mean": round(float(np.mean([s[m_key]["pair_cov"] * 100.0 for s in per_seed])), 2),
            "pair_cov_pct_std": round(float(np.std([s[m_key]["pair_cov"] * 100.0 for s in per_seed], ddof=1)), 2),
            "view_cov_pct_mean": round(float(np.mean([s[m_key]["view_cov"] * 100.0 for s in per_seed])), 2),
            "total_invalid_views_3seeds": int(sum(s[m_key]["invalid_views"] for s in per_seed)),
            "latency_median_ms": round(float(np.mean([s[m_key]["latency_median_ms"] for s in per_seed])), 2),
            "family": fams_agg,
        }
    return {"aggregate_3seed_heldout": agg, "per_seed": per_seed}


def run_part2_12family_792views() -> dict:
    t0 = time.perf_counter()
    cohorts = load_benchmark_items()
    tasks = []
    for cname, items in cohorts.items():
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
            rig_cls, _, _ = _classify_dual_gauge(tgt, list(f.constraints))
            tasks.append(
                {
                    "cohort": cname,
                    "family": fam,
                    "rig_cls": rig_cls,
                    "px": f.graph.points_2d.double().cpu(),
                    "adj": f.graph.adjacency,
                    "sty": f.graph.edge_style,
                    "cs": list(f.constraints),
                    "tgt": tgt,
                }
            )

    rows = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_worker_12fam_view, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            rows[futs[fut]] = fut.result()

    def _agg(sub: list[dict]) -> dict:
        return {
            "views": len(sub),
            "marill_msda_pa": round(fmean([r["msda_pa"] for r in sub]), 4),
            "marill_msda_valid_pct": round(100.0 * fmean([1.0 if r["msda_valid"] else 0.0 for r in sub]), 2),
            "marill_msda_lat_median_ms": round(float(np.median([r["msda_lat_ms"] for r in sub])), 2),
            "lipson_shpitalni_pa": round(fmean([r["ls_pa"] for r in sub]), 4),
            "lipson_shpitalni_proper_pa": round(fmean([r["ls_proper_pa"] for r in sub]), 4),
            "lipson_shpitalni_valid_pct": round(100.0 * fmean([1.0 if r["ls_valid"] else 0.0 for r in sub]), 2),
            "lipson_shpitalni_lat_median_ms": round(float(np.median([r["ls_lat_ms"] for r in sub])), 2),
        }

    by_class = {}
    for cls_k in ("Class_I_RayRigid", "Class_II_CoupledTilt", "Class_III_BasRelief"):
        sub_c = [r for r in rows if r["rig_cls"] == cls_k]
        by_class[cls_k] = _agg(sub_c)

    by_fam = {}
    for fam_k in sorted({r["family"] for r in rows}):
        by_fam[fam_k] = _agg([r for r in rows if r["family"] == fam_k])

    pilot144 = [r for r in rows if "final_confirmation" in r["cohort"] or "pilot144" in r["cohort"]]
    res = {
        "overall_792_views": _agg(rows),
        "by_rigidity_class": by_class,
        "by_family": by_fam,
    }
    if pilot144:
        res["pilot144_subset"] = _agg(pilot144)

    print(
        f"[Suite A2 792 Views] ({time.perf_counter()-t0:.1f}s) | "
        f"MSDA PA={res['overall_792_views']['marill_msda_pa']:.4f} | "
        f"LS PA={res['overall_792_views']['lipson_shpitalni_pa']:.4f} ({res['overall_792_views']['lipson_shpitalni_valid_pct']:.2f}% valid)",
        flush=True,
    )
    return res


def run_part3_cad_120parts() -> dict:
    t0 = time.perf_counter()
    archetypes = [
        "chamfered_block",
        "slanted_wedge_housing",
        "l_bracket_extrusion",
        "v_groove_guide",
        "stepped_slotted_block",
        "frustum_hopper_die",
    ]
    rng = np.random.default_rng(20261029)
    tasks = []
    for arch in archetypes:
        for inst in range(20):
            pts_raw, faces, full_cs, affine_cs = _build_cad_archetype(arch, rng)
            pts_c = pts_raw - pts_raw.mean(axis=0, keepdims=True)
            pts_n = pts_c / float(np.sqrt(np.mean(np.sum(pts_c**2, axis=1))))

            yaw = float(rng.uniform(0.48, 0.92)) * (1.0 if inst % 2 == 0 else -1.0)
            pitch = float(rng.uniform(0.42, 0.72))
            rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
            rx = np.array([[1.0, 0.0, 0.0], [0.0, math.sin(pitch), math.cos(pitch)], [0.0, -math.cos(pitch), math.sin(pitch)]])
            gt_cam = pts_n @ (rx @ rz).T
            if np.linalg.det(rx @ rz) < 0:
                rx_p = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(pitch - 1.1), -math.sin(pitch - 1.1)], [0.0, math.sin(pitch - 1.1), math.cos(pitch - 1.1)]])
                gt_cam = pts_n @ (rx_p @ rz).T

            edge_styles = _compute_hidden_line_styles(gt_cam, faces)
            n_pts = len(gt_cam)
            px_2d = np.zeros((n_pts, 2), dtype=np.float32)
            px_2d[:, 0] = gt_cam[:, 0] * 110.0 + 224.0
            px_2d[:, 1] = gt_cam[:, 1] * 110.0 + 160.0

            adj = torch.zeros((n_pts, n_pts), dtype=torch.bool)
            estyle = torch.full((n_pts, n_pts), EDGE_STYLE_UNKNOWN, dtype=torch.long)
            for (u, v), st in edge_styles.items():
                adj[u, v] = adj[v, u] = True
                estyle[u, v] = estyle[v, u] = st

            cam_gt = fit_isotropic_orthographic_camera(
                torch.tensor(gt_cam, dtype=torch.float64), torch.tensor(px_2d, dtype=torch.float64)
            )
            tasks.append(
                {
                    "arch": arch,
                    "inst": inst,
                    "gt_cam": gt_cam,
                    "faces": faces,
                    "full_cs": full_cs,
                    "affine_cs": affine_cs,
                    "px_2d": px_2d,
                    "adj": adj,
                    "estyle": estyle,
                }
            )

    rows = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_worker_cad_part, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            rows[futs[fut]] = fut.result()

    def _agg_cad(sub: list[dict], key: str) -> dict:
        return {
            "shape_pa_mean": round(fmean([r[key]["shape_pa"] for r in sub]), 4),
            "proper_pa_mean": round(fmean([r[key]["proper_pa"] for r in sub]), 4),
            "max_face_planarity_mm_mean": round(fmean([r[key]["max_face_planarity_mm"] for r in sub]), 4),
            "cad_sewing_valid_pct": round(100.0 * fmean([1.0 if r[key]["cad_sewing_valid_010mm"] else 0.0 for r in sub]), 2),
            "semantically_feasible_pct": round(100.0 * fmean([1.0 if r[key]["semantically_feasible"] else 0.0 for r in sub]), 2),
            "affine_feasible_pct": round(100.0 * fmean([1.0 if r[key]["affine_feasible"] else 0.0 for r in sub]), 2),
            "dihedral_angle_error_deg_mean": round(fmean([r[key]["mean_dihedral_angle_error_deg"] for r in sub]), 3),
            "relative_edge_length_error_pct": round(100.0 * fmean([r[key]["mean_relative_edge_length_error"] for r in sub]), 3),
            "step_valid_pct": round(100.0 * fmean([1.0 if r[key]["step_syntax_and_refs_valid"] else 0.0 for r in sub]), 1),
            "watertight_brep_pct": round(100.0 * fmean([1.0 if r[key]["watertight_manifold_solid_brep"] else 0.0 for r in sub]), 1),
            "latency_median_ms": round(float(np.median([r[key]["latency_ms"] for r in sub])), 2),
        }

    overall = {
        "marill_msda": _agg_cad(rows, "marill_msda"),
        "lipson_shpitalni": _agg_cad(rows, "lipson_shpitalni"),
        "lipson_shpitalni_full_gdt": _agg_cad(rows, "lipson_shpitalni_full_gdt"),
    }
    per_arch = {}
    for arch in archetypes:
        sub_a = [r for r in rows if r["arch"] == arch]
        per_arch[arch] = {
            "marill_msda": _agg_cad(sub_a, "marill_msda"),
            "lipson_shpitalni": _agg_cad(sub_a, "lipson_shpitalni"),
            "lipson_shpitalni_full_gdt": _agg_cad(sub_a, "lipson_shpitalni_full_gdt"),
        }

    print(
        f"[Suite C2 120 CAD Parts] ({time.perf_counter()-t0:.1f}s) | "
        f"MSDA PA={overall['marill_msda']['shape_pa_mean']:.4f} Sew={overall['marill_msda']['cad_sewing_valid_pct']:.2f}% | "
        f"LS-96 PA={overall['lipson_shpitalni']['shape_pa_mean']:.4f} Sew={overall['lipson_shpitalni']['cad_sewing_valid_pct']:.2f}% | "
        f"LS-FullGDT PA={overall['lipson_shpitalni_full_gdt']['shape_pa_mean']:.4f} Sew={overall['lipson_shpitalni_full_gdt']['cad_sewing_valid_pct']:.2f}%",
        flush=True,
    )
    return {"overall_120_parts": overall, "per_archetype": per_arch}


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    p1 = run_part1_heldout_3seeds()
    p2 = run_part2_12family_792views()
    p3 = run_part3_cad_120parts()
    report = {
        "status": "PASS",
        "experiment": "Classical_NonNeural_Marill_MSDA_and_Lipson_Shpitalni_Benchmarks_V9",
        "part1_heldout_3seeds_1536views": p1,
        "part2_12family_792views": p2,
        "part3_cad_reverse_engineering_120parts": p3,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved classical baseline report to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
