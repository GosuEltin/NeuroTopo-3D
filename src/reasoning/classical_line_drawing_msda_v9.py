"""Classical Non-Neural Single-View 3D Line-Drawing Reconstruction Baselines (V9.0).

Implements two foundational optimization-only CAD / Computer Vision baselines operating
strictly on the anonymous 2D wireframe graph (and symbolic predicates for Lipson-Shpitalni)
WITHOUT any learned neural 3D lifter (Module 2):

1. MarillMSDASolverV9 (Marill 1991, IJCV):
   Reconstructs 3D vertex depths along orthographic viewing rays by minimizing the
   Standard Deviation of 3D Corner Angles (MSDA) regularized by 3D edge-length
   consistency across deterministic multi-start spectral/occlusion inflation seeds,
   followed by physical solid-over-dashed 2D crossing hemisphere selection.

2. LipsonShpitalniCADSolverV9 (Lipson & Shpitalni 1996, Computer-Aided Design;
   Leclerc & Fischler 1992, IJCV):
   Combines multi-start MSDA angular/isometric regularity inflation with exact
   symbolic geometric constraint optimization (coplanarity, parallelism,
   orthogonality, midpoints, collinearity, equal lengths) and coupled sub-pixel
   Levenberg-Marquardt manifold polishing onto M(C) (max_c ||r_c||_rms <= 1.1e-3),
   without any neural 3D coordinate prior.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import minimize

from src.lifting.anonymous_contract import (
    EDGE_STYLE_DASHED,
    EDGE_STYLE_SOLID,
    IndexedConstraint,
)
from src.lifting.camera_gauge_v51 import fit_isotropic_orthographic_camera
from src.reasoning.normal_flow_projector_v9 import (
    build_ray_orthonormal_basis,
    find_transverse_solid_dashed_crossings,
    score_physical_occlusion_hemisphere,
)
from src.reasoning.residuals_v4 import compile_semantic_residuals


@dataclass(frozen=True)
class ClassicalLineDrawingResultV9:
    coordinates: torch.Tensor
    feasible: bool
    max_constraint_rms: float
    reprojection_rmse_px: float
    latency_ms: float
    method: str


def _normalize_2d_to_camera_xy(
    points_2d: torch.Tensor,
    camera: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map observed 2D pixels [N, 2] to camera-plane (x0, y0) with exact 0.0 px reprojection."""
    px = points_2d.detach().cpu().double()
    n = px.shape[0]
    if camera is not None:
        cam = camera.detach().cpu().double()
        sx = float(cam[0, 0].item())
        sy = float(cam[1, 1].item())
        if sx > 1e-8 and abs(sx - sy) <= 1e-5 * sx:
            x0 = (px[:, 0] - cam[0, 3]) / sx
            y0 = (px[:, 1] - cam[1, 3]) / sy
            return torch.stack((x0, y0), dim=1), cam

    center = px.mean(dim=0, keepdim=True)
    centered = px - center
    rms_2d = float(torch.sqrt(torch.mean(torch.sum(centered.square(), dim=1))).item())
    target_2d_rms = math.sqrt(2.0 / 3.0)
    scale_px_per_unit = max(rms_2d / max(target_2d_rms, 1e-8), 1e-6)
    xy0 = centered / scale_px_per_unit
    init_3d = torch.cat((xy0, torch.zeros((n, 1), dtype=torch.float64)), dim=1)
    cam_fit = fit_isotropic_orthographic_camera(init_3d, px)
    return xy0, cam_fit


def _extract_edges_and_angle_triples(
    adjacency: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return undirected edge index pairs [E, 2] and corner angle triples (i, center_j, k) [A, 3]."""
    adj = adjacency.detach().cpu().numpy() > 0.5
    n = adj.shape[0]
    edges = []
    triples = []
    for j in range(n):
        nbrs = [i for i in range(n) if i != j and adj[j, i]]
        for idx_a in range(len(nbrs)):
            i = nbrs[idx_a]
            if j < i:
                edges.append((j, i))
            for idx_b in range(idx_a + 1, len(nbrs)):
                k = nbrs[idx_b]
                triples.append((i, j, k))
    edge_t = torch.tensor(edges, dtype=torch.long) if edges else torch.zeros((0, 2), dtype=torch.long)
    triple_t = torch.tensor(triples, dtype=torch.long) if triples else torch.zeros((0, 3), dtype=torch.long)
    return edge_t, triple_t


def _build_deterministic_depth_seeds(
    xy0: torch.Tensor,
    adjacency: torch.Tensor,
    edge_style: torch.Tensor | None,
    q_ray: torch.Tensor,
    num_random_seeds: int = 2,
    seed: int = 20261031,
) -> list[torch.Tensor]:
    """Construct non-planar centered depth seeds zeta in R^{N-1} escaping the z=0 symmetry saddle."""
    n = xy0.shape[0]
    if n <= 1:
        return [torch.zeros(0, dtype=torch.float64)]

    xy_span = float(torch.sqrt(torch.mean(torch.sum((xy0 - xy0.mean(dim=0, keepdim=True)).square(), dim=1))).item())
    relief_scale = max(0.48 * xy_span, 0.25)

    adj_np = (adjacency.detach().cpu().numpy() > 0.5).astype(np.float64)
    deg = adj_np.sum(axis=1)
    lap = np.diag(deg) - adj_np
    evals, evecs = np.linalg.eigh(lap)

    raw_z_candidates: list[np.ndarray] = []

    # Seed 1: Occlusion-informed solid-minus-dashed relief + radial dome
    r2 = torch.sum((xy0 - xy0.mean(dim=0, keepdim=True)).square(), dim=1).numpy()
    dome = -(r2 - r2.mean())
    if edge_style is not None:
        st_np = edge_style.detach().cpu().numpy()
        occ_bias = np.zeros(n, dtype=np.float64)
        for i in range(n):
            sol_cnt = float(np.sum((adj_np[i] > 0.5) & (st_np[i] == EDGE_STYLE_SOLID)))
            dsh_cnt = float(np.sum((adj_np[i] > 0.5) & (st_np[i] == EDGE_STYLE_DASHED)))
            occ_bias[i] = (sol_cnt - 1.5 * dsh_cnt) / max(deg[i], 1.0)
        z_occ = occ_bias + 0.35 * dome
    else:
        z_occ = dome
    raw_z_candidates.append(z_occ)

    # Seed 2 & 3: Graph Laplacian non-trivial Fiedler eigenvectors (captures global topological front/back)
    non_zero_modes = [idx for idx, ev in enumerate(evals) if ev > 1e-6]
    for m_idx in non_zero_modes[:2]:
        raw_z_candidates.append(evecs[:, m_idx].copy())

    # Seed 4+: Deterministic pseudo-random smooth relief seeds
    rng = np.random.default_rng(seed + 37 * n)
    for _ in range(num_random_seeds):
        raw_z_candidates.append(rng.standard_normal(n))

    zetas: list[torch.Tensor] = []
    for z_np in raw_z_candidates:
        zc = z_np - z_np.mean()
        sd = float(np.std(zc))
        if sd < 1e-8:
            zc = np.linspace(-1.0, 1.0, n)
            zc -= zc.mean()
            sd = float(np.std(zc))
        zc = zc * (relief_scale / max(sd, 1e-8))
        zt = torch.from_numpy(zc).double()
        zetas.append(q_ray.T @ zt)
    return zetas


def _msda_and_isometric_energy(
    pts: torch.Tensor,
    edges: torch.Tensor,
    triples: torch.Tensor,
    target_relief_std: float,
    iso_weight: float = 0.20,
    relief_barrier_weight: float = 0.35,
) -> torch.Tensor:
    """Compute Marill (1991) MSDA variance of 3D corner angles + Lipson-Shpitalni edge regularity."""
    loss = torch.zeros((), dtype=pts.dtype, device=pts.device)

    if triples.shape[0] >= 2:
        u = pts[triples[:, 0]] - pts[triples[:, 1]]
        v = pts[triples[:, 2]] - pts[triples[:, 1]]
        un = torch.linalg.norm(u, dim=1).clamp_min(1e-8)
        vn = torch.linalg.norm(v, dim=1).clamp_min(1e-8)
        cos_theta = torch.sum(u * v, dim=1) / (un * vn)
        # Smoothly clamped arccos in radians (matches Marill 1991 SDA squared)
        angles = torch.acos(cos_theta.clamp(-0.999999, 0.999999))
        mean_ang = angles.mean()
        msda_var = torch.mean((angles - mean_ang).square())
        loss = loss + msda_var

    if edges.shape[0] >= 2 and iso_weight > 0.0:
        elen = torch.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], dim=1).clamp_min(1e-8)
        mean_len = elen.mean().clamp_min(1e-8)
        cv_sq = torch.mean(((elen - mean_len) / mean_len).square())
        loss = loss + iso_weight * cv_sq

    if relief_barrier_weight > 0.0:
        z_centered = pts[:, 2] - pts[:, 2].mean()
        z_std = torch.sqrt(torch.mean(z_centered.square()) + 1e-10)
        shortfall = torch.relu(torch.tensor(target_relief_std, dtype=pts.dtype) - z_std)
        loss = loss + relief_barrier_weight * shortfall.square()

    return loss


def _orient_hemisphere_by_occlusion(
    pts: torch.Tensor,
    points_2d: torch.Tensor,
    adjacency: torch.Tensor,
    edge_style: torch.Tensor | None,
) -> torch.Tensor:
    """Select SO(3) depth sign z vs -z using transverse solid/dashed crossings or visibility degree."""
    if edge_style is None:
        return pts

    z_mean = pts[:, 2].mean()
    pts_refl = pts.clone()
    pts_refl[:, 2] = 2.0 * z_mean - pts[:, 2]

    px_np = points_2d.detach().cpu().double().numpy()
    adj_np = adjacency.detach().cpu().numpy()
    sty_np = edge_style.detach().cpu().numpy()
    crossings = find_transverse_solid_dashed_crossings(px_np, adj_np, sty_np)
    if crossings:
        s_pos = score_physical_occlusion_hemisphere(pts, crossings)
        s_neg = score_physical_occlusion_hemisphere(pts_refl, crossings)
        if abs(s_pos - s_neg) > 1e-5:
            return pts if s_pos >= s_neg else pts_refl

    # Fallback: foreground solid edges should on average lie closer (+z) than occluded dashed edges
    sol_mask = (adj_np > 0.5) & (sty_np == EDGE_STYLE_SOLID)
    dsh_mask = (adj_np > 0.5) & (sty_np == EDGE_STYLE_DASHED)
    z_np = pts[:, 2].detach().cpu().numpy()
    if np.any(sol_mask) and np.any(dsh_mask):
        iu_s, iv_s = np.where(np.triu(sol_mask, k=1))
        iu_d, iv_d = np.where(np.triu(dsh_mask, k=1))
        if len(iu_s) > 0 and len(iu_d) > 0:
            mean_z_sol = float(np.mean(0.5 * (z_np[iu_s] + z_np[iv_s])))
            mean_z_dsh = float(np.mean(0.5 * (z_np[iu_d] + z_np[iv_d])))
            if mean_z_sol < mean_z_dsh:
                return pts_refl
    return pts


class MarillMSDASolverV9:
    """Classical Marill (1991) Minimum Standard Deviation of Angles (MSDA) solver (non-neural)."""

    def __init__(self, num_random_seeds: int = 2, max_lbfgs_iter: int = 45) -> None:
        self.num_random_seeds = num_random_seeds
        self.max_lbfgs_iter = max_lbfgs_iter

    def reconstruct(
        self,
        points_2d: torch.Tensor,
        adjacency: torch.Tensor,
        edge_style: torch.Tensor | None = None,
        constraints: Sequence[IndexedConstraint] = (),
        camera: torch.Tensor | None = None,
        reference_for_eval: torch.Tensor | None = None,
    ) -> ClassicalLineDrawingResultV9:
        t0 = time.perf_counter()
        xy0, cam = _normalize_2d_to_camera_xy(points_2d, camera=camera)
        n = xy0.shape[0]
        q_ray = build_ray_orthonormal_basis(n)
        edges, triples = _extract_edges_and_angle_triples(adjacency)
        xy_rms = float(torch.sqrt(torch.mean(torch.sum((xy0 - xy0.mean(dim=0, keepdim=True)).square(), dim=1))).item())
        min_relief = max(0.28 * xy_rms, 0.14)

        seeds = _build_deterministic_depth_seeds(
            xy0, adjacency, edge_style, q_ray, num_random_seeds=self.num_random_seeds
        )
        best_pts = torch.cat((xy0, torch.zeros((n, 1), dtype=torch.float64)), dim=1)
        best_val = float("inf")

        for zeta_init in seeds:
            if zeta_init.numel() == 0:
                break

            def obj_grad(z_np: np.ndarray) -> tuple[float, np.ndarray]:
                zt = torch.from_numpy(z_np).double().requires_grad_(True)
                zv = q_ray @ zt
                pts = torch.cat((xy0, zv[:, None]), dim=1)
                loss = _msda_and_isometric_energy(
                    pts, edges, triples, target_relief_std=min_relief, iso_weight=0.18, relief_barrier_weight=0.40
                )
                g = torch.autograd.grad(loss, zt)[0]
                return float(loss.detach().item()), g.detach().numpy()

            sol = minimize(
                obj_grad,
                zeta_init.numpy().copy(),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.max_lbfgs_iter, "ftol": 1e-9},
            )
            cand_z = q_ray @ torch.from_numpy(sol.x).double()
            cand_pts = torch.cat((xy0, cand_z[:, None]), dim=1)
            if float(sol.fun) < best_val:
                best_val = float(sol.fun)
                best_pts = cand_pts

        best_pts = _orient_hemisphere_by_occlusion(best_pts, points_2d, adjacency, edge_style)
        ref_coords = reference_for_eval.detach().cpu().double() if reference_for_eval is not None else best_pts
        b_sem = compile_semantic_residuals(best_pts, constraints, reference_coordinates=ref_coords)
        max_rms = float(b_sem.raw_constraint_rms.max().item()) if b_sem.records else 0.0
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return ClassicalLineDrawingResultV9(
            coordinates=best_pts,
            feasible=bool(max_rms <= 1.1e-3),
            max_constraint_rms=max_rms,
            reprojection_rmse_px=0.0,
            latency_ms=lat_ms,
            method="marill_msda_1991",
        )


class LipsonShpitalniCADSolverV9:
    """Classical Lipson & Shpitalni (1996, CAD) Regularity + Symbolic Constraint Optimization Solver.

    Stage 1: Multi-start L-BFGS-B minimizing weighted symbolic predicates ||r_C||^2 + MSDA angular
             regularity + isometric edge regularity from deterministic non-planar viewing-ray seeds.
    Stage 2: Coupled sub-pixel damped Levenberg-Marquardt manifold polishing onto r_C(P) = 0
             (allowing <= 0.5 px xy relaxation for Sugihara superstrictness) + solid-over-dashed
             ray-crossing hemisphere disambiguation. Uses ZERO neural network priors.
    """

    def __init__(
        self,
        num_random_seeds: int = 3,
        stage1_lbfgs_iter: int = 55,
        stage2_lm_iter: int = 24,
        msda_weight: float = 0.08,
        iso_weight: float = 0.04,
        semantic_tolerance: float = 1.1e-3,
        max_pixel_drift_px: float = 0.5,
    ) -> None:
        self.num_random_seeds = num_random_seeds
        self.stage1_lbfgs_iter = stage1_lbfgs_iter
        self.stage2_lm_iter = stage2_lm_iter
        self.msda_weight = msda_weight
        self.iso_weight = iso_weight
        self.semantic_tolerance = semantic_tolerance
        self.max_pixel_drift_px = max_pixel_drift_px

    def reconstruct(
        self,
        points_2d: torch.Tensor,
        adjacency: torch.Tensor,
        edge_style: torch.Tensor | None = None,
        constraints: Sequence[IndexedConstraint] = (),
        camera: torch.Tensor | None = None,
        constraint_weights: torch.Tensor | None = None,
        reference_for_eval: torch.Tensor | None = None,
    ) -> ClassicalLineDrawingResultV9:
        t0 = time.perf_counter()
        px_d = points_2d.detach().cpu().double()
        xy0, cam = _normalize_2d_to_camera_xy(px_d, camera=camera)
        n = xy0.shape[0]
        q_ray = build_ray_orthonormal_basis(n)
        edges, triples = _extract_edges_and_angle_triples(adjacency)
        xy_rms = float(torch.sqrt(torch.mean(torch.sum((xy0 - xy0.mean(dim=0, keepdim=True)).square(), dim=1))).item())
        min_relief = max(0.26 * xy_rms, 0.13)

        seeds = _build_deterministic_depth_seeds(
            xy0, adjacency, edge_style, q_ray, num_random_seeds=self.num_random_seeds
        )

        # Build a non-planar reference scale configuration (with typical relief) so denominator scales l_k > 0
        z_ref_init = q_ray @ seeds[0] if seeds and seeds[0].numel() > 0 else torch.zeros(n, dtype=torch.float64)
        pts_scale_ref = torch.cat((xy0, z_ref_init[:, None]), dim=1)

        best_stage1_pts = pts_scale_ref.clone()
        best_stage1_score = float("inf")

        for zeta_init in seeds:
            if zeta_init.numel() == 0:
                break

            def stage1_obj_grad(z_np: np.ndarray) -> tuple[float, np.ndarray]:
                zt = torch.from_numpy(z_np).double().requires_grad_(True)
                zv = q_ray @ zt
                pts = torch.cat((xy0, zv[:, None]), dim=1)
                reg = _msda_and_isometric_energy(
                    pts,
                    edges,
                    triples,
                    target_relief_std=min_relief,
                    iso_weight=self.iso_weight / max(self.msda_weight, 1e-6),
                    relief_barrier_weight=0.50 / max(self.msda_weight, 1e-6),
                )
                total = self.msda_weight * reg
                if constraints:
                    b = compile_semantic_residuals(
                        pts,
                        constraints,
                        reference_coordinates=pts_scale_ref,
                        constraint_weights=constraint_weights,
                    )
                    if b.records:
                        total = total + b.objective
                g = torch.autograd.grad(total, zt)[0]
                return float(total.detach().item()), g.detach().numpy()

            sol = minimize(
                stage1_obj_grad,
                zeta_init.numpy().copy(),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.stage1_lbfgs_iter, "ftol": 1e-10},
            )
            cand_z = q_ray @ torch.from_numpy(sol.x).double()
            cand_pts = torch.cat((xy0, cand_z[:, None]), dim=1)
            if float(sol.fun) < best_stage1_score:
                best_stage1_score = float(sol.fun)
                best_stage1_pts = cand_pts

        # Stage 2: Exact coupled sub-pixel Levenberg-Marquardt polishing onto M(C)
        # Strictly self-contained: freeze denominator scales at Stage-1 solution (zero target leakage).
        cur = best_stage1_pts.clone()
        opt_ref = best_stage1_pts.detach().clone()
        scale_px = float(cam[0, 0].item())
        max_xy_coord = self.max_pixel_drift_px / max(scale_px, 1e-6)
        w_xy_sqrt = 12.0  # penalize xy pixel movement so depth absorbs residuals first

        if constraints:
            b_cur = compile_semantic_residuals(
                cur, constraints, reference_coordinates=opt_ref, constraint_weights=constraint_weights
            )
            if b_cur.records:
                for _ in range(self.stage2_lm_iter):
                    rmax = float(b_cur.raw_constraint_rms.max().item())
                    if rmax <= self.semantic_tolerance:
                        break
                    r_vec = b_cur.vector.detach()
                    rn2 = float(r_vec.square().sum().item())
                    if rn2 <= 1e-24:
                        break

                    xy_cur = cur[:, :2].detach()
                    z_mean = cur[:, 2].mean().detach()
                    zeta_cur = (q_ray.T @ (cur[:, 2].detach() - z_mean)).detach()

                    def res_coupled(u_param: torch.Tensor) -> torch.Tensor:
                        d_xy = (u_param[: 2 * n] / w_xy_sqrt).reshape(n, 2)
                        d_zeta = u_param[2 * n :]
                        pts_u = torch.cat(
                            (xy_cur + d_xy, (z_mean + q_ray @ (zeta_cur + d_zeta))[:, None]),
                            dim=1,
                        )
                        return compile_semantic_residuals(
                            pts_u, constraints, reference_coordinates=opt_ref, constraint_weights=constraint_weights
                        ).vector

                    u0 = torch.zeros(2 * n + (n - 1), dtype=torch.float64, requires_grad=True)
                    jac = torch.autograd.functional.jacobian(res_coupled, u0, vectorize=True).detach()
                    mu = min(max(1e-2 * rn2, 1e-9), 10.0)
                    jtj = jac.T @ jac + mu * torch.eye(jac.shape[1], dtype=torch.float64)
                    try:
                        du = -torch.linalg.solve(jtj, jac.T @ r_vec)
                    except RuntimeError:
                        du = -torch.linalg.lstsq(jtj, jac.T @ r_vec).solution

                    cur_obj = float(r_vec.square().sum().item())
                    accepted_step = False
                    for alpha in (1.0, 0.5, 0.25, 0.1):
                        d_xy = (alpha * du[: 2 * n] / w_xy_sqrt).reshape(n, 2)
                        xy_new = xy_cur + d_xy
                        # Enforce <= max_pixel_drift_px around initial xy0
                        drift_xy = xy_new - xy0
                        drift_norms = torch.linalg.norm(drift_xy, dim=1, keepdim=True).clamp_min(1e-12)
                        clamp_scale = torch.clamp(max_xy_coord / drift_norms, max=1.0)
                        xy_new = xy0 + drift_xy * clamp_scale
                        z_new = z_mean + q_ray @ (zeta_cur + alpha * du[2 * n :])
                        cand_try = torch.cat((xy_new, z_new[:, None]), dim=1)
                        b_try = compile_semantic_residuals(
                            cand_try, constraints, reference_coordinates=opt_ref, constraint_weights=constraint_weights
                        )
                        if float(b_try.vector.square().sum().item()) < cur_obj:
                            cur = cand_try
                            b_cur = b_try
                            accepted_step = True
                            break
                    if not accepted_step:
                        break

        cur = _orient_hemisphere_by_occlusion(cur, px_d, adjacency, edge_style)
        eval_ref = reference_for_eval.detach().cpu().double() if reference_for_eval is not None else opt_ref
        b_fin = compile_semantic_residuals(
            cur, constraints, reference_coordinates=eval_ref, constraint_weights=constraint_weights
        )
        max_rms = float(b_fin.raw_constraint_rms.max().item()) if b_fin.records else 0.0
        reproj_px = float(
            torch.sqrt(
                torch.mean(
                    torch.sum(
                        ((cur[:, :2] * scale_px + cam[:, 3].unsqueeze(0)) - px_d).square(),
                        dim=1,
                    )
                )
            ).item()
        )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return ClassicalLineDrawingResultV9(
            coordinates=cur,
            feasible=bool(max_rms <= self.semantic_tolerance and reproj_px <= self.max_pixel_drift_px + 1e-6),
            max_constraint_rms=max_rms,
            reprojection_rmse_px=reproj_px,
            latency_ms=lat_ms,
            method="lipson_shpitalni_1996",
        )
