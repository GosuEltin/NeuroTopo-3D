"""Unified Iterative Normal-Flow Row-Space Levenberg-Marquardt & Multi-Branch Necker Projector (V9.0).

Implements the Q1 Journal algorithmic contributions:
1. Camera-Anchored Normal-Flow Row-Space Levenberg-Marquardt (NF-RSLM):
   Eliminates the viewing-ray depth nullspace ker_tau(J_bar_z) algebraically at EVERY iteration
   (V_ker^T delta_zeta^(k) == 0 to machine precision, Theorem 3), achieving local quadratic
   convergence onto M(C) with purely second-order O(||delta_z_row||^2) curvature drift.
2. Multi-Branch Discrete Necker & Single-Vertex Reflection Continuation:
   Escapes singular planar folds (z = z_bar) and local apex inversions on depth-rigid geometries
   (such as near-degenerate Tetrahedra and Orthogonal Projections) without relaxing safety gates.
3. Physical Solid-Over-Dashed 2D Ray-Crossing Disambiguation (Proposition 2):
   Selects the physically valid SO(3) depth hemisphere using foreground-solid over background-dashed
   transverse 2D crossing orderings when available, falling back to neural anchor correlation.
4. Parametric Nullspace CAD Family Explorer:
   Extracts an orthonormal basis of ker(J_bar_z) and traces exact nonlinear manifold hypotheses
   V(eta) preserving 0.0 px image reprojection and r_max <= 1.1e-3 for interactive CAD exploration.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from src.lifting.anonymous_contract import (
    EDGE_STYLE_DASHED,
    EDGE_STYLE_SOLID,
    IndexedConstraint,
)
from src.lifting.camera_gauge_v51 import snap_coordinates_to_observed_image_plane
from src.reasoning.nullspace_guarded_projector_v843 import (
    align_depth_hemisphere_v843,
    depth_jacobian_nullity_v843,
)
from src.reasoning.projector_v6 import FeasibilityProjectorConfigV6
from src.reasoning.residuals_v4 import compile_semantic_residuals
from src.reasoning.shared_anchor_projector_v827 import (
    SharedAnchorProjectorV827,
    SharedAnchorResultV827,
)


@dataclass(frozen=True)
class NormalFlowProjectorConfigV9:
    semantic_tolerance: float = 1.1e-3
    xy_stiffness_weight: float = 1.0
    max_nf_iterations: int = 28
    damping_scale: float = 1e-2
    min_damping: float = 1e-11
    max_damping: float = 1e2
    tau_svd: float = 1e-3
    relative_rank_tol: float = 1e-2
    step_tau_svd: float = 1e-4
    step_ray_rel_tol: float = 1e-4
    base_stage1_max_rms: float = 0.15
    base_stage1_max_coord: float = 0.35
    # Regime displacement & nullspace drift gates
    underconstrained_max_row_rms: float = 0.28
    underconstrained_max_null_rms: float = 0.035
    underconstrained_max_coord: float = 0.55
    rigid_max_rms: float = 0.65
    rigid_max_coord: float = 1.35
    ood_max_row_rms: float = 0.95
    ood_max_null_rms: float = 0.12
    ood_max_coord: float = 1.95
    max_reprojection_increase_px: float = 1.0
    min_rigid_singular_value: float = 1e-3
    monotone_stage1_anchor: bool = False
    enable_multibranch_reflection: bool = True
    enable_crossing_occlusion_selector: bool = True
    enable_ood_continuation: bool = False
    enable_slsqp_fallback_on_rigid: bool = True
    occlusion_crossing_margin_threshold: float = 0.02


def build_ray_orthonormal_basis(n: int) -> torch.Tensor:
    """Deterministic orthonormal basis Q_ray in R^{N x (N-1)} of 1_N^perp (Helmert matrix)."""
    if n <= 1:
        return torch.zeros((n, 0), dtype=torch.float64)
    q = torch.zeros((n, n - 1), dtype=torch.float64)
    for k in range(1, n):
        denom = math.sqrt(k * (k + 1))
        q[:k, k - 1] = 1.0 / denom
        q[k, k - 1] = -float(k) / denom
    return q


def find_transverse_solid_dashed_crossings(
    points_2d: np.ndarray,
    adjacency: np.ndarray,
    edge_style: np.ndarray,
    margin: float = 0.06,
) -> list[tuple[int, int, int, int, float, float]]:
    """Return interior transverse 2D crossings (i, j, p, q, s, t) between solid (i,j) and dashed (p,q)."""
    n = len(points_2d)
    solid_edges = []
    dashed_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0.5:
                if int(edge_style[i, j]) == EDGE_STYLE_SOLID:
                    solid_edges.append((i, j))
                elif int(edge_style[i, j]) == EDGE_STYLE_DASHED:
                    dashed_edges.append((i, j))

    crossings = []
    for i, j in solid_edges:
        p1, p2 = points_2d[i], points_2d[j]
        d1 = p2 - p1
        for p, q in dashed_edges:
            if p in (i, j) or q in (i, j):
                continue
            q1, q2 = points_2d[p], points_2d[q]
            d2 = q2 - q1
            det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
            if abs(det) < 1e-6:
                continue
            rhs = q1 - p1
            s = (rhs[0] * (-d2[1]) - rhs[1] * (-d2[0])) / det
            t = (d1[0] * rhs[1] - d1[1] * rhs[0]) / det
            if margin < s < (1.0 - margin) and margin < t < (1.0 - margin):
                crossings.append((i, j, p, q, float(s), float(t)))
    return crossings


def score_physical_occlusion_hemisphere(
    candidate: torch.Tensor,
    crossings: list[tuple[int, int, int, int, float, float]],
) -> float:
    """Compute mean depth gap z_solid(s) - z_dashed(t) (+z points toward observer)."""
    if not crossings:
        return 0.0
    z = candidate[:, 2].detach().cpu().double().numpy()
    gaps = [
        ((1.0 - s) * z[i] + s * z[j]) - ((1.0 - t) * z[p] + t * z[q])
        for (i, j, p, q, s, t) in crossings
    ]
    return float(np.mean(gaps))


def select_optimal_hemisphere_v9(
    candidate: torch.Tensor,
    raw: torch.Tensor,
    constraints: Sequence[IndexedConstraint],
    *,
    semantic_tolerance: float = 1.1e-3,
    crossings: list[tuple[int, int, int, int, float, float]] | None = None,
    use_crossing_selector: bool = True,
    crossing_margin_threshold: float = 0.02,
    viewer_z_sign: float = 1.0,
) -> tuple[torch.Tensor, bool, str]:
    """Select between candidate and its depth reflection around mean z."""
    cand_d = candidate.detach().cpu().double()
    raw_d = raw.detach().cpu().double()
    z_mean = cand_d[:, 2].mean()
    z_centered = cand_d[:, 2] - z_mean

    flipped = cand_d.clone()
    flipped[:, 2] = z_mean - z_centered

    # Check if flipped is also semantically valid
    b_flip = compile_semantic_residuals(flipped, constraints, reference_coordinates=raw_d)
    flip_valid = bool(
        b_flip.records and float(b_flip.raw_constraint_rms.max().item()) <= semantic_tolerance
    )
    if not flip_valid:
        return candidate, False, "only_primary_valid"

    # Check physical crossing order if crossings exist AND neural anchor agrees on viewer chart sign
    if use_crossing_selector and crossings:
        raw_occ = score_physical_occlusion_hemisphere(raw_d, crossings)
        cand_occ = score_physical_occlusion_hemisphere(cand_d, crossings)
        flip_occ = score_physical_occlusion_hemisphere(flipped, crossings)
        # Viewer-canonical convention in V833/V840 has +z toward observer -> solid in front of dashed means occ > 0
        target_sign = viewer_z_sign if abs(raw_occ) < 1e-4 else (1.0 if raw_occ >= 0.0 else -1.0)
        if target_sign * flip_occ > crossing_margin_threshold and target_sign * cand_occ < -crossing_margin_threshold:
            ret = flipped.to(candidate.dtype)
            b_ret = compile_semantic_residuals(ret.double(), constraints, reference_coordinates=raw_d)
            if float(b_ret.raw_constraint_rms.max().item()) <= semantic_tolerance:
                return ret, True, "crossing_occlusion_flipped"
        if target_sign * cand_occ > crossing_margin_threshold:
            return candidate, False, "crossing_occlusion_kept"

    # Fallback: correlate with neural anchor depth
    aligned, did_flip = align_depth_hemisphere_v843(
        candidate, raw, constraints, semantic_tolerance=semantic_tolerance
    )
    return aligned, did_flip, ("anchor_corr_flipped" if did_flip else "anchor_corr_kept")


def decompose_depth_displacement_v9(
    candidate: torch.Tensor,
    raw: torch.Tensor,
    constraints: Sequence[IndexedConstraint],
    *,
    tau_svd: float = 1e-3,
    relative_rank_tol: float = 1e-2,
) -> dict[str, object]:
    """Compute viewing-ray Jacobian SVD at candidate and decompose Delta_z into row and null components."""
    cand_d = candidate.detach().cpu().double()
    raw_d = raw.detach().cpu().double()
    n = cand_d.shape[0]
    q_ray = build_ray_orthonormal_basis(n)
    xy = cand_d[:, :2].clone()

    def res_z(zeta: torch.Tensor) -> torch.Tensor:
        z_vec = cand_d[:, 2].mean() + q_ray @ zeta
        pts = torch.cat((xy, z_vec[:, None]), dim=1)
        return compile_semantic_residuals(pts, constraints, reference_coordinates=raw_d).vector

    zeta_cand = q_ray.T @ (cand_d[:, 2] - cand_d[:, 2].mean())
    zeta_raw = q_ray.T @ (raw_d[:, 2] - raw_d[:, 2].mean())
    delta_zeta = zeta_cand - zeta_raw

    b_cand = compile_semantic_residuals(cand_d, constraints, reference_coordinates=raw_d)
    if b_cand.vector.numel() == 0 or n <= 1:
        null_rms = float((q_ray @ delta_zeta).square().mean().sqrt().item()) if n > 1 else 0.0
        return {
            "d_null_ray": max(0, n - 1),
            "rank_ray": 0,
            "sigma_max": 0.0,
            "sigma_min_ray": 0.0,
            "row_rms": 0.0,
            "null_rms": null_rms,
            "singular_values": [],
        }

    jac_ray = torch.autograd.functional.jacobian(
        res_z, zeta_cand.clone().requires_grad_(True), vectorize=True
    ).detach()
    _, s_vals, vh = torch.linalg.svd(jac_ray, full_matrices=True)
    s_max = float(s_vals[0].item()) if s_vals.numel() else 0.0
    s_min_ray = float(s_vals[n - 2].item()) if s_vals.numel() >= (n - 1) else 0.0
    thresh = max(tau_svd, relative_rank_tol * s_max)
    r_rank = int((s_vals >= thresh).sum().item())
    d_null = int((n - 1) - r_rank)

    v_row = vh[:r_rank, :].T  # (N-1, r)
    v_ker = vh[r_rank:, :].T  # (N-1, d_null)

    dz_row = q_ray @ (v_row @ (v_row.T @ delta_zeta)) if r_rank > 0 else torch.zeros(n, dtype=torch.float64)
    dz_ker = q_ray @ (v_ker @ (v_ker.T @ delta_zeta)) if d_null > 0 else torch.zeros(n, dtype=torch.float64)

    return {
        "d_null_ray": d_null,
        "rank_ray": r_rank,
        "sigma_max": s_max,
        "sigma_min_ray": s_min_ray,
        "row_rms": float(dz_row.square().mean().sqrt().item()),
        "null_rms": float(dz_ker.square().mean().sqrt().item()),
        "singular_values": s_vals.tolist(),
    }


def extract_nullspace_cad_family(
    feasible_coordinates: torch.Tensor,
    raw_reference: torch.Tensor,
    constraints: Sequence[IndexedConstraint],
    *,
    amplitudes: Sequence[float] = (-0.20, -0.10, 0.0, 0.10, 0.20),
    mode_index: int = 0,
    tau_svd: float = 1e-3,
    relative_rank_tol: float = 1e-2,
    polish_steps: int = 8,
) -> dict[str, object]:
    """Trace an exact nonlinear manifold family V(eta) along the viewing-ray nullspace ker(J_bar_z)
    while keeping (x, y) fixed (0.0 px reprojection drift) and polishing strictly in row(J_bar_z)!
    """
    v_star = feasible_coordinates.detach().cpu().double()
    raw_d = raw_reference.detach().cpu().double()
    n = v_star.shape[0]
    q_ray = build_ray_orthonormal_basis(n)
    xy_fixed = v_star[:, :2].clone()
    z_mean = v_star[:, 2].mean()
    zeta_star = q_ray.T @ (v_star[:, 2] - z_mean)

    def res_z(zeta: torch.Tensor) -> torch.Tensor:
        z_vec = z_mean + q_ray @ zeta
        pts = torch.cat((xy_fixed, z_vec[:, None]), dim=1)
        return compile_semantic_residuals(pts, constraints, reference_coordinates=raw_d).vector

    jac0 = torch.autograd.functional.jacobian(
        res_z, zeta_star.clone().requires_grad_(True), vectorize=True
    ).detach()
    _, s_vals, vh = torch.linalg.svd(jac0, full_matrices=True)
    s_max = float(s_vals[0].item()) if s_vals.numel() else 0.0
    thresh = max(tau_svd, relative_rank_tol * s_max)
    r_rank = int((s_vals >= thresh).sum().item())
    d_null = int((n - 1) - r_rank)
    if d_null <= 0 or mode_index >= d_null:
        return {
            "d_null_ray": d_null,
            "basis_vectors_z": [],
            "members": [],
        }

    v_ker = vh[r_rank:, :].T  # (N-1, d_null)
    basis_z = (q_ray @ v_ker)  # (N, d_null), columns have unit L2 norm -> scale by sqrt(N) for RMS amplitude!
    dir_zeta = v_ker[:, mode_index] * math.sqrt(n)

    members = []
    for amp in amplitudes:
        zeta_cur = zeta_star + float(amp) * dir_zeta
        # Polish strictly orthogonal to v_ker so tangential coordinate amp is preserved!
        for _ in range(max(polish_steps, 32)):
            f_val = res_z(zeta_cur)
            if float(f_val.norm().item()) < 1e-13:
                break
            jc = torch.autograd.functional.jacobian(
                res_z, zeta_cur.clone().requires_grad_(True), vectorize=True
            ).detach()
            uc, sc, vhc = torch.linalg.svd(jc, full_matrices=True)
            ur = uc[:, :r_rank]
            sr = sc[:r_rank]
            vr = vhc[:r_rank, :].T
            mu = 1e-12 + 1e-4 * float(f_val.square().sum().item())
            dz = -vr @ ((sr / (sr.square() + mu)) * (ur.T @ f_val))
            zeta_cur = zeta_cur + dz

        pts_m = torch.cat((xy_fixed, (z_mean + q_ray @ zeta_cur)[:, None]), dim=1)
        b_m = compile_semantic_residuals(pts_m, constraints, reference_coordinates=raw_d)
        max_r = float(b_m.raw_constraint_rms.max().item()) if b_m.records else 0.0
        reproj_drift = float((pts_m[:, :2] - v_star[:, :2]).abs().max().item())
        members.append(
            {
                "amplitude": float(amp),
                "coordinates": pts_m,
                "max_constraint_rms": max_r,
                "xy_drift": reproj_drift,
            }
        )

    return {
        "d_null_ray": d_null,
        "basis_vectors_z": basis_z.T.tolist(),
        "members": members,
    }


class NormalFlowMultiBranchProjectorV9:
    """Q1 Journal Unified Normal-Flow Row-Space & Multi-Branch Reflection Projector (V9.0)."""

    def __init__(self, config: NormalFlowProjectorConfigV9 | None = None) -> None:
        self.config = config or NormalFlowProjectorConfigV9()
        if self.config.monotone_stage1_anchor:
            self._stage1_anchor = SharedAnchorProjectorV827(
                FeasibilityProjectorConfigV6(
                    semantic_tolerance=self.config.semantic_tolerance,
                    max_rms_displacement=self.config.base_stage1_max_rms,
                    max_coordinate_displacement=self.config.base_stage1_max_coord,
                    max_reprojection_increase_px=self.config.max_reprojection_increase_px,
                )
            )
        else:
            self._stage1_anchor = None
        if self.config.enable_slsqp_fallback_on_rigid:
            fb_rms = self.config.ood_max_row_rms if self.config.enable_ood_continuation else self.config.rigid_max_rms
            fb_coord = self.config.ood_max_coord if self.config.enable_ood_continuation else self.config.rigid_max_coord
            fb_tol = min(self.config.semantic_tolerance, 1e-4) if self.config.enable_ood_continuation else self.config.semantic_tolerance
            self._rigid_fallback = SharedAnchorProjectorV827(
                FeasibilityProjectorConfigV6(
                    semantic_tolerance=fb_tol,
                    max_rms_displacement=fb_rms,
                    max_coordinate_displacement=fb_coord,
                    max_reprojection_increase_px=self.config.max_reprojection_increase_px,
                )
            )
        else:
            self._rigid_fallback = None

    def _reprojection_error_px(
        self,
        xyz: torch.Tensor,
        cam: torch.Tensor | None,
        pixels: torch.Tensor | None,
    ) -> float:
        if cam is None or pixels is None:
            return 0.0
        err = xyz @ cam[:, :3].T + cam[:, 3] - pixels
        return float(err.square().sum(-1).mean().sqrt().item())

    def _clamp_xy_reprojection(
        self,
        xyz: torch.Tensor,
        cam: torch.Tensor | None,
        pixels: torch.Tensor | None,
        max_allowed_px: float,
    ) -> torch.Tensor:
        """If pixel RMSE exceeds max_allowed_px, contract (x, y) radially toward exact inverse projection."""
        if cam is None or pixels is None:
            return xyz
        cur_px = self._reprojection_error_px(xyz, cam, pixels)
        if cur_px <= max_allowed_px or cur_px <= 1e-12:
            return xyz
        snapped = snap_coordinates_to_observed_image_plane(xyz, pixels, cam)
        ratio = max(0.0, (max_allowed_px - 1e-6) / cur_px)
        out = xyz.clone()
        out[:, :2] = snapped[:, :2] + ratio * (xyz[:, :2] - snapped[:, :2])
        return out

    def _solve_normal_flow_branch(
        self,
        start_xyz: torch.Tensor,
        raw_xyz: torch.Tensor,
        constraints: Sequence[IndexedConstraint],
        weights: torch.Tensor | None,
        cam: torch.Tensor | None,
        pixels: torch.Tensor | None,
        before_px: float,
    ) -> tuple[torch.Tensor, float, int, float]:
        """Run Iterative Normal-Flow Row-Space Levenberg-Marquardt from start_xyz.

        Returns (best_xyz, best_max_rms, iterations_used, max_stepwise_null_velocity).
        """
        cfg = self.config
        n = start_xyz.shape[0]
        q_ray = build_ray_orthonormal_basis(n)
        inv_sqrt_wxy = 1.0 / math.sqrt(cfg.xy_stiffness_weight)
        max_allowed_px = before_px + cfg.max_reprojection_increase_px

        cur = self._clamp_xy_reprojection(start_xyz.clone(), cam, pixels, max_allowed_px)
        z_bar = cur[:, 2].mean()
        zeta = q_ray.T @ (cur[:, 2] - z_bar)
        xy = cur[:, :2].clone()

        def eval_bundle(xy_t: torch.Tensor, zeta_t: torch.Tensor):
            pts = torch.cat((xy_t, (z_bar + q_ray @ zeta_t)[:, None]), dim=1)
            return pts, compile_semantic_residuals(
                pts, constraints, reference_coordinates=raw_xyz, constraint_weights=weights
            )

        pts_cur, b_cur = eval_bundle(xy, zeta)
        if not b_cur.records:
            return pts_cur, 0.0, 0, 0.0

        best_pts = pts_cur.clone()
        best_max_rms = float(b_cur.raw_constraint_rms.max().item())
        nu_scale = cfg.damping_scale
        max_null_vel = 0.0
        total_iters = 0

        # Pure Viewing-Ray Normal-Flow ONLY when strict 0.0 px reprojection preservation is requested
        if n > 1 and cfg.max_reprojection_increase_px <= 1e-9:
            for _ in range(cfg.max_nf_iterations):
                if best_max_rms <= cfg.semantic_tolerance:
                    break
                f_vec = b_cur.vector.detach()
                f_norm_sq = float(f_vec.square().sum().item())
                if f_norm_sq <= 1e-24:
                    break
                total_iters += 1

                def res_z_only(zeta_s: torch.Tensor) -> torch.Tensor:
                    pts_s = torch.cat((xy, (z_bar + q_ray @ zeta_s)[:, None]), dim=1)
                    return compile_semantic_residuals(
                        pts_s, constraints, reference_coordinates=raw_xyz, constraint_weights=weights
                    ).vector

                j_ray = torch.autograd.functional.jacobian(
                    res_z_only, zeta.clone().requires_grad_(True), vectorize=True
                ).detach()
                if j_ray.numel() == 0:
                    break
                u_z, s_z, vh_z = torch.linalg.svd(j_ray, full_matrices=True)
                sz_max = float(s_z[0].item()) if s_z.numel() else 0.0
                if sz_max <= 1e-12:
                    break
                z_thresh = max(cfg.step_tau_svd, cfg.step_ray_rel_tol * sz_max)
                r_z = int((s_z >= z_thresh).sum().item())
                if r_z == 0:
                    break
                u_row = u_z[:, :r_z]
                s_row = s_z[:r_z]
                v_row_z = vh_z[:r_z, :].T
                v_ker_z = vh_z[r_z:, :].T

                mu_k = min(cfg.max_damping, max(cfg.min_damping, nu_scale * f_norm_sq))
                d_zeta = -v_row_z @ ((s_row / (s_row.square() + mu_k)) * (u_row.T @ f_vec))
                if v_ker_z.shape[1] > 0:
                    d_zeta = d_zeta - v_ker_z @ (v_ker_z.T @ d_zeta)
                    max_null_vel = max(max_null_vel, float((v_ker_z.T @ d_zeta).abs().max().item()))

                accepted_z = False
                for alpha in (1.0, 0.5, 0.25, 0.125):
                    cand_zeta = zeta + alpha * d_zeta
                    cand_pts, b_cand = eval_bundle(xy, cand_zeta)
                    cand_obj = float(b_cand.vector.square().sum().item())
                    cand_max_rms = float(b_cand.raw_constraint_rms.max().item()) if b_cand.records else 0.0
                    if cand_max_rms < best_max_rms:
                        best_max_rms = cand_max_rms
                        best_pts = cand_pts.clone()
                    if cand_obj < f_norm_sq * (1.0 - 1e-4 * alpha) or cand_max_rms <= cfg.semantic_tolerance:
                        zeta = cand_zeta
                        pts_cur = cand_pts
                        b_cur = b_cand
                        nu_scale = max(1e-4, nu_scale * 0.5)
                        accepted_z = True
                        break
                if not accepted_z:
                    nu_scale = min(1e3, nu_scale * 8.0)
                    if nu_scale >= 1e3:
                        break
            return best_pts, best_max_rms, total_iters, max_null_vel

        # Coupled Sub-Pixel (x, y) + Viewing-Ray Row-Space Normal-Flow:
        # Eliminates viewing-ray nullspace ker_tau(J_bar_z) at EVERY iteration (V_ker_z^T d_zeta == 0)
        # while keeping ALL active constraint directions down to 1e-8 * sigma_max!
        target_stop_tol = min(cfg.semantic_tolerance, 1e-5) if cfg.enable_ood_continuation else cfg.semantic_tolerance
        eff_max_iters = max(cfg.max_nf_iterations, 42) if cfg.enable_ood_continuation else cfg.max_nf_iterations
        for _ in range(eff_max_iters):
            if best_max_rms <= target_stop_tol:
                break

            f_vec = b_cur.vector.detach()
            f_norm_sq = float(f_vec.square().sum().item())
            if f_norm_sq <= 1e-24:
                break
            total_iters += 1

            def res_state(state_vec: torch.Tensor) -> torch.Tensor:
                xy_s = state_vec[: 2 * n].reshape(n, 2)
                zeta_s = state_vec[2 * n :]
                pts_s = torch.cat((xy_s, (z_bar + q_ray @ zeta_s)[:, None]), dim=1)
                return compile_semantic_residuals(
                    pts_s, constraints, reference_coordinates=raw_xyz, constraint_weights=weights
                ).vector

            state0 = torch.cat((xy.reshape(-1), zeta), dim=0).requires_grad_(True)
            jac_full = torch.autograd.functional.jacobian(res_state, state0, vectorize=True).detach()
            j_xy = jac_full[:, : 2 * n]
            j_ray = jac_full[:, 2 * n :]

            if n > 1 and j_ray.numel() > 0:
                _, s_z, vh_z = torch.linalg.svd(j_ray, full_matrices=True)
                sz_max = float(s_z[0].item()) if s_z.numel() else 0.0
                z_thresh = max(cfg.step_tau_svd, cfg.step_ray_rel_tol * sz_max)
                r_z = int((s_z >= z_thresh).sum().item())
                v_row_z = vh_z[:r_z, :].T
                v_ker_z = vh_z[r_z:, :].T
            else:
                r_z = 0
                v_row_z = torch.zeros((max(0, n - 1), 0), dtype=torch.float64)
                v_ker_z = torch.eye(max(0, n - 1), dtype=torch.float64)

            j_red = torch.cat((inv_sqrt_wxy * j_xy, j_ray @ v_row_z), dim=1)
            u_w, s_w, vh_w = torch.linalg.svd(j_red, full_matrices=False)
            sw_max = float(s_w[0].item()) if s_w.numel() else 0.0
            if sw_max <= 1e-12:
                break
            keep_w = s_w >= (1e-8 * sw_max)
            u_act = u_w[:, keep_w]
            s_act = s_w[keep_w]
            v_act = vh_w[keep_w, :].T

            mu_k = min(cfg.max_damping, max(cfg.min_damping, nu_scale * f_norm_sq))
            du = -v_act @ ((s_act / (s_act.square() + mu_k)) * (u_act.T @ f_vec))

            d_xy = (inv_sqrt_wxy * du[: 2 * n]).reshape(n, 2)
            d_xi_z = du[2 * n :]
            d_zeta = v_row_z @ d_xi_z if r_z > 0 else torch.zeros_like(zeta)

            if v_ker_z.shape[1] > 0 and d_zeta.numel() > 0:
                d_zeta = d_zeta - v_ker_z @ (v_ker_z.T @ d_zeta)
                null_vel = float((v_ker_z.T @ d_zeta).abs().max().item())
                max_null_vel = max(max_null_vel, null_vel)

            accepted_step = False
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                cand_xy = xy + alpha * d_xy
                cand_zeta = zeta + alpha * d_zeta
                cand_pts_raw = torch.cat((cand_xy, (z_bar + q_ray @ cand_zeta)[:, None]), dim=1)
                cand_pts = self._clamp_xy_reprojection(cand_pts_raw, cam, pixels, max_allowed_px)
                cand_xy_clamped = cand_pts[:, :2]
                _, b_cand = eval_bundle(cand_xy_clamped, cand_zeta)
                cand_obj = float(b_cand.vector.square().sum().item())
                cand_max_rms = float(b_cand.raw_constraint_rms.max().item()) if b_cand.records else 0.0

                if cand_max_rms < best_max_rms:
                    best_max_rms = cand_max_rms
                    best_pts = cand_pts.clone()

                    if cand_obj < f_norm_sq * (1.0 - 1e-4 * alpha) or cand_max_rms <= target_stop_tol:
                        xy = cand_xy_clamped
                        zeta = cand_zeta
                        pts_cur = cand_pts
                        b_cur = b_cand
                        nu_scale = max(1e-4, nu_scale * 0.5)
                        accepted_step = True
                        break

            if not accepted_step:
                nu_scale = min(1e3, nu_scale * 8.0)
                if nu_scale >= 1e3:
                    break

        # Terminal KKT Nullspace Retraction under severe OOD shift (when d_null_ray >= 1):
        # Removes second-order path curvature accumulation in ker(J_ray(zeta_NF)) and polishes along row(J_ray).
        if cfg.enable_ood_continuation and best_max_rms <= cfg.semantic_tolerance and n > 1:
            zeta_raw0 = q_ray.T @ (raw_xyz[:, 2] - raw_xyz[:, 2].mean())
            zeta_nf = q_ray.T @ (best_pts[:, 2] - z_bar)
            xy_nf = best_pts[:, :2].clone()
            for _ in range(2):
                def _res_only_z(zv: torch.Tensor) -> torch.Tensor:
                    return compile_semantic_residuals(
                        torch.cat((xy_nf, (z_bar + q_ray @ zv)[:, None]), dim=1),
                        constraints,
                        reference_coordinates=raw_xyz,
                        constraint_weights=weights,
                    ).vector

                jz_t = torch.autograd.functional.jacobian(
                    _res_only_z, zeta_nf.clone().requires_grad_(True), vectorize=True
                ).detach()
                if jz_t.numel() == 0:
                    break
                uz_t, sz_t, vhz_t = torch.linalg.svd(jz_t, full_matrices=True)
                smax_t = float(sz_t[0].item()) if sz_t.numel() else 0.0
                rk_t = int((sz_t >= max(cfg.tau_svd, cfg.relative_rank_tol * smax_t)).sum().item())
                if rk_t <= 0 or rk_t >= (n - 1):
                    break
                v_ker_t = vhz_t[rk_t:, :].T
                null_drift_vec = v_ker_t @ (v_ker_t.T @ (zeta_nf - zeta_raw0))
                if float((q_ray @ null_drift_vec).square().mean().sqrt().item()) <= cfg.ood_max_null_rms * 0.5:
                    break
                zeta_ret = zeta_nf - 0.85 * null_drift_vec
                for _ in range(5):
                    fv = _res_only_z(zeta_ret)
                    if float(fv.norm().item()) <= 1e-9:
                        break
                    jp = torch.autograd.functional.jacobian(
                        _res_only_z, zeta_ret.clone().requires_grad_(True), vectorize=True
                    ).detach()
                    up, sp, vhp = torch.linalg.svd(jp, full_matrices=True)
                    ur_p, sr_p, vr_p = up[:, :rk_t], sp[:rk_t], vhp[:rk_t, :].T
                    mu_p = 1e-10 + 1e-3 * float(fv.square().sum().item())
                    zeta_ret = zeta_ret - vr_p @ ((sr_p / (sr_p.square() + mu_p)) * (ur_p.T @ fv))
                pts_ret, b_ret = eval_bundle(xy_nf, zeta_ret)
                ret_rms = float(b_ret.raw_constraint_rms.max().item()) if b_ret.records else 0.0
                if ret_rms <= cfg.semantic_tolerance:
                    zeta_nf = zeta_ret
                    best_pts = pts_ret.clone()
                    best_max_rms = ret_rms

        return best_pts, best_max_rms, total_iters, max_null_vel

    def _generate_reflection_seeds(self, raw_d: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Generate discrete reflection continuation seeds to escape z=0 singular folds."""
        z = raw_d[:, 2]
        z_bar = z.mean()
        zc = z - z_bar
        seeds: list[tuple[str, torch.Tensor]] = []

        # Seed 1: Global Necker reflection -zc
        s_necker = raw_d.clone()
        s_necker[:, 2] = z_bar - zc
        seeds.append(("global_necker_flip", s_necker))

        # Seed 2: Relief-amplified seeds (+/- 1.6x) to escape near-flat z ~ 0 stalls
        z_std = float(zc.square().mean().sqrt().item())
        if z_std < 0.25:
            scale_up = 0.35 / max(z_std, 1e-3)
            s_amp_pos = raw_d.clone()
            s_amp_pos[:, 2] = z_bar + min(scale_up, 3.0) * zc
            seeds.append(("relief_amp_pos", s_amp_pos))
            s_amp_neg = raw_d.clone()
            s_amp_neg[:, 2] = z_bar - min(scale_up, 3.0) * zc
            seeds.append(("relief_amp_neg", s_amp_neg))

        # Seed 3..: Single-vertex apex/foot reflections across z_bar
        n = raw_d.shape[0]
        max_vert_n = 16 if self.config.enable_ood_continuation else 11
        if n <= max_vert_n:
            # Prioritize vertices with largest |z_i - z_bar| or smallest |z_i - z_bar|
            order = torch.argsort(zc.abs(), descending=True).tolist()
            for v_idx in order[: min(n, 6)]:
                sv = raw_d.clone()
                sv[:, 2] = z.clone()
                sv[v_idx, 2] = 2.0 * z_bar - z[v_idx]
                # Re-center depth mean to z_bar
                sv[:, 2] = sv[:, 2] - sv[:, 2].mean() + z_bar
                seeds.append((f"vertex_flip_{v_idx}", sv))

        # Seed 4..: Canonical axonometric oblique depth-ramp continuation for unseen OOD / CAD topologies
        if self.config.enable_ood_continuation and n >= 6:
            ux = raw_d[:, 0] - raw_d[:, 0].mean()
            uy = raw_d[:, 1] - raw_d[:, 1].mean()
            ux_s = ux / ux.square().mean().sqrt().clamp_min(1e-6)
            uy_s = uy / uy.square().mean().sqrt().clamp_min(1e-6)
            for sx, sy in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
                sr = raw_d.clone()
                ramp = 0.42 * (sx * ux_s + sy * uy_s) + 0.15 * zc
                sr[:, 2] = z_bar + (ramp - ramp.mean())
                seeds.append((f"axon_ramp_{int(sx)}_{int(sy)}", sr))

        return seeds

    def project(
        self,
        raw_coordinates: torch.Tensor,
        constraints: Sequence[IndexedConstraint],
        *,
        warm_start: torch.Tensor | None = None,
        camera: torch.Tensor | None = None,
        points_2d: torch.Tensor | None = None,
        constraint_confidences=None,
        adjacency: torch.Tensor | None = None,
        edge_style: torch.Tensor | None = None,
    ) -> SharedAnchorResultV827:
        started = time.perf_counter()
        cfg = self.config
        raw = raw_coordinates.detach().cpu().double()
        start0 = raw if warm_start is None else warm_start.detach().cpu().double()
        cam = None if camera is None else camera.detach().cpu().double()
        pixels = None if points_2d is None else points_2d.detach().cpu().double()
        weights = (
            None
            if constraint_confidences is None
            else torch.as_tensor(constraint_confidences, dtype=torch.float64).cpu()
        )

        init_bundle = compile_semantic_residuals(
            raw, constraints, reference_coordinates=raw, constraint_weights=weights
        )
        informative = len(init_bundle.records)
        raw_max_res = float(init_bundle.raw_constraint_rms.max().item()) if informative else 0.0
        before_px = self._reprojection_error_px(raw, cam, pixels)

        if informative == 0:
            return SharedAnchorResultV827(
                raw_coordinates.clone(),
                True,
                "no_informative_constraints",
                {
                    "runtime_seconds": time.perf_counter() - started,
                    "regime": "abstained_no_constraints",
                    "raw_maximum_residual": 0.0,
                },
            )
        if raw_max_res <= cfg.semantic_tolerance:
            return SharedAnchorResultV827(
                raw_coordinates.clone(),
                False,
                "already_feasible",
                {
                    "runtime_seconds": time.perf_counter() - started,
                    "regime": "already_feasible",
                    "raw_maximum_residual": raw_max_res,
                    "displacement_rms": 0.0,
                    "nullspace_drift_rms": 0.0,
                },
            )

        # Optional Stage 0: Exact Stage-1 Monotonicity Anchor (r <= 0.15)
        if self._stage1_anchor is not None:
            res_s1 = self._stage1_anchor.project(
                raw_coordinates,
                constraints,
                warm_start=warm_start,
                camera=camera,
                points_2d=points_2d,
                constraint_confidences=constraint_confidences,
            )
            if not res_s1.abstained or res_s1.status != "no_safe_feasible_candidate":
                diag_s1 = dict(res_s1.diagnostics)
                diag_s1["regime"] = "base_r015"
                diag_s1["zero_nullity_rescued"] = False
                diag_s1["nullspace_drift_rms"] = 0.0
                return SharedAnchorResultV827(res_s1.coordinates, res_s1.abstained, res_s1.status, diag_s1)

        # Precompute transverse 2D solid-over-dashed crossings if graph style is provided
        crossings: list[tuple[int, int, int, int, float, float]] = []
        if (
            cfg.enable_crossing_occlusion_selector
            and adjacency is not None
            and edge_style is not None
        ):
            pts2d_for_cross = (
                pixels.numpy() if pixels is not None else raw[:, :2].numpy()
            )
            crossings = find_transverse_solid_dashed_crossings(
                pts2d_for_cross,
                adjacency.detach().cpu().double().numpy(),
                edge_style.detach().cpu().numpy(),
            )

        def certify_candidate(
            cand_pts: torch.Tensor, branch_tag: str, iters: int, null_vel: float
        ) -> tuple[bool, torch.Tensor, dict[str, object]]:
            aligned, flipped, flip_reason = select_optimal_hemisphere_v9(
                cand_pts,
                raw,
                constraints,
                semantic_tolerance=cfg.semantic_tolerance,
                crossings=crossings,
                use_crossing_selector=cfg.enable_crossing_occlusion_selector,
                crossing_margin_threshold=cfg.occlusion_crossing_margin_threshold,
            )
            # Screen in target output dtype (exact float32/float64 round-trip safety!)
            ret_typed = aligned.to(raw_coordinates.dtype)
            cand_screen = ret_typed.double().cpu()
            b_fin = compile_semantic_residuals(
                cand_screen, constraints, reference_coordinates=raw, constraint_weights=weights
            )
            fin_res = float(b_fin.raw_constraint_rms.max().item()) if b_fin.records else 0.0
            if fin_res > cfg.semantic_tolerance:
                return False, raw, {"reject_reason": "residual_above_tol", "fin_res": fin_res}

            after_px = self._reprojection_error_px(cand_screen, cam, pixels)
            px_inc = after_px - before_px
            if px_inc > cfg.max_reprojection_increase_px + 1e-8:
                return False, raw, {"reject_reason": "reprojection_exceeded", "px_inc": px_inc}

            delta = cand_screen - raw
            disp_rms = float(delta.square().sum(-1).mean().sqrt().item())
            coord_max = float(delta.abs().max().item())

            spec = decompose_depth_displacement_v9(
                cand_screen,
                raw,
                constraints,
                tau_svd=cfg.tau_svd,
                relative_rank_tol=cfg.relative_rank_tol,
            )
            d_null = int(spec["d_null_ray"])
            s_min = float(spec["sigma_min_ray"])
            row_rms = float(spec["row_rms"])
            null_rms = float(spec["null_rms"])

            # Regime-specific spectral safety gates:
            if d_null == 0 and s_min >= cfg.min_rigid_singular_value:
                max_rms_cap = cfg.ood_max_row_rms if cfg.enable_ood_continuation else cfg.rigid_max_rms
                max_coord_cap = cfg.ood_max_coord if cfg.enable_ood_continuation else cfg.rigid_max_coord
                if disp_rms <= max_rms_cap + 1e-9 and coord_max <= max_coord_cap + 1e-9:
                    return True, ret_typed, {
                        "regime": f"v9_rigid_d0_{branch_tag}",
                        "zero_nullity_rescued": disp_rms > 0.15,
                        "hemisphere_flipped": flipped,
                        "hemisphere_reason": flip_reason,
                        "d_null_ray": 0,
                        "sigma_min_ray": s_min,
                        "displacement_rms": disp_rms,
                        "row_rms": row_rms,
                        "nullspace_drift_rms": null_rms,
                        "coordinate_max": coord_max,
                        "reprojection_increase_px": px_inc,
                        "nf_iterations": iters,
                        "max_stepwise_null_velocity": null_vel,
                        "final_max_residual": fin_res,
                    }
            else:
                # Under-constrained (d_null >= 1) OR near-singular rigid:
                # Safe ONLY if tangential nullspace drift is tightly bounded AND row-space is within cap!
                max_row_cap = cfg.ood_max_row_rms if cfg.enable_ood_continuation else cfg.underconstrained_max_row_rms
                max_null_cap = cfg.ood_max_null_rms if cfg.enable_ood_continuation else cfg.underconstrained_max_null_rms
                max_coord_cap = cfg.ood_max_coord if cfg.enable_ood_continuation else cfg.underconstrained_max_coord
                if (
                    row_rms <= max_row_cap + 1e-9
                    and null_rms <= max_null_cap + 1e-9
                    and disp_rms <= max_row_cap + 0.05
                    and coord_max <= max_coord_cap + 1e-9
                ):
                    return True, ret_typed, {
                        "regime": f"v9_rowspace_d{d_null}_{branch_tag}",
                        "zero_nullity_rescued": False,
                        "rowspace_rescued_beyond_015": disp_rms > 0.15,
                        "hemisphere_flipped": flipped,
                        "hemisphere_reason": flip_reason,
                        "d_null_ray": d_null,
                        "sigma_min_ray": s_min,
                        "displacement_rms": disp_rms,
                        "row_rms": row_rms,
                        "nullspace_drift_rms": null_rms,
                        "coordinate_max": coord_max,
                        "reprojection_increase_px": px_inc,
                        "nf_iterations": iters,
                        "max_stepwise_null_velocity": null_vel,
                        "final_max_residual": fin_res,
                    }

            return False, raw, {
                "reject_reason": "spectral_bounds_exceeded",
                "d_null_ray": d_null,
                "sigma_min_ray": s_min,
                "disp_rms": disp_rms,
                "row_rms": row_rms,
                "null_rms": null_rms,
            }

        # Stage 1: Primary Normal-Flow Row-Space solve from neural anchor
        cand0, res0, it0, nvel0 = self._solve_normal_flow_branch(
            start0, raw, constraints, weights, cam, pixels, before_px
        )
        if res0 <= cfg.semantic_tolerance:
            ok, out_coords, diag = certify_candidate(cand0, "primary", it0, nvel0)
            if ok:
                diag["runtime_seconds"] = time.perf_counter() - started
                diag["raw_maximum_residual"] = raw_max_res
                return SharedAnchorResultV827(
                    out_coords.to(raw_coordinates.device), False, "feasible_update_v9_primary", diag
                )

        # Stage 2: Multi-Branch Discrete Necker & Single-Vertex Reflection Continuation
        # Only allow non-trivial reflection continuation on depth-rigid (d_null == 0) or 1D-coupled (d_null == 1) topologies
        best_valid_choice: tuple[float, torch.Tensor, dict[str, object]] | None = None
        if cfg.enable_multibranch_reflection:
            for seed_tag, seed_xyz in self._generate_reflection_seeds(raw):
                cand_b, res_b, it_b, nvel_b = self._solve_normal_flow_branch(
                    seed_xyz, raw, constraints, weights, cam, pixels, before_px
                )
                if res_b <= cfg.semantic_tolerance:
                    ok_b, out_b, diag_b = certify_candidate(cand_b, seed_tag, it_b, nvel_b)
                    if ok_b and int(diag_b["d_null_ray"]) <= 1:
                        score_b = float(diag_b["displacement_rms"])
                        if best_valid_choice is None or score_b < best_valid_choice[0]:
                            best_valid_choice = (score_b, out_b, diag_b)
                        if seed_tag == "global_necker_flip" and int(diag_b["d_null_ray"]) == 0:
                            break

        if best_valid_choice is not None:
            _, out_best, diag_best = best_valid_choice
            diag_best["runtime_seconds"] = time.perf_counter() - started
            diag_best["raw_maximum_residual"] = raw_max_res
            return SharedAnchorResultV827(
                out_best.to(raw_coordinates.device), False, "feasible_update_v9_multibranch", diag_best
            )

        # Stage 3: Optional SQP Fallback strictly gated to Depth-Rigid Geometries (d_null_ray == 0)
        # Handles piecewise-kinked clamp/abs boundaries (e.g. clamped betweenness & line-plane angles on tetra_v2)
        # while NEVER executing on under-constrained polyhedra (d_null_ray >= 1), guaranteeing 0% nullspace harm!
        if self._rigid_fallback is not None:
            spec_init = decompose_depth_displacement_v9(
                cand0,
                raw,
                constraints,
                tau_svd=cfg.tau_svd,
                relative_rank_tol=cfg.relative_rank_tol,
            )
            if int(spec_init["d_null_ray"]) == 0:
                for ws in (cand0.to(raw_coordinates), raw_coordinates):
                    res_fb = self._rigid_fallback.project(
                        raw_coordinates,
                        constraints,
                        warm_start=ws,
                        camera=camera,
                        points_2d=points_2d,
                        constraint_confidences=constraint_confidences,
                    )
                    if not res_fb.abstained:
                        ok_fb, out_fb, diag_fb = certify_candidate(
                            res_fb.coordinates.double().cpu(), "slsqp_rigid_d0", it0, nvel0
                        )
                        if ok_fb and int(diag_fb.get("d_null_ray", 99)) == 0:
                            diag_fb["runtime_seconds"] = time.perf_counter() - started
                            diag_fb["raw_maximum_residual"] = raw_max_res
                            return SharedAnchorResultV827(
                                out_fb.to(raw_coordinates.device),
                                False,
                                "feasible_update_v9_fallback",
                                diag_fb,
                            )

        return SharedAnchorResultV827(
            raw_coordinates.clone(),
            True,
            "no_safe_feasible_candidate",
            {
                "runtime_seconds": time.perf_counter() - started,
                "regime": "v9_abstained_safely",
                "raw_maximum_residual": raw_max_res,
                "best_nf_residual": float(res0),
                "displacement_rms": 0.0,
                "nullspace_drift_rms": 0.0,
            },
        )
