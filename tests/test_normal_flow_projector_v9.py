from __future__ import annotations

import math
import torch

from src.lifting.anonymous_contract import (
    EDGE_STYLE_DASHED,
    EDGE_STYLE_SOLID,
    IndexedConstraint,
)
from src.lifting.camera_gauge_v51 import fit_isotropic_orthographic_camera
from src.reasoning.normal_flow_projector_v9 import (
    NormalFlowMultiBranchProjectorV9,
    NormalFlowProjectorConfigV9,
    build_ray_orthonormal_basis,
    extract_nullspace_cad_family,
    find_transverse_solid_dashed_crossings,
    select_optimal_hemisphere_v9,
)
from src.reasoning.residuals_v4 import compile_semantic_residuals


def test_ray_orthonormal_basis_exact_orthogonality() -> None:
    for n in (4, 8, 12):
        q = build_ray_orthonormal_basis(n)
        assert q.shape == (n, n - 1)
        ident = q.T @ q
        assert torch.allclose(ident, torch.eye(n - 1, dtype=torch.float64), atol=1e-14)
        ones_sum = q.sum(dim=0).abs().max().item()
        assert ones_sum < 1e-14


def test_normal_flow_cube_zero_nullspace_velocity_and_cad_family() -> None:
    # Construct an orthographic cube (N=8) rotated in 3D with coplanarity + parallelism constraints
    pts_cube = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    # Apply non-degenerate 3D rotation
    rx, ry = 0.42, -0.55
    rot_x = torch.tensor([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=torch.float64)
    rot_y = torch.tensor([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=torch.float64)
    gt = pts_cube @ (rot_x @ rot_y).T
    gt = gt - gt.mean(dim=0, keepdim=True)
    gt = gt / gt.square().sum(-1).mean().sqrt()

    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]
    constraints = [
        IndexedConstraint(
            operator="coplanar",
            entities=tuple(f),
            arguments={"pts": f},
            solver_active=True,
        )
        for f in faces
    ] + [
        IndexedConstraint(
            operator="parallel",
            entities=(0, 4, 1, 5),
            arguments={"lines": [[0, 4], [1, 5]]},
            solver_active=True,
        ),
        IndexedConstraint(
            operator="parallel",
            entities=(1, 5, 2, 6),
            arguments={"lines": [[1, 5], [2, 6]]},
            solver_active=True,
        ),
    ]

    # Perturb depth out of coplanarity
    torch.manual_seed(7)
    raw = gt.clone()
    raw[:, 2] = raw[:, 2] + 0.18 * torch.randn(8, dtype=torch.float64)
    px = gt[:, :2] * 140.0 + 220.0
    cam = fit_isotropic_orthographic_camera(raw, px)

    proj = NormalFlowMultiBranchProjectorV9()
    res = proj.project(raw, constraints, camera=cam, points_2d=px)
    assert not res.abstained
    assert float(res.diagnostics["max_stepwise_null_velocity"]) < 1e-14
    assert float(res.diagnostics["nullspace_drift_rms"]) < 0.05
    assert float(res.diagnostics["final_max_residual"]) <= 1.1e-3

    # Test strict 0.0 px CAD mode + Parametric Nullspace CAD Family Explorer
    proj_cad = NormalFlowMultiBranchProjectorV9(NormalFlowProjectorConfigV9(max_reprojection_increase_px=0.0))
    res_cad = proj_cad.project(raw, constraints, camera=cam, points_2d=px)
    assert not res_cad.abstained
    assert abs(float(res_cad.diagnostics["reprojection_increase_px"])) < 1e-12
    fam = extract_nullspace_cad_family(res_cad.coordinates, raw, constraints, amplitudes=(-0.15, 0.0, 0.15))
    assert int(fam["d_null_ray"]) >= 1
    assert len(fam["members"]) == 3
    for m in fam["members"]:
        assert float(m["max_constraint_rms"]) < 1e-9
        assert float(m["xy_drift"]) < 1e-14


def test_multibranch_rescues_apex_inverted_rigid_tetrahedron() -> None:
    # Regular / right-corner tetrahedron: V0=(0,0,0), V1=(1,0,0), V2=(0,1,0), V3=(0,0,1) rotated
    pts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.3]],
        dtype=torch.float64,
    )
    rx, ry = 0.5, 0.6
    rot_x = torch.tensor([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=torch.float64)
    rot_y = torch.tensor([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=torch.float64)
    gt = pts @ (rot_x @ rot_y).T
    gt = gt - gt.mean(dim=0, keepdim=True)
    gt = gt / gt.square().sum(-1).mean().sqrt()

    constraints = [
        IndexedConstraint("orthogonal", (0, 1, 2), {"lines": [[0, 1], [0, 2]]}, True),
        IndexedConstraint("orthogonal", (0, 1, 3), {"lines": [[0, 1], [0, 3]]}, True),
        IndexedConstraint("orthogonal", (0, 2, 3), {"lines": [[0, 2], [0, 3]]}, True),
    ]

    # Create a trapped near-flat / single-vertex inverted seed where primary branch stalls or lands far
    raw = gt.clone()
    raw[:, 2] = 0.02 * torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float64)
    px = gt[:, :2] * 150.0 + 200.0
    cam = fit_isotropic_orthographic_camera(raw, px)

    proj = NormalFlowMultiBranchProjectorV9(
        NormalFlowProjectorConfigV9(rigid_max_rms=0.85, rigid_max_coord=1.6)
    )
    res = proj.project(raw, constraints, camera=cam, points_2d=px)
    assert not res.abstained
    b_check = compile_semantic_residuals(res.coordinates, constraints, reference_coordinates=raw)
    assert float(b_check.raw_constraint_rms.max().item()) <= 1.1e-3


def test_transverse_solid_dashed_crossing_detector() -> None:
    pts2d = torch.tensor([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0], [10.0, 0.0]], dtype=torch.float64).numpy()
    adj = torch.zeros((4, 4), dtype=torch.float64).numpy()
    sty = torch.zeros((4, 4), dtype=torch.long).numpy()
    adj[0, 1] = adj[1, 0] = 1.0
    sty[0, 1] = sty[1, 0] = EDGE_STYLE_SOLID
    adj[2, 3] = adj[3, 2] = 1.0
    sty[2, 3] = sty[3, 2] = EDGE_STYLE_DASHED
    crossings = find_transverse_solid_dashed_crossings(pts2d, adj, sty)
    assert len(crossings) == 1
    i, j, p, q, s, t = crossings[0]
    assert (i, j) == (0, 1) and (p, q) == (2, 3)
    assert abs(s - 0.5) < 1e-6 and abs(t - 0.5) < 1e-6

    # Verify select_optimal_hemisphere_v9 flips an inverted depth hemisphere when solid (0,1) is behind dashed (2,3)
    raw = torch.tensor(
        [[0.0, 0.0, 0.2], [1.0, 1.0, 0.2], [0.0, 1.0, -0.2], [1.0, 0.0, -0.2]],
        dtype=torch.float64,
    )
    inverted_cand = raw.clone()
    inverted_cand[:, 2] = -raw[:, 2]  # solid at z=-0.2 < dashed at z=+0.2!
    constraints = [
        IndexedConstraint("equal_length", (0, 1, 2, 3), {"pairs": [[0, 1], [2, 3]]}, True),
    ]
    selected, did_flip, reason = select_optimal_hemisphere_v9(
        inverted_cand,
        raw,
        constraints,
        crossings=crossings,
        use_crossing_selector=True,
    )
    assert did_flip is True
    assert reason == "crossing_occlusion_flipped"
    assert float(selected[0, 2].item()) > float(selected[2, 2].item())


def test_phase1b_subpixel_xy_polish_on_noisy_midpoint() -> None:
    # 5 points: 4 coplanar base vertices + 1 midpoint of (0, 1) with slight 2D pixel noise
    pts = torch.tensor(
        [
            [-1.0, -0.5, 0.2],
            [1.0, -0.5, -0.2],
            [1.0, 0.8, 0.1],
            [-1.0, 0.8, 0.5],
            [0.0, -0.5, 0.0],
        ],
        dtype=torch.float64,
    )
    constraints = [
        IndexedConstraint("coplanar", (0, 1, 2, 3), {"pts": [0, 1, 2, 3]}, True),
        IndexedConstraint("midpoint", (0, 1, 4), {"args": [0, 1, 4]}, True),
    ]
    raw = pts.clone()
    raw[4, 0] += 0.005  # 2D midpoint noise (~0.5 px at scale 100)
    raw[3, 2] += 0.12   # 3D coplanarity noise
    px = raw[:, :2] * 100.0 + 200.0
    cam = fit_isotropic_orthographic_camera(raw, px)

    proj = NormalFlowMultiBranchProjectorV9()
    res = proj.project(raw, constraints, camera=cam, points_2d=px)
    assert not res.abstained
    assert float(res.diagnostics["final_max_residual"]) <= 1.1e-3
    assert float(res.diagnostics["reprojection_increase_px"]) <= 1.0 + 1e-8
    assert float(res.diagnostics["max_stepwise_null_velocity"]) < 1e-14

