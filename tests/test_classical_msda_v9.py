"""Unit tests for Classical Non-Neural Line-Drawing Reconstruction Baselines (V9.0)."""
from __future__ import annotations

import math

import numpy as np
import torch

from scripts.analyze_module3_v835_audit import numpy_pa
from src.lifting.anonymous_contract import (
    EDGE_STYLE_DASHED,
    EDGE_STYLE_SOLID,
    EDGE_STYLE_UNKNOWN,
    IndexedConstraint,
)
from src.reasoning.classical_line_drawing_msda_v9 import (
    LipsonShpitalniCADSolverV9,
    MarillMSDASolverV9,
)


def _make_orthographic_cube_case():
    pts = np.array(
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
        dtype=np.float64,
    )
    yaw, pitch = 0.62, 0.54
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(pitch), -math.sin(pitch)], [0.0, math.sin(pitch), math.cos(pitch)]])
    gt = pts @ (rx @ rz).T
    gt = gt / math.sqrt(float(np.mean(np.sum(gt**2, axis=1))))

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    adj = torch.zeros((8, 8), dtype=torch.bool)
    sty = torch.full((8, 8), EDGE_STYLE_UNKNOWN, dtype=torch.long)
    back_v = int(np.argmin(gt[:, 2]))
    for u, v in edges:
        adj[u, v] = adj[v, u] = True
        st = EDGE_STYLE_DASHED if (u == back_v or v == back_v) else EDGE_STYLE_SOLID
        sty[u, v] = sty[v, u] = st

    px = torch.tensor(gt[:, :2] * 120.0 + np.array([224.0, 160.0]), dtype=torch.float64)
    cs = [IndexedConstraint("coplanar", tuple(f), {"pts": list(f)}, True) for f in faces] + [
        IndexedConstraint("orthogonal", (0, 1, 0, 3), {"lines": [[0, 1], [0, 3]]}, True),
        IndexedConstraint("orthogonal", (0, 1, 0, 4), {"lines": [[0, 1], [0, 4]]}, True),
        IndexedConstraint("orthogonal", (0, 3, 0, 4), {"lines": [[0, 3], [0, 4]]}, True),
        IndexedConstraint("parallel", (0, 4, 1, 5), {"lines": [[0, 4], [1, 5]]}, True),
        IndexedConstraint("parallel", (0, 4, 2, 6), {"lines": [[0, 4], [2, 6]]}, True),
    ]
    return torch.tensor(gt, dtype=torch.float64), px, adj, sty, cs


def test_marill_msda_and_lipson_shpitalni_reconstruct_cube():
    gt, px, adj, sty, cs = _make_orthographic_cube_case()
    mask = torch.ones(8, dtype=torch.bool)

    flat = torch.cat((gt[:, :2], torch.zeros((8, 1), dtype=torch.float64)), dim=1)
    pa_flat = float(numpy_pa(flat, gt, mask, True))

    msda = MarillMSDASolverV9(num_random_seeds=2, max_lbfgs_iter=45)
    r_msda = msda.reconstruct(px, adj, sty, constraints=cs, reference_for_eval=gt)
    pa_msda = float(numpy_pa(r_msda.coordinates, gt, mask, True))
    assert r_msda.reprojection_rmse_px == 0.0
    assert pa_msda < pa_flat * 0.35, f"Expected MSDA to inflate cube accurately, got {pa_msda} vs flat {pa_flat}"

    ls = LipsonShpitalniCADSolverV9(num_random_seeds=2, stage1_lbfgs_iter=50, stage2_lm_iter=20)
    r_ls = ls.reconstruct(px, adj, sty, constraints=cs, reference_for_eval=gt)
    pa_ls = float(numpy_pa(r_ls.coordinates, gt, mask, True))
    proper_pa_ls = float(numpy_pa(r_ls.coordinates, gt, mask, False))
    assert r_ls.feasible, f"Expected Lipson-Shpitalni to satisfy cube constraints, got rms={r_ls.max_constraint_rms}"
    assert pa_ls < 0.05, f"Expected near-exact 3D cube from Lipson-Shpitalni, got PA={pa_ls}"
    assert proper_pa_ls < 0.05, f"Expected crossing/occlusion sign disambiguation to recover proper SO(3) cube, got {proper_pa_ls}"
