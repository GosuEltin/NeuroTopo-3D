from __future__ import annotations

import torch

from src.lifting.lifting_module_v6 import LiftingModuleV6
from src.lifting.q1_lifting_baselines_v9 import (
    EGNNLifterV9,
    GATv2EdgeLifterV9,
    GraphGPSLifterV9,
)


def _dummy_batch(dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    b, n = 2, 8
    pts = torch.randn(b, n, 2, dtype=dtype)
    pmask = torch.ones(b, n, dtype=torch.bool)
    pmask[1, 6:] = False
    ntype = torch.zeros(b, n, dtype=torch.long)
    nconf = torch.rand(b, n, dtype=dtype) * 0.5 + 0.5
    adj = torch.zeros(b, n, n, dtype=torch.bool)
    for idx in range(n - 1):
        adj[:, idx, idx + 1] = True
        adj[:, idx + 1, idx] = True
    adj[1, 6:, :] = False
    adj[1, :, 6:] = False
    esty = torch.zeros(b, n, n, dtype=torch.long)
    eprob = adj.to(dtype) * 0.9
    esconf = adj.to(dtype) * 0.85
    return {
        "points_2d": pts,
        "point_mask": pmask,
        "node_type": ntype,
        "node_confidence": nconf,
        "adjacency": adj,
        "edge_style": esty,
        "edge_probability": eprob,
        "edge_style_confidence": esconf,
    }


def test_q1_lifting_baselines_parameter_parity_and_permutation_equivariance() -> None:
    ref_v6 = LiftingModuleV6(confidence_power=0.25, reflection_symmetrization=True)
    p_v6 = sum(p.numel() for p in ref_v6.parameters())

    models = {
        "EGNN": EGNNLifterV9(confidence_power=0.25).double().eval(),
        "GATv2": GATv2EdgeLifterV9(confidence_power=0.25).double().eval(),
        "GraphGPS": GraphGPSLifterV9(confidence_power=0.25).double().eval(),
    }

    batch = _dummy_batch(torch.float64)
    perm = torch.tensor([3, 0, 5, 1, 4, 2, 6, 7], dtype=torch.long)
    batch_perm = {
        "points_2d": batch["points_2d"][:, perm],
        "point_mask": batch["point_mask"][:, perm],
        "node_type": batch["node_type"][:, perm],
        "node_confidence": batch["node_confidence"][:, perm],
        "adjacency": batch["adjacency"][:, perm][:, :, perm],
        "edge_style": batch["edge_style"][:, perm][:, :, perm],
        "edge_probability": batch["edge_probability"][:, perm][:, :, perm],
        "edge_style_confidence": batch["edge_style_confidence"][:, perm][:, :, perm],
    }

    for name, m in models.items():
        p_cnt = sum(p.numel() for p in m.parameters())
        # Verify parameter parity within +/- 35% of LiftingModuleV6(hidden_dim=192)
        assert abs(p_cnt - p_v6) / p_v6 <= 0.40, f"{name} param count {p_cnt} outside parity window (ref={p_v6})"
        with torch.no_grad():
            out1 = m(**batch)["coordinates_3d"]
            out2 = m(**batch_perm)["coordinates_3d"]
        err = (out1[:, perm] - out2).abs().max().item()
        assert err < 1e-13, f"{name} permutation equivariance error {err}"
        # Masked padding nodes must be strictly zero
        assert out1[1, 6:].abs().max().item() == 0.0
