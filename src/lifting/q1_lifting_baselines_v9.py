"""Modern Graph & Equivariant Neural Lifting Baselines for Q1 Journal Evaluation (V9.0).

Implements three parameter-matched (~1.63M params) modern neural 2D-to-3D graph lifting
architectures sharing the exact anonymous input/output contract and confidence-weighted
gauge normalization of LiftingModuleV6:
1. EGNNLifterV9: E(2)/Coordinate-Difference Equivariant Graph Neural Network (Satorras et al., ICML 2021)
   adapted to wireframe lifting with explicit axial depth lift head.
2. GATv2EdgeLifterV9: Multi-Head Edge-Style-Conditioned Dynamic Graph Attention Network
   (Brody et al., ICLR 2022) operating over topological wireframe neighborhoods.
3. GraphGPSLifterV9: Hybrid Local Edge-Gated MPNN + Global Multi-Head Self-Attention Graph Transformer
   (Rampasek et al., NeurIPS 2022).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lifting.lifting_module_v4 import CoordinateHead, LiftingModuleV4
from src.lifting.lifting_module_v6 import LiftingModuleV6


class _BaseWireframeLifterV9(nn.Module):
    """Shared input featurization and confidence-weighted camera-residual gauge head."""

    NODE_FEATURE_DIM = LiftingModuleV4.NODE_FEATURE_DIM  # 11
    PAIR_FEATURE_DIM = LiftingModuleV4.PAIR_FEATURE_DIM  # 10

    def __init__(self, *, confidence_power: float = 0.25, reflection_symmetrization: bool = False) -> None:
        super().__init__()
        self.confidence_power = float(confidence_power)
        self.reflection_symmetrization = bool(reflection_symmetrization)

    def _confidence_weights(
        self, point_mask: torch.Tensor, node_confidence: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        conf = node_confidence.to(dtype).clamp(0.0, 1.0).pow(self.confidence_power)
        w = conf * point_mask.to(dtype)
        empty = w.sum(dim=1, keepdim=True) <= 1e-8
        return torch.where(empty, point_mask.to(dtype), w)

    def _node_and_pair_features(
        self,
        *,
        points_2d: torch.Tensor,
        point_mask: torch.Tensor,
        node_type: torch.Tensor,
        node_confidence: torch.Tensor,
        adjacency: torch.Tensor,
        edge_style: torch.Tensor,
        edge_probability: torch.Tensor,
        edge_style_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = self._confidence_weights(point_mask, node_confidence, dtype=points_2d.dtype)
        centered, rms = LiftingModuleV6._weighted_center_and_rms(points_2d, w)
        centered = centered / rms.unsqueeze(-1)
        radius = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        type_idx = torch.where(
            (node_type >= 0) & (node_type <= 2), node_type, torch.full_like(node_type, 3)
        )
        type_oh = F.one_hot(type_idx, num_classes=4).to(points_2d.dtype)
        w_edges = edge_probability * w[:, None, :]
        denom = (w.sum(dim=1)[:, None, None] - w[:, :, None]).clamp_min(1e-8)
        deg = w_edges.sum(dim=-1, keepdim=True) / denom
        node_feat = torch.cat(
            (points_2d, centered, radius, node_confidence.unsqueeze(-1), type_oh, deg), dim=-1
        ).masked_fill(~point_mask.unsqueeze(-1), 0.0)
        pair_feat = LiftingModuleV4._pair_features(
            points_2d, point_mask, adjacency, edge_style, edge_probability, edge_style_confidence
        )
        return node_feat, pair_feat, w

    def forward(
        self,
        *,
        points_2d: torch.Tensor,
        point_mask: torch.Tensor,
        node_type: torch.Tensor,
        node_confidence: torch.Tensor,
        adjacency: torch.Tensor,
        edge_style: torch.Tensor,
        edge_probability: torch.Tensor,
        edge_style_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inputs = {
            "points_2d": points_2d,
            "point_mask": point_mask,
            "node_type": node_type,
            "node_confidence": node_confidence,
            "adjacency": adjacency,
            "edge_style": edge_style,
            "edge_probability": edge_probability,
            "edge_style_confidence": edge_style_confidence,
        }
        orig = self._forward_once(**inputs)
        if not self.reflection_symmetrization:
            orig["reflection_disagreement"] = points_2d.new_zeros((points_2d.shape[0],))
            return orig

        mirrored_pts = points_2d.clone()
        mirrored_pts[..., 0] = -mirrored_pts[..., 0]
        mir = self._forward_once(**{**inputs, "points_2d": mirrored_pts})
        refl = points_2d.new_tensor([-1.0, 1.0, -1.0])
        mir_back = mir["coordinates_3d"] * refl
        comb = 0.5 * (orig["coordinates_3d"] + mir_back)
        coords = LiftingModuleV6._normalize_gauge_weighted(
            comb, point_mask, orig["confidence_weights"]
        )
        dis = torch.linalg.vector_norm(orig["coordinates_3d"] - mir_back, dim=-1)
        w = orig["confidence_weights"]
        orig["coordinates_3d"] = coords
        orig["reflection_disagreement"] = torch.sum(dis * w, dim=1) / w.sum(dim=1).clamp_min(1e-8)
        return orig


class _EGNNLayerV9(nn.Module):
    def __init__(self, hidden_dim: int, edge_attr_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_attr_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)
        self.z_lift_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.z_lift_mlp[-1].weight)
        nn.init.zeros_(self.z_lift_mlp[-1].bias)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        edge_attr: torch.Tensor,
        neighbor_mask: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, d = h.shape
        diff = coords[:, :, None, :] - coords[:, None, :, :]  # (B, N, N, 3)
        dist_sq = diff.square().sum(dim=-1, keepdim=True)     # (B, N, N, 1)
        hi = h[:, :, None, :].expand(b, n, n, d)
        hj = h[:, None, :, :].expand(b, n, n, d)
        m_in = torch.cat((hi, hj, dist_sq, edge_attr), dim=-1)
        m_ij = self.edge_mlp(m_in) * neighbor_mask.unsqueeze(-1).to(h.dtype)

        deg = neighbor_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(h.dtype)
        m_agg = m_ij.sum(dim=2) / deg
        h_out = h + self.node_mlp(torch.cat((self.node_norm(h), m_agg), dim=-1))
        h_out = h_out.masked_fill(~point_mask.unsqueeze(-1), 0.0)

        trans_weights = torch.tanh(self.coord_mlp(m_ij)) * neighbor_mask.unsqueeze(-1).to(h.dtype)
        delta_coord = (diff * trans_weights).sum(dim=2) / deg
        dz = self.z_lift_mlp(h_out)
        delta_z_vec = torch.cat((torch.zeros_like(dz), torch.zeros_like(dz), dz), dim=-1)
        coords_out = (coords + 0.25 * delta_coord + 0.25 * delta_z_vec).masked_fill(
            ~point_mask.unsqueeze(-1), 0.0
        )
        return h_out, coords_out


class EGNNLifterV9(_BaseWireframeLifterV9):
    """Sparse Wireframe E(2)/Coordinate-Difference Equivariant GNN Lifter (~1.63M params)."""

    def __init__(
        self,
        *,
        hidden_dim: int = 192,
        num_layers: int = 6,
        dropout: float = 0.1,
        confidence_power: float = 0.25,
        reflection_symmetrization: bool = False,
    ) -> None:
        super().__init__(
            confidence_power=confidence_power,
            reflection_symmetrization=reflection_symmetrization,
        )
        # Invariant node features exclude raw signed (x, y) so E(2) invariance holds strictly on scalar stream
        self.node_in = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        # Invariant edge features: [distance_2d, adj, edge_prob, style_oh(4), style_conf] = 8 dims
        self.layers = nn.ModuleList(
            _EGNNLayerV9(hidden_dim, edge_attr_dim=8, dropout=dropout) for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.residual_head = CoordinateHead(hidden_dim, zero_initialize=True)

    def _forward_once(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        points_2d = kwargs["points_2d"]
        point_mask = kwargs["point_mask"]
        adjacency = kwargs["adjacency"]
        edge_probability = kwargs["edge_probability"]
        node_feat, pair_feat, w = self._node_and_pair_features(**kwargs)
        # Scalar invariant subset of node_feat: [radius(1), conf(1), type_oh(4), deg(1)] = 7 dims
        inv_node = node_feat[..., 4:]
        # Invariant subset of pair_feat: exclude signed delta_xy (first 2 dims) -> 8 dims
        inv_edge = pair_feat[..., 2:]

        h = self.node_in(inv_node).masked_fill(~point_mask.unsqueeze(-1), 0.0)
        coords = LiftingModuleV6._camera_plane_coordinates_weighted(points_2d, point_mask, w)
        eye = torch.eye(points_2d.shape[1], dtype=torch.bool, device=points_2d.device).unsqueeze(0)
        neighbor_mask = (adjacency | (edge_probability >= 0.35)) & (~eye) & point_mask[:, :, None] & point_mask[:, None, :]

        for layer in self.layers:
            h, coords = layer(h, coords, inv_edge, neighbor_mask, point_mask)

        h = self.out_norm(h)
        raw_coords = coords + self.residual_head(h)
        norm_coords = LiftingModuleV6._normalize_gauge_weighted(raw_coords, point_mask, w)
        graph_emb = torch.sum(h * w.unsqueeze(-1), dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return {
            "coordinates_3d": norm_coords,
            "graph_embedding": graph_emb,
            "node_embeddings": h,
            "router_logits": h.new_empty((h.shape[0], 0)),
            "router_weights": h.new_empty((h.shape[0], 0)),
            "confidence_weights": w,
        }


class _GATv2EdgeBlockV9(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, pair_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.lin_l = nn.Linear(hidden_dim, hidden_dim)
        self.lin_r = nn.Linear(hidden_dim, hidden_dim)
        self.lin_e = nn.Linear(pair_dim, hidden_dim)
        self.att_vec = nn.Parameter(torch.empty(1, num_heads, 1, 1, self.head_dim))
        nn.init.xavier_uniform_(self.att_vec)
        self.lin_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 3 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: torch.Tensor,
        pair_feat: torch.Tensor,
        local_mask: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = h.shape
        x = self.norm1(h)
        xl = self.lin_l(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, N, Dh)
        xr = self.lin_r(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        xe = self.lin_e(pair_feat).reshape(b, n, n, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        logits = (F.leaky_relu(xl[:, :, :, None, :] + xr[:, :, None, :, :] + xe, 0.2) * self.att_vec).sum(dim=-1)
        logits = logits.masked_fill(~local_mask[:, None, :, :], -torch.inf)
        alpha = self.dropout(torch.softmax(logits, dim=-1))
        val = self.lin_v(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        agg = (alpha @ val).transpose(1, 2).reshape(b, n, d)
        h = h + self.out_proj(agg).masked_fill(~point_mask.unsqueeze(-1), 0.0)
        h = h + self.ffn(self.norm2(h))
        return h.masked_fill(~point_mask.unsqueeze(-1), 0.0)


class GATv2EdgeLifterV9(_BaseWireframeLifterV9):
    """Multi-Head Edge-Style-Conditioned GATv2 Lifter (~1.65M params)."""

    def __init__(
        self,
        *,
        hidden_dim: int = 192,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        confidence_power: float = 0.25,
        reflection_symmetrization: bool = False,
    ) -> None:
        super().__init__(
            confidence_power=confidence_power,
            reflection_symmetrization=reflection_symmetrization,
        )
        self.node_in = nn.Sequential(
            nn.Linear(self.NODE_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            _GATv2EdgeBlockV9(hidden_dim, num_heads, self.PAIR_FEATURE_DIM, dropout)
            for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = CoordinateHead(hidden_dim, zero_initialize=True)

    def _forward_once(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        points_2d = kwargs["points_2d"]
        point_mask = kwargs["point_mask"]
        adjacency = kwargs["adjacency"]
        edge_probability = kwargs["edge_probability"]
        node_feat, pair_feat, w = self._node_and_pair_features(**kwargs)

        eye = torch.eye(points_2d.shape[1], dtype=torch.bool, device=points_2d.device).unsqueeze(0)
        # Keep self-loop eye active on all rows (including padded rows) so softmax never sees an all -inf row in backward!
        local_mask = ((adjacency | (edge_probability >= 0.35)) & point_mask[:, :, None] & point_mask[:, None, :]) | eye

        h = self.node_in(node_feat).masked_fill(~point_mask.unsqueeze(-1), 0.0)
        for blk in self.blocks:
            h = blk(h, pair_feat, local_mask, point_mask)
        h = self.out_norm(h)
        base = LiftingModuleV6._camera_plane_coordinates_weighted(points_2d, point_mask, w)
        coords = LiftingModuleV6._normalize_gauge_weighted(base + self.head(h), point_mask, w)
        graph_emb = torch.sum(h * w.unsqueeze(-1), dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return {
            "coordinates_3d": coords,
            "graph_embedding": graph_emb,
            "node_embeddings": h,
            "router_logits": h.new_empty((h.shape[0], 0)),
            "router_weights": h.new_empty((h.shape[0], 0)),
            "confidence_weights": w,
        }


class _GraphGPSBlockV9(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, pair_dim: int, dropout: float) -> None:
        super().__init__()
        # Local Gated Edge MPNN branch
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.local_q = nn.Linear(hidden_dim, hidden_dim)
        self.local_kv = nn.Linear(hidden_dim, hidden_dim)
        self.local_edge = nn.Linear(pair_dim, hidden_dim)
        self.local_out = nn.Linear(hidden_dim, hidden_dim)
        # Global Multi-Head Self-Attention branch (no dense pair/visibility bias)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.global_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.global_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # FFN fusion
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: torch.Tensor,
        pair_feat: torch.Tensor,
        adj_mask: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = h.shape
        # 1. Local Edge-Gated MPNN
        xl = self.local_norm(h)
        gate = torch.sigmoid(self.local_q(xl)[:, :, None, :] + self.local_kv(xl)[:, None, :, :] + self.local_edge(pair_feat))
        gate = gate * adj_mask.unsqueeze(-1).to(h.dtype)
        deg = adj_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(h.dtype)
        msg_local = self.local_out((gate * self.local_kv(xl)[:, None, :, :]).sum(dim=2) / deg)

        # 2. Global Self-Attention
        xg = self.global_norm(h)
        qkv = self.global_qkv(xg).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~point_mask[:, None, None, :], -torch.inf)
        att = self.dropout(torch.softmax(scores, dim=-1))
        msg_global = self.global_out((att @ v).transpose(1, 2).reshape(b, n, d))

        h = h + (msg_local + msg_global).masked_fill(~point_mask.unsqueeze(-1), 0.0)
        h = h + self.ffn(self.ffn_norm(h))
        return h.masked_fill(~point_mask.unsqueeze(-1), 0.0)


class GraphGPSLifterV9(_BaseWireframeLifterV9):
    """Hybrid Local Edge-Gated MPNN + Global Self-Attention GraphGPS Lifter (~1.63M params)."""

    def __init__(
        self,
        *,
        hidden_dim: int = 176,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        confidence_power: float = 0.25,
        reflection_symmetrization: bool = False,
    ) -> None:
        super().__init__(
            confidence_power=confidence_power,
            reflection_symmetrization=reflection_symmetrization,
        )
        self.node_in = nn.Sequential(
            nn.Linear(self.NODE_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            _GraphGPSBlockV9(hidden_dim, num_heads, self.PAIR_FEATURE_DIM, dropout)
            for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = CoordinateHead(hidden_dim, zero_initialize=True)

    def _forward_once(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        points_2d = kwargs["points_2d"]
        point_mask = kwargs["point_mask"]
        adjacency = kwargs["adjacency"]
        edge_probability = kwargs["edge_probability"]
        node_feat, pair_feat, w = self._node_and_pair_features(**kwargs)

        eye = torch.eye(points_2d.shape[1], dtype=torch.bool, device=points_2d.device).unsqueeze(0)
        adj_mask = (adjacency | (edge_probability >= 0.35)) & (~eye) & point_mask[:, :, None] & point_mask[:, None, :]

        h = self.node_in(node_feat).masked_fill(~point_mask.unsqueeze(-1), 0.0)
        for blk in self.blocks:
            h = blk(h, pair_feat, adj_mask, point_mask)
        h = self.out_norm(h)
        base = LiftingModuleV6._camera_plane_coordinates_weighted(points_2d, point_mask, w)
        coords = LiftingModuleV6._normalize_gauge_weighted(base + self.head(h), point_mask, w)
        graph_emb = torch.sum(h * w.unsqueeze(-1), dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return {
            "coordinates_3d": coords,
            "graph_embedding": graph_emb,
            "node_embeddings": h,
            "router_logits": h.new_empty((h.shape[0], 0)),
            "router_weights": h.new_empty((h.shape[0], 0)),
            "confidence_weights": w,
        }
