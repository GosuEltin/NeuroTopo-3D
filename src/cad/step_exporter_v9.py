"""ISO 10303-21 (.STEP) & Wavefront (.OBJ) B-Rep / Wireframe CAD Exporter & Validator (V9.0).

Converts reconstructed 3D polyhedral / mechanical wireframe coordinates and cyclic planar faces
into standard-compliant ISO 10303-21 STEP files (ADVANCED_FACE + CLOSED_SHELL / MANIFOLD_SOLID_BREP
for watertight solids, or SHELL_BASED_SURFACE_MODEL + GEOMETRIC_CURVE_SET) and Wavefront OBJ files,
and computes CAD kernel B-Rep manufacturability metrics:
- Syntactic & referential integrity of exported .STEP entity graph,
- Face planarity max/RMS residual (mm and dimensionless),
- Inter-face 3D dihedral angle error (deg),
- Edge-length ratio error.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(v))
    if nrm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / nrm


def fit_face_plane(points_3d: np.ndarray, face_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit best least-squares plane to face vertices.

    Returns (centroid, unit_normal, unit_ref_dir, max_point_to_plane_distance).
    """
    pts = points_3d[list(face_indices)]
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    if len(face_indices) >= 3:
        # Orient normal consistently with polygon Newell winding first
        newell = np.zeros(3, dtype=np.float64)
        m = len(face_indices)
        for k in range(m):
            p_cur = pts[k]
            p_nxt = pts[(k + 1) % m]
            newell[0] += (p_cur[1] - p_nxt[1]) * (p_cur[2] + p_nxt[2])
            newell[1] += (p_cur[2] - p_nxt[2]) * (p_cur[0] + p_nxt[0])
            newell[2] += (p_cur[0] - p_nxt[0]) * (p_cur[1] + p_nxt[1])
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = _normalize(vh[-1])
        if float(np.dot(normal, newell)) < 0.0:
            normal = -normal
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Orthonormal tangent reference direction in plane
    edge0 = pts[1] - pts[0] if len(pts) >= 2 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    edge0_proj = edge0 - float(np.dot(edge0, normal)) * normal
    if float(np.linalg.norm(edge0_proj)) <= 1e-10:
        cand = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(cand, normal))) > 0.9:
            cand = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        edge0_proj = cand - float(np.dot(cand, normal)) * normal
    ref_dir = _normalize(edge0_proj)

    dists = np.abs(centered @ normal)
    max_dist = float(np.max(dists)) if len(dists) > 0 else 0.0
    return centroid, normal, ref_dir, max_dist


def orient_faces_outward(points_3d: np.ndarray, faces: Sequence[Sequence[int]]) -> list[list[int]]:
    """Orient all cyclic face loops consistently outward via BFS half-edge propagation + signed volume."""
    if not faces:
        return []
    oriented: list[list[int]] = [[int(i) for i in f] for f in faces]
    num_faces = len(oriented)

    # Map each undirected edge {u, v} to incident face indices
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi, f in enumerate(oriented):
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            edge_to_faces.setdefault((min(a, b), max(a, b)), []).append(fi)

    visited = [False] * num_faces
    for start_f in range(num_faces):
        if visited[start_f]:
            continue
        visited[start_f] = True
        queue = [start_f]
        while queue:
            cur = queue.pop(0)
            f_cur = oriented[cur]
            m_cur = len(f_cur)
            cur_dir = {(f_cur[k], f_cur[(k + 1) % m_cur]) for k in range(m_cur)}
            for k in range(m_cur):
                a, b = f_cur[k], f_cur[(k + 1) % m_cur]
                for nb in edge_to_faces.get((min(a, b), max(a, b)), []):
                    if visited[nb]:
                        continue
                    visited[nb] = True
                    f_nb = oriented[nb]
                    m_nb = len(f_nb)
                    nb_dir = {(f_nb[j], f_nb[(j + 1) % m_nb]) for j in range(m_nb)}
                    # Shared edge must be traversed in OPPOSITE directions across adjacent faces!
                    if (a, b) in cur_dir and (a, b) in nb_dir:
                        oriented[nb] = list(reversed(f_nb))
                    elif (b, a) in cur_dir and (b, a) in nb_dir:
                        oriented[nb] = list(reversed(f_nb))
                    queue.append(nb)

    # Compute polyhedral signed volume via Newell surface integral (handles non-convex faces!)
    pts_c = points_3d - points_3d.mean(axis=0, keepdims=True)
    vol6 = 0.0
    for f in oriented:
        m = len(f)
        if m < 3:
            continue
        p0 = pts_c[f[0]]
        newell = np.zeros(3, dtype=np.float64)
        for k in range(m):
            pa = pts_c[f[k]]
            pb = pts_c[f[(k + 1) % m]]
            newell[0] += (pa[1] - pb[1]) * (pa[2] + pb[2])
            newell[1] += (pa[2] - pb[2]) * (pa[0] + pb[0])
            newell[2] += (pa[0] - pb[0]) * (pa[1] + pb[1])
        vol6 += float(np.dot(p0, newell))

    if vol6 < 0.0:
        oriented = [list(reversed(f)) for f in oriented]
    return oriented


def is_watertight_2manifold(faces: Sequence[Sequence[int]]) -> bool:
    """Verify every undirected edge belongs to exactly 2 faces with opposite half-edge orientations."""
    if len(faces) < 4:
        return False
    directed_counts: dict[tuple[int, int], int] = {}
    undirected_counts: dict[tuple[int, int], int] = {}
    for f in faces:
        m = len(f)
        if m < 3:
            return False
        for k in range(m):
            u, v = int(f[k]), int(f[(k + 1) % m])
            if u == v:
                return False
            directed_counts[(u, v)] = directed_counts.get((u, v), 0) + 1
            key = (min(u, v), max(u, v))
            undirected_counts[key] = undirected_counts.get(key, 0) + 1
    if not undirected_counts:
        return False
    for (a, b), cnt in undirected_counts.items():
        if cnt != 2:
            return False
        if directed_counts.get((a, b), 0) != 1 or directed_counts.get((b, a), 0) != 1:
            return False
    return True


def export_cad_to_step(
    points_3d: np.ndarray,
    faces: Sequence[Sequence[int]],
    edges: Sequence[tuple[int, int]] | None = None,
    output_path: str | Path | None = None,
    *,
    part_name: str = "NEUROTOPO_CAD_SOLID",
    scale_mm: float = 100.0,
) -> str:
    """Export 3D polyhedral B-Rep solid/shell and wireframe to ISO 10303-21 STEP format."""
    pts = np.asarray(points_3d, dtype=np.float64) * float(scale_mm)
    oriented_faces = orient_faces_outward(pts, faces)
    watertight = is_watertight_2manifold(oriented_faces)

    # Collect unique undirected edges from faces + explicit wireframe edges
    edge_set: set[tuple[int, int]] = set()
    for f in oriented_faces:
        for k in range(len(f)):
            u, v = int(f[k]), int(f[(k + 1) % len(f)])
            if u != v:
                edge_set.add((min(u, v), max(u, v)))
    if edges is not None:
        for u, v in edges:
            if int(u) != int(v):
                edge_set.add((min(int(u), int(v)), max(int(u), int(v))))
    sorted_edges = sorted(edge_set)

    entities: list[str] = []

    def add_ent(body: str) -> int:
        eid = len(entities) + 1
        entities.append(f"#{eid} = {body};")
        return eid

    def fmt_vec(vec: np.ndarray) -> str:
        return f"({vec[0]:.8f}, {vec[1]:.8f}, {vec[2]:.8f})"

    # Context & Units
    app_ctx = add_ent("APPLICATION_CONTEXT('configuration controlled 3d designs of mechanical parts and assemblies')")
    add_ent(f"APPLICATION_PROTOCOL_DEFINITION('international standard', 'config_control_design', 1994, #{app_ctx})")
    prod_ctx = add_ent(f"MECHANICAL_CONTEXT('', #{app_ctx}, 'mechanical')")
    prod = add_ent(f"PRODUCT('{part_name}', '{part_name}', 'Reconstructed by NeuroTopo V9.0', (#{prod_ctx}))")
    p_form = add_ent(f"PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('1', '', #{prod}, .NOT_KNOWN.)")
    pd_ctx = add_ent(f"DESIGN_CONTEXT('', #{app_ctx}, 'design')")
    p_def = add_ent(f"PRODUCT_DEFINITION('design', '', #{p_form}, #{pd_ctx})")
    p_shape = add_ent(f"PRODUCT_DEFINITION_SHAPE('', '', #{p_def})")

    len_unit = add_ent("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI., .METRE.))")
    ang_unit = add_ent("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($, .RADIAN.))")
    sol_unit = add_ent("(NAMED_UNIT(*) SI_UNIT($, .STERADIAN.) SOLID_ANGLE_UNIT())")
    unc_unit = add_ent(
        f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.0E-04), #{len_unit}, 'distance_accuracy_value', 'cad sewing tolerance')"
    )
    geom_ctx = add_ent(
        f"(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc_unit})) "
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{len_unit}, #{ang_unit}, #{sol_unit})) REPRESENTATION_CONTEXT('3D', '3D Context'))"
    )

    # Vertices
    cart_ids: list[int] = []
    vert_ids: list[int] = []
    for idx, p in enumerate(pts):
        cid = add_ent(f"CARTESIAN_POINT('V_{idx}', {fmt_vec(p)})")
        vid = add_ent(f"VERTEX_POINT('VP_{idx}', #{cid})")
        cart_ids.append(cid)
        vert_ids.append(vid)

    # Canonical Undirected Edges (LINE + EDGE_CURVE from u -> v with u < v)
    edge_curve_map: dict[tuple[int, int], int] = {}
    line_ids: list[int] = []
    for u, v in sorted_edges:
        dir_uv = _normalize(pts[v] - pts[u])
        mag = max(float(np.linalg.norm(pts[v] - pts[u])), 1e-6)
        did = add_ent(f"DIRECTION('DIR_{u}_{v}', {fmt_vec(dir_uv)})")
        vec_id = add_ent(f"VECTOR('VEC_{u}_{v}', #{did}, {mag:.8f})")
        lid = add_ent(f"LINE('LINE_{u}_{v}', #{cart_ids[u]}, #{vec_id})")
        ec_id = add_ent(f"EDGE_CURVE('EC_{u}_{v}', #{vert_ids[u]}, #{vert_ids[v]}, #{lid}, .T.)")
        edge_curve_map[(u, v)] = ec_id
        line_ids.append(lid)

    # Planar B-Rep Faces
    adv_face_ids: list[int] = []
    for f_idx, f_loop in enumerate(oriented_faces):
        c_f, n_f, ref_f, _ = fit_face_plane(pts, f_loop)
        cp_f = add_ent(f"CARTESIAN_POINT('FC_{f_idx}', {fmt_vec(c_f)})")
        dn_f = add_ent(f"DIRECTION('FN_{f_idx}', {fmt_vec(n_f)})")
        dr_f = add_ent(f"DIRECTION('FR_{f_idx}', {fmt_vec(ref_f)})")
        ax_f = add_ent(f"AXIS2_PLACEMENT_3D('FAX_{f_idx}', #{cp_f}, #{dn_f}, #{dr_f})")
        pl_f = add_ent(f"PLANE('PLANE_{f_idx}', #{ax_f})")

        oriented_edge_ids: list[int] = []
        m = len(f_loop)
        for k in range(m):
            a, b = int(f_loop[k]), int(f_loop[(k + 1) % m])
            canon_key = (min(a, b), max(a, b))
            ec_id = edge_curve_map[canon_key]
            sense = ".T." if a < b else ".F."
            oe_id = add_ent(f"ORIENTED_EDGE('', *, *, #{ec_id}, {sense})")
            oriented_edge_ids.append(f"#{oe_id}")

        eloop_id = add_ent(f"EDGE_LOOP('EL_{f_idx}', ({', '.join(oriented_edge_ids)}))")
        fbound_id = add_ent(f"FACE_OUTER_BOUND('FOB_{f_idx}', #{eloop_id}, .T.)")
        aface_id = add_ent(f"ADVANCED_FACE('FACE_{f_idx}', (#{fbound_id}), #{pl_f}, .T.)")
        adv_face_ids.append(f"#{aface_id}")

    if adv_face_ids:
        if watertight:
            shell_id = add_ent(f"CLOSED_SHELL('SOLID_SHELL', ({', '.join(adv_face_ids)}))")
            rep_item = add_ent(f"MANIFOLD_SOLID_BREP('{part_name}_BREP', #{shell_id})")
            shape_rep = add_ent(f"ADVANCED_BREP_SHAPE_REPRESENTATION('{part_name}_REP', (#{rep_item}), #{geom_ctx})")
        else:
            shell_id = add_ent(f"OPEN_SHELL('SURFACE_SHELL', ({', '.join(adv_face_ids)}))")
            rep_item = add_ent(f"SHELL_BASED_SURFACE_MODEL('{part_name}_SBSM', (#{shell_id}))")
            shape_rep = add_ent(f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION('{part_name}_REP', (#{rep_item}), #{geom_ctx})")
    else:
        curve_refs = ", ".join(f"#{lid}" for lid in line_ids)
        rep_item = add_ent(f"GEOMETRIC_CURVE_SET('{part_name}_WIREFRAME', ({curve_refs}))")
        shape_rep = add_ent(f"GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION('{part_name}_REP', (#{rep_item}), #{geom_ctx})")

    add_ent(f"SHAPE_DEFINITION_REPRESENTATION(#{p_shape}, #{shape_rep})")

    step_text = "\n".join(
        [
            "ISO-10303-21;",
            "HEADER;",
            "FILE_DESCRIPTION(('NeuroTopo V9.0 Reconstructed CAD B-Rep Solid'), '2;1');",
            f"FILE_NAME('{part_name}.step', '2026-10-29T12:00:00', ('NeuroTopo V9.0'), ('Geometry Solver Research'), 'NeuroTopo STEP Exporter V9.0', 'NeuroTopo V9.0', '');",
            "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));",
            "ENDSEC;",
            "DATA;",
            *entities,
            "ENDSEC;",
            "END-ISO-10303-21;",
            "",
        ]
    )
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(step_text, encoding="utf-8")
    return step_text


def export_cad_to_obj(
    points_3d: np.ndarray,
    faces: Sequence[Sequence[int]],
    edges: Sequence[tuple[int, int]] | None = None,
    output_path: str | Path | None = None,
    *,
    scale_mm: float = 100.0,
) -> str:
    """Export 3D CAD solid/wireframe to Wavefront .OBJ format."""
    pts = np.asarray(points_3d, dtype=np.float64) * float(scale_mm)
    oriented = orient_faces_outward(pts, faces)
    lines = ["# NeuroTopo V9.0 Reconstructed CAD Mesh & Wireframe"]
    for p in pts:
        lines.append(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}")
    for f in oriented:
        _, n_f, _, _ = fit_face_plane(pts, f)
        lines.append(f"vn {n_f[0]:.8f} {n_f[1]:.8f} {n_f[2]:.8f}")
    for f_idx, f in enumerate(oriented, start=1):
        v_tokens = " ".join(f"{int(v)+1}//{f_idx}" for v in f)
        lines.append(f"f {v_tokens}")
    if edges is not None:
        for u, v in edges:
            lines.append(f"l {int(u)+1} {int(v)+1}")
    text = "\n".join(lines) + "\n"
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text, encoding="utf-8")
    return text


def validate_step_and_brep_quality(
    step_text: str,
    reconstructed_3d: np.ndarray,
    gt_3d: np.ndarray,
    faces: Sequence[Sequence[int]],
    *,
    scale_mm: float = 100.0,
    planarity_sewing_tol_mm: float = 0.10,
) -> dict[str, object]:
    """Validate ISO 10303-21 STEP entity graph and compute CAD B-Rep geometric quality metrics."""
    has_header = step_text.startswith("ISO-10303-21;") and "END-ISO-10303-21;" in step_text
    defined_ids: set[int] = set()
    referenced_ids: set[int] = set()
    balanced_parens = True
    entity_types: dict[str, int] = {}

    in_data = False
    for line in step_text.splitlines():
        s = line.strip()
        if s == "DATA;":
            in_data = True
            continue
        if s == "ENDSEC;":
            in_data = False
            continue
        if not in_data or not s.startswith("#"):
            continue
        if s.count("(") != s.count(")"):
            balanced_parens = False
        m = re.match(r"^#(\d+)\s*=\s*(.+);$", s)
        if not m:
            balanced_parens = False
            continue
        eid = int(m.group(1))
        body = m.group(2)
        defined_ids.add(eid)
        t_match = re.match(r"^\(?([A-Z0-9_]+)", body)
        if t_match:
            etype = t_match.group(1)
            entity_types[etype] = entity_types.get(etype, 0) + 1
        for ref in re.findall(r"#(\d+)", body):
            referenced_ids.add(int(ref))

    dangling_refs = sorted(referenced_ids - defined_ids)
    step_valid = bool(has_header and balanced_parens and len(defined_ids) > 0 and len(dangling_refs) == 0)

    # Align reconstructed_3d to gt_3d via optimal similarity/rigid Procrustes (reflection-aware for intrinsic shape)
    rec = np.asarray(reconstructed_3d, dtype=np.float64)
    gt = np.asarray(gt_3d, dtype=np.float64)
    rec_c = rec - rec.mean(axis=0)
    gt_c = gt - gt.mean(axis=0)
    rec_rms = max(float(np.sqrt(np.mean(np.sum(rec_c**2, axis=1)))), 1e-12)
    gt_rms = max(float(np.sqrt(np.mean(np.sum(gt_c**2, axis=1)))), 1e-12)
    rec_n = rec_c / rec_rms
    gt_n = gt_c / gt_rms

    u_p, _, vh_p = np.linalg.svd(rec_n.T @ gt_n)
    r_orth = u_p @ vh_p
    rec_aligned = rec_n @ r_orth
    shape_pa = float(np.sqrt(np.mean(np.sum((rec_aligned - gt_n) ** 2, axis=1))))

    # Proper SO(3) Procrustes
    d_det = np.linalg.det(r_orth)
    u_so3 = u_p.copy()
    if d_det < 0:
        u_so3[:, -1] *= -1.0
    r_so3 = u_so3 @ vh_p
    proper_pa = float(np.sqrt(np.mean(np.sum((rec_n @ r_so3 - gt_n) ** 2, axis=1))))

    # Face planarity residuals (in normalized gauge units and physical mm)
    rec_mm = rec_n * float(scale_mm)
    gt_mm = gt_n * float(scale_mm)
    oriented_rec = orient_faces_outward(rec_mm, faces)
    oriented_gt = orient_faces_outward(gt_mm, faces)

    face_planarity_mm = []
    rec_normals = []
    gt_normals = []
    for idx in range(len(faces)):
        _, n_r, _, max_d_mm = fit_face_plane(rec_mm, oriented_rec[idx])
        _, n_g, _, _ = fit_face_plane(gt_mm, oriented_gt[idx])
        face_planarity_mm.append(max_d_mm)
        rec_normals.append(n_r)
        gt_normals.append(n_g)

    max_planarity_mm = float(np.max(face_planarity_mm)) if face_planarity_mm else 0.0
    mean_planarity_mm = float(np.mean(face_planarity_mm)) if face_planarity_mm else 0.0
    sewing_valid = bool(max_planarity_mm <= planarity_sewing_tol_mm)

    # Inter-face 3D dihedral angle error across all shared edges!
    # Use aligned coordinates so reflection/rotation gauge doesn't flip face pairing
    rec_al_mm = rec_aligned * float(scale_mm)
    oriented_al = orient_faces_outward(rec_al_mm, faces)
    al_normals = [fit_face_plane(rec_al_mm, f)[1] for f in oriented_al]

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for f_idx, f in enumerate(faces):
        m = len(f)
        for k in range(m):
            a, b = int(f[k]), int(f[(k + 1) % m])
            key = (min(a, b), max(a, b))
            edge_to_faces.setdefault(key, []).append(f_idx)

    dihedral_errors_deg = []
    edge_len_rec = []
    edge_len_gt = []
    for (a, b), f_list in edge_to_faces.items():
        edge_len_rec.append(float(np.linalg.norm(rec_aligned[b] - rec_aligned[a])))
        edge_len_gt.append(float(np.linalg.norm(gt_n[b] - gt_n[a])))
        if len(f_list) == 2:
            i1, i2 = f_list[0], f_list[1]
            cos_rec = float(np.clip(np.dot(al_normals[i1], al_normals[i2]), -1.0, 1.0))
            cos_gt = float(np.clip(np.dot(gt_normals[i1], gt_normals[i2]), -1.0, 1.0))
            ang_rec = math.degrees(math.acos(cos_rec))
            ang_gt = math.degrees(math.acos(cos_gt))
            dihedral_errors_deg.append(abs(ang_rec - ang_gt))

    mean_dihedral_err_deg = float(np.mean(dihedral_errors_deg)) if dihedral_errors_deg else 0.0
    max_dihedral_err_deg = float(np.max(dihedral_errors_deg)) if dihedral_errors_deg else 0.0
    rel_edge_err = float(
        np.mean(np.abs(np.array(edge_len_rec) - np.array(edge_len_gt)) / np.clip(edge_len_gt, 1e-8, None))
    ) if edge_len_gt else 0.0

    return {
        "step_syntax_and_refs_valid": step_valid,
        "watertight_manifold_solid_brep": "MANIFOLD_SOLID_BREP" in entity_types,
        "entity_count": len(defined_ids),
        "dangling_reference_count": len(dangling_refs),
        "cad_sewing_valid_010mm": sewing_valid,
        "max_face_planarity_mm": max_planarity_mm,
        "mean_face_planarity_mm": mean_planarity_mm,
        "shape_pa": shape_pa,
        "proper_pa": proper_pa,
        "mean_dihedral_angle_error_deg": mean_dihedral_err_deg,
        "max_dihedral_angle_error_deg": max_dihedral_err_deg,
        "mean_relative_edge_length_error": rel_edge_err,
    }
