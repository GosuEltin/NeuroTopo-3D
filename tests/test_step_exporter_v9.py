from __future__ import annotations

import numpy as np

from src.cad.step_exporter_v9 import (
    export_cad_to_obj,
    export_cad_to_step,
    is_watertight_2manifold,
    orient_faces_outward,
    validate_step_and_brep_quality,
)


def test_step_and_obj_exporter_watertight_chamfered_block(tmp_path) -> None:
    # Construct a 10-vertex watertight Chamfered Rectangular Block (classic CAD solid!)
    # Base z = 0: (0,0,0), (2,0,0), (2,1.2,0), (0,1.2,0)
    # Top z = 1 with 45-deg chamfer along x >= 1.4 -> z drops from 1.0 to 0.6 at x=2.0
    pts = np.array(
        [
            [0.0, 0.0, 0.0],    # 0
            [2.0, 0.0, 0.0],    # 1
            [2.0, 1.2, 0.0],    # 2
            [0.0, 1.2, 0.0],    # 3
            [0.0, 0.0, 1.0],    # 4
            [1.4, 0.0, 1.0],    # 5
            [2.0, 0.0, 0.55],   # 6
            [2.0, 1.2, 0.55],   # 7
            [1.4, 1.2, 1.0],    # 8
            [0.0, 1.2, 1.0],    # 9
        ],
        dtype=np.float64,
    )
    faces = [
        [0, 1, 2, 3],          # bottom (quad)
        [4, 5, 8, 9],          # top flat (quad)
        [5, 6, 7, 8],          # slanted chamfer face (quad)
        [0, 3, 9, 4],          # left (quad)
        [1, 2, 7, 6],          # right lower wall (quad)
        [0, 1, 6, 5, 4],       # front pentagon
        [3, 2, 7, 8, 9],       # back pentagon
    ]

    oriented = orient_faces_outward(pts, faces)
    assert is_watertight_2manifold(oriented) is True

    step_file = tmp_path / "chamfered_block.step"
    obj_file = tmp_path / "chamfered_block.obj"
    step_text = export_cad_to_step(pts, faces, output_path=step_file, part_name="CHAMFERED_BLOCK")
    obj_text = export_cad_to_obj(pts, faces, output_path=obj_file)

    assert step_file.exists() and obj_file.exists()
    assert "MANIFOLD_SOLID_BREP" in step_text
    assert "CLOSED_SHELL" in step_text
    assert obj_text.count("\nv ") + (1 if obj_text.startswith("v ") else 0) == 10

    qual = validate_step_and_brep_quality(step_text, pts, pts, faces)
    assert qual["step_syntax_and_refs_valid"] is True
    assert qual["watertight_manifold_solid_brep"] is True
    assert qual["dangling_reference_count"] == 0
    assert qual["cad_sewing_valid_010mm"] is True
    assert float(qual["max_face_planarity_mm"]) < 1e-10
    assert float(qual["shape_pa"]) < 1e-12
    assert float(qual["mean_dihedral_angle_error_deg"]) < 1e-6
