# NeuroTopo-3D — Official Reproducibility & CAD B-Rep Benchmark Repository

**Paper Title:** *NeuroTopo-3D: Spectral Normal-Flow Manifold Projection for Single-View Polyhedral and CAD B-Rep Reconstruction*  
**Authors:** Tin Nguyen Anh (ORCID: [0009-0009-4454-9541](https://orcid.org/0009-0009-4454-9541)) and Thuyen Phan Thi Le  
**Affiliation:** FPT University, Quy Nhon AI Campus, Gia Lai, Vietnam  
**Repository:** [https://github.com/GosuEltin/NeuroTopo-3D](https://github.com/GosuEltin/NeuroTopo-3D)

---

## Overview

This repository contains the reference implementation, symbolic/spectral theorem verification suite, classical non-neural baselines (`Marill-MSDA 1991` and `Lipson–Shpitalni 1996`), multi-seed experimental logs, and exported **ISO 10303-21 (`.STEP` AP214 `MANIFOLD_SOLID_BREP`)** and **Wavefront (`.OBJ`)** 3D CAD models for **NeuroTopo-3D**.

### Relationship to Prior 2D Conference Work
Our earlier 2D planar diagram benchmark paper (*"NeuroTopo: A Benchmark for Neuro-Symbolic Topological Reasoning on Geometric Diagrams"*, IEEE ATiGB 2026, pp. 421–431, DOI: [10.1109/ATiGB70203.2026.11628217](https://doi.org/10.1109/ATiGB70203.2026.11628217)) addressed 2D image-plane keypoint refinement without 3D depth recovery. **NeuroTopo-3D** extends that line into full 2D-to-3D polyhedral lifting, Dual-Gauge Spectral Rigidity Theory, Iterative Normal-Flow Row-Space Levenberg–Marquardt (`NF-RSLM`), and watertight ISO 10303-21 `.STEP` CAD B-Rep reverse engineering.

### Key Modules Included
- `src/reasoning/normal_flow_projector_v9.py`: Iterative Normal-Flow Row-Space Levenberg–Marquardt (`NF-RSLM`) with multi-branch Necker reflection continuation and target-free solid-over-dashed ray-crossing disambiguation.
- `src/reasoning/classical_line_drawing_msda_v9.py`: Non-neural classical single-view polyhedral & CAD sketch reconstruction baselines (`Marill-MSDA 1991` and `Lipson–Shpitalni 1996`).
- `src/lifting/q1_lifting_baselines_v9.py`: Permutation-equivariant and occlusion-asymmetric neural 2D-to-3D lifters (`PairGT-Asym`, `PairGT-Sym`, `EGNN`, `GATv2-Edge`, `GraphGPS`).
- `src/cad/step_exporter_v9.py`: Outward 2-manifold B-Rep orientation (half-edge BFS + exact Newell signed-volume test), ISO 10303-21 `.STEP` solid synthesis, CAD kernel sewing tolerance auditor, and Parametric Nullspace CAD Family Explorer.
- `scripts/verify_q1_theorems_v9.py`: Automated `SymPy` symbolic computer algebra and IEEE-754 `float64` verification of Theorems 1–3, Lemma 1, and Proposition 1.
- `scripts/verify_q1_journal_claims_v9.py`: End-to-end 5-gate reproducibility and numerical claim verifier.
- `runs_q1_journal_v9/`: All 10 JSON benchmark reports and 17 exported `.STEP` / `.OBJ` CAD B-Rep solids (`runs_q1_journal_v9/cad_exports/`).

## Quick Verification

```bash
pytest tests/test_q1_theorems_v9.py tests/test_normal_flow_projector_v9.py tests/test_step_exporter_v9.py tests/test_classical_msda_v9.py
python scripts/verify_q1_theorems_v9.py
python scripts/verify_q1_journal_claims_v9.py
```

## License
MIT License.
