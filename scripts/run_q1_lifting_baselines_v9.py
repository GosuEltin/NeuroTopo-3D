"""Train and evaluate Modern Neural Graph & Equivariant Lifting Baselines for Q1 Journal (V9.0).

Trains 5 parameter-matched architectures from scratch across 3 seeds (128 epochs / 1,024 updates
on V831 TRAIN with confidence-weighted complete/partial gauge supervision, bfloat16 autocast on GPU):
1. EGNNLifterV9 (E(2)/Coordinate-Difference Equivariant MPNN, Satorras et al. 2021)
2. GATv2EdgeLifterV9 (Multi-Head Edge-Style-Conditioned GATv2, Brody et al. 2022)
3. GraphGPSLifterV9 (Hybrid Local Edge-Gated MPNN + Global Self-Attention, Rampasek et al. 2022)
4. LiftingModuleV6_NoSym (Pair-Biased Confidence Graph Transformer without SO(3) reflection symmetrization)
5. LiftingModuleV6_Sym (Pair-Biased Confidence Graph Transformer WITH exact SO(3) reflection symmetrization, Ours-Scratch)

Also evaluates the full warm-started checkpoints from V840 (LiftingModuleV6 b96_canonical & b512_weighted)
on V831 TRAIN (512 views), V836 FIT DEVELOPMENT (1,024 views), and V842 HELDOUT (512 views),
with Paired Wilcoxon Signed-Rank + Holm-Bonferroni corrected p-values and inference latency profiling.
"""
from __future__ import annotations

import gc
import json
import random
import sys
import time
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

from scripts.end2end_v842_contract import heldout_metadata
from scripts.evaluate_module2_v840_factorial import (
    aggregate_v840,
    row_measurements,
    weighted_reference_single,
)
from scripts.export_module2_v6_projector_payload import _model
from scripts.module2_v834_contract import TRAIN_SEEDS, model_inputs, read_json
from scripts.module2_v838_audit import full_permutation
from scripts.module2_v840_contract import (
    RECIPE,
    cohort_metadata,
    prepare_supervision_targets_v840,
)
from scripts.run_module2_v832_visibility_adaptation import inference_records
from src.lifting.cached_dataset import CachedModule2Dataset, MixedModule2Dataset
from src.lifting.lifting_module_v6 import LiftingModuleV6
from src.lifting.losses_v4 import Module2LossWeights, compute_module2_loss
from src.lifting.q1_lifting_baselines_v9 import (
    EGNNLifterV9,
    GATv2EdgeLifterV9,
    GraphGPSLifterV9,
)
from src.lifting.v4_dataset import collate_module2_samples

OUT_DIR = ROOT / "runs_q1_journal_v9" / "module2_baselines"


def build_scratch_model(arch: str, device: torch.device) -> torch.nn.Module:
    if arch == "egnn_v9":
        return EGNNLifterV9(hidden_dim=192, num_layers=6, confidence_power=0.25).to(device)
    if arch == "gatv2_edge_v9":
        return GATv2EdgeLifterV9(hidden_dim=192, num_layers=6, num_heads=8, confidence_power=0.25).to(device)
    if arch == "graphgps_v9":
        return GraphGPSLifterV9(hidden_dim=176, num_layers=6, num_heads=8, confidence_power=0.25).to(device)
    if arch == "pair_gt_nosym_v6":
        return LiftingModuleV6(
            hidden_dim=192,
            num_layers=6,
            num_heads=8,
            confidence_power=0.25,
            reflection_symmetrization=False,
        ).to(device)
    if arch == "pair_gt_sym_v6":
        return LiftingModuleV6(
            hidden_dim=192,
            num_layers=6,
            num_heads=8,
            confidence_power=0.25,
            reflection_symmetrization=True,
        ).to(device)
    raise ValueError(f"Unknown architecture: {arch}")


def measure_reflection_disagreement(
    model: torch.nn.Module, ds: CachedModule2Dataset, device: torch.device
) -> float:
    """Measure horizontal reflection SO(3) equivariance error ||f(R_x G) - diag(-1,1,-1) f(G)||."""
    model.eval()
    errs = []
    refl = torch.tensor([-1.0, 1.0, -1.0], device=device)
    with torch.no_grad():
        for start in range(0, len(ds), 64):
            batch = collate_module2_samples([ds[i] for i in range(start, min(start + 64, len(ds)))])
            inp = model_inputs(batch, device)
            out1 = model(**inp)["coordinates_3d"]
            inp_mir = dict(inp)
            pts_mir = inp["points_2d"].clone()
            pts_mir[..., 0] = -pts_mir[..., 0]
            inp_mir["points_2d"] = pts_mir
            out2 = model(**inp_mir)["coordinates_3d"] * refl
            pmask = inp["point_mask"]
            diff = torch.linalg.vector_norm(out1 - out2, dim=-1)
            for b in range(diff.shape[0]):
                errs.append(float(diff[b, pmask[b]].mean().item()))
    return float(np.mean(errs))


def measure_inference_latency_ms(
    model: torch.nn.Module, ds: CachedModule2Dataset, device: torch.device
) -> dict[str, float]:
    model.eval()
    batch1 = collate_module2_samples([ds[0]])
    inp1 = model_inputs(batch1, device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(**inp1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for idx in range(min(64, len(ds))):
            b_i = collate_module2_samples([ds[idx]])
            in_i = model_inputs(b_i, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(**in_i)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def evaluate_model_on_cohort(
    model: torch.nn.Module,
    pred_ds: CachedModule2Dataset,
    oracle_ds: CachedModule2Dataset,
    meta: dict[str, dict],
    device: torch.device,
) -> dict[str, object]:
    oracle_by_id = {r["sample_id"]: r["contract"] for r in oracle_ds.samples}
    pred_conf_by_id = {r["sample_id"]: r["graph"].node_confidence.double() for r in pred_ds.samples}
    completeness = {
        r["sample_id"]: (full_permutation(r, oracle_by_id[r["sample_id"]]) is not None)
        for r in pred_ds.samples
    }

    recs = inference_records(model, pred_ds, project=False)
    rows = []
    for rec in recs:
        sid = rec["sample_id"]
        sign = float(meta[sid]["viewer_camera_unit"][2])
        y_can = rec["target"] * torch.tensor([1.0, 1.0, sign], dtype=torch.float64)
        tmask = rec["target_mask"]
        y_w = weighted_reference_single(y_can, tmask, pred_conf_by_id[sid], confidence_power=0.25)
        m = row_measurements(rec["raw"], y_can, y_w, tmask)
        rows.append(
            {
                "sample_id": sid,
                "group": rec["geometry_group"],
                "family": rec["family"],
                "complete": completeness[sid],
                **m,
            }
        )
    agg = aggregate_v840(rows)
    return agg


def holm_bonferroni(p_vals: dict[str, float]) -> dict[str, float]:
    sorted_items = sorted(p_vals.items(), key=lambda kv: kv[1])
    m = len(sorted_items)
    adjusted = {}
    running_max = 0.0
    for idx, (k, p) in enumerate(sorted_items):
        adj = min(1.0, p * (m - idx))
        running_max = max(running_max, adj)
        adjusted[k] = running_max
    return adjusted


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Module 2 Q1 Baselines on device={device}", flush=True)

    meta_train = cohort_metadata("train")
    meta_dev = cohort_metadata("development")
    meta_ho = heldout_metadata()

    oracle_train = CachedModule2Dataset(ROOT / "runs_module1_v832/train/oracle.pt")
    oracle_dev = CachedModule2Dataset(ROOT / "runs_module1_v836/fit/oracle.pt")
    oracle_ho = CachedModule2Dataset(ROOT / "runs_module1_v842/oracle.pt")

    scratch_archs = [
        "egnn_v9",
        "gatv2_edge_v9",
        "graphgps_v9",
        "pair_gt_nosym_v6",
        "pair_gt_sym_v6",
    ]
    epochs = 128  # 1,024 updates on V831 TRAIN (64 batch size -> 8 steps/epoch)
    lr = 2.5e-4
    loss_weights = Module2LossWeights(**RECIPE["loss_weights"])

    all_results: dict[str, list[dict]] = {
        a: [] for a in scratch_archs + ["v840_b96_canonical", "v840_b512_weighted"]
    }

    for branch, seed in enumerate(TRAIN_SEEDS):
        pred_train = CachedModule2Dataset(ROOT / f"runs_module1_v832/train/branch_{branch}.pt")
        pred_dev = CachedModule2Dataset(ROOT / f"runs_module1_v836/fit/branch_{branch}.pt")
        pred_ho = CachedModule2Dataset(ROOT / f"runs_module1_v842/branch_{branch}.pt")

        # 1. Train & evaluate each scratch baseline
        for arch in scratch_archs:
            ckpt_file = OUT_DIR / f"{arch}_branch_{branch}.pt"
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = build_scratch_model(arch, device)
            param_count = sum(p.numel() for p in model.parameters())

            if ckpt_file.exists():
                state = torch.load(ckpt_file, map_location=device, weights_only=False)
                model.load_state_dict(state["model"])
                train_time_s = float(state.get("train_time_s", 0.0))
                final_train_loss = float(state.get("final_loss", 0.0))
            else:
                ds_mix = MixedModule2Dataset(oracle_train, pred_train, predicted_probability=0.75)
                loader = DataLoader(
                    ds_mix,
                    batch_size=64,
                    shuffle=True,
                    num_workers=0,
                    generator=torch.Generator().manual_seed(seed),
                    collate_fn=collate_module2_samples,
                )
                opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
                t0 = time.perf_counter()
                final_train_loss = 0.0
                for ep in range(1, epochs + 1):
                    model.train()
                    ep_losses = []
                    for batch in loader:
                        sup_targets = prepare_supervision_targets_v840(
                            batch, meta_train, gauge_mode="weighted", confidence_power=0.25
                        )
                        inp = model_inputs(batch, device)
                        tgts = {k: v.to(device) for k, v in sup_targets.items()}
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                            pred_out = model(**inp)
                            loss, _ = compute_module2_loss(pred_out, tgts, weights=loss_weights)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Non-finite loss in {arch} branch={branch} ep={ep}")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                        ep_losses.append(float(loss.detach().item()))
                    sched.step()
                    final_train_loss = fmean(ep_losses)
                train_time_s = time.perf_counter() - t0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "arch": arch,
                        "branch": branch,
                        "seed": seed,
                        "parameters": param_count,
                        "train_time_s": train_time_s,
                        "final_loss": final_train_loss,
                    },
                    ckpt_file,
                )

            model.eval()
            ev_tr = evaluate_model_on_cohort(model, pred_train, oracle_train, meta_train, device)
            ev_dev = evaluate_model_on_cohort(model, pred_dev, oracle_dev, meta_dev, device)
            ev_ho = evaluate_model_on_cohort(model, pred_ho, oracle_ho, meta_ho, device)
            refl_err = measure_reflection_disagreement(model, pred_ho, device)
            lat = measure_inference_latency_ms(model, pred_ho, device)

            print(
                f"[Branch {branch}] {arch:18s} ({param_count/1e6:.2f}M) | "
                f"Dev Macro PA={ev_dev['macro']['shape_pa']:.4f} | "
                f"HO Macro PA={ev_ho['macro']['shape_pa']:.4f} (Proper={ev_ho['macro']['proper_pa']:.4f}) | "
                f"ReflErr={refl_err:.4e} | Lat={lat['median_ms']:.2f}ms",
                flush=True,
            )
            all_results[arch].append(
                {
                    "branch": branch,
                    "seed": seed,
                    "parameters": param_count,
                    "train_time_s": train_time_s,
                    "final_train_loss": final_train_loss,
                    "reflection_equivariance_error": refl_err,
                    "latency_gpu_ms": lat,
                    "train": ev_tr,
                    "development": ev_dev,
                    "heldout": ev_ho,
                }
            )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # 2. Evaluate frozen V840 checkpoints (b96_canonical & b512_weighted) on exact same pipeline
        for v840_arm in ("b96_canonical", "b512_weighted"):
            key = f"v840_{v840_arm}"
            ckpt_p = ROOT / f"runs_module2_v840/branch_{branch}/{v840_arm}/final.pt"
            ckpt_obj = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            m_v840 = _model(ckpt_obj, device)
            m_v840.eval()
            p_cnt = sum(p.numel() for p in m_v840.parameters())
            ev_tr = evaluate_model_on_cohort(m_v840, pred_train, oracle_train, meta_train, device)
            ev_dev = evaluate_model_on_cohort(m_v840, pred_dev, oracle_dev, meta_dev, device)
            ev_ho = evaluate_model_on_cohort(m_v840, pred_ho, oracle_ho, meta_ho, device)
            refl_err = measure_reflection_disagreement(m_v840, pred_ho, device)
            lat = measure_inference_latency_ms(m_v840, pred_ho, device)
            print(
                f"[Branch {branch}] {key:18s} ({p_cnt/1e6:.2f}M) | "
                f"Dev Macro PA={ev_dev['macro']['shape_pa']:.4f} | "
                f"HO Macro PA={ev_ho['macro']['shape_pa']:.4f} (Proper={ev_ho['macro']['proper_pa']:.4f}) | "
                f"ReflErr={refl_err:.4e} | Lat={lat['median_ms']:.2f}ms",
                flush=True,
            )
            all_results[key].append(
                {
                    "branch": branch,
                    "seed": seed,
                    "parameters": p_cnt,
                    "reflection_equivariance_error": refl_err,
                    "latency_gpu_ms": lat,
                    "train": ev_tr,
                    "development": ev_dev,
                    "heldout": ev_ho,
                }
            )
            del m_v840
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Summarize 3-seed mean +/- std and Paired Wilcoxon + Holm-Bonferroni tests on Development & Heldout!
    summary_table = {}
    for arch, br_list in all_results.items():
        summary_table[arch] = {
            "parameters": br_list[0]["parameters"],
            "reflection_equivariance_error_mean": float(np.mean([b["reflection_equivariance_error"] for b in br_list])),
            "latency_median_ms": float(np.mean([b["latency_gpu_ms"]["median_ms"] for b in br_list])),
        }
        for split in ("train", "development", "heldout"):
            macro_pas = [b[split]["macro"]["shape_pa"] for b in br_list]
            proper_pas = [b[split]["macro"]["proper_pa"] for b in br_list]
            z_shares = [b[split]["macro"]["z_mse_share"] for b in br_list]
            fams = {}
            for fam in ("tetra_v2", "cube_v2", "prism_v2", "polyhedra_v2"):
                fpas = [b[split]["family"][fam]["shape_pa"] for b in br_list]
                fams[fam] = {"mean": float(np.mean(fpas)), "std": float(np.std(fpas, ddof=1))}
            summary_table[arch][split] = {
                "macro_shape_pa_mean": float(np.mean(macro_pas)),
                "macro_shape_pa_std": float(np.std(macro_pas, ddof=1)),
                "macro_proper_pa_mean": float(np.mean(proper_pas)),
                "macro_proper_pa_std": float(np.std(proper_pas, ddof=1)),
                "z_mse_share_mean": float(np.mean(z_shares)),
                "family_shape_pa": fams,
            }

    # Paired Wilcoxon signed-rank tests (seed-averaged per geometry group)
    # 1) Controlled scratch comparison: pair_gt_sym_v6 vs {egnn_v9, gatv2_edge_v9, graphgps_v9, pair_gt_nosym_v6}
    # 2) Full V840 comparison: v840_b512_weighted vs all baselines
    wilcoxon_tests = {}
    for split in ("development", "heldout"):
        group_scores: dict[str, np.ndarray] = {}
        for arch, br_list in all_results.items():
            # Average group shape_pa across the 3 seeds
            per_seed = []
            for b in br_list:
                dets = sorted(b[split]["details"], key=lambda d: d["group"])
                per_seed.append([d["shape_pa"] for d in dets])
            group_scores[arch] = np.mean(np.array(per_seed), axis=0)

        # Scratch contrasts against pair_gt_sym_v6
        raw_p_scratch = {}
        stats_scratch = {}
        ref_s = group_scores["pair_gt_sym_v6"]
        for comp in ("egnn_v9", "gatv2_edge_v9", "graphgps_v9", "pair_gt_nosym_v6"):
            diff = ref_s - group_scores[comp]
            stat, pval = wilcoxon(diff, alternative="less")
            raw_p_scratch[comp] = float(pval)
            stats_scratch[comp] = {
                "mean_delta_pa": float(np.mean(diff)),
                "median_delta_pa": float(np.median(diff)),
                "wilcoxon_stat": float(stat),
                "p_raw": float(pval),
            }
        adj_scratch = holm_bonferroni(raw_p_scratch)
        for comp in stats_scratch:
            stats_scratch[comp]["p_holm_bonferroni"] = adj_scratch[comp]

        # Full system contrasts against v840_b512_weighted
        raw_p_full = {}
        stats_full = {}
        ref_f = group_scores["v840_b512_weighted"]
        for comp in ("egnn_v9", "gatv2_edge_v9", "graphgps_v9", "pair_gt_nosym_v6", "pair_gt_sym_v6", "v840_b96_canonical"):
            diff = ref_f - group_scores[comp]
            stat, pval = wilcoxon(diff, alternative="less")
            raw_p_full[comp] = float(pval)
            stats_full[comp] = {
                "mean_delta_pa": float(np.mean(diff)),
                "median_delta_pa": float(np.median(diff)),
                "wilcoxon_stat": float(stat),
                "p_raw": float(pval),
            }
        adj_full = holm_bonferroni(raw_p_full)
        for comp in stats_full:
            stats_full[comp]["p_holm_bonferroni"] = adj_full[comp]

        wilcoxon_tests[split] = {
            "scratch_vs_pair_gt_sym_v6": stats_scratch,
            "full_vs_v840_b512_weighted": stats_full,
        }

    # Strip bulky per-group details before writing compact JSON report
    compact_branches = {}
    for arch, br_list in all_results.items():
        compact_branches[arch] = []
        for b in br_list:
            b_copy = dict(b)
            for sp in ("train", "development", "heldout"):
                sp_copy = dict(b[sp])
                sp_copy.pop("details", None)
                b_copy[sp] = sp_copy
            compact_branches[arch].append(b_copy)

    report = {
        "status": "PASS",
        "experiment": "Q1_Module2_Modern_Equivariant_and_Graph_Baselines_3Seeds",
        "summary_table": summary_table,
        "wilcoxon_holm_bonferroni": wilcoxon_tests,
        "branches": compact_branches,
    }
    out_json = ROOT / "runs_q1_journal_v9" / "module2_modern_baselines_3seeds.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved Module 2 baselines report to {out_json}", flush=True)


if __name__ == "__main__":
    main()
