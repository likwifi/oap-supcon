# Architecture V3 completion audit and analysis

Analyzed on 7 October 2026 from the copied run artifacts and Slurm logs. Original artifacts were preserved.

## Decision

Prioritize mean-plus-max pooling for a matched 160-epoch confirmation against the control. It is the strongest completed architecture arm and adds no parameters. Its gains remain exploratory: all architecture-versus-control paired 95% intervals include zero under crossed seed/identity resampling.

Complete the two missing anatomical-refinement seeds before ranking that arm. The full candidate has no measured recognition result. Retain the preregistered raw_reliable descriptor at fusion weight 0.5, with raw_uniform as its weighting control; this screen does not establish a substantive advantage for reliability weighting.

## Completion and integrity

- Array 703: 16 of 21 tasks completed; failed tasks are 13, 14, 18, 19, 20.
- Five arms have all three seeds (control, angles, late_fusion, adaptive, pooling); parts has seed 11 only; full has none.
- All completed runs have 80 consecutive history epochs, the first best validation checkpoint, all required checkpoints and matching config/metric fingerprints.
- All 21 configs share one source hash and one dataset hash; completed runs share Python 3.12.3 and PyTorch 2.14.0+cu126.
- Source hash matches current local source: f8a66f27214560f9e00a0938d9c7f17745ffe2d2ad47c0cba1dee0417de33dc2.
- Training uses 62 identities from original identities 001–074. The same 12 disjoint identities are held out; all 792 probes belong to them. Original identities 075–124 are not used by these configs/artifacts.
- Probe order and labels match across runs. All synthetic Rank-1 rows agree with their saved probe outcomes within 1e-6.
- Regenerated runs.csv, robustness_auc.csv and clean_rank1.tex reproduce the copied files byte-for-byte. summary.csv and official_per_view.csv differ only by floating point rounding, at most 1.39e-17.

## Accuracy

Numbers are percentages averaged over available training seeds. Clean means Rank-1 against the pooled multi-view gallery with identical views excluded. Robust means the equally weighted average of 16 positive-severity clean-gallery corruption cells: four families × severities 0.1, 0.3, 0.5, 0.7. These are synthetic corruptions, not real-occlusion annotations.

| Arm | Seeds | Global clean | Global robust | Raw reliable clean | Raw reliable robust | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2-equivalent control | 3 | 83.00 | 66.04 | 86.70 | 71.22 | 1,687,570 |
| Angle inputs | 3 | 83.84 | 65.56 | 85.86 | 69.08 | 1,691,922 |
| Later fusion | 3 | 83.84 | 67.41 | 86.24 | 71.40 | 1,753,435 |
| Adaptive graph | 3 | 80.60 | 66.15 | 84.09 | 69.17 | 1,851,416 |
| Mean-plus-max pooling | 3 | 86.36 | 69.86 | 87.88 | 72.62 | 1,687,570 |
| Anatomical refinement | 1 | 88.01 | 71.08 | 86.49 | 71.59 | 3,488,897 |
| Full candidate | 0 | — | — | — | — | 4,175,004 (planned) |

The parts row is a single seed and must not be ranked as a three-seed mean. Against the control with the same seed 11, its global gains are +3.66 clean and +4.77 robust points; raw-reliable gains are only +0.13 and +1.62 points.

## Paired architecture differences

Differences and intervals are percentage points. Each candidate is paired with the same control seed and probe identity. Intervals below use 20,000 bootstrap draws, resampling seeds and identities independently as crossed clusters; the identity draw is shared across sampled seeds. This preserves the identity clustering across seeds. All intervals are pointwise and unadjusted for screening multiple arms.

| Arm | Descriptor | Seeds | Clean Δ [95% CI] | Robust Δ [95% CI] |
| --- | --- | ---: | ---: | ---: |
| Angle inputs | global | 3 | +0.84 [-2.78, +3.75] | -0.48 [-4.73, +4.03] |
| Angle inputs | raw_reliable | 3 | -0.84 [-4.42, +2.27] | -2.14 [-7.32, +2.28] |
| Later fusion | global | 3 | +0.84 [-3.16, +5.05] | +1.37 [-1.42, +4.42] |
| Later fusion | raw_reliable | 3 | -0.46 [-3.70, +2.86] | +0.17 [-2.84, +3.39] |
| Adaptive graph | global | 3 | -2.40 [-6.90, +1.77] | +0.11 [-3.35, +3.65] |
| Adaptive graph | raw_reliable | 3 | -2.61 [-7.28, +1.60] | -2.05 [-5.42, +1.24] |
| Mean-plus-max pooling | global | 3 | +3.37 [-1.39, +8.08] | +3.83 [-0.75, +8.33] |
| Mean-plus-max pooling | raw_reliable | 3 | +1.18 [-2.48, +5.56] | +1.40 [-2.56, +5.54] |
| Anatomical refinement | global | 1 | +3.66 [-0.13, +8.46] | +4.76 [-0.03, +9.32] |
| Anatomical refinement | raw_reliable | 1 | +0.13 [-4.17, +4.42] | +1.62 [-2.13, +5.19] |

Pooling global clean gains by seed 11/22/33: +0.76, +4.29, +5.05 points. Robust gains: +2.37, +3.16, +5.94. Its raw-reliable robust gains are +1.82, −1.11, +3.48, so the full retrieval-system advantage is less consistent than the global-encoder advantage.
Later fusion has positive global robustness gains in all three seeds (+2.00, +0.28, +1.84), but little average gain after raw-part matching (+0.17). Angles and adaptive graphs do not show a consistent advantage. Adaptive graphs reduce clean global accuracy in every seed.

## Raw-part matching and weighting

On the control, raw_reliable minus global yields +3.70 clean points [1.22, 5.85] and +5.19 robust points [2.84, 7.49]. Raw uniform matching yields +3.87 clean and +5.08 robust points. The reliability-minus-uniform comparison is −0.17 clean points [−0.63, 0.29] and +0.11 robust points [−0.03, 0.24]. The main supported improvement is anatomical raw-part matching, rather than confidence weighting.
On pooling, raw_reliable adds +1.52 clean points [−0.51, 3.24] and +2.76 robust points [0.53, 4.49] over its global descriptor. Raw uniform matching is almost identical: 88.01% clean and 72.54% robust, compared with 87.88% and 72.62% for raw_reliable.

## Official single-view gallery conditions

These values use the official CASIA-B single-view gallery procedure on the 12 development identities, averaging gallery views and excluding identical views. They are distinct from the pooled-gallery scores above and are not official 50-test-identity benchmark results. Paired probe archives represent pooled galleries; no single-view paired intervals were inferred from them.

| Arm | Descriptor | Seeds | NM | BG | CL |
| --- | --- | ---: | ---: | ---: | ---: |
| V2-equivalent control | global | 3 | 83.03 | 64.77 | 59.95 |
| V2-equivalent control | raw_reliable | 3 | 86.46 | 67.58 | 63.22 |
| Angle inputs | global | 3 | 82.56 | 65.11 | 59.87 |
| Angle inputs | raw_reliable | 3 | 85.96 | 66.55 | 61.44 |
| Later fusion | global | 3 | 83.31 | 68.13 | 61.65 |
| Later fusion | raw_reliable | 3 | 86.54 | 69.77 | 62.99 |
| Adaptive graph | global | 3 | 82.40 | 65.62 | 55.77 |
| Adaptive graph | raw_reliable | 3 | 86.06 | 68.66 | 60.25 |
| Mean-plus-max pooling | global | 3 | 82.92 | 67.79 | 65.43 |
| Mean-plus-max pooling | raw_reliable | 3 | 86.94 | 70.90 | 67.94 |
| Anatomical refinement | global | 1 | 84.89 | 71.21 | 62.69 |
| Anatomical refinement | raw_reliable | 1 | 86.97 | 69.96 | 64.39 |

Pooling improves raw-reliable BG by 3.32 and CL by 4.72 points, with a 0.48-point NM gain. Its global NM score is 0.11 points lower than the control even though its overall pooled clean score improves.

## Failure diagnosis and recovery

| Task IDs | Arm / seeds | Failure location | GPU slice | Recovery |
| --- | --- | --- | --- | --- |
| 13, 14 | parts / 22, 33 | First training forward, part-trunk spatial convolution | 1g.12gb; 10.75 GiB usable | Retry on at least the 2g.24gb profile that completed parts seed 11; verify forward/backward memory before the full retry |
| 18, 19, 20 | full / 11, 22, 33 | First training forward, part-trunk residual/ReLU | 2g.24gb; 21.62 GiB usable | Use a larger profile/GPU (roughly 40–48 GB or more as a starting resource target), then pilot the unchanged training settings |

All five failures occurred before any epoch completed. None has history.json or a checkpoint. Evaluation-only recovery is unavailable. Each identity batch contains 64 original clips; two augmentation views and one clean anchor produce a single forward pass of 192 sequences × 60 frames. The five local trunks retain training activations, which explains the peak memory pressure. Allocation near failure is 10.06 GiB for parts and 20.87 GiB for full, before successful completion of the forward/backward pass.
Other processes listed by the error do not by themselves establish sharing of one MIG instance: these are hardware-partitioned GPU slices. Request the appropriate GPU/MIG type through this cluster’s configured Slurm GRES. --mem changes host RAM; --exclusive does not enlarge a MIG slice. Merely reducing array concurrency is not a demonstrated solution to these per-task failures.
Inspect available resource names on the cluster with `sinfo -N -p compute -o "%N %G %f"` and `scontrol show node hopper1`; use the exact configured type rather than inventing a GRES name. Keep P=16, K=4, 60-frame training clips, augmentation/loss settings, and 80 epochs fixed on retries. A smaller training batch changes BatchNorm and batch-hard-triplet sampling; if required, treat it as a new matched experiment. Allocator tuning alone has not been demonstrated to solve these OOMs.
Failed directories can remain as historical records. The architecture analyzer handles failed attempts and detects completed arm/seed pairs. Avoid rerunning successful tasks or pooling a changed-source/config retry into the existing arm.

## Gait3D log audit

- Array 690: dataset path error (`processed/dataset-available.npz`).
- Array 693: interpreter path error (`logs/.venv/bin/python`).
- Array 696: seed 11 completed; seed 22 failed at its first validation (after epoch 4 was printed) with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH; seed 33 failed with no compatible CUDA kernel on volta2.
- Array 700: all three seeds 11, 22, 33 completed 160 epochs and wrote status=complete.
- Arrays 696 and 700 both produced a completed seed-11 run. Use the three matched array-700 runs, after checking config/source/data fingerprints, rather than selecting the better duplicate seed-11 result.
- The Gait3D result directory was not copied locally. These logs establish run completion, not descriptor scores or paired evidence; do not freeze final training from logs alone.

## Next experiments

1. Recover parts tasks 13 and 14 on a sufficiently large MIG profile; pilot full on a larger profile before tasks 18–20. Preserve the successful results.
2. After finishing the screen, confirm pooling and the V2-equivalent control with seeds 11/22/33 at the same 160-epoch budget in a separate result directory. Keep global, raw_reliable and raw_uniform descriptors with fusion weight 0.5.
3. If parts retains its gain across all seeds, add a capacity-matched control: it has about twice the parameters of V2. Do not attribute its gain solely to anatomical structure.
4. Copy the Gait3D array-700 outputs and run its descriptor analyzer on a directory containing only that matched three-seed stage. Decide the final Gait3D epoch budget after reviewing its validation histories and descriptor results.
5. Keep final CASIA-B test evaluation for the frozen design; the current screen is validation-only.

Training time varies with MIG slice and scheduling, so observed per-run train_hours are not controlled inference-latency evidence. No new GPU jobs were submitted and no training/model settings were changed.

## Reproduction and files

```bash
python3 scripts/analyze_architecture.py results_v3_dev --output audits/benchmark_v3_completed
PYTHONPATH=src python3 -m oap_supcon.cli aggregate --results-root results_v3_dev --duplicates error --output audits/benchmark_v3_completed/aggregate
```

- `analysis.json`: validated manifest, endpoint summaries and all paired comparisons.
- `architecture_summary.csv`: means and between-seed SDs.
- `per_seed.csv`: endpoint/official-condition values and run costs per seed.
- `paired_comparisons.csv`: architecture and descriptor paired intervals.
- `run_manifest.csv`: arm/task mapping, failures and checkpoint availability.
- `architecture_comparison.png` / `.svg`: architecture difference figure.
- `aggregate/`: regenerated ordinary run/summary and official-view tables.

Resource references: [NVIDIA MIG profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html), [Slurm sbatch allocation semantics](https://slurm.schedmd.com/sbatch.html), [PyTorch CUDA memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf).
