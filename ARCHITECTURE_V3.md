# Architecture V3: improvement plan and runnable development code

The new `part_adaptive` backbone is an opt-in coordinate model motivated by the
literature comparison. Recognition improvement remains an experimental question.
The existing V2 models, checkpoints and result artifacts retain their interfaces.
New architecture runs write to `results_v3_dev/`.

## Implemented candidate

| Change | Intended benefit | Implementation |
|---|---|---|
| Joint-angle cosine input | Supply articulated geometry independently of translation, rotation and scale | A fourth stream; missing or degenerate endpoints have zero confidence |
| Graph processing before fusion | Allow each input stream to learn spatial/motion features independently | One graph/temporal block per stream before concatenation |
| Per-frame adaptive graph | Model relations that vary with the observed pose | Anatomical partitions plus feature-derived attention with a source-confidence prior |
| Anatomical refinement | Learn local features before retrieval pooling | Five region-restricted graph branches, merged into the global feature map |
| Masked mean-plus-max pooling | Retain distinctive activations that a sequence average can dilute | Confidence-weighted mean combined with a presence-masked maximum |

All local branches receive gradients from global CE and triplet through their
fusion into the global feature map. The development screen uses `metric_baseline`
with no auxiliary part metric loss. It reports `global`, `raw_uniform` and
`raw_reliable` matching from the same checkpoint.

The model is implemented in `src/oap_supcon/adaptive_model.py`. It reuses the V2
heads and retrieval interface. Its disabled control has the same state-dict keys,
initialization and outputs as V2 at a fixed seed. A dedicated regression test
checks this and loads a V2 state dict strictly.

Missing sources receive zero graph attention, empty neighbourhoods are finite,
and padding is excluded using `frame_mask`. This candidate retains the observed
joint mask: it does not infer persistent features at unobserved joints.
Source confidence is a prior, not a learned estimate of recognition utility.
The maximum-pooling branch may be more sensitive to outliers, which is why it
has an independent control.

With 62 development identities, 64 hidden channels and 256 embedding dimensions:

| Configuration | Parameters |
|---|---:|
| V2-equivalent control | 1,687,570 |
| Angle input only | 1,691,922 |
| Later fusion only | 1,753,435 |
| Adaptive graph only | 1,851,416 |
| Anatomical refinement only | 3,488,897 |
| Mean-plus-max pooling only | 1,687,570 |
| Full candidate | 4,175,004 |

Parameter differences are part of these interventions. They do not isolate the
benefit of architecture at identical capacity. If a larger arm wins, add a
capacity-matched V2 control before attributing its gain to anatomical structure.

## Stage 1: establish a stronger external reference

The first empirical priority remains the existing official GPGait and GPGait++
implementations in `external/FastPoseGait/`. The new preparation script exports
the canonical CASIA-B NPZ into their pickle layout and creates development
configs. It preserves raw coordinates and confidence before OAP normalization
or masking, and keeps the official encoder, losses, optimizer, scheduler and
pose transformations.

Run preparation on the GPU host so generated absolute paths match that host:

```bash
python3 scripts/prepare_fastposegait.py \
  --output external_runs/fastposegait_v3_dev --gpus 1 --dry-run
python3 scripts/prepare_fastposegait.py \
  --output external_runs/fastposegait_v3_dev --gpus 1
```

The script selects the existing fixed 12 validation identities from the original
74 training identities. It exports 62 training identities and those 12 held-out
identities; identities 075–124 are excluded. The official framework calls its
evaluation partition `TEST_SET`; that key contains validation identities here.
`trainer_cfg.with_test` is disabled, and a separately invoked evaluation uses
the predetermined checkpoint. The development run is explicitly different
from a 74/50 published-score reproduction.

For the one-GPU configuration, after preparing data and the official framework's
own dependencies:

```bash
export OAP_REFERENCE_ROOT="$PWD/external_runs/fastposegait_v3_dev"
cd external/FastPoseGait
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 fastposegait/main.py \
  --cfgs "$OAP_REFERENCE_ROOT/gpgait++.yaml" --phase train --log_to_file
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 fastposegait/main.py \
  --cfgs "$OAP_REFERENCE_ROOT/gpgait++.yaml" --phase test --log_to_file
```

Use `gpgait.yaml` for the original GPGait comparison. Official results stay in
the external framework's output tree. The exporter writes template and source
hashes, partition identities, frame counts and a source-data checksum to a
manifest, and refuses to overwrite an existing preparation directory.

The exporter has passed raw-coordinate round-trip and partition tests. Its real
CASIA-B dry run covers 8,140 sequences with no official test sequences exported.
The official CUDA training framework has not run on this CPU host. Its published
accuracy has therefore not been reproduced. Transferring OAP raw-part matching
to that framework remains a subsequent experiment after obtaining the reference
checkpoints and reconciling its descriptor/evaluator interface.

## Stage 2: run the controlled architecture screen

All seven arms share the data, mixed-partial augmentation, metric objective,
optimizer, training clip length, descriptor weights and validation selection
rule from Benchmark V2. Individual recipes enable one addition at a time;
`benchmark_v3` enables the full candidate. Architecture switches are recorded
in the saved configuration and fingerprint.

The array has 21 tasks: seven arms × seeds 11, 22 and 33. Its initial fixed budget
is 80 epochs per arm, with three concurrent GPU jobs. A short pilot of the full
candidate can first check GPU memory and runtime; pilot accuracy is not used
to select a model.

The job defaults match the current cloud cluster: partition `compute`, one GPU
per task, and `volta1` excluded. Set `OAP_PYTHON` to the cloud environment's Python
before submitting (for example, `export OAP_PYTHON="$PWD/.venv/bin/python"`).

```bash
# Inspect the full candidate's seed-11 command without training.
python3 scripts/run_architecture_array.py --task-id 18 --dry-run

# Optional resource pilot on the GPU host.
python3 scripts/run_architecture_array.py --task-id 18 --epochs 10

# Run the prespecified 80-epoch development matrix.
sbatch slurm/17_architecture_v3_dev.slurm

# Rebuild summaries after completion.
PYTHONPATH=src python3 -m oap_supcon.cli aggregate \
  --results-root results_v3_dev --duplicates error
```

A direct full-candidate run uses:

```bash
PYTHONPATH=src python3 -m oap_supcon.cli run \
  --dataset casia_b_pose --preset benchmark_v3 --method metric_baseline \
  --seed 11 --epochs 80 --device cuda
```

The screen has no final-test option. Recipe defaults retain validation-only
evaluation and the established 12-identity split. Compare clean NM/BG/CL and
each corruption family, rather than selecting from one favorable corruption
cell. The common checkpoint-selection endpoint remains global matching with
0.75 clean and 0.25 fixed-corruption validation weight; raw-part results are
reported separately. This leaves the existing global-selection limitation
explicit instead of silently changing the experiment.

Confirm promising arms at the same 160-epoch budget as their control before
freezing the final recipe. A persuasive improvement needs a consistent paired
validation gain across seeds, acceptable clean performance, and a measured cost
in runtime/memory. Increase the identity/dataset evidence rather than repeatedly
tuning on the 50 CASIA-B test identities, which earlier project work has already
exposed.

## Stage 3: validate real occlusion and richer representations

After an encoder gain is established, stage the official SUSTech1K normal-gallery
and occlusion-probe protocol. A heatmap branch requires original pose-estimator
heatmaps; Gaussian renderings of stored coordinates do not reconstruct their
uncertainty. Keep estimator and representation changes separate from encoder
changes. The current candidate supports 2D and 3D coordinates and has 3D software
tests, but recognition evidence remains based on the available 2D poses.

Pretraining, root-motion features, heatmap inputs and latent pose completion
remain later stages because each changes the data or supervision available.
They are not prerequisites for testing this coordinate architecture.

## Software verification

On 5 October 2026, **124 tests passed**. The full 4.18M-parameter development
candidate completed three optimizer steps on eight real CASIA-B training
sequences with finite losses and gradients. `audits/benchmark_v3/verification.json`
checks the tested source fingerprint against the integration record. The array
dry run, shell syntax and Python compilation checks also passed. No GPU jobs
were submitted and no recognition improvement has been measured.

Run the regression suite and the real-training-data integration check:

```bash
OMP_NUM_THREADS=2 python3 -m pytest -q
PYTHONPATH=src OMP_NUM_THREADS=2 python3 scripts/verify_benchmark_recipe.py \
  --preset benchmark_v3 --method metric_baseline --device cpu --steps 3
```

Verification includes V2 checkpoint/output compatibility, absent/padded evidence
invariance, angle geometry in 2D and 3D, empty graph neighbourhoods, CPU BF16
backpropagation, global-loss gradients in every anatomical branch, checkpoint
reload and end-to-end validation with guards against test-set access. The real
integration check uses only training identities and records its settings and
finite-gradient evidence in `audits/benchmark_v3/integration.json`. These checks
establish executable code, not improved recognition accuracy or CUDA performance.
