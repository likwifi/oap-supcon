# Next experiment gate after `benchmark_v2_factorial`

The completed 18-run factorial should **not** be repeated. It already establishes
that the tested part-SupCon variants do not outperform `metric_baseline` on the
held-out CASIA-B development split.

## P0 — checkpoint-only descriptor controls (complete; no training)

All three baseline checkpoints completed successfully. At the preregistered
0.5 fusion weight, reliability-weighted raw parts improved clean Rank-1 from
81.52% to 84.47% and the positive-severity clean-gallery aggregate from 63.90%
to 68.23%. Hierarchical paired 95% intervals for the gains are [1.47, 4.67]
and [3.25, 5.35] percentage points, respectively. All four corruption families
improved and all ten fresh random-head seeds passed the numerical gate. Uniform
raw matching reached 84.43% and 68.11%, showing that reliability weighting has
only a small incremental effect. The P0 decision is `advance_to_p1`.

Freeze `raw_reliable` with fusion weight 0.5 as the primary development
candidate. Keep weight 0.75 as sensitivity only; do not select it after viewing
the validation results.

Use the three saved `metric_baseline` `checkpoint.pt` files and keep the backbone
fixed. Re-evaluate the same probe masks with:

1. `global` (existing reference);
2. normalized raw confidence-weighted part pools, with no projection head;
3. raw part pools with uniform rather than reliability-weighted matching;
4. the existing frozen part heads;
5. at least 10 freshly initialized frozen part-head seeds per backbone
   checkpoint; and
6. fusion weights 0.0, 0.25, 0.5, 0.75, and 1.0, reported as a sensitivity
   analysis rather than selected on the same validation identities.

This experiment isolates anatomical pooling, reliability weighting, random
projection, and score fusion. It is the minimum required control because the
baseline part heads receive no direct training gradient.

### P0 runbook

The control runner performs inference from the three saved baseline checkpoints;
it does not repeat training. On the cluster, from the repository root:

```bash
export OAP_PROJECT_ROOT="$PWD"
export OAP_PYTHON="$PWD/.venv/bin/python"
find results_v2_factorial -path '*metric_baseline*' -name checkpoint.pt | wc -l
sbatch slurm/15_descriptor_controls.slurm
```

The checkpoint count must be `3`. The SLURM file uses tasks `0-2`, the `compute`
partition, one GPU per task, and excludes `volta1`. Completed tasks are safe to
resubmit: the runner detects and reuses a complete output rather than replacing
it. An incomplete output is never overwritten and must be moved aside before a
retry.

Monitor and validate the array with:

```bash
squeue -u "$USER" -o "%.18i %.10P %.24j %.2t %.10M %R"
find results_descriptor_controls -name metrics.json | wc -l
grep -R "Traceback\|FAILED\|Error" logs/casia_p0_controls_*.out
```

After all three `metrics.json` files exist, run:

```bash
"$OAP_PYTHON" scripts/analyze_descriptor_controls.py results_descriptor_controls
```

This writes `results_descriptor_controls/summary.csv` and
`results_descriptor_controls/analysis.json`. The latter reports the preregistered
0.5-fusion acceptance gate, hierarchical paired intervals, all fusion-weight
sensitivities, and the ten-random-head stability range.

### Acceptance gate

Continue with a deterministic part descriptor only if it:

- keeps clean Rank-1 within 0.5 percentage points of the global descriptor;
- improves the positive-severity clean-gallery aggregate by at least 2 points;
- improves at least three of the four corruption families; and
- does not depend on a favorable random projection seed.

If only the random heads improve performance, report the finding as exploratory
and do not promote the descriptor as the proposed method.

## P1 — frozen-backbone head training (cheap GPU run)

If P0 supports part matching, freeze each metric-baseline backbone and train only
the local heads under three alternatives:

1. global-to-part stop-gradient distillation;
2. low-weight part triplet with an explicit variance floor; and
3. no local loss (frozen deterministic control).

Use seeds 11, 22, and 33. This is a head-only experiment, not a repetition of
the 160-epoch backbone training. Reject any head whose final part variance falls
below `1e-4` or whose clean accuracy violates the P0 gate.

The implementation freezes every non-head parameter and all BatchNorm state.
It starts from each selected baseline checkpoint and runs two preregistered
40-epoch head conditions: global-to-part stop-gradient distillation and part
triplet with a normalized `1e-4` variance floor. Submit all six runs with:

```bash
export OAP_PROJECT_ROOT="$PWD"
export OAP_PYTHON="$PWD/.venv/bin/python"
sbatch slurm/16_head_controls.slurm
```

The array is `0-5%3`. After all six tasks complete:

```bash
find results_head_controls -name metrics.json | wc -l
"$OAP_PYTHON" scripts/analyze_head_controls.py results_head_controls \
  --p0-results results_descriptor_controls
```

The count must be `6`. A trained head replaces the raw descriptor only if it
retains the variance floor and global clean safety gate, stays within 0.5 points
of the raw descriptor's clean result, and its paired robust interval against
`raw_reliable` has a nonnegative lower bound. Otherwise retain the parameter-
free raw descriptor and do not launch P2.

## P2 — one redesigned end-to-end candidate

Launch P2 only if a P1 trained head satisfies the replacement rule above. In
that case, train one candidate end to end with three development seeds, keep
`metric_baseline` unchanged as the reference, and add parameter-level gradient
diagnostics because activation-level clean-CE cosines are structurally zero in
the current logging scheme. If P1 retains the raw descriptor, skip P2: the raw
method has no learned local head to train end to end and should advance directly
to final validation.

Do not launch a new broad loss-weight sweep. The development candidate advances
only if it passes the same clean/robustness gate and its paired identity-level
bootstrap interval is not centered on a degradation.

## P3 — final evidence after the design is frozen

1. Train `metric_baseline` and the single frozen candidate on all CASIA-B
   training identities and evaluate identities 075--124. Fix the epoch count in
   advance; the median selected baseline epoch in the current factorial is 85.
2. Run at least five seeds for the two primary CASIA-B comparisons.
3. Add a real-occlusion endpoint, preferably SUSTech1K OCC or another dataset
   with an explicit occlusion protocol.
4. Add one large cross-view or in-the-wild pose-gait dataset.
5. Reproduce at least one established external pose-gait backbone and repeat the
   primary descriptor comparison on it.
6. Measure inference latency, gallery storage, and scoring throughput for both
   global and part-aware descriptors.

## Release requirements

- Initialize the canonical source as a Git repository and tag the exact code
  used for final runs. The present artifacts record `not-a-git-checkout`.
- Lock the Python/PyTorch environment.
- Preserve `config.yaml`, `history.json`, `metrics.json`, and
  `probe_outcomes.npz` for every reported run.
- Regenerate manuscript evidence with:

  ```bash
  python3 scripts/analyze_factorial.py results_v2_factorial \
    --output Paper/generated --bootstrap-samples 20000
  ```
